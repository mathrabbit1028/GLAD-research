#!/usr/bin/env python3
"""Posthoc saved-model audit of prior adaptation and component-ratio quality.

No model is trained or selected. Each prior comparison holds the final mixture
weights, generator noise, graph sizes, and classifier parameters fixed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ['no_ae', 'initial_ae', 'fixed_prior', 'level_c']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--modes', choices=MODES, nargs='+', default=MODES)
    parser.add_argument('--pilots-per-component', type=int, default=64)
    parser.add_argument('--ratio-per-component', type=int, default=16)
    args = parser.parse_args()
    if min(args.pilots_per_component, args.ratio_per_component) < 1:
        raise ValueError('Pilot counts must be positive')
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '1'
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch_geometric.data import Data
    from src.utils.config import _wrap
    from src.utils.device import require_cuda
    from src.models.factory import create_model
    from scripts.diagnose_prior_mixture_gap import attributes
    torch.set_num_threads(1)
    device = require_cuda('cuda:0')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def graph_list(batch):
        x, adjacency, mask = [value.detach().cpu() for value in batch]
        return [Data(x=features[:int(valid.sum())].clone(),
                     edge_index=(edges[:int(valid.sum()), :int(valid.sum())] > .5).nonzero().T.contiguous(),
                     num_nodes=int(valid.sum()))
                for features, edges, valid in zip(x, adjacency, mask)]

    def digest(batch):
        result = hashlib.sha256()
        for value in batch:
            value = value.detach().cpu().contiguous()
            result.update(str((tuple(value.shape), str(value.dtype))).encode())
            result.update(value.numpy().tobytes())
        return result.hexdigest()

    def weighted_moments(values, wn, wa, components):
        value = np.asarray(values, dtype=float)
        means = value.reshape(components, -1).mean(1)
        return dict(normal=float(value @ wn), abnormal=float(value @ wa), uniform=float(value.mean()),
                    component_mean_range=[float(means.min()), float(means.max())])

    def node_weighted_categories(batch, weights):
        x, _, mask = batch
        weights = torch.as_tensor(weights, device=x.device, dtype=x.dtype)
        numerator = (x*mask.unsqueeze(-1)*weights[:, None, None]).sum((0, 1))
        denominator = (mask*weights[:, None]).sum()
        return (numerator/denominator).detach().cpu().tolist()

    def ratio_metrics(model, batch):
        logits = torch.cat([model.ratio(tuple(value[offset:offset+128] for value in batch))
                            for offset in range(0, len(batch[0]), 128)])
        components = model.bank.components
        target = torch.arange(components, device=device).repeat_interleave(args.ratio_per_component)
        if len(logits) != len(target):
            raise ValueError('Component-major sample count differs from ratio targets')
        results = {}
        for name, scores in [('uncalibrated', logits), ('temperature_scaled', logits/model.temperature)]:
            log_prob = F.log_softmax(scores, -1)
            probability = log_prob.exp()
            correct = scores.argmax(-1) == target
            nll = float(F.nll_loss(log_prob, target))
            results[name] = dict(accuracy=float(correct.float().mean()), nll=nll,
                nll_minus_uniform=nll-math.log(components),
                mean_max_probability=float(probability.max(-1).values.mean()),
                mean_entropy=float(-(probability*log_prob).sum(-1).mean()),
                component_accuracy=correct.reshape(components, args.ratio_per_component).float().mean(1).cpu().tolist())
        return results

    for seed in args.seeds:
        for mode in args.modes:
            output = args.root.resolve()/f'seed{seed}'/mode
            manifest = json.loads((output/'manifest.json').read_text())
            if manifest['status'] != 'completed':
                raise ValueError(f'Cannot audit incomplete run: {output}')
            start = time.monotonic()
            checkpoint_path = output/'checkpoint.pt'
            saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
            cfg = _wrap(saved['model_config'])
            model = create_model(saved['input_dim'], cfg, device)
            model.load_state_dict(saved['model_state'])
            components = model.bank.components
            m = args.pilots_per_component
            wn, wa = [value.detach().cpu().numpy().repeat(m)/m for value in model.weights]
            structural_keys = ['nodes', 'mean_degree', 'connected', 'max_degree', 'triangle_count',
                               'isolated_fraction', 'degree_over4_fraction', 'label_homophily']
            structural_keys += [f'category_{index}' for index in range(saved['input_dim'])]
            prior_states = {'raw':model.raw_eta, 'initial_ae':model.initial_eta, 'selected':model.eta}
            populations, hard_batches = {}, {}
            with torch.no_grad():
                real_attributes = [attributes(graph) for graph in graph_list(model.fit_data)]
                real_fit = {key:float(np.mean([row[key] for row in real_attributes])) for key in structural_keys}
                real_fit['node_weighted_category_frequencies'] = node_weighted_categories(
                    model.fit_data, np.ones(len(model.fit_data[0]))/len(model.fit_data[0]))
                for name, eta in prior_states.items():
                    soft = model.bank.sample(eta, m, 84000)
                    hard = model.bank.sample(eta, m, 84000, hard=True)
                    hard_batches[name] = tuple(value.detach().cpu() for value in hard)
                    # Graph moments use graph-balanced mixture weights; category
                    # counts additionally expose the node-weighted statistic.
                    attrs = [attributes(graph) for graph in graph_list(hard)]
                    hard_summary = {key:weighted_moments([row[key] for row in attrs], wn, wa, components)
                                    for key in structural_keys}
                    x, adjacency, mask = soft
                    count = mask.sum(1)
                    soft_degree = (adjacency.sum((1, 2))/count).cpu().numpy()
                    per_graph_categories = (x.sum(1)/count[:, None]).cpu().numpy()
                    soft_summary = {'mean_degree':weighted_moments(soft_degree, wn, wa, components)}
                    soft_summary.update({f'category_{index}':weighted_moments(per_graph_categories[:, index], wn, wa, components)
                                         for index in range(saved['input_dim'])})
                    for representation, batch, result in [('soft', soft, soft_summary), ('hard', hard, hard_summary)]:
                        result['node_weighted_category_frequencies'] = dict(
                            normal=node_weighted_categories(batch, wn), abnormal=node_weighted_categories(batch, wa),
                            uniform=node_weighted_categories(batch, np.ones(len(wn))/len(wn)))
                    ratio_audit = {}
                    for representation, is_hard in [('soft', False), ('hard', True)]:
                        ratio_batch = model.bank.sample(eta, args.ratio_per_component, 84002, hard=is_hard)
                        ratio_audit[representation] = ratio_metrics(model, ratio_batch)
                        del ratio_batch
                    populations[name] = dict(eta={key:value.detach().cpu().tolist() for key, value in eta.items()},
                        soft=soft_summary, hard=hard_summary, soft_digest=digest(soft), hard_digest=digest(hard),
                        final_component_classifier=ratio_audit)
                    print('LEVEL_C_POSTHOC '+json.dumps(dict(seed=seed, mode=mode, prior=name,
                        normal_soft_degree=soft_summary['mean_degree']['normal'],
                        normal_hard_connected=hard_summary['connected']['normal'],
                        ratio_accuracy=ratio_audit['soft']['temperature_scaled']['accuracy'],
                        ratio_nll=ratio_audit['soft']['temperature_scaled']['nll'])), flush=True)
                    del soft, hard
            differences = {}
            for baseline in ['initial_ae', 'raw']:
                comparison = {}
                for representation in ['soft', 'hard']:
                    selected, previous = populations['selected'][representation], populations[baseline][representation]
                    comparison[representation] = {key:{mixture:selected[key][mixture]-previous[key][mixture]
                        for mixture in ['normal', 'abnormal', 'uniform']}
                        for key in selected if key != 'node_weighted_category_frequencies'}
                    selected_categories = selected['node_weighted_category_frequencies']
                    previous_categories = previous['node_weighted_category_frequencies']
                    comparison[representation]['node_category_tv'] = {mixture:float(np.abs(
                        np.asarray(selected_categories[mixture])-np.asarray(previous_categories[mixture])).sum()/2)
                        for mixture in ['normal', 'abnormal', 'uniform']}
                xs, adjacency_s, mask_s = hard_batches['selected']
                xb, adjacency_b, mask_b = hard_batches[baseline]
                torch.testing.assert_close(mask_s, mask_b)
                edges = ((adjacency_s-adjacency_b).abs().sum((1, 2))/2).numpy()
                nodes = ((xs.argmax(-1) != xb.argmax(-1))*mask_s).sum(1).numpy()
                changed = ((edges+nodes) > 0).astype(float)
                comparison['paired_hard_sample_changes'] = {
                    'edge_edits':weighted_moments(edges, wn, wa, components),
                    'node_label_edits':weighted_moments(nodes, wn, wa, components),
                    'fraction_graphs_changed':weighted_moments(changed, wn, wa, components)}
                differences[f'selected_minus_{baseline}'] = comparison
            result = dict(seed=seed, mode=mode,
                protocol=dict(posthoc=True, checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    audit_script_sha256=source_hash, pilot_noise_seed=84000, pilots_per_component=m,
                    ratio_noise_seed=84002, ratio_per_component=args.ratio_per_component,
                    fixed_weights='selected normal and abnormal mixture weights for all prior comparisons',
                    fixed_classifier='selected component GIN, normalization buffers and temperature for all prior comparisons',
                    soft_moments='Weighted moments of relaxed adjacency and categorical samples; not claimed to equal hard-sampling expectations',
                    ratio_reference='Uniform component distribution; same final classifier evaluated under each prior state',
                    no_fitting_or_selection=True),
                real_fit=real_fit, populations=populations, paired_prior_differences=differences,
                component_classifier_reference=dict(components=components, chance_accuracy=1/components,
                    uniform_nll=math.log(components), maximum_entropy=math.log(components),
                    temperature=float(model.temperature)), seconds=time.monotonic()-start)
            (output/'posthoc_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            print('LEVEL_C_POSTHOC_COMPLETE '+json.dumps(dict(seed=seed, mode=mode,
                seconds=result['seconds'], output=str(output/'posthoc_audit.json'))), flush=True)
            del model, saved, hard_batches
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
