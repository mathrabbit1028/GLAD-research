#!/usr/bin/env python3
"""Monitored Level C experiment with matched fixed-prior and initialization controls."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ['no_ae', 'initial_ae', 'fixed_prior', 'level_c']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=['all'] + MODES, default='all')
    parser.add_argument('--override', action='append', default=[])
    parser.add_argument('--pilots-per-component', type=int, default=64)
    args = parser.parse_args()
    if args.pilots_per_component < 2:
        raise ValueError('At least two pilots per component are required')
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '1'
    import numpy as np
    import torch
    import yaml
    from sklearn.metrics import roc_auc_score
    from torch_geometric.data import Batch, Data
    from src.utils.config import load_config, to_plain
    from src.utils.device import require_cuda
    from src.utils.seed import set_seed
    from src.data.datamodule import TUDataModule
    from src.models.factory import create_model
    from src.models.candidates.prior_mixture import CandidateAdapter as Proposal
    from src.models.candidates.prior_mixture_gin import graph_digest
    from src.models.candidates.ae_prior_mixture_impl.core import reconstruction
    from src.evaluation.metrics import classification_metrics
    from scripts.diagnose_prior_mixture_gap import attributes
    torch.set_num_threads(1)
    device = require_cuda('cuda:0')
    cfg = load_config(['model=ae_prior_mixture_level_c', 'dataset=mutag', 'seed=0'] + args.override)
    cfg.evaluation.model_selection = 'internal_normal_split'
    set_seed(0)
    data = TUDataModule(cfg.dataset, 0)
    graphs = Proposal._clean_training(data.train_dataloader(shuffle=False))
    cal = [data.dataset[i] for i in data.split['validation_indices']]
    test = [data.dataset[i] for i in data.split['test_indices']]
    labels = np.array([int(int(g.y.item()) != data.split['normal_label']) for g in test])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_file = Path(cfg.dataset.data_root)/'MUTAG/processed/data.pt'
    data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    structural_keys = ['nodes', 'mean_degree', 'connected', 'max_degree', 'triangle_count',
                       'isolated_fraction', 'degree_over4_fraction', 'label_homophily'] + [
                           f'category_{i}' for i in range(data.dataset.num_features)]
    m = args.pilots_per_component

    def tensor_digest(batch):
        digest = hashlib.sha256()
        for tensor in batch:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def values(eta):
        return {key: value.detach().cpu().tolist() for key, value in eta.items()}

    def hard_graphs(batch):
        x, adjacency, mask = [t.detach().cpu() for t in batch]
        result = []
        for features, edges, valid in zip(x, adjacency, mask):
            n = int(valid.sum())
            result.append(Data(x=features[:n].clone(),
                edge_index=(edges[:n, :n] > .5).nonzero().T.contiguous(), num_nodes=n))
        return result

    def score_dense(model, batch):
        with torch.no_grad():
            chunks = [model.network(tuple(t[start:start+128] for t in batch)).flatten().cpu()
                      for start in range(0, len(batch[0]), 128)]
        result = torch.cat(chunks).numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite pilot logits')
        return result

    def kernel_sum(x, y, wx, wy):
        # Exact biased weighted kernel statistic, chunked to keep pilot audits small.
        total = 0.
        for offset in range(0, len(x), 256):
            delta = np.square(x[offset:offset+256, None] - y[None, :])
            kernel = sum(np.exp(-delta/(2*sigma*sigma)) for sigma in [.5, 1., 2.])/3
            total += float(wx[offset:offset+256] @ (kernel @ wy))
        return total

    modes = MODES if args.mode == 'all' else [args.mode]
    for mode in modes:
        output = root/f'seed{args.seed}'/mode
        output.mkdir(parents=True, exist_ok=False)
        run_cfg = copy.deepcopy(cfg.model)
        run_cfg.mode = mode
        resolved = copy.deepcopy(cfg)
        resolved.model = run_cfg
        resolved.classifier_seed = args.seed
        (output/'resolved_config.yaml').write_text(yaml.safe_dump(to_plain(resolved), sort_keys=False))
        manifest = dict(status='running', mode=mode, seed=args.seed, split_seed=0,
            gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__),
            data_sha256=data_hash, training_digest=graph_digest(graphs), pilot_noise_seed=84000,
            pilots_per_component=m, selection='normal_only_internal_selection', source_sha256={})
        for pattern in ['src/models/candidates/ae_prior_mixture*.py',
                        'src/models/candidates/ae_prior_mixture_impl/*.py',
                        'scripts/run_ae_prior_mixture_level_c.py',
                        'configs/models/ae_prior_mixture_level_c.yaml']:
            for path in ROOT.glob(pattern):
                relative = path.relative_to(ROOT)
                destination = output/'source'/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                manifest['source_sha256'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        start = time.monotonic()
        def save_manifest():
            (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        save_manifest()
        try:
            set_seed(args.seed)
            model = create_model(data.dataset.num_features, run_cfg, device)
            model.fit_graphs(graphs, args.seed)
            roles = model.roles
            (output/'data_roles.json').write_text(json.dumps(dict(split=data.split,
                roles_dataset_indices={role:[data.split['train_indices'][i] for i in ids]
                                       for role, ids in roles.items()}), indent=2)+'\n')
            checkpoint = dict(model_state=model.state_dict(), model_config=to_plain(run_cfg),
                input_dim=data.dataset.num_features, split=data.split, classifier_seed=args.seed)
            torch.save(checkpoint, output/'checkpoint.pt')
            restored = create_model(data.dataset.num_features, run_cfg, device)
            restored.load_state_dict(torch.load(output/'checkpoint.pt', weights_only=True,
                                               map_location=device)['model_state'])
            torch.testing.assert_close(model.anomaly_score(Batch.from_data_list(test[:5])),
                                       restored.anomaly_score(Batch.from_data_list(test[:5])))
            for key in model.eta:
                torch.testing.assert_close(model.eta[key], restored.eta[key])
            with torch.no_grad():
                before = model.bank.sample(model.eta, 2, 84001)
                after = restored.bank.sample(restored.eta, 2, 84001)
                for a, b in zip(before, after):
                    torch.testing.assert_close(a, b)
            del restored, before, after
            manifest.update(stage='fresh_pilots_and_heldout_test', checkpoint_roundtrip='passed',
                            sampler_roundtrip='passed')
            save_manifest()
            wn, wa = [w.detach().cpu().numpy().astype(float) for w in model.weights]
            component_ids = np.repeat(np.arange(len(wn)), m)
            weights_n, weights_a = wn[component_ids]/m, wa[component_ids]/m
            with torch.no_grad():
                pilots = model.bank.sample(model.eta, m, 84000)
                hard_pilots = model.bank.sample(model.eta, m, 84000, hard=True)
                raw_hard_pilots = model.bank.sample(model.raw_eta, m, 84000, hard=True)
            manifest['pilot_digest'] = tensor_digest(pilots)
            manifest['prior_specification'] = model.bank.specification
            pilot_logits = score_dense(model, pilots)
            hard_pilot_logits = score_dense(model, hard_pilots)
            pilot_attributes = [attributes(g) for g in hard_graphs(hard_pilots)]
            raw_pilot_attributes = [attributes(g) for g in hard_graphs(raw_hard_pilots)]
            fit_attributes = [attributes(graphs[i]) for i in roles['fit']]
            del pilots, hard_pilots, raw_hard_pilots
            def weighted_auc(logits):
                return float(roc_auc_score(np.r_[np.zeros(len(logits)), np.ones(len(logits))],
                    np.r_[logits, logits], sample_weight=np.r_[weights_n, weights_a]))
            real_scores = {}
            with (output/'real_scores.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['split', 'dataset_index', 'is_anomaly', 'raw_logit', 'soft_logit', 'hard_logit'])
                for name, gs, indices, ys in [
                    ('train', graphs, data.split['train_indices'], np.zeros(len(graphs), dtype=int)),
                    ('calibration', cal, data.split['validation_indices'], np.zeros(len(cal), dtype=int)),
                    ('test', test, data.split['test_indices'], labels)]:
                    raw = model.score_graphs(gs, transform=False)
                    soft = model.score_graphs(gs)
                    hard = model.score_graphs(gs, hard=True)
                    real_scores[name] = dict(raw=raw, soft=soft, hard=hard)
                    writer.writerows(zip([name]*len(gs), indices, ys, raw, soft, hard))
            with (output/'pilot_scores.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['component', 'logit', 'hard_logit', 'normal_weight', 'abnormal_weight'])
                writer.writerows(zip(component_ids, pilot_logits, hard_pilot_logits, weights_n, weights_a))
            with (output/'pilot_structure.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['component', 'normal_weight', 'abnormal_weight']+structural_keys+
                                ['raw_prior_'+key for key in structural_keys])
                writer.writerows([int(component), float(n), float(a)]+[row[k] for k in structural_keys]+
                                 [raw[k] for k in structural_keys]
                                 for component, n, a, row, raw in zip(component_ids, weights_n, weights_a,
                                                                    pilot_attributes, raw_pilot_attributes))
            threshold = float(np.quantile(real_scores['calibration']['soft'], .99))
            metrics = classification_metrics(labels.tolist(), real_scores['test']['soft'].tolist(), threshold)
            prediction = real_scores['test']['soft'] >= threshold
            metrics.update(tp=int((prediction & (labels == 1)).sum()), fp=int((prediction & (labels == 0)).sum()),
                           fn=int((~prediction & (labels == 1)).sum()), tn=int((~prediction & (labels == 0)).sum()),
                           hard_auc=float(roc_auc_score(labels, real_scores['test']['hard'])),
                           raw_auc=float(roc_auc_score(labels, real_scores['test']['raw'])))
            rng = np.random.default_rng(85000)
            nids, aids = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
            boot = []
            for _ in range(1000):
                ids = np.r_[rng.choice(nids, len(nids)), rng.choice(aids, len(aids))]
                boot.append(float(roc_auc_score(labels[ids], real_scores['test']['soft'][ids])))
            metrics['conditional_bootstrap95'] = np.quantile(boot, [.025, .975]).tolist()
            with torch.no_grad():
                mixture_input = model.fit_data if mode == 'no_ae' else model.ae(model.fit_data)
                log_ratios = torch.log_softmax(model.ratio(mixture_input)/model.temperature, -1) + np.log(len(wn))
                def likelihood(w):
                    return float(torch.logsumexp(log_ratios+w.clamp_min(1e-30).log(), -1).mean())
                uniform = torch.ones_like(model.weights[0])/len(wn)
                mixture_audit = dict(normal_log_likelihood=likelihood(model.weights[0]),
                    abnormal_log_likelihood=likelihood(model.weights[1]), uniform_log_likelihood=likelihood(uniform),
                    normal_effective_components=float(1/model.weights[0].square().sum()),
                    abnormal_effective_components=float(1/model.weights[1].square().sum()),
                    normal_weight_tv_from_raw=float((model.weights[0]-model.raw_weights[0]).abs().sum()/2),
                    abnormal_weight_tv_from_raw=float((model.weights[1]-model.raw_weights[1]).abs().sum()/2),
                    mixture_overlap=float(torch.minimum(*model.weights).sum()))
                for w in model.weights:
                    torch.testing.assert_close(w.sum(), torch.ones((), device=device), rtol=1e-5, atol=1e-6)
                    if float(w.min()) < -1e-7:
                        raise ValueError('Negative mixture weight')
                if float(model.weights[1].max()) > run_cfg.mixture.cap+1e-6:
                    raise ValueError('Abnormal mixture cap violated')
                audit = {}
                for name, gs in [('train', graphs), ('test', test)]:
                    original = model._dense(gs)
                    soft = original if mode == 'no_ae' else model.ae(original)
                    hard = original if mode == 'no_ae' else model.ae(original, hard=True)
                    _, _, edges, nodes = reconstruction(original, soft, run_cfg.loss.edge_budget, run_cfg.loss.node_budget)
                    _, _, hard_edges, hard_nodes = reconstruction(original, hard, run_cfg.loss.edge_budget, run_cfg.loss.node_budget)
                    audit[name] = dict(mean_expected_edge_edits=float(edges.mean()), max_expected_edge_edits=float(edges.max()),
                        mean_expected_node_edits=float(nodes.mean()), max_expected_node_edits=float(nodes.max()),
                        mean_hard_edge_edits=float(hard_edges.mean()), mean_hard_node_edits=float(hard_nodes.mean()),
                        hard_changed_graphs=int(((hard_edges+hard_nodes)>0).sum()),
                        mean_abs_soft_hard_logit=float(np.abs(real_scores[name]['soft']-real_scores[name]['hard']).mean()))
            p = np.asarray(pilot_logits, dtype=float)/float(model.scale)
            pilot_self = kernel_sum(p, p, weights_n, weights_n)
            def gap_values(logits):
                x = np.asarray(logits, dtype=float)/float(model.scale)
                w = np.ones(len(x))/len(x)
                return kernel_sum(x, x, w, w)+pilot_self-2*kernel_sum(x, p, w, weights_n)
            gap = {}
            for role, indices in {**roles, 'train':list(range(len(graphs)))}.items():
                raw, soft = [real_scores['train'][key][indices] for key in ['raw', 'soft']]
                gap[role] = dict(raw_synthetic_mmd=float(gap_values(raw)), soft_synthetic_mmd=float(gap_values(soft)),
                    raw_mean=float(np.mean(raw)), soft_mean=float(np.mean(soft)),
                    synthetic_mean=float(pilot_logits @ weights_n))
            gap['test_normal'] = dict(soft_synthetic_mmd=float(gap_values(real_scores['test']['soft'][labels == 0])))
            def coverage_for(logits, normal_scores, abnormal_scores):
                order = np.argsort(logits)
                position = min(np.searchsorted(np.cumsum(weights_n[order]), .95), len(order)-1)
                cutoff = float(logits[order[position]])
                def relative_auc(real_values, pilot_weights):
                    return float(np.mean([np.sum(pilot_weights*(logits > v)) +
                        .5*np.sum(pilot_weights*(logits == v)) for v in real_values]))
                return dict(synthetic_normal95_cutoff=cutoff,
                    test_normals_accepted=int((normal_scores <= cutoff).sum()), test_normals=len(normal_scores),
                    test_anomalies_accepted=int((abnormal_scores <= cutoff).sum()), test_anomalies=len(abnormal_scores),
                    real_normal_vs_abnormal_pilot_auc=relative_auc(normal_scores, weights_a),
                    synthetic_normal_vs_real_normal_auc=1-relative_auc(normal_scores, weights_n))
            coverage = coverage_for(pilot_logits, real_scores['test']['soft'][labels == 0], real_scores['test']['soft'][labels == 1])
            hard_coverage = coverage_for(hard_pilot_logits, real_scores['test']['hard'][labels == 0], real_scores['test']['hard'][labels == 1])
            structure = {}
            raw_wn = model.raw_weights[0].detach().cpu().numpy()[component_ids]/m
            for key in structural_keys:
                selected = np.array([row[key] for row in pilot_attributes])
                raw = np.array([row[key] for row in raw_pilot_attributes])
                by_component = selected.reshape(len(wn), m).mean(1)
                structure[key] = dict(real_fit_mean=float(np.mean([row[key] for row in fit_attributes])),
                    synthetic_normal_mean=float(selected @ weights_n), synthetic_abnormal_mean=float(selected @ weights_a),
                    raw_prior_at_selected_weights=float(raw @ weights_n), raw_prior_at_raw_weights=float(raw @ raw_wn),
                    component_mean_range=[float(by_component.min()), float(by_component.max())])
            parameter_audit = dict(raw=values(model.raw_eta), selected=values(model.eta))
            if hasattr(model, 'initial_eta'):
                parameter_audit['initial_ae'] = values(model.initial_eta)
            parameter_audit['shift_from_raw'] = {key:dict(l2=float((value-model.raw_eta[key]).norm()),
                max_abs=float((value-model.raw_eta[key]).abs().max())) for key, value in model.eta.items()}
            if hasattr(model, 'initial_eta'):
                parameter_audit['shift_from_initial_ae'] = {key:dict(l2=float((value-model.initial_eta[key]).norm()),
                    max_abs=float((value-model.initial_eta[key]).abs().max())) for key, value in model.eta.items()}
            if 'category_probs' in model.eta:
                parameter_audit['category_tv_from_raw'] = float((model.eta['category_probs']-model.raw_eta['category_probs']).abs().sum()/2)
            result = dict(mode=mode, seed=args.seed, synthetic_auc=weighted_auc(pilot_logits),
                hard_synthetic_auc=weighted_auc(hard_pilot_logits),
                real_test=metrics, real_normal99_threshold=threshold, gap=gap, edits=audit,
                relative_coverage=coverage, hard_relative_coverage=hard_coverage,
                structural_coverage=structure, prior_parameters=parameter_audit, mixture_audit=mixture_audit,
                weights_normal=wn.tolist(), weights_abnormal=wa.tolist(),
                diagnostics=model.diagnostics, seconds=time.monotonic()-start,
                protocol=dict(pilots_per_component=m, pilot_noise_seed=84000, roles=roles,
                    primary_pilot_representation='relaxed', structural_pilot_representation='hard',
                    fixed_prior_control='Freeze eta at initial AE fit; same ratio, mixture and GIN updates',
                    limitation='Finite relaxed surrogate; same split across seeds; test labels evaluated after selection.'))
            (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            manifest.update(status='completed', stage='done', seconds=result['seconds'])
            save_manifest()
            print('RESULT '+json.dumps({key:result[key] for key in ['mode', 'seed', 'synthetic_auc', 'real_test',
                'gap', 'edits', 'relative_coverage', 'prior_parameters', 'seconds']}), flush=True)
            del model
            torch.cuda.empty_cache()
        except BaseException as error:
            manifest.update(status='failed', error=str(error), seconds=time.monotonic()-start)
            save_manifest()
            (output/'error.txt').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
