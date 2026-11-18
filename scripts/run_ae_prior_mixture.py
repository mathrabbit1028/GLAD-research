#!/usr/bin/env python3
"""Fixed protocol, monitored Level B experiment and matched controls."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stage', choices=['bank', 'run'], default='run')
    parser.add_argument('--mode', choices=['all', 'no_ae', 'frozen_gin', 'level_b'], default='all')
    parser.add_argument('--override', action='append', default=[])
    args = parser.parse_args()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '1'
    import numpy as np
    import torch
    import yaml
    from sklearn.metrics import roc_auc_score
    from torch_geometric.data import Batch
    from src.utils.config import load_config, to_plain
    from src.utils.device import require_cuda
    from src.utils.seed import set_seed
    from src.data.datamodule import TUDataModule
    from src.models.factory import create_model
    from src.models.candidates.prior_mixture import CandidateAdapter as Proposal
    from src.models.candidates.prior_mixture_gin import graph_digest
    from src.models.candidates.prior_mixture_impl.sampling import PriorBank
    from src.models.candidates.ae_prior_mixture_gin import prepare_bank
    from src.models.candidates.ae_prior_mixture_impl.core import mmd, reconstruction
    from src.evaluation.metrics import classification_metrics
    torch.set_num_threads(1)
    device = require_cuda('cuda:0')
    cfg = load_config(['model=ae_prior_mixture_gin', 'dataset=mutag', 'seed=0'] + args.override)
    cfg.evaluation.model_selection = 'internal_normal_split'
    set_seed(0)
    data = TUDataModule(cfg.dataset, 0)
    graphs = Proposal._clean_training(data.train_dataloader(shuffle=False))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cached = root/'bank.pt'
    if args.stage == 'bank':
        if cached.exists():
            raise ValueError('Bank checkpoint already exists')
        sources = {}
        for directory in ['src', 'configs', 'scripts', 'tests']:
            for path in (ROOT/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    relative = path.relative_to(ROOT)
                    dest = root/'source'/relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'source_sha256.json').write_text(json.dumps(sources, indent=2)+'\n')
        (root/'resolved_config.yaml').write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
        bank = prepare_bank(graphs, cfg.model, device)
        bank.update(model_config=to_plain(cfg.model), split=data.split,
                    data_sha256=hashlib.sha256((Path(cfg.dataset.data_root)/'MUTAG/processed/data.pt').read_bytes()).hexdigest())
        torch.save(bank, cached)
        (root/'data_roles.json').write_text(json.dumps(dict(split=data.split,
            roles_dataset_indices={role:[data.split['train_indices'][i] for i in ids] for role,ids in bank['roles'].items()}), indent=2)+'\n')
        print('BANK_READY '+str(cached), flush=True)
        return
    bank = torch.load(cached, map_location='cpu', weights_only=True)
    if bank['model_config'] != to_plain(cfg.model) or bank['split'] != data.split:
        raise ValueError('Bank configuration or dataset split mismatch')
    if bank['data_sha256'] != hashlib.sha256((Path(cfg.dataset.data_root)/'MUTAG/processed/data.pt').read_bytes()).hexdigest():
        raise ValueError('Dataset hash mismatch')
    modes = ['no_ae', 'frozen_gin', 'level_b'] if args.mode == 'all' else [args.mode]
    fit = [graphs[i] for i in bank['roles']['fit']]
    # Fresh, component-stratified pilots: weighted estimates do not differentiate sampling.
    pilot_bank = PriorBank(fit, cfg.model.proposal.prior, 84000)
    pilot_graphs, component_ids = pilot_bank.labelled(64)
    component_ids = component_ids.numpy()
    from scripts.diagnose_prior_mixture_gap import attributes
    structural_keys = ['connected', 'max_degree', 'triangle_count', 'isolated_fraction', 'degree_over4_fraction']
    pilot_attributes = [attributes(g) for g in pilot_graphs]
    fit_attributes = [attributes(g) for g in fit]
    cal = [data.dataset[i] for i in data.split['validation_indices']]
    test = [data.dataset[i] for i in data.split['test_indices']]
    labels = np.array([int(int(g.y.item()) != data.split['normal_label']) for g in test])
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
            bank_sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
            pilot_digest=graph_digest(pilot_graphs), selection='normal_only_internal_selection')
        manifest['source_sha256'] = {}
        for pattern in ['src/models/candidates/ae_prior_mixture*.py',
                        'src/models/candidates/ae_prior_mixture_impl/*.py',
                        'scripts/run_ae_prior_mixture.py', 'configs/models/ae_prior_mixture_gin.yaml']:
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
            model.fit_prepared(graphs, bank, args.seed)
            checkpoint = dict(model_state=model.state_dict(), model_config=to_plain(run_cfg),
                              input_dim=data.dataset.num_features, split=data.split)
            torch.save(checkpoint, output/'checkpoint.pt')
            restored = create_model(data.dataset.num_features, run_cfg, device)
            restored.load_state_dict(torch.load(output/'checkpoint.pt', weights_only=True, map_location=device)['model_state'])
            torch.testing.assert_close(model.anomaly_score(Batch.from_data_list(test[:5])),
                                       restored.anomaly_score(Batch.from_data_list(test[:5])))
            manifest.update(stage='fresh_pilots_and_heldout_test', checkpoint_roundtrip='passed')
            save_manifest()
            wn, wa = [w.detach().cpu().numpy() for w in model.weights]
            weights_n, weights_a = wn[component_ids]/64, wa[component_ids]/64
            pilot_logits = model.score_graphs(pilot_graphs, transform=False)
            pilot_auc = roc_auc_score(np.r_[np.zeros(len(pilot_logits)), np.ones(len(pilot_logits))],
                np.r_[pilot_logits, pilot_logits], sample_weight=np.r_[weights_n, weights_a])
            real_scores = {}
            with (output/'real_scores.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['split', 'dataset_index', 'is_anomaly', 'raw_logit', 'soft_logit', 'hard_logit'])
                for name, gs, indices, ys in [
                    ('train', graphs, data.split['train_indices'], np.zeros(len(graphs), dtype=int)),
                    ('calibration', cal, data.split['validation_indices'], np.zeros(len(cal), dtype=int)),
                    ('test', test, data.split['test_indices'], labels)]:
                    raw, soft, hard = model.score_graphs(gs, False), model.score_graphs(gs), model.score_graphs(gs, hard=True)
                    real_scores[name] = dict(raw=raw, soft=soft, hard=hard)
                    for idx, y, r, s, h in zip(indices, ys, raw, soft, hard):
                        writer.writerow([name, idx, y, r, s, h])
            with (output/'pilot_scores.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['component', 'logit', 'normal_weight', 'abnormal_weight'])
                writer.writerows(zip(component_ids, pilot_logits, weights_n, weights_a))
            threshold = float(np.quantile(real_scores['calibration']['soft'], .99))
            metrics = classification_metrics(labels.tolist(), real_scores['test']['soft'].tolist(), threshold)
            prediction = real_scores['test']['soft'] >= threshold
            metrics.update(tp=int((prediction & (labels == 1)).sum()), fp=int((prediction & (labels == 0)).sum()),
                           fn=int((~prediction & (labels == 1)).sum()), tn=int((~prediction & (labels == 0)).sum()))
            metrics['hard_auc'] = float(roc_auc_score(labels, real_scores['test']['hard']))
            metrics['raw_auc'] = float(roc_auc_score(labels, real_scores['test']['raw']))
            rng = np.random.default_rng(85000)
            nids, aids = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
            boot = []
            for _ in range(1000):
                ids = np.r_[rng.choice(nids, len(nids)), rng.choice(aids, len(aids))]
                boot.append(float(roc_auc_score(labels[ids], real_scores['test']['soft'][ids])))
            metrics['conditional_bootstrap95'] = np.quantile(boot, [.025, .975]).tolist()
            with torch.no_grad():
                mixture_input = model.ae(model.fit_data) if mode == 'level_b' else model.fit_data
                log_ratios = torch.log_softmax(model.ratio(mixture_input)/model.temperature, -1) + np.log(len(wn))
                def likelihood(w):
                    return float(torch.logsumexp(log_ratios+w.log(), -1).mean())
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
                p = torch.tensor(pilot_logits, device=device)
                pw = torch.tensor(weights_n, device=device)
                gap = {}
                for role, indices in {**bank['roles'], 'train':list(range(len(graphs)))}.items():
                    raw = torch.tensor(real_scores['train']['raw'][indices], device=device)
                    soft = torch.tensor(real_scores['train']['soft'][indices], device=device)
                    gap[role] = dict(raw_synthetic_mmd=float(mmd(raw, p, model.scale, wy=pw)),
                        soft_synthetic_mmd=float(mmd(soft, p, model.scale, wy=pw)),
                        raw_mean=float(raw.mean()), soft_mean=float(soft.mean()),
                        synthetic_mean=float((p*pw).sum()))
                tn = torch.tensor(real_scores['test']['soft'][labels == 0], device=device)
                gap['test_normal'] = dict(soft_synthetic_mmd=float(mmd(tn, p, model.scale, wy=pw)))
                audit = {}
                for name, gs in [('train', graphs), ('test', test)]:
                    d = model._dense(gs)
                    t = d if mode == 'no_ae' else model.ae(d)
                    h = d if mode == 'no_ae' else model.ae(d, hard=True)
                    _, _, edges, nodes = reconstruction(d, t, run_cfg.loss.edge_budget, run_cfg.loss.node_budget)
                    _, _, hard_edges, hard_nodes = reconstruction(d, h, run_cfg.loss.edge_budget, run_cfg.loss.node_budget)
                    audit[name] = dict(mean_expected_edge_edits=float(edges.mean()), max_expected_edge_edits=float(edges.max()),
                        mean_expected_node_edits=float(nodes.mean()), max_expected_node_edits=float(nodes.max()),
                        mean_hard_edge_edits=float(hard_edges.mean()), mean_hard_node_edits=float(hard_nodes.mean()),
                        hard_changed_graphs=int(((hard_edges+hard_nodes)>0).sum()),
                        mean_abs_soft_hard_logit=float(np.abs(real_scores[name]['soft']-real_scores[name]['hard']).mean()))
            order = np.argsort(pilot_logits)
            cutoff = float(pilot_logits[order[np.searchsorted(np.cumsum(weights_n[order]), .95)]])
            score_normal = real_scores['test']['soft'][labels == 0]
            score_abnormal = real_scores['test']['soft'][labels == 1]
            # Relative-pilot checks use each pilot population's weighted ranks.
            def relative_auc(real_values, pilot_weights):
                return float(np.mean([np.sum(pilot_weights*(pilot_logits > v)) + .5*np.sum(pilot_weights*(pilot_logits == v)) for v in real_values]))
            coverage = dict(synthetic_normal95_cutoff=cutoff,
                test_normals_accepted=int((score_normal < cutoff).sum()), test_normals=len(score_normal),
                test_anomalies_accepted=int((score_abnormal < cutoff).sum()), test_anomalies=len(score_abnormal),
                real_normal_vs_abnormal_pilot_auc=relative_auc(score_normal, weights_a),
                synthetic_normal_vs_real_normal_auc=1-relative_auc(score_normal, weights_n))
            structure = {}
            for key in structural_keys:
                values = np.array([row[key] for row in pilot_attributes])
                by_component = values.reshape(len(wn), 64).mean(1)
                structure[key] = dict(real_fit_mean=float(np.mean([row[key] for row in fit_attributes])),
                    synthetic_normal_mean=float(values @ weights_n),
                    synthetic_abnormal_mean=float(values @ weights_a),
                    component_mean_range=[float(by_component.min()), float(by_component.max())])
            result = dict(mode=mode, seed=args.seed, synthetic_auc=float(pilot_auc),
                real_test=metrics, real_normal99_threshold=threshold, gap=gap, edits=audit,
                relative_coverage=coverage, structural_coverage=structure, mixture_audit=mixture_audit,
                weights_normal=wn.tolist(), weights_abnormal=wa.tolist(),
                diagnostics=model.diagnostics, seconds=time.monotonic()-start)
            (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            manifest.update(status='completed', stage='done', seconds=result['seconds'])
            save_manifest()
            print('RESULT '+json.dumps({k:result[k] for k in ['mode','seed','synthetic_auc','real_test','gap','edits','relative_coverage','seconds']}), flush=True)
            del model, restored
            torch.cuda.empty_cache()
        except BaseException as error:
            manifest.update(status='failed', error=str(error), seconds=time.monotonic()-start)
            save_manifest()
            (output/'error.txt').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
