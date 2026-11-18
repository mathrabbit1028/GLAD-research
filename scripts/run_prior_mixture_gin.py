"""Controlled GIN replacement: frozen priors, independent pilots, real-data transfer."""
import argparse
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
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classifier-seed', type=int, default=0)
    parser.add_argument('--pilots', type=int, default=8192)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.pilots, args.bootstrap) < 1 or args.classifier_seed < 0:
        parser.error('Counts must be positive and seed nonnegative')
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '1'
    import numpy as np
    import torch
    import yaml
    from torch_geometric.data import Batch
    from src.utils.config import _wrap, load_config, to_plain
    from src.utils.seed import set_seed
    from src.utils.device import require_cuda
    from src.data.datamodule import TUDataModule
    from src.models.factory import create_model
    from src.models.candidates.prior_mixture_gin import sample_mixture, graph_digest
    from src.models.candidates.prior_mixture import CandidateAdapter as ProposalAdapter
    from src.models.candidates.prior_mixture_impl.sampling import PriorBank
    from src.models.candidates.prior_mixture_impl.diagnostics import separation_auc
    from src.evaluation.evaluator import Evaluator
    torch.set_num_threads(1)
    device = require_cuda(args.device)
    output, reference = args.output.resolve(), args.reference_run.resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    cached_path = reference/'proposal_checkpoint.pt'
    cached = torch.load(cached_path, weights_only=True, map_location='cpu')
    if cached['stage'] != 'proposal_diagnostic_only' or cached['config']['model']['mixture']['min_shell_rate'] != 0:
        raise ValueError('Requires frozen, cap-only proposal checkpoint')
    cfg = _wrap(cached['config'])
    source_cfg = cfg.model
    cfg.model = load_config(['model=prior_mixture_gin']).model
    cfg.model.proposal = source_cfg
    cfg.output_dir, cfg.device = str(output), str(device)
    cfg.classifier_seed = args.classifier_seed
    split = cached['split']
    processed = Path(cfg.dataset.data_root)/cfg.dataset.name/'processed/data.pt'
    data_hash = hashlib.sha256(processed.read_bytes()).hexdigest()
    reference_manifest = json.loads((reference/'manifest.json').read_text())
    if reference_manifest['data_sha256'].get(str(processed)) != data_hash:
        raise ValueError('Dataset differs from the frozen-proposal fit')
    data = TUDataModule(cfg.dataset, int(cfg.seed), split=split)
    graphs = ProposalAdapter._clean_training(data.train_dataloader())
    current_spec = PriorBank(graphs, source_cfg.prior, int(cfg.seed)).specification
    if current_spec != cached['model_state']['prior_specification']:
        raise ValueError('Prior specification differs from the saved training normals')
    output.mkdir(parents=True)
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
    (output/'split.json').write_text(json.dumps(split, indent=2)+'\n')
    manifest = dict(status='running', experiment='prior_mixture_gin', classifier_seed=args.classifier_seed,
                    split_seed=int(cfg.seed), sampling_seed=cfg.model.sampling.seed,
                    reference=str(reference), reference_sha256=hashlib.sha256(cached_path.read_bytes()).hexdigest(),
                    data_sha256=data_hash, train_digest=graph_digest(graphs), source_sha256={},
                    gpu=torch.cuda.get_device_name(device), device=str(device), torch=str(torch.__version__),
                    epochs=cfg.model.training.epochs, pilots_per_mixture=args.pilots, bootstrap=args.bootstrap,
                    selection='final_epoch', unfiltered=True, ellipsoid_used_for_training_or_gin_score=False)
    for directory in ['src', 'configs', 'scripts']:
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT)
            destination = output/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            manifest['source_sha256'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save_manifest()
    start = time.monotonic()
    try:
        set_seed(args.classifier_seed, True)
        model = create_model(data.dataset.num_features, cfg.model, device)
        normal = cached['model_state']['normal_weights'].numpy()
        abnormal = cached['model_state']['abnormal_weights'].numpy()
        model.fit_from_proposals(graphs, normal, abnormal, current_spec)
        checkpoint_path = output/'classifier_checkpoint.pt'
        torch.save(dict(stage='gin_classifier', model_state=model.state_dict(), config=to_plain(cfg),
                        split=split, input_dim=data.dataset.num_features), checkpoint_path)
        restored = create_model(data.dataset.num_features, cfg.model, device)
        restored.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)['model_state'])
        expected = model.anomaly_score(Batch.from_data_list(graphs[:16]))
        torch.testing.assert_close(expected, restored.anomaly_score(Batch.from_data_list(graphs[:16])), rtol=2e-5, atol=2e-6)
        manifest.update(stage='fresh_pilots_and_heldout_evaluation', checkpoint_roundtrip='passed')
        save_manifest()
        frozen = create_model(data.dataset.num_features, source_cfg, device)
        frozen.load_state_dict(cached['model_state'])
        pilots, pilot_hashes = {}, {}
        # These streams are shared across classifier seeds, and disjoint from fitting/calibration.
        with (output/'pilot_scores.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['mixture','draw','component','num_nodes','gin_logit','ellipsoid_score','log_mixture_ratio'])
            for index, (name, weights) in enumerate([('normal', normal), ('abnormal', abnormal),
                                                     ('uniform', np.full(len(normal), 1/len(normal)))]):
                generated, components = sample_mixture(graphs, source_cfg.prior, weights, args.pilots, 52001+index)
                pilot_hashes[name] = graph_digest(generated)
                values = model.score_graphs(generated)
                reference_scores, ratios = [], []
                for offset in range(0, len(generated), 64):
                    batch = generated[offset:offset+64]
                    reference_scores.extend(frozen.anomaly_score(Batch.from_data_list(batch)).cpu().tolist())
                    logits, _ = frozen._encode(batch)
                    logs = frozen._log_ratios(logits)
                    contrast = (torch.logsumexp(logs+frozen.weights_abnormal.log(), 1) -
                                torch.logsumexp(logs+frozen.weights_normal.log(), 1))
                    ratios.extend(contrast.cpu().tolist())
                pilots[name] = dict(gin=np.asarray(values), ellipsoid=np.asarray(reference_scores), ratio=np.asarray(ratios))
                for i, (graph, component) in enumerate(zip(generated, components)):
                    writer.writerow([name,i,int(component),graph.num_nodes,float(values[i]),reference_scores[i],ratios[i]])
                handle.flush()
                model._log('fresh_pilots', mixture=name, graphs=len(generated), median_logit=float(np.median(values)))
        def auc_interval(n, a, seed):
            rng = np.random.default_rng(seed)
            estimates = [separation_auc(rng.choice(n,len(n)), rng.choice(a,len(a))) for _ in range(args.bootstrap)]
            return dict(auc=separation_auc(n,a), interval95=np.quantile(estimates,[.025,.975]).tolist())
        relative = {score:auc_interval(pilots['normal'][score],pilots['abnormal'][score],60001)
                    for score in ['gin','ellipsoid','ratio']}
        # Synthetic-normal cutoffs are set without real calibration/test observations.
        synthetic_cal, _ = sample_mixture(graphs, source_cfg.prior, normal,
                                          cfg.model.sampling.validation_per_class, cfg.model.sampling.seed+1000)
        synthetic_cal_scores = model.score_graphs(synthetic_cal)
        synthetic_threshold = float(np.quantile(synthetic_cal_scores,.95))
        evaluator = Evaluator(device)
        _, cal_scores, _, _ = evaluator.collect(model, data.validation_dataloader())
        threshold = float(np.quantile(cal_scores,cfg.evaluation.threshold_quantile))
        metrics, labels, test_scores = evaluator.evaluate(model,data.test_dataloader(),split['normal_label'],threshold)
        labels, test_scores = np.asarray(labels), np.asarray(test_scores)
        train_scores = model.score_graphs(graphs)
        real_normal = test_scores[labels == 0]
        real_abnormal = test_scores[labels == 1]
        metrics['bootstrap'] = auc_interval(real_normal,real_abnormal,60002)
        def summary(values):
            values = np.asarray(values)
            return dict(n=len(values), median_logit=float(np.median(values)),
                        q05_q95=np.quantile(values,[.05,.95]).tolist(),
                        normal_at_equal_priors=int((values < 0).sum()),
                        normal_fraction_at_equal_priors=float((values < 0).mean()),
                        accepted_at_synthetic_normal95=int((values < synthetic_threshold).sum()),
                        coverage_at_synthetic_normal95=float((values < synthetic_threshold).mean()))
        coverage = {name:summary(values['gin']) for name,values in pilots.items()}
        coverage.update(real_training=summary(train_scores),real_calibration=summary(cal_scores),
                        real_test_normal=summary(real_normal),real_test_abnormal=summary(real_abnormal))
        # Rank position tests directional compatibility of real normals with the proposals.
        transfer = dict(real_normal_vs_abnormal_pilot=auc_interval(real_normal,pilots['abnormal']['gin'],60003),
                        synthetic_normal_vs_real_normal=auc_interval(pilots['normal']['gin'],real_normal,60004))
        with (output/'real_scores.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['split','dataset_index','is_anomaly','gin_logit'])
            # _clean_training uses reproducible shuffled order; use unshuffled collection for IDs.
            _, ordered_train, _, _ = evaluator.collect(model,data.train_dataloader(shuffle=False))
            for name, indices, ys, values in [('train',split['train_indices'],np.zeros(len(graphs)),ordered_train),
                                             ('calibration',split['validation_indices'],np.zeros(len(cal_scores)),cal_scores),
                                             ('test',split['test_indices'],labels,test_scores)]:
                for idx,y,value in zip(indices,ys,values):
                    writer.writerow([name,idx,int(y),float(value)])
        result = dict(classifier_seed=args.classifier_seed,split_seed=int(cfg.seed),relative=relative,
                      coverage=coverage,transfer=transfer,real_test=metrics,
                      synthetic_normal95_threshold=synthetic_threshold,real_normal99_threshold=threshold,
                      pilot_sha256=pilot_hashes,diagnostics=model.diagnostics,
                      uncertainty='Bootstrap within populations, conditional on frozen priors and this classifier; no seed or split uncertainty',
                      seconds=time.monotonic()-start)
        (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        manifest.update(status='completed',stage='done',seconds=result['seconds'])
        save_manifest()
        print('RESULT '+json.dumps({k:result[k] for k in ['classifier_seed','relative','coverage','transfer','real_test','seconds']}),flush=True)
    except BaseException as error:
        manifest.update(status='failed',error=str(error),seconds=time.monotonic()-start)
        save_manifest()
        (output/'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
