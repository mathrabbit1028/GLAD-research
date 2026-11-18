"""Compare fresh, unfiltered proposal pilots using the existing model and splits.

This is a proposal diagnostic, not a trained-detector benchmark. No anomaly
labels, threshold calibration, radius training, or test evaluation are used.
"""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pilots', type=int, default=8192)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--resume-geometry', type=Path, help='Reuse a saved frozen geometry stage')
    args = parser.parse_args()
    if min(args.pilots, args.bootstrap, args.threads) < 1:
        parser.error('Pilot, bootstrap, and thread counts must be positive')
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[variable] = str(args.threads)
    import numpy as np
    import torch
    import yaml
    from src.utils.config import _wrap, to_plain
    from src.utils.seed import set_seed
    from src.utils.device import require_cuda
    from src.data.datamodule import TUDataModule
    from src.models.factory import create_model
    from src.models.candidates.prior_mixture_impl.sampling import PriorBank
    from src.models.candidates.prior_mixture_impl.geometry import distances, scores
    from src.models.candidates.prior_mixture_impl.diagnostics import compare_scores, summarize_scores
    torch.set_num_threads(args.threads)
    device = require_cuda(args.device)
    reference = args.reference_run.resolve()
    cfg = _wrap(yaml.safe_load((reference/'resolved_config.yaml').read_text()))
    if cfg.model.name != 'prior_mixture':
        raise ValueError('Reference run must use prior_mixture')
    cfg.device = str(device)
    cfg.output_dir = str(output)
    cfg.model.mixture.min_shell_rate = 0.
    split = json.loads((reference/'split.json').read_text())
    output.mkdir(parents=True)
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
    (output/'split.json').write_text(json.dumps(split, indent=2)+'\n')
    manifest = dict(status='running', experiment='relative_unfiltered_prior_pilots',
                    reference_run=str(reference), seed=int(cfg.seed), pilots_per_mixture=args.pilots,
                    bootstrap=args.bootstrap, device=str(device),
                    gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__),
                    changes={'model.mixture.min_shell_rate':0.},
                    unfiltered=True, radius_training=False, test_evaluation=False,
                    source_sha256={}, data_sha256={})
    manifest['resume_geometry'] = str(args.resume_geometry.resolve()) if args.resume_geometry else None
    # Snapshot the complete shared implementation and the diagnostic entry point.
    for directory in ['src', 'configs', 'scripts']:
        for path in sorted((ROOT/directory).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT)
            destination = output/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            manifest['source_sha256'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    processed = Path(cfg.dataset.data_root)/cfg.dataset.name/'processed/data.pt'
    manifest['data_sha256'][str(processed)] = hashlib.sha256(processed.read_bytes()).hexdigest()
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save_manifest()
    started = time.monotonic()
    try:
        set_seed(int(cfg.seed), bool(cfg.reproducibility.deterministic))
        data = TUDataModule(cfg.dataset, int(cfg.seed), split=split)
        model = create_model(data.dataset.num_features, cfg.model, device)
        def save_geometry(fitted, bank, log_ratios):
            torch.save(dict(stage='frozen_geometry_only', model_state=fitted.state_dict(),
                            config=to_plain(cfg), split=split, input_dim=data.dataset.num_features,
                            log_ratios=torch.tensor(log_ratios), bank_rng=bank.rng.bit_generator.state,
                            data_sha256=manifest['data_sha256']), output/'geometry_checkpoint.pt')
            fitted._log('geometry_checkpoint_saved', path=str(output/'geometry_checkpoint.pt'))
        if args.resume_geometry:
            cached = torch.load(args.resume_geometry, weights_only=True, map_location='cpu')
            if cached['stage'] != 'frozen_geometry_only' or cached['split'] != split or cached['config']['seed'] != cfg.seed:
                raise ValueError('Geometry checkpoint has a different stage, split, or seed')
            if cached['data_sha256'] != manifest['data_sha256']:
                raise ValueError('Geometry checkpoint uses different input data')
            for key in ['architecture', 'prior', 'training', 'geometry']:
                if cached['config']['model'][key] != to_plain(cfg.model[key]):
                    raise ValueError(f'Geometry checkpoint uses different {key} settings')
            model.load_state_dict(cached['model_state'])
            graphs = model._clean_training(data.train_dataloader())
            bank = PriorBank(graphs, cfg.model.prior, int(cfg.seed))
            bank.rng.bit_generator.state = cached['bank_rng']
            model.fit_abnormal_proposal(bank, cached['log_ratios'].numpy(), np.random.default_rng(int(cfg.seed)), cfg.model)
            _, embeddings = model._encode(graphs)
        else:
            bank, graphs, embeddings, _ = model.fit_proposals(data.train_dataloader(), cfg.model, device,
                                                             on_geometry=save_geometry)
        torch.save(dict(stage='proposal_diagnostic_only', model_state=model.state_dict(),
                        config=to_plain(cfg), split=split, input_dim=data.dataset.num_features),
                   output/'proposal_checkpoint.pt')
        manifest['stage'] = 'fresh_unfiltered_pilots'
        save_manifest()
        real = scores(distances(embeddings, model.geometry), model.geometry, preliminary=True).cpu().numpy()
        mixtures = dict(normal=model.weights_normal.cpu().numpy(),
                        abnormal=model.weights_abnormal.cpu().numpy(),
                        uniform=np.full(cfg.model.prior.components, 1/cfg.model.prior.components))
        pilots, summary = {}, {}
        with (output/'pilot_scores.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['mixture', 'draw', 'component', 'num_nodes', 'score', 'log_abnormal_normal_ratio'])
            for index, (name, weights) in enumerate(mixtures.items()):
                sampler = PriorBank(graphs, cfg.model.prior, int(cfg.seed)+10001+index)
                rng = np.random.default_rng(int(cfg.seed)+20001+index)
                chunks, counts, sizes = [], np.zeros(len(weights), dtype=int), []
                size = cfg.model.training.batch_size
                for start in range(0, args.pilots, size):
                    ids = rng.choice(len(weights), size=min(size, args.pilots-start), p=weights/weights.sum())
                    generated = [sampler.sample(int(k)) for k in ids]
                    logits, z = model._encode(generated)
                    ds = scores(distances(z, model.geometry), model.geometry, preliminary=True).cpu().numpy()
                    ratios = model._log_ratios(logits)
                    contrast = (torch.logsumexp(ratios+model.weights_abnormal.log(), 1) -
                                torch.logsumexp(ratios+model.weights_normal.log(), 1)).cpu().numpy()
                    for offset, (component, graph, value, log_ratio) in enumerate(zip(ids, generated, ds, contrast)):
                        writer.writerow([name, start+offset, int(component), graph.num_nodes, float(value), float(log_ratio)])
                    chunks.append(ds)
                    np.add.at(counts, ids, 1)
                    sizes.extend(g.num_nodes for g in generated)
                    if (start+len(ids)) % 1024 == 0 or start+len(ids) == args.pilots:
                        handle.flush()
                        model._log('unfiltered_pilots', mixture=name, completed=start+len(ids), target=args.pilots)
                pilots[name] = np.concatenate(chunks)
                summary[name] = dict(**summarize_scores(pilots[name], cfg.model.generation.shell_inner,
                                                       cfg.model.generation.shell_outer),
                                     component_counts=counts.tolist(), mean_nodes=float(np.mean(sizes)))
                model._log('unfiltered_summary', mixture=name, **summary[name])
        result = dict(experiment=manifest['experiment'], seed=int(cfg.seed),
                      pilots_per_mixture=args.pilots, unfiltered=True,
                      normal_vs_abnormal=compare_scores(pilots['normal'], pilots['abnormal'], seed=30001+int(cfg.seed), bootstrap=args.bootstrap),
                      normal_vs_uniform=compare_scores(pilots['normal'], pilots['uniform'], seed=30002+int(cfg.seed), bootstrap=args.bootstrap),
                      uniform_vs_abnormal=compare_scores(pilots['uniform'], pilots['abnormal'], seed=30003+int(cfg.seed), bootstrap=args.bootstrap),
                      pilots=summary,
                      real_training=summarize_scores(real, cfg.model.generation.shell_inner, cfg.model.generation.shell_outer),
                      diagnostics=model.diagnostics, seconds=time.monotonic()-started,
                      interpretation='Pilot discrimination; not real-anomaly detection AUROC')
        (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        manifest.update(status='completed', stage='done', seconds=result['seconds'])
        save_manifest()
        print('RESULT '+json.dumps({k:result[k] for k in ['normal_vs_abnormal', 'normal_vs_uniform', 'uniform_vs_abnormal', 'seconds']}),flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=str(error), seconds=time.monotonic()-started)
        save_manifest()
        (output/'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
