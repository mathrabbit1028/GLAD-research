"""Optional CLI for the same experiment engine used by the terminal UI."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.catalog import discover
from src.experiments.plan import ExperimentSpec, new_spec
from src.experiments.runner import run_experiment


def main():
    catalog = discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--models', nargs='+', choices=list(catalog['methods']))
    parser.add_argument('--datasets', nargs='+', choices=list(catalog['datasets']))
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device', help='GPU device: cuda or cuda:N (CPU execution is disabled)')
    parser.add_argument('--gpus', nargs='+', help='Visible GPU devices, e.g. cuda:0 cuda:1; auto discovers all')
    parser.add_argument('--max-parallel-jobs', type=int)
    parser.add_argument('--min-free-gpu-memory-mb', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--train-only', action='store_true', default=None)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec = ExperimentSpec.from_dict(json.loads(args.profile.read_text())) if args.profile else new_spec(catalog)
    for key in ['models', 'datasets', 'seeds', 'device', 'threads', 'train_only',
                'max_parallel_jobs', 'min_free_gpu_memory_mb']:
        if getattr(args, key) is not None:
            setattr(spec, key, getattr(args, key))
    if args.gpus is not None:
        spec.gpu_devices = [] if args.gpus == ['auto'] else args.gpus
        spec.device = 'cuda'
    elif args.device is not None:
        spec.gpu_devices = []
    if args.output:
        spec.output_dir = str(args.output)
    def event(value):
        if value['type'] == 'job_finished':
            job = value['job']
            print(f"{job.get('gpu_device', '-')}: {job['model']}/{job['dataset']}/{job['seed']}: {job['status']}", flush=True)
    output = run_experiment(spec, on_event=event)
    print(output)
    if json.loads((output / 'manifest.json').read_text())['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
