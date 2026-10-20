"""Aggregate completed jobs from one benchmark manifest into runs and summaries."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METRICS = ('auc', 'average_precision', 'f1', 'accuracy', 'latency_ms_per_graph')


def aggregate(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    rows, seen, splits = [], set(), {}
    expected, completed, roles = defaultdict(set), defaultdict(list), {}
    for job in manifest['runs']:
        cfg = job['config']
        group = (job['model'], cfg['dataset']['name'])
        roles[group] = job.get('role', cfg['model'].get('role', 'benchmark'))
        expected[group].add(job['seed'])
        if job['status'] != 'completed':
            continue
        key = (*group, job['seed'])
        if key in seen:
            raise ValueError(f'Duplicate completed job: {key}')
        seen.add(key)
        pair = (group[1], job['seed'])
        split_hash = job['split_sha256']
        if splits.setdefault(pair, split_hash) != split_hash:
            raise ValueError(f'Unpaired graph split: {pair}')
        work = run_dir / Path(job['output_dir']).name
        stem = f'{group[0]}_{group[1]}_seed{job["seed"]}'
        row = json.loads((work / 'results' / f'{stem}.json').read_text())
        if (row['algorithm'], row['dataset'], row['seed']) != key:
            raise ValueError(f'Result does not match manifest job: {key}')
        row.update(role=roles[group], split_sha256=split_hash, wall_seconds=job['seconds'],
                   gpu_device=job.get('gpu_device'), gpu_uuid=job.get('gpu_uuid'))
        rows.append(row)
        completed[group].append(row)
    if not rows:
        raise ValueError('No completed evaluation jobs in this benchmark')
    summaries = []
    for (model, dataset), seeds in sorted(expected.items()):
        group = completed[(model, dataset)]
        actual = {row['seed'] for row in group}
        item = dict(algorithm=model, role=roles[(model, dataset)], dataset=dataset, n_expected=len(seeds), n_seeds=len(group),
                    complete=actual == seeds, seeds=','.join(map(str, sorted(actual))))
        for metric in METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            item[metric + '_mean'] = mean(values) if values else None
            item[metric + '_std'] = stdev(values) if len(values) > 1 else None
        summaries.append(item)
    rows.sort(key=lambda row: (row['algorithm'], row['dataset'], row['seed']))
    return rows, summaries


def write_summary(run_dir, output=None):
    run_dir = Path(run_dir).resolve()
    rows, summaries = aggregate(run_dir)
    output = Path(output) if output else run_dir / 'summary'
    output.mkdir(parents=True, exist_ok=False)
    for name, records in [('runs', rows), ('summary', summaries)]:
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (output / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        (output / f'{name}.json').write_text(json.dumps(records, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(write_summary(args.run_dir, args.output_dir))


if __name__ == '__main__':
    main()
