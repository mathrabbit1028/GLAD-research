"""Plot the saved unfiltered pilot diagnostic without rerunning the model."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'prior-mixture-matplotlib'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads((args.run/'result.json').read_text())
    values = {name: [] for name in ['normal', 'abnormal', 'uniform']}
    contrasts = {name: [] for name in values}
    with (args.run/'pilot_scores.csv').open() as handle:
        for row in csv.DictReader(handle):
            values[row['mixture']].append(float(row['score']))
            contrasts[row['mixture']].append(float(row['log_abnormal_normal_ratio']))
    colors = dict(normal='#2374AB', abnormal='#C44E52', uniform='#777777')
    labels = dict(normal='Normal mixture', abnormal='Abnormal mixture', uniform='Uniform mixture')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, sequence in values.items():
        sorted_values = np.sort(sequence)
        axes[0].plot(sorted_values, np.arange(1, len(sequence)+1)/len(sequence),
                     label=labels[name], color=colors[name], linewidth=2)
    axes[0].set_xscale('log')
    axes[0].axvline(1, color='#555555', linestyle='--', linewidth=1, label='Normal-region boundary')
    axes[0].set(xlabel='Distance score to real normal regions (log scale)',
                ylabel='Fraction of pilots at or below score', ylim=(0, 1),
                title='Unfiltered pilot score distributions')
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8, loc='lower right')
    for name, sequence in contrasts.items():
        axes[1].plot(np.sort(sequence), np.arange(1,len(sequence)+1)/len(sequence),
                     label=labels[name], color=colors[name], linewidth=2)
    axes[1].set(xlabel='Log estimated abnormal / normal likelihood ratio',
                ylabel='Fraction of pilots at or below score', ylim=(0,1))
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(alpha=.2)
    auc = result['normal_vs_abnormal']['auc']
    low, high = result['normal_vs_abnormal']['bootstrap_95_interval']
    axes[0].set_title(f'Ellipsoid distance: AUROC {auc:.3f}\n95% interval {low:.3f}–{high:.3f}', fontsize=11)
    secondary = args.run/'secondary_ratio_analysis.json'
    if secondary.exists():
        ratio = json.loads(secondary.read_text())
        lo, hi = ratio['bootstrap_95_interval']
        axes[1].set_title(f"Likelihood ratio: AUROC {ratio['auc']:.3f}\n95% interval {lo:.3f}–{hi:.3f} (secondary)", fontsize=11)
    else:
        axes[1].set_title('Likelihood-ratio score distribution (secondary)', fontsize=11)
    fig.suptitle('Relative comparison of normal and abnormal graph pilots', fontsize=13)
    fig.text(.5, .01, f"MUTAG · seed {result['seed']} · {result['pilots_per_mixture']:,} pilots per mixture · synthetic-pilot comparison, not real-anomaly AUROC",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(args.run/'pilot_comparison.png', dpi=180)
    fig.savefig(args.run/'pilot_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
