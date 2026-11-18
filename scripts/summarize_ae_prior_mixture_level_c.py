#!/usr/bin/env python3
"""Summarize matched Level C controls and make interpretable scientific figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

MODES = ['no_ae', 'initial_ae', 'fixed_prior', 'level_c']
NAMES = ['No AE', 'Initial AE', 'Fixed prior', 'Level C']
COLORS = dict(raw='#2878b5', ae='#8560a9', normal='#20874d', abnormal='#e6952e')


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--plot-seed', type=int, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    plot_seed = args.seeds[0] if args.plot_seed is None else args.plot_seed
    if plot_seed not in args.seeds:
        raise ValueError('Plot seed must be in the reported seeds')
    records = {mode:[] for mode in MODES}
    scores = {mode:[] for mode in MODES}
    test_ids = None
    for seed in args.seeds:
        audit_keys = []
        raw_eta = None
        initial_eta = None
        for mode in MODES:
            path = root/f'seed{seed}'/mode
            manifest = json.loads((path/'manifest.json').read_text())
            if manifest['status'] != 'completed':
                raise ValueError(f'Unfinished experiment: {path}')
            audit_keys.append((manifest['data_sha256'], manifest['training_digest'],
                               manifest['pilot_noise_seed'], manifest['pilots_per_component']))
            record = json.loads((path/'result.json').read_text())
            records[mode].append(record)
            parameters = record['prior_parameters']
            if raw_eta is None:
                raw_eta = parameters['raw']
            for key in raw_eta:
                np.testing.assert_allclose(raw_eta[key], parameters['raw'][key], atol=1e-7)
            if 'initial_ae' in parameters:
                if initial_eta is None:
                    initial_eta = parameters['initial_ae']
                for key in initial_eta:
                    np.testing.assert_allclose(initial_eta[key], parameters['initial_ae'][key], atol=1e-7)
            if mode == 'fixed_prior':
                for key in parameters['selected']:
                    np.testing.assert_allclose(parameters['selected'][key], parameters['initial_ae'][key], atol=1e-7)
            real = [row for row in read_csv(path/'real_scores.csv') if row['split'] == 'test']
            ids = [(row['dataset_index'], row['is_anomaly']) for row in real]
            if test_ids is None:
                test_ids = ids
            if test_ids != ids:
                raise ValueError('Test observations or order differ across conditions')
            scores[mode].append(np.array([float(row['soft_logit']) for row in real]))
        if len(set(audit_keys)) != 1:
            raise ValueError('Controls used different data or pilot base-noise protocols')
    labels = np.array([int(label) for _, label in test_ids])
    scores = {mode:np.array(value) for mode, value in scores.items()}
    summary = dict(seeds=args.seeds, conditions={}, paired_differences={},
        protocol='Matched relaxed generator and finite inner training. Fixed prior freezes eta at the initial AE fit; Level C refits eta through the AE.',
        caveat=f'Same data split across seeds; variability reflects training seeds, not independent datasets. Test: {int((labels==0).sum())} normals and {int((labels==1).sum())} anomalies. Bootstrap intervals condition on these fitted models.')
    def stats(value):
        value = [float(x) for x in value]
        return dict(mean=float(np.mean(value)), sample_sd=float(np.std(value, ddof=1)) if len(value)>1 else 0., values=value)
    for mode, rows in records.items():
        condition = dict(
            synthetic_auc=stats([r['synthetic_auc'] for r in rows]),
            hard_synthetic_auc=stats([r['hard_synthetic_auc'] for r in rows]),
            real_auc=stats([r['real_test']['auc'] for r in rows]),
            real_hard_auc=stats([r['real_test']['hard_auc'] for r in rows]),
            real_raw_auc=stats([r['real_test']['raw_auc'] for r in rows]),
            select_raw_mmd=stats([r['gap']['select']['raw_synthetic_mmd'] for r in rows]),
            select_soft_mmd=stats([r['gap']['select']['soft_synthetic_mmd'] for r in rows]),
            test_normal_soft_mmd=stats([r['gap']['test_normal']['soft_synthetic_mmd'] for r in rows]),
            mean_edge_edits=stats([r['edits']['train']['mean_expected_edge_edits'] for r in rows]),
            mean_node_edits=stats([r['edits']['train']['mean_expected_node_edits'] for r in rows]),
            hard_changed_test_graphs=[r['edits']['test']['hard_changed_graphs'] for r in rows],
            selected_steps=[r['diagnostics']['selected_step'] for r in rows],
            seconds=stats([r['seconds'] for r in rows]),
            prior_mean_degree=stats([r['prior_parameters']['selected']['mean_degree'] for r in rows]),
            prior_category_tv_from_raw=stats([r['prior_parameters']['category_tv_from_raw'] for r in rows]),
            prior_degree_shift_from_initial=stats([r['prior_parameters']['shift_from_initial_ae']['mean_degree']['l2'] for r in rows]),
            prior_category_shift_from_initial=stats([r['prior_parameters']['shift_from_initial_ae']['category_probs']['l2'] for r in rows]))
        for key in ['connected', 'mean_degree', 'max_degree', 'triangle_count', 'isolated_fraction']:
            condition['structure_'+key] = stats([r['structural_coverage'][key]['synthetic_normal_mean'] for r in rows])
        for key in ['test_normals_accepted', 'test_anomalies_accepted',
                    'real_normal_vs_abnormal_pilot_auc', 'synthetic_normal_vs_real_normal_auc']:
            condition['coverage_'+key] = stats([r['relative_coverage'][key] for r in rows])
        condition['calibrated_confusion'] = [{key:r['real_test'][key] for key in ['tp', 'fp', 'fn', 'tn']} for r in rows]
        summary['conditions'][mode] = condition
    ni, ai = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    for control in ['no_ae', 'initial_ae', 'fixed_prior']:
        rng, boot = np.random.default_rng(86000), []
        for _ in range(1000):
            ids = np.r_[rng.choice(ni, len(ni)), rng.choice(ai, len(ai))]
            boot.append(np.mean([roc_auc_score(labels[ids], adaptive[ids])-roc_auc_score(labels[ids], baseline[ids])
                                 for baseline, adaptive in zip(scores[control], scores['level_c'])]))
        summary['paired_differences'][control] = dict(
            mean_auc_gain=summary['conditions']['level_c']['real_auc']['mean']-summary['conditions'][control]['real_auc']['mean'],
            per_seed_auc_gain=(np.array(summary['conditions']['level_c']['real_auc']['values'])-
                               np.array(summary['conditions'][control]['real_auc']['values'])).tolist(),
            conditional_paired_bootstrap95=np.quantile(boot, [.025, .975]).tolist())
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.3))
    def ecdf(ax, x, weights=None, **kwargs):
        x = np.asarray(x)
        order = np.argsort(x)
        weights = np.ones(len(x))/len(x) if weights is None else np.asarray(weights)
        ax.step(x[order], np.cumsum(weights[order])/weights.sum(), where='post', **kwargs)
    for index, (mode, name) in enumerate(zip(MODES, NAMES)):
        path = root/f'seed{plot_seed}'/mode
        pilots, real = read_csv(path/'pilot_scores.csv'), read_csv(path/'real_scores.csv')
        ax = axes[0, index]
        ps = [float(row['logit']) for row in pilots]
        ecdf(ax, ps, [float(row['normal_weight']) for row in pilots], color=COLORS['normal'], label='Synthetic normal', linewidth=1.7)
        ecdf(ax, ps, [float(row['abnormal_weight']) for row in pilots], color=COLORS['abnormal'], label='Synthetic abnormal', linewidth=1.7)
        train = [row for row in real if row['split'] == 'train']
        ecdf(ax, [float(row['raw_logit']) for row in train], color=COLORS['raw'], label='Raw training normal', linewidth=2)
        if mode != 'no_ae':
            ecdf(ax, [float(row['soft_logit']) for row in train], color=COLORS['ae'], linestyle='--', label='AE training normal', linewidth=2)
        abnormal = [row for row in real if row['split'] == 'test' and row['is_anomaly'] == '1']
        ecdf(ax, [float(row['soft_logit']) for row in abnormal], color='#c64747', linestyle=':', label='Real test anomaly', alpha=.8)
        ax.set(title=f'{name}: score distributions', xlabel='Anomaly logit →', ylabel='Cumulative fraction')
        ax.grid(alpha=.15)
        ax.legend(fontsize=7.5, loc='lower right')
    ax = axes[1, 0]
    for index, seed in enumerate(args.seeds):
        ax.plot(range(4), [summary['conditions'][mode]['real_auc']['values'][index] for mode in MODES], marker='o', label=f'Seed {seed}')
    ax.axhline(.5, color='gray', linestyle=':', label='Chance')
    ax.set(xticks=range(4), xticklabels=NAMES, ylabel='AUROC', title='Real anomaly detection ↑', ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
    ax = axes[1, 1]
    xx = np.arange(4)
    for offset, key, label, color in [(-.18, 'select_raw_mmd', 'Raw vs synthetic normal', COLORS['raw']),
                                    (.18, 'select_soft_mmd', 'AE vs synthetic normal', COLORS['ae'])]:
        condition = [summary['conditions'][mode][key] for mode in MODES]
        ax.bar(xx+offset, [item['mean'] for item in condition], .34,
               yerr=[item['sample_sd'] for item in condition], color=color, label=label, capsize=3)
    ax.set(xticks=xx, xticklabels=NAMES, ylabel='Squared MMD', title='Selection-normal score gap ↓')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.15)
    ax = axes[1, 2]
    degree_changes = [[r['prior_parameters']['selected']['mean_degree']-
                       r['prior_parameters']['initial_ae']['mean_degree'] for r in records[mode]]
                      for mode in MODES]
    condition = [stats(value) for value in degree_changes]
    ax.bar(xx, [item['mean'] for item in condition], color=['#aaa', '#aaa', '#2878b5', '#8560a9'],
           yerr=[item['sample_sd'] for item in condition], capsize=3)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xticks=xx, xticklabels=NAMES, ylabel='Mean-degree parameter − initial AE value', title='Prior adaptation from matched start')
    ax.grid(axis='y', alpha=.15)
    ax = axes[1, 3]
    condition = [summary['conditions'][mode]['structure_connected'] for mode in MODES]
    ax.bar(xx, [item['mean'] for item in condition], color=['#aaa', '#aaa', '#2878b5', '#8560a9'],
           yerr=[item['sample_sd'] for item in condition], capsize=3)
    raw_connected = records['no_ae'][0]['structural_coverage']['connected']['real_fit_mean']
    ax.axhline(raw_connected, color='#c64747', linestyle=':', label='Real fit normals')
    ax.set(xticks=xx, xticklabels=NAMES, ylabel='Connected fraction', title='Hard synthetic-normal graphs', ylim=(0, 1.07))
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.15)
    fig.suptitle(f'Level C: differentiating prior fitting, component GIN, mixture weights and anomaly GIN • ECDF seed {plot_seed}', fontsize=16)
    fig.text(.5, .005, f'Fixed-prior control shares initial AE prior; identical base noise • bars: mean ± sample SD across {len(args.seeds)} seeds • test labels used only after selection', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(root/'comparison.png', dpi=180, bbox_inches='tight')
    fig.savefig(root/'comparison.pdf', bbox_inches='tight')
    # Plot selection history separately, retaining every seed rather than choosing a good-looking run.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    any_history = False
    for column, mode in enumerate(['fixed_prior', 'level_c']):
        for seed, record in zip(args.seeds, records[mode]):
            history = record['diagnostics'].get('history', [])
            selection = [row for row in history if row.get('split') == 'select' and 'total' in row]
            updates = [row for row in history if row.get('split') == 'meta' and 'gradient_norm' in row]
            if selection:
                any_history = True
                axes[0, column].plot([row['step'] for row in selection], [row['total'] for row in selection], marker='.', label=f'Seed {seed}')
                selected = record['diagnostics']['selected_step']
                selected_record = next((row for row in selection if row['step'] == selected), None)
                if selected_record:
                    axes[0, column].scatter([selected], [selected_record['total']], marker='*', s=100, zorder=4)
            if updates:
                axes[1, column].plot([row['step'] for row in updates], [row['gradient_norm'] for row in updates], marker='.', label=f'Seed {seed}')
        axes[0, column].set(title=f'{NAMES[MODES.index(mode)]}: normal-only selection', xlabel='Outer AE step', ylabel='Selection objective ↓')
        axes[1, column].set(title='Unclipped outer gradient', xlabel='Outer AE step', ylabel='Gradient norm', yscale='symlog')
        for row in range(2):
            axes[row, column].grid(alpha=.15)
            if axes[row, column].get_legend_handles_labels()[0]:
                axes[row, column].legend(fontsize=8)
    if any_history:
        fig.suptitle('Learning stability; stars mark selected checkpoints', fontsize=14)
        fig.tight_layout()
        fig.savefig(root/'learning_curves.png', dpi=180, bbox_inches='tight')
        fig.savefig(root/'learning_curves.pdf', bbox_inches='tight')
    plt.close('all')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
