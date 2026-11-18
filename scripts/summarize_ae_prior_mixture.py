#!/usr/bin/env python3
"""Summarize prespecified matched Level B controls without selecting on test."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    modes = ['no_ae', 'frozen_gin', 'level_b']
    names = ['No AE', 'Frozen GIN', 'Level B']
    records = {mode:[] for mode in modes}
    for seed in args.seeds:
        pilot_hashes, bank_hashes = [], []
        for mode in modes:
            path = args.root/f'seed{seed}'/mode
            manifest = json.loads((path/'manifest.json').read_text())
            if manifest['status'] != 'completed':
                raise ValueError(f'Unfinished experiment: {path}')
            pilot_hashes.append(manifest['pilot_digest'])
            bank_hashes.append(manifest['bank_sha256'])
            records[mode].append(json.loads((path/'result.json').read_text()))
        if len(set(pilot_hashes)) != 1 or len(set(bank_hashes)) != 1:
            raise ValueError('Controls used different pilots or fitted banks')
        np.testing.assert_allclose(records['no_ae'][-1]['gap']['select']['raw_synthetic_mmd'],
                                   records['frozen_gin'][-1]['gap']['select']['raw_synthetic_mmd'], atol=1e-6)
    summary = dict(seeds=args.seeds, conditions={}, paired_differences={},
        caveat='Same split and bank; variability over classifier seeds only. Test has 13 normals and 63 anomalies.')
    def stats(values):
        return dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)) if len(values)>1 else 0., values=values)
    for mode, rows in records.items():
        summary['conditions'][mode] = dict(
            synthetic_auc=stats([r['synthetic_auc'] for r in rows]),
            real_auc=stats([r['real_test']['auc'] for r in rows]),
            real_hard_auc=stats([r['real_test']['hard_auc'] for r in rows]),
            real_raw_auc=stats([r['real_test']['raw_auc'] for r in rows]),
            select_raw_mmd=stats([r['gap']['select']['raw_synthetic_mmd'] for r in rows]),
            select_soft_mmd=stats([r['gap']['select']['soft_synthetic_mmd'] for r in rows]),
            test_normal_soft_mmd=stats([r['gap']['test_normal']['soft_synthetic_mmd'] for r in rows]),
            mean_edge_edits=stats([r['edits']['train']['mean_expected_edge_edits'] for r in rows]),
            mean_node_edits=stats([r['edits']['train']['mean_expected_node_edits'] for r in rows]),
            hard_changed_graphs=[r['edits']['test']['hard_changed_graphs'] for r in rows],
            selected_steps=[r['diagnostics']['selected_step'] for r in rows])
    audit_paths = [args.root/f'seed{seed}/initialization_audit.json' for seed in args.seeds]
    audits = [json.loads(p.read_text()) for p in audit_paths] if all(p.exists() for p in audit_paths) else []
    if audits:
        summary['conditions']['initial_ae'] = dict(
            synthetic_auc=stats([r['synthetic_auc'] for r in audits]),
            real_auc=stats([r['real_auc'] for r in audits]),
            select_raw_mmd=stats([r['select_raw_mmd'] for r in audits]),
            select_soft_mmd=stats([r['select_soft_mmd'] for r in audits]),
            note='Post-run attribution control; same full inner refit with untrained near-identity AE; not used to tune or select.')
    # Paired graph bootstrap of average AUROC differences across these fixed seeds.
    from sklearn.metrics import roc_auc_score
    scores = {}
    test_ids = None
    for mode in modes:
        values = []
        for seed in args.seeds:
            with (args.root/f'seed{seed}'/mode/'real_scores.csv').open() as handle:
                rows = [r for r in csv.DictReader(handle) if r['split'] == 'test']
            current_ids = [(r['dataset_index'], r['is_anomaly']) for r in rows]
            if test_ids is None:
                test_ids = current_ids
            if current_ids != test_ids:
                raise ValueError('Test observations or order differ across controls')
            values.append(np.array([float(r['soft_logit']) for r in rows]))
        scores[mode] = np.array(values)
    labels = np.array([int(r['is_anomaly']) for r in rows])
    if audits:
        for audit in audits:
            if [(str(i),str(y)) for i,y in zip(audit['test_indices'],audit['test_labels'])] != test_ids:
                raise ValueError('Initialization audit used different test observations')
        scores['initial_ae'] = np.array([r['test_scores'] for r in audits])
    ni, ai = np.flatnonzero(labels==0), np.flatnonzero(labels==1)
    for control in ['no_ae', 'frozen_gin'] + (['initial_ae'] if audits else []):
        rng, boot = np.random.default_rng(86000), []
        for _ in range(1000):
            ids = np.r_[rng.choice(ni,len(ni)), rng.choice(ai,len(ai))]
            boot.append(np.mean([roc_auc_score(labels[ids], b[ids])-roc_auc_score(labels[ids], a[ids])
                                 for a,b in zip(scores[control],scores['level_b'])]))
        summary['paired_differences'][control] = dict(
            mean_auc_gain=summary['conditions']['level_b']['real_auc']['mean']-summary['conditions'][control]['real_auc']['mean'],
            conditional_paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(16,8.7))
    def ecdf(ax, x, weights=None, **kwargs):
        order = np.argsort(x)
        weights = np.ones(len(x))/len(x) if weights is None else np.asarray(weights)
        ax.step(np.asarray(x)[order], np.cumsum(weights[order])/weights.sum(), where='post', **kwargs)
    for index, (mode, name) in enumerate(zip(modes,names)):
        path = args.root/f'seed{args.seeds[0]}'/mode
        with (path/'pilot_scores.csv').open() as handle:
            pilots = list(csv.DictReader(handle))
        with (path/'real_scores.csv').open() as handle:
            real = list(csv.DictReader(handle))
        ax = axes[0,index]
        ps = [float(r['logit']) for r in pilots]
        ecdf(ax, ps, [float(r['normal_weight']) for r in pilots], color='#2878b5', label='Synthetic normal')
        ecdf(ax, ps, [float(r['abnormal_weight']) for r in pilots], color='#e6952e', label='Synthetic abnormal', alpha=.8)
        train = [r for r in real if r['split']=='train']
        ecdf(ax, [float(r['raw_logit']) for r in train], color='#8560a9', linestyle='--', label='Raw training normal', linewidth=2)
        if mode != 'no_ae':
            ecdf(ax, [float(r['soft_logit']) for r in train], color='#20874d', label='AE training normal', linewidth=2)
        abnormal = [r for r in real if r['split']=='test' and r['is_anomaly']=='1']
        ecdf(ax, [float(r['soft_logit']) for r in abnormal], color='#c64747', linestyle=':', label='Real test abnormal', alpha=.8)
        ax.set(title=f'{name}: score distributions (seed {args.seeds[0]})', xlabel='Anomaly logit →', ylabel='Cumulative fraction')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8,loc='lower right')
    ax = axes[1,0]
    plot_modes = ['no_ae', 'frozen_gin', 'initial_ae', 'level_b'] if audits else modes
    plot_names = ['No AE', 'Frozen GIN', 'Initial AE', 'Level B'] if audits else names
    for j, seed in enumerate(args.seeds):
        ax.plot(range(len(plot_modes)), [summary['conditions'][mode]['real_auc']['values'][j] for mode in plot_modes], marker='o', label=f'Seed {seed}')
    ax.axhline(.5,color='gray',linestyle=':',label='Chance')
    ax.set(xticks=range(len(plot_modes)),xticklabels=plot_names,ylabel='AUROC',title='Real anomaly detection (higher is better)',ylim=(.25,1))
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
    ax = axes[1,1]
    xx=np.arange(len(plot_modes))
    for offset,key,label,color in [(-.18,'select_raw_mmd','Raw vs synthetic normal','#8560a9'),(.18,'select_soft_mmd','AE vs synthetic normal','#20874d')]:
        values=[summary['conditions'][mode][key]['mean'] for mode in plot_modes]
        sd=[summary['conditions'][mode][key]['sample_sd'] for mode in plot_modes]
        ax.bar(xx+offset,values,.34,yerr=sd,color=color,label=label,capsize=3)
    ax.set(xticks=xx,xticklabels=plot_names,ylabel='Squared MMD',title='Selection-normal gap (lower is better)')
    ax.legend(fontsize=9)
    ax.grid(axis='y',alpha=.15)
    ax = axes[1,2]
    xx=np.arange(3)
    for offset,key,label,color in [(-.18,'mean_edge_edits','Expected edge edits','#2878b5'),(.18,'mean_node_edits','Expected label edits','#e6952e')]:
        values=[summary['conditions'][mode][key]['mean'] for mode in modes]
        sd=[summary['conditions'][mode][key]['sample_sd'] for mode in modes]
        ax.bar(xx+offset,values,.34,yerr=sd,color=color,label=label,capsize=3)
    ax.axhline(2,color='#2878b5',linestyle=':',alpha=.6)
    ax.axhline(.5,color='#e6952e',linestyle=':',alpha=.6)
    ax.set(xticks=xx,xticklabels=names,ylabel='Mean per training-normal graph',title='Reconstruction changes (dotted: budgets)')
    ax.legend(fontsize=9)
    ax.grid(axis='y',alpha=.15)
    fig.suptitle('Level B: fixed-bank AE → mixture weights → GIN training',fontsize=17,y=1.01)
    fig.text(.5,-.005,'70 fit / 15 feedback / 15 selection normals • test: 13 normals + 63 anomalies • bars: mean ± sample SD across classifier seeds',ha='center',fontsize=10)
    fig.tight_layout()
    fig.savefig(args.root/'comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(args.root/'comparison.pdf',bbox_inches='tight')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
