"""Unfiltered proposal comparisons, separate from detector fitting."""
import numpy as np


def separation_auc(normal, abnormal):
    """P(score_A > score_N) + 0.5 P(tie), without a pairwise matrix."""
    normal, abnormal = np.sort(np.asarray(normal, dtype=float)), np.asarray(abnormal, dtype=float)
    if not len(normal) or not len(abnormal) or not (np.isfinite(normal).all() and np.isfinite(abnormal).all()):
        raise ValueError('Comparison requires two finite, nonempty score arrays')
    lower = np.searchsorted(normal, abnormal, side='left')
    upper = np.searchsorted(normal, abnormal, side='right')
    return float(((lower + upper) / (2 * len(normal))).mean())


def compare_scores(normal, abnormal, seed=0, bootstrap=1000):
    normal, abnormal = np.asarray(normal), np.asarray(abnormal)
    estimate = separation_auc(normal, abnormal)
    rng = np.random.default_rng(seed)
    samples = [separation_auc(rng.choice(normal, len(normal), replace=True),
                              rng.choice(abnormal, len(abnormal), replace=True)) for _ in range(bootstrap)]
    return dict(auc=estimate, bootstrap_95_interval=np.quantile(samples, [.025, .975]).tolist(),
                bootstrap_replicates=bootstrap,
                interval_scope='pilot Monte Carlo uncertainty conditional on this fitted model',
                median_ratio=float(np.median(abnormal) / max(np.median(normal), np.finfo(float).tiny)),
                mean_log1p_difference=float(np.log1p(abnormal).mean() - np.log1p(normal).mean()))


def summarize_scores(values, inner, outer):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Scores must be finite, nonnegative, and nonempty')
    inside = values <= 1
    shell = (values >= 1 + inner) & (values <= 1 + outer)
    return dict(n=len(values), mean=float(values.mean()),
                quantiles=dict(zip(['q01', 'q10', 'q25', 'median', 'q75', 'q90', 'q99'],
                                   np.quantile(values, [.01, .1, .25, .5, .75, .9, .99]).tolist())),
                inside_count=int(inside.sum()), inside_fraction=float(inside.mean()),
                shell_count=int(shell.sum()), shell_fraction=float(shell.mean()))
