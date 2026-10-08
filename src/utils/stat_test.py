"""Paired tests on results with matching dataset and seed."""
import csv
from collections import defaultdict
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon


def compare_paired(
    csv_path: str | Path, proposed: str, baseline: str, dataset: str, metric: str = "auc",
) -> dict[str, int | float]:
    """Compare two methods using scores from matching dataset seeds."""
    if metric not in ("auc", "f1", "accuracy"):
        raise ValueError("Unsupported metric")
    by_seed = defaultdict(dict)
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["algorithm"] in (proposed, baseline):
                by_seed[int(row["seed"])][row["algorithm"]] = float(row[metric])
    pairs = [values for _, values in sorted(by_seed.items())
             if proposed in values and baseline in values]
    if len(pairs) < 2:
        raise ValueError("At least two matching seeds are required")
    proposed_values = [pair[proposed] for pair in pairs]
    baseline_values = [pair[baseline] for pair in pairs]
    return {"n_pairs": len(pairs),
            "paired_t_p": float(ttest_rel(proposed_values, baseline_values).pvalue),
            "wilcoxon_p": float(wilcoxon(proposed_values, baseline_values).pvalue)}
