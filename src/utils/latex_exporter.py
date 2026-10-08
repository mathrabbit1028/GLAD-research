"""Produce a paper-ready mean ± sample standard deviation table."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def export_mean_std(csv_path: str | Path, output_path: str | Path) -> Path:
    """Summarize completed runs as a LaTeX mean ± sample deviation table."""
    groups = defaultdict(list)
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["algorithm"], row["dataset"])].append(row)
    if not groups:
        raise ValueError("No completed experiment rows")
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{llrcc}",
             r"\hline", "Model & Dataset & Trials & AUC & F1 " + r"\\", r"\hline"]
    for (algorithm, dataset), rows in sorted(groups.items()):
        values = []
        for metric in ("auc", "f1"):
            series = [float(row[metric]) for row in rows if row[metric]]
            values.append(f"{mean(series):.4f} $\\pm$ {stdev(series):.4f}" if len(series) > 1
                          else f"{series[0]:.4f}" if series else "N/A")
        safe_algorithm = algorithm.replace("_", r"\_")
        safe_dataset = dataset.replace("_", r"\_")
        lines.append(f"{safe_algorithm} & {safe_dataset} & {len(rows)} & {values[0]} & {values[1]} " + r"\\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{TU graph anomaly detection results}", r"\label{tab:tu_glad}",
              r"\end{table}"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
