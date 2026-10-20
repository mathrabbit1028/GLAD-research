"""Produce a paper-ready mean ± sample standard deviation table."""
import csv
import argparse
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
    lines = [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{llrccc}",
             r"\hline", "Model & Dataset & Trials & AUROC & AP & F1 " + r"\\", r"\hline"]
    for (algorithm, dataset), rows in sorted(groups.items()):
        values = []
        for metric in ("auc", "average_precision", "f1"):
            series = [100 * float(row[metric]) for row in rows if row.get(metric)]
            values.append(f"{mean(series):.2f} $\\pm$ {stdev(series):.2f}" if len(series) > 1
                          else f"{series[0]:.2f}" if series else "N/A")
        safe_algorithm = algorithm.replace("_", r"\_")
        safe_dataset = dataset.replace("_", r"\_")
        lines.append(f"{safe_algorithm} & {safe_dataset} & {len(rows)} & " + " & ".join(values) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Normal-only graph anomaly detection: mean and sample standard deviation in percent.}",
              r"\label{tab:tu_glad}",
              r"\end{table}"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_mean_std(args.csv, args.output))


if __name__ == "__main__":
    main()
