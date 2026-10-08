"""Build results.csv from completed per-run JSON files."""
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    """Combine per-run JSON records into a sorted results CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/results")
    args = parser.parse_args()
    directory = Path(args.results_dir)
    rows = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    if not rows:
        raise SystemExit(f"No result JSON files in {directory}")
    rows.sort(key=lambda row: (row["algorithm"], row["dataset"], row["seed"]))
    fields = list(rows[0])
    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(directory / "results.csv")


if __name__ == "__main__":
    main()
