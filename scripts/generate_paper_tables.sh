#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m src.utils.aggregate
"$PYTHON_BIN" -c 'from src.utils.latex_exporter import export_mean_std; export_mean_std("outputs/results/results.csv", "outputs/results/table.tex")'
