#!/usr/bin/env bash
# Consume one completed benchmark; never discovers or mixes unrelated runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
if (( $# != 1 )); then
  echo "Usage: $0 BENCHMARK_DIRECTORY" >&2
  exit 2
fi
RUN_DIRECTORY="$1"
"$PYTHON_BIN" -m src.utils.aggregate --run-dir "$RUN_DIRECTORY"
"$PYTHON_BIN" -m src.utils.latex_exporter \
  --csv "$RUN_DIRECTORY/summary/runs.csv" --output "$RUN_DIRECTORY/summary/table.tex"
