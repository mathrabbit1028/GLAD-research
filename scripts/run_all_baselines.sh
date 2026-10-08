#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
trials="$(awk '$1 == "trials:" {print $2}' configs/config.yaml)"
if [[ ! "$trials" =~ ^[1-9][0-9]*$ ]]; then
  echo "configs/config.yaml must define a positive integer trials value" >&2
  exit 2
fi
mkdir -p outputs/logs
for model_config in configs/model/*.yaml; do
  model="${model_config##*/}"
  model="${model%.yaml}"
  for dataset_config in configs/dataset/*.yaml; do
    dataset="${dataset_config##*/}"
    dataset="${dataset%.yaml}"
    dataset_name="$(awk '/^name:/ {print $2}' "configs/dataset/${dataset}.yaml")"
    for ((seed=0; seed<trials; seed++)); do
      log="outputs/logs/${model}_${dataset_name}_seed${seed}.log"
      "$PYTHON_BIN" -m src.train "model=$model" "dataset=$dataset" "seed=$seed" 2>&1 | tee "$log"
      "$PYTHON_BIN" -m src.evaluate --checkpoint "outputs/checkpoints/${model}_${dataset_name}_seed${seed}.pt" 2>&1 | tee -a "$log"
    done
  done
done
"$PYTHON_BIN" -m src.utils.aggregate
