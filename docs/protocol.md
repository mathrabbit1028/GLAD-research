# Default normal-only protocol

The active protocol lives in
[`../configs/protocol/normal_only.yaml`](../configs/protocol/normal_only.yaml).
The suite selects methods, datasets and seeds; dataset configs hold dataset
metadata and batching; model configs hold architecture and optimizer settings.
CLI overrides are applied after these defaults.

Training and evaluation require CUDA. Experiments discover visible usable GPUs
by default; an explicit `gpu_devices` list restricts the pool. A legacy
`device: cuda:N` selection restricts an empty pool to that GPU. `auto` never
falls back to CPU. The runner checks the selected interpreter's GPUs before
creating a run directory. Workers also check their assigned GPU before loading
datasets. LGKDE checks CUDA-enabled DGL.
Install the single CUDA 12.1 environment from `requirements.txt`.

Jobs execute concurrently with at most one job per pool GPU and a configurable
`max_parallel_jobs` cap (`0` means the pool's size). Training/calibration and
evaluation for a job stay on the same GPU. Workers receive a single-device
`CUDA_VISIBLE_DEVICES` setting and use local `cuda:0`; original device IDs and
UUIDs are stored per job. Only the coordinator writes the manifest and checks
paired split hashes. Completion order has no effect on seed aggregation.
Parallel runs can contend for host CPU, memory and I/O; timing remains
descriptive and should not be compared as isolated hardware measurements.

## Composition

`configs/config.yaml` selects `protocol=normal_only`, `suite=default`,
`model=lgkde` and `dataset=mutag`. Both single-run and benchmark entry points
compose these defaults. Benchmark IDs resolve from `configs/benchmarks/`; candidate IDs resolve from
`configs/models/`. Additional dataset names resolve from `configs/extra/dataset/`.
The terminal UI discovers new configs without changing the default baseline suite. Copied upstream sources live in
`src/models/vendor/`; benchmark adapters live in `src/models/benchmarks/` and your adapters in
`src/models/candidates/`. Model imports and saved adapter paths use these
directories directly.

Full-value references such as `${dataset.max_nodes}` retain their types.
Embedded references form output paths such as
`outputs/runs/${model.name}/${dataset.name}/seed${seed}`.
Resolved runtime configs record the actual graph padding size.

## Splits and class direction

For each seed, normal graphs are shuffled and partitioned into 80% training,
10% calibration and remaining test graphs. Calibration has at least one graph
when requested, and testing retains at least one normal. Every graph from a
non-normal class is an anomaly, held out for final testing. All methods use
the same split and feature policy for a given dataset/seed.

The normal class is the majority class, with the lowest normalized PyG index
breaking ties. `dataset.normal_label=<index>` fixes another class explicitly.
This supports multiclass ENZYMES as one-versus-rest. Label direction is chosen
before fitting and is never selected by test scores. This shared split is a
local protocol choice; the source report does not specify every split detail.

## Features and structural metadata

TU attributes and node-label one-hot features supplied by PyG are retained.
A featureless graph receives two node features: a constant one and the
logarithm of one plus its degree. These use the graph itself and do not fit
statistics on validation/test data. No dataset-wide degree vocabulary is
learned. Features follow TU's provided schema.

`max_nodes: auto` reads the maximum node count from the raw graph-indicator
metadata, with a graph-size fallback if that file is unavailable. This only
sets tensor padding and does not fit graph feature distributions or labels.
No graph is silently truncated. Dense methods can need substantial memory
for the largest DD and REDDIT-BINARY graphs.

## Fitting, calibration and scoring

Only training normals fit model parameters, representation classifiers,
centers, density references, contrastive contexts and score-normalization
statistics. Synthetic negatives derive from these normals. LGKDE rejects a
nonzero `mixed_anomaly_ratio`.

Training uses fixed final epochs. Held-out normal calibration scores set the
99th percentile threshold. Calibration graphs do not optimize model weights
or choose epochs. All non-normal labels are reserved for final evaluation.

Scores have one value per graph, with higher values indicating anomalies.
AUROC and anomaly-positive average precision evaluate rankings; F1 and accuracy
use the saved normal-calibrated threshold. Average precision uses stepwise
precision-recall integration rather than trapezoidal PR AUC. Latency and memory
are descriptive unless hardware and concurrent work are controlled.

## Artifacts

A single-run directory holds `resolved_config.yaml`, `split.json` and
`checkpoints/<model>_<DATASET>_seed<seed>.pt`. Checkpoints include model state,
feature dimension, config, split and threshold. Final evaluation restores
those values and writes metric JSON plus graph-indexed score CSV under
`results/`.

A TUI or CLI experiment directory adds `manifest.json`, `environment.txt`, source snapshots,
source/data hashes and a log per job. Workers run the frozen source snapshot
and planned per-job config with absolute dataset/output paths. The TUI and CLI
use the same experiment engine under `src/experiments/`. Results aggregate only from jobs marked
completed in that manifest. The summary records expected/completed seeds and
uses sample standard deviations; one seed has no estimated deviation.
Failed, interrupted and training-only jobs do not become evaluation results.

Implementation differences and reproduction limits are recorded in
[`methods.md`](methods.md) and [`../reports/methods.json`](../reports/methods.json).
