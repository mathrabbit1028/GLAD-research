# GLAD Lab

A terminal framework with GPU-only execution for developing graph-level anomaly
detectors and comparing them with nine published benchmark methods. Your candidate models
and the benchmark baselines use the same data splits, features, calibration
and evaluation pipeline.

The source comparison is
[`reports/normal_only_graph_anomaly_detection_report.pdf`](reports/normal_only_graph_anomaly_detection_report.pdf).
The nine methods are comparison baselines: **LGKDE, SIGNET, MUSE, UniFORM,
CVTGAD, GLADC, OCGIN, OCGTL and GLocalKD**. Your methods belong in the separate
candidate model area. Candidate documentation: [AGDiff](docs/agdiff.md) and
[GRAFT-AD](docs/graft_ad.md).

MUSE is independently implemented and UniFORM is an experimental paper-based
adaptation with unverified fidelity. Configuring a baseline does not establish
reproduction of its published scores. [Method notes](docs/methods.md) and
[source metadata](reports/methods.json) record these distinctions.

**Terminal UI execution, training and evaluation remain deferred at the
user's request. This interface has not been runtime validated.**

## 1. Install and open the terminal UI

Use Python 3.10 on Linux x86_64 with an NVIDIA GPU and a driver supporting
CUDA 12.1. `requirements.txt` is the single environment definition. It pins
CUDA builds of PyTorch, torch-scatter and DGL alongside all framework/UI
dependencies:

```bash
python -m pip install -r requirements.txt
python launch_tui.py
```

`python -m src.tui` opens the same workbench. It runs in an interactive terminal,
including SSH, using Python's curses module. No desktop display is required.
Use `ssh -t` when your connection does not allocate a terminal. For long runs,
start the UI inside tmux or screen so you can reconnect after an SSH disconnect.
A 100-column by 30-row terminal is comfortable; the minimum is 60 by 16.
If curses is missing, use a Python build with ncurses support.

In **Execution settings**, set **Experiment Python** to this GPU environment's
absolute Python path. The UI reads configs without importing model dependencies.
Opening the UI, editing profiles, previewing and exporting results do not require
a GPU. Only **Start experiment** launches training/evaluation workers.

Training and evaluation require CUDA. Start discovers usable GPUs in the
selected experiment interpreter and checks a small CUDA operation on each.
If LGKDE is selected, it also checks DGL graph transfers. With no available
GPU, run-directory creation and all training/evaluation jobs are blocked. Direct
`src.train` and `src.evaluate` entry points enforce the same GPU requirement
before loading datasets. There is no CPU fallback. Legacy profiles using
`auto` select CUDA and fail if it is unavailable; CPU selections are rejected.
The manifest records the GPU pool, assignments, PyTorch/CUDA and, when used, DGL.

### Repair an existing DGL installation

`Device API cuda is not enabled` means LGKDE is moving graphs to CUDA with a
CPU-only DGL installation. In an existing environment whose PyTorch build uses
CUDA 12.1, replace just DGL using that environment's Python:

```bash
conda activate agdiff
python -m pip install --force-reinstall --no-deps --no-cache-dir --only-binary=:all: \
  --no-index --find-links https://data.dgl.ai/wheels/cu121/repo.html \
  'dgl==1.1.3+cu121'
```

This command preserves installed PyTorch/CUDA and other dependencies. For a
complete environment matching the repository's pins, install `requirements.txt`.
The [official DGL CUDA wheel index](https://data.dgl.ai/wheels/cu121/repo.html)
provides the GPU build; the plain PyPI DGL wheel is CPU-only.

`ModuleNotFoundError: No module named 'torchdata.datapipes'` can arise when DGL
2.1 imports GraphBolt against a TorchData release without DataPipes. This
framework uses DGL 1.1.3, which does not need that GraphBolt import. Its CUDA
wheel fixes both the DataPipes dependency and the CPU/CUDA build mismatch.
See [DGL's GraphBolt import](https://github.com/dmlc/dgl/blob/v2.1.0/python/dgl/graphbolt/base.py)
and [TorchData's removal notice](https://github.com/pytorch/data/releases).

The `torch.load` FutureWarnings in older PyG releases are separate from this
CUDA exception. Restart the terminal UI after repairing the environment, select
the repaired interpreter, and use a new experiment output directory. Existing
failed run artifacts and frozen source snapshots remain unchanged.

## 2. Configure an experiment

Navigate with **Up/Down** or **k/j**, select with **Enter**, and return with
**q** or **Esc**. **PgUp/PgDn** and **Home/End** navigate longer menus and views.
Text fields support arrows, Backspace/Delete and **Ctrl-U** to clear. Enter
accepts a field; Esc cancels it. See [terminal controls](docs/tui.md).

1. In **Methods**, toggle individual candidates/benchmarks using Space or Enter.
   **a** selects all, **n** clears, **b** selects only benchmarks, and **c** selects
   only candidates. UniFORM is marked experimental.
2. In **Datasets**, include/exclude datasets the same way. Missing processed
   data is marked and must be prepared before starting.
3. In **Seeds**, choose a count and first seed, or enter exact comma/space-separated
   seed IDs. Count 5 and first seed 0 produce `0,1,2,3,4`.
4. In **Execution settings**, set name, **GPUs** (`auto` or IDs such as `0,1`),
   **Maximum parallel jobs**, minimum free GPU memory, numerical-library threads
   per job and experiment interpreter. Leave the output directory blank for a
   timestamped directory; a fixed path must not exist.
5. In **Hyperparameters**, select a scope, target and parameter, then press Enter
   to edit its YAML value. **r** clears overrides for that scope/target.

The header shows the job count. The default selection is nine benchmarks x
12 datasets x five seeds = **540 jobs**. Candidates are selected explicitly.
Jobs execute in parallel with at most one job per available GPU. The default
**Maximum parallel jobs = 0** uses the discovered pool; set `1` for sequential
execution or another positive value to cap concurrency. Each slot covers a
job's training, calibration and evaluation. **Train + calibrate only** skips final evaluation;
training-only jobs do not enter comparison results.

**GPUs = auto** discovers all usable devices visible to the experiment interpreter.
Explicit IDs use that interpreter's `CUDA_VISIBLE_DEVICES` numbering. The default
minimum free memory is 1024 MiB at discovery; this is not a model memory estimate.
Reported external compute processes are checked at discovery and before dispatch
when `nvidia-smi` is available. Checks are advisory and cannot reserve hardware
against other applications. Use an assigned GPU pool on shared servers.
See [parallel scheduling details](docs/tui.md#parallel-gpu-jobs).

| Parameter scope | Examples | Applies to |
|---|---|---|
| Model | hidden dimension, layers, learning rate, epochs, sample generation | One method across its selected datasets/seeds |
| Dataset | batch size, maximum node count | One dataset across all selected methods/seeds |
| Shared protocol | train/calibration fractions, normal class, threshold quantile | Every selected job |

Values retain their types: use `true`/`false` and lists such as `[0.1, 1.0, 10.0]`.
`max_nodes` accepts `auto` or a positive integer; `normal_label` accepts `majority`
or an integer PyG class index. Leave references such as `${dataset.max_nodes}`
at their default unless deliberately overriding them. Modified fields have a `*`.
Types and supported settings are checked when previewing, saving and starting.
Changes stay in the profile; source YAML configs are not rewritten. The UI does
not search hyperparameters or select settings using test scores.

**Profiles** opens JSON profiles from `experiments/` or another typed path, saves
selections/settings/overrides, or restores suite defaults. Existing desktop GUI
JSON profiles remain compatible. Update their Python path for this server and
clear/change a fixed output path before rerunning. **Refresh configs** discovers
new or edited configs. Keep config IDs stable so saved profiles can find them.
Paths typed in the UI resolve relative to the repository root; `~` is supported.

**Preview plan** shows the job count, missing processed datasets, profile and
first resolved job config. It starts no workers and creates no run directory.

## 3. Run, monitor and compare

Choose **Start experiment** to validate the plan, check the selected interpreter's
GPU, snapshot sources/configs and launch isolated subprocesses. Editing a profile
while a run is active does not change its frozen setup. Keep environment
packages and datasets stable during the run.

**Run monitor** shows progress, concurrent jobs with GPU assignments, job status
and a live log tail. Up/Down selects a job; **Tab** cycles active jobs; **f**
follows an active job; **s** stops all active process groups and cancels
pending jobs. **q** returns to the menu while work continues. Full logs remain
in the job folders. The UI allows one active experiment at a time.

Quitting during a run offers **Keep running** or **Stop and quit**. Ctrl-C also
requests cancellation and waits for worker cleanup. Completed artifacts remain;
stopped jobs are not resumable. Start a new experiment for unfinished combinations.

**Results / export** loads the active run or another directory containing
`manifest.json`. **Show / refresh results** displays AUROC, average precision
and F1 as mean +/- sample standard deviation in percent, benchmark/candidate
roles and completed/expected seed counts. CSV/JSON exports include original
GPU device IDs and UUIDs, while workers use isolated local `cuda:0` devices.
Partial groups are marked explicitly;
one seed has no estimated deviation. Return to the results menu and show again
to refresh a running experiment. **Export CSV / JSON / LaTeX** writes a new folder.

Failed, cancelled and training-only jobs do not contribute evaluation metrics.
The manifest/logs retain errors; aggregation checks result identities, duplicate
runs and paired split hashes. Small score differences alone do not establish
statistical significance.

## 4. Add your own model

### Create a candidate

Use **New candidate model** in the terminal UI and enter a unique lowercase ID, such as
`my_detector`. It creates:

```text
src/models/candidates/my_detector.py
configs/models/my_detector.yaml
```

The starter is a complete example of the interface: it reconstructs pooled
node features with an MLP. It ignores most graph structure and is not a novel
method or a published benchmark. Replace its model and objective with your
own implementation. The terminal UI selects the new candidate, but does not execute it.

You can also copy [`templates/model_adapter.py`](templates/model_adapter.py)
and [`templates/model_config.yaml`](templates/model_config.yaml) manually,
rename the YAML ID and adapter module, then choose **Refresh configs**.
No registry edit or benchmark-file change is needed.

### Implement the adapter contract

Subclass [`BaseAnomalyModel`](src/models/base_model.py). The constructor receives
`input_dim`, the model config and the selected `torch.device`. Implement:

| Method | Responsibility |
|---|---|
| `fit(train_loader, config, device)` | Fit on normal training graphs only. |
| `anomaly_score(batch)` | Return a finite tensor of shape `[batch.num_graphs]`; larger means more anomalous. |
| `state_dict()` | Return checkpointable parameters and all fitted scoring state. |
| `load_state_dict(state)` | Restore everything needed to reproduce scores on the selected device. |

Useful rules for a comparable implementation:

- Infer feature width from `input_dim`; do not hardcode a dataset's feature count.
- Use the provided device. Handle different graph sizes, isolated nodes,
  edgeless graphs and final minibatches, including one graph where applicable.
- Iterate the supplied training loader/subset. Do not reach into its backing
  dataset to fit on calibration/test graphs, inspect anomaly labels, or choose
  checkpoints using final-test metrics.
- Fit reference embeddings, centers, preprocessing statistics and negative
  generation on training normals only. Keep test graphs out of training and
  contrastive reference construction.
- Return one score per graph. Scoring a graph alone or in another batch should
  retain its meaning; do not use other held-out graphs as an implicit reference.
- Save training references, normalization statistics and centers along with
  network weights. Checkpoints load with `weights_only=True`: use tensors and
  standard dict/list/scalar containers, not pickled custom objects.
- Use `eval()` and deterministic scoring where appropriate. Expose tunable
  values in YAML rather than embedding dataset-specific constants in code.

For an ordinary `nn.Module`, saving/restoring its `state_dict()` may be enough.
A density or memory-based model also needs its fitted reference state.
The framework handles split creation, threshold calibration, checkpoint files,
final metrics and score exports.

### Define the model config

A candidate config looks like:

```yaml
name: my_detector
role: candidate
display_name: My detector
adapter: src.models.candidates.my_detector.CandidateAdapter
architecture:
  hidden_dim: 32
training:
  epochs: 100
  learning_rate: 0.001
```

The filename must be `my_detector.yaml`, and `name` must match it. IDs must be
unique across candidates and benchmarks. The adapter must be a subclass under
`src.models.*`. The terminal UI discovers parameter leaves automatically; mappings
under `architecture`, `training` or additional model settings become editable.
Metadata fields (`name`, `adapter`, `validator`, `role`, `display_name`, `description`) are
not hyperparameters.

An optional `validator: src.models.candidates.my_config.validate_config` points
to a function accepting the resolved model config and raising `ValueError` for
unsupported settings. Keep that module free of Torch/PyG/DGL imports: the terminal UI
calls it during planning. The factory also calls it before constructing the
adapter. See the [validator example](src/models/candidates/agdiff_config.py).

For multiple variants, create separate candidate modules/configs with different
IDs, or save several terminal UI profiles with different overrides. Model code in
`src/models/benchmarks/` and pinned sources in `src/models/vendor/` are reserved
for comparison baselines and their upstream dependencies.

## 5. Data and protocol

The default datasets are **MUTAG, PROTEINS, DD, ENZYMES, DHFR, BZR, COX2, AIDS,
IMDB-BINARY, NCI1, COLLAB and REDDIT-BINARY**. Their raw/processed files have
already been prepared by the user. Additional existing datasets are selectable
separately in the terminal UI.

[`configs/protocol/normal_only.yaml`](configs/protocol/normal_only.yaml) defines:

- 80% of the chosen normal class for fitting, 10% for normal-only calibration,
  and remaining normals for testing; integer rounding applies.
- Every other class held out as anomalies for final testing. The majority class
  is normal, with the lowest PyG index breaking ties. ENZYMES defaults to class
  0 versus its other five classes.
- TU node features retained; featureless nodes receive `[1, log(1 + degree)]`.
- Fixed final epochs for model selection and a threshold at the 99th percentile
  of held-out normal calibration scores.
- AUROC, anomaly-positive average precision, F1, accuracy, latency and CUDA memory
  where available. AP uses stepwise precision-recall integration.

These are disclosed local comparison settings, not recovered optimal settings
for every paper. [Protocol notes](docs/protocol.md) explain details. Avoid using
test scores to repeatedly tune your candidate if you want an unbiased final
comparison. Shared terminal UI protocol changes are saved in the profile and manifest.

To add another dataset, create `configs/dataset/<id>.yaml` with `name`,
`data_root`, `max_nodes: auto`, `batch_size` and any method-related metadata
such as `latent_dim`. Prepare its TU files under
`data/TU/<NAME>/{raw,processed}/`, then choose **Refresh configs** in the terminal UI. A future rebuild from
local raw files can use `python scripts/prepare_tu_data.py --dataset NAME`.
Only add its ID to `configs/suite/default.yaml` if it should become preselected
by default. The preparation script does not download data.

## 6. Files and reproducibility

```text
configs/
  benchmarks/            nine published baseline configs
  models/                your candidate configs
  dataset/               default and newly registered datasets
  extra/dataset/         additional existing datasets
  protocol/              shared normal-only comparison rules
  suite/default.yaml     preselected benchmarks/datasets/seeds
  config.yaml            single-run and protocol composition defaults
src/
  tui/                   curses terminal interface; no eager model imports
  experiments/           catalog, profile/plan validation and shared runner
  utils/device.py        GPU requirement and CUDA runtime checks
  models/
    benchmarks/          published baseline adapters
    candidates/          your model adapters
    vendor/              pinned upstream cores, licenses and hashes
    base_model.py        shared adapter contract
  data/, evaluation/     splitting, graph features, calibration and metrics
  train.py, evaluate.py  worker entry points used by the experiment engine
experiments/             saved JSON profiles
templates/               candidate module/config starter files
outputs/experiments/     isolated experiment artifacts
reports/, docs/          research sources, provenance and implementation notes
scripts/                 optional automation; terminal UI is the main workflow
```

An experiment directory contains `experiment.json`, `manifest.json`, source and
data hashes, environment packages and a source snapshot. Each job has its
planned/resolved configs, exact split indices, log, checkpoint, calibrated
threshold and final metrics/scores when evaluation was enabled. Completed
metrics automatically generate `summary/`; terminal UI exports can write new folders.
Raw/processed data stays in place and is not copied into each source snapshot.
Model imports use `src.models.benchmarks.*` or `src.models.candidates.*`.
Saved configs that name an adapter must use these paths; the former root-level
model import wrappers have been removed.

The optional CLI uses the same engine and saved profiles:

```bash
python scripts/run_benchmarks.py --profile experiments/my_comparison.json
```

For example, append `--gpus cuda:0 cuda:1 --max-parallel-jobs 2` to restrict an
automated run to two visible GPUs. Saved profiles store `gpu_devices`,
`max_parallel_jobs` and `min_free_gpu_memory_mb`; older profiles inherit defaults.

Existing single-run entry points remain available for development, for example
`python -m src.train model=my_detector dataset=mutag seed=0`.
They use `outputs/runs/<model>/<DATASET>/seed<seed>/` by default and can overwrite
a previous direct run at the same path. TUI/matrix experiments require new
output directories and preserve prior runs.

The current framework uses the files and entry points listed above. Obsolete
framework documentation, dependency snapshots and model wrappers have been
removed. Existing research data and experiment outputs are retained; historical
scores do not validate this implementation or terminal UI.
