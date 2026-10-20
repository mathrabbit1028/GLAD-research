# Terminal workbench

Launch `python launch_tui.py` or `python -m src.tui` in an interactive Linux
terminal. Remote sessions can use `ssh -t`; no X server, browser or desktop
session is needed. Use a terminal of at least 60 columns by 16 rows; 100 by 30
or larger gives more space for parameters and logs. Run inside tmux or screen
to keep the session available across SSH disconnects.

Install the CUDA 12.1 environment from `requirements.txt`. The UI uses Python's
curses module and PyYAML. Select this environment's Python under **Execution
settings**. Planning and editing remain available without a GPU; training and
evaluation require a working CUDA device.

## Controls

| Screen | Keys |
|---|---|
| Menus | Up/Down or k/j; Enter selects; q/Esc returns |
| Long menus and preview/results | PgUp/PgDn, Home/End |
| Methods/datasets | Space or Enter toggles; a selects all; n clears |
| Methods | b selects only benchmarks; c selects only candidates |
| Parameter list | Enter edits; r clears overrides for the displayed scope |
| Text entry | Enter accepts; Esc cancels; Ctrl-U clears; arrows/Home/End edit |
| Run monitor | Up/Down selects a job; Tab cycles active jobs; f follows an active job; s stops all jobs |
| Main menu | q/Esc or Quit exits; an active run offers Keep running or Stop and quit |

**Seeds** supports a count and first seed, or explicit comma/space-separated IDs.
**Hyperparameters** exposes model, dataset and shared protocol scopes. Values
are parsed as YAML/JSON data; use `true`/`false` for booleans. Modified values
have a `*`. Types and supported settings are checked during preview, save and
start. The editor does not rewrite source YAML configs.

**Profiles** opens JSON files from `experiments/` or a typed path, saves profiles
and restores suite defaults. Existing desktop-workbench JSON profiles use the
same schema. Set the experiment Python path for this server and clear/change a
fixed output directory before rerunning. Paths entered in the UI resolve
relative to the repository root; `~` expands to the current user's home.

**Preview plan** displays job count, missing data, profile and first resolved
job config. **Start experiment** is the only action that launches workers.
Opening the application or a profile, previewing, editing and creating a model
do not train or evaluate anything.

## Parallel GPU jobs

Under **Execution settings**, configure:

| Setting | Default | Behavior |
|---|---|---|
| GPUs | `auto` | Discover all visible usable GPUs; enter `0,1` or `cuda:0,cuda:1` to restrict the pool |
| Maximum parallel jobs | `0` | Automatic: one job per available GPU; `1` runs sequentially; larger values cap concurrency |
| Minimum free GPU memory | `1024` MiB | Skip GPUs below this amount during startup discovery |
| Threads per job | `1` | Numerical-library thread limit for each concurrent job |

GPU numbers refer to the selected experiment interpreter's CUDA visibility,
including any inherited `CUDA_VISIBLE_DEVICES` mapping. Profiles record
`gpu_devices` as a list (empty means automatic), `max_parallel_jobs` and
`min_free_gpu_memory_mb`. Older profiles inherit these defaults. For compatibility,
`device: cuda`/`auto` allows automatic discovery; `device: cuda:N` restricts an
otherwise empty pool to that GPU. CPU execution is disabled. Editing **GPUs**
replaces that legacy device selection.

Start checks each candidate GPU with a small CUDA operation in the selected
interpreter. LGKDE also checks DGL graph transfers. Unusable GPUs and GPUs below
the free-memory threshold are skipped. Where NVIDIA monitoring is available,
GPUs reporting another compute process are skipped. If no GPU qualifies, no
run directory or training/evaluation jobs are created. The discovered pool and
skip reasons are recorded in the manifest after successful startup.

The scheduler runs at most one job per GPU and retains that slot through
training, calibration and evaluation. As a job finishes, that GPU takes the
next queued combination. Each worker sees only its assigned GPU as `cuda:0`;
the manifest, log and CSV/JSON exports also record the original visible device
and GPU UUID. Parallel completion order does not change split pairing or seed
aggregation. Direct workers retain their GPU checks before loading datasets.

Before dispatch, the scheduler rechecks reported compute processes. If a GPU
in the discovered pool becomes busy externally, dispatch to it waits; other
idle pool GPUs can continue. Stop cancels that wait. Startup-skipped GPUs are
not added later during this experiment. NVIDIA process queries are advisory:
they cannot reserve a GPU against another application, and monitoring may be
unavailable in some environments. Without process monitoring, scheduling uses
the startup usability/memory checks. Choose an explicit pool or a cluster GPU
allocation on shared servers. The free-memory threshold does not predict a
model's memory requirement.

**Run monitor** shows progress, active jobs and GPU assignments, selected job
status and a live tail of its log. **Tab** cycles active job logs; **f** follows
the selected active job until it finishes.
Returning to the main menu leaves the run active. **Results / export** opens
the active run or another directory containing `manifest.json`; Show / refresh
loads completed results, including partial seed groups. Exports write CSV,
JSON and LaTeX into a new directory. Training-only, failed and cancelled jobs
are excluded from evaluation summaries. Full logs remain in each job folder.

## Implementation

- `src/tui/`: curses screens, profile editing, queued worker events and log tails.
- `src/experiments/catalog.py`: discovers configs and scaffolds candidates.
- `src/experiments/plan.py`: profile schema and resolved job-plan validation.
- `src/experiments/runner.py`: source snapshots, subprocesses, manifests,
  cancellation and aggregation.
- `src/utils/device.py`: GPU-only device selection and runtime checks.
- `src/train.py` and `src/evaluate.py`: isolated training/evaluation workers.

Only the main thread draws the UI. A background coordinator supplies copied
events, and model code executes in the selected interpreter's subprocesses.
Planning imports no Torch, PyG, DGL or model adapters; optional validators must
remain lightweight. Editing the next profile during a run cannot change its
frozen spec. Child commands use argument lists without a shell.

Cancellation terminates all active child process groups and cancels pending
jobs. Completed outputs remain. Quitting during a run waits for cleanup;
Ctrl-C, SIGTERM and SIGHUP request the same cleanup. Stopped jobs are not
resumable. Use a new experiment for unfinished combinations. For persistent
remote work, launch inside tmux/screen rather than relying on a live SSH link.

No terminal UI, tests, training or evaluation runs were performed for this
change, following the user's deferred-running instruction. Runtime GPU discovery
and parallel scheduling have not been executed for this update.
