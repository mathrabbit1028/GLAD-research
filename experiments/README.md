# Experiment profiles

Use the terminal UI's Profiles menu to save JSON profiles here when you want
them versioned with your research.
Profiles select benchmarks/candidates, datasets, seeds and parameter overrides.
They also record GPU selection (`gpu_devices`, empty for automatic), concurrency
(`max_parallel_jobs`, 0 for one job per available GPU) and startup minimum free
memory (`min_free_gpu_memory_mb`, default 1024). GPU indices refer to the worker
interpreter's visible devices. Older profiles inherit these defaults.
Run outputs are separate under outputs/experiments/. Opening a profile never
starts an experiment. Existing JSON profiles remain compatible. See the project
README and docs/tui.md for the terminal workflow.
