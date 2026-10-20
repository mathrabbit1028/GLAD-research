# AGDiff candidate integration

AGDiff is a user candidate, selected in the terminal UI alongside the nine benchmark
methods. This is the repository's local paper-based implementation; integration
does not establish an exact reproduction or high performance. The reference
paper is [`papers/AGDiff(2025).pdf`](<../papers/AGDiff(2025).pdf>).
No tests, training, evaluation or terminal UI validation were run for this adaptation.

## Files and contract

- `configs/models/agdiff.yaml`: candidate identity and actual model controls.
- `src/models/candidates/agdiff.py`: the `BaseAnomalyModel` adapter.
- `src/models/candidates/agdiff_config.py`: lightweight configuration validation.
- `src/models/candidates/agdiff_impl/agdiff_model.py`: local neural components.

Use `src.models.candidates.agdiff.AGDiffAdapter` for the adapter and
`src.models.candidates.agdiff_impl.agdiff_model.AGDiff` for the core. The earlier
root-level import wrappers have been removed; update custom imports or saved
adapter paths accordingly. Network state keys are retained. Older saved configs
can omit the new execution controls; their defaults preserve the previous
dense/sparse boundary and reverse-checkpointing settings.

The adapter infers feature width from the framework's `input_dim`, uses the
selected device, and passes only node features, edge indices and graph sizes
to the model. Dataset labels, split metadata and input edge attributes are not
used by this node-feature/unweighted-edge implementation. The loader batch is
not mutated. Variable graph sizes, a singleton minibatch, isolated nodes and
edgeless graphs follow the same graph-batch interface as the benchmarks.

## Terminal UI settings

Select **agdiff** in **Methods**, or open
[`experiments/agdiff_vs_benchmarks.json`](../experiments/agdiff_vs_benchmarks.json)
from **Profiles** and choose the experiment interpreter under **Execution
settings**. The full profile selects 600 jobs; loading
it does not execute them. Under **Hyperparameters → Model → agdiff**,
configure:

| Parameter | Default | Effect |
|---|---:|---|
| `architecture.hidden_dim` | 64 | GIN and detector hidden width |
| `architecture.latent_dim` | 16 | Node latent width, owned by the model |
| `architecture.gin_layers` | 3 | GIN depth |
| `architecture.backbone` | GIN | Supported encoder; other names are rejected |
| `architecture.detector_spectral_norm` | false | Optional detector spectral normalization |
| `training.optimizer` | Adam | Adam or AdamW in both training phases |
| `training.learning_rate` | 0.001 | Optimizer learning rate in both phases |
| `training.weight_decay` | 0 | Passed explicitly to the optimizer |
| `training.pretrain_epochs` | 100 | VAE pretraining budget |
| `training.joint_epochs` | 200 | Diffusion/detector training budget |
| `training.kl_coefficient` | 0.001 | VAE KL penalty |
| `training.diffusion_steps` | 1000 | Forward/reverse diffusion schedule length |
| `training.lambda_diff` | 1 | Diffusion loss weight in the joint objective |
| `training.gradient_clip_norm` | 5 | Per-component gradient norm cap |

**Dataset → batch_size** controls the real loader batch size. The former
`model.training.batch_size` entry was redundant and is removed from the active
config. Latent width is now a model hyperparameter, rather than a reference to
dataset metadata. Neither training phases nor diffusion steps are silently
shortened for terminal UI runs.

The optional config `validator` hook checks supported choices, finite numeric
settings and positive dimensions/budgets during plan preview/start and model
construction. It imports no Torch, PyG or DGL, preserving lightweight planning.

## Normal-only training and scoring

1. Pretrain the VAE on the supplied normal training loader. Diffusion and
   detector parameters are inactive during this phase.
2. Freeze the VAE parameters and its training mode. Fit diffusion and detector
   using real normal graphs and generated pseudo-anomalies. Gradients from
   classification still pass through the frozen decoder to generated latents.
3. Switch to evaluation mode. Inference returns the **negative detector normality
   logit**, one finite scalar per graph, so higher means more anomalous. No
   diffusion sampling or held-out-graph contrastive context is used for scoring.

Each `fit` re-enables the appropriate phase parameters. The adapter logs total
loss plus classification and diffusion losses in the joint phase, and rejects
non-finite losses/gradient norms. Parameters and diffusion schedule buffers are
saved as a plain network state dictionary compatible with the framework's
`weights_only=True` loading. No extra fitted reference set is required.

The framework, rather than AGDiff, creates seeded splits, calibrates the operating
threshold on held-out normal graphs, saves checkpoints and computes final
metrics. Test labels do not affect model fitting or threshold selection.

## Larger graphs and memory

The following are editable model settings:

- `architecture.reconstruction_block_size` (512): evaluate the VAE edge BCE
  objective over row blocks, excluding self pairs and cross-graph pairs. Blocks
  of large graphs use gradient checkpointing. All within-graph edge/non-edge
  pairs are retained; floating-point reduction order differs from the former
  full matrix calculation. This bounds stored pairwise activations, while
  total reconstruction computation remains quadratic in node count.
- `architecture.dense_decode_max_nodes` (512): use the existing dense soft
  pseudo-graph decoder up to this size. Above it, use observed edges plus sampled
  non-edges and the existing straight-through edge weighting. This sparse
  branch is an approximation, not the full dense decoder. A pool with no
  candidate edges produces an empty edge tensor rather than a concatenation
  failure.
- `training.reverse_checkpointing` (true) and `training.reverse_chunk_size` (16):
  recompute reverse-diffusion segments during backward to limit stored
  activations. This preserves the configured number of diffusion steps at
  additional computation cost. The small-schedule path retains the existing
  behavior of reversing without checkpoint chunks below 100 steps.

These controls do not guarantee a given hardware budget. Large DD/REDDIT graphs
and long diffusion schedules can remain expensive. Changes to the dense/sparse
boundary alter generation and must be recorded when comparing results.
