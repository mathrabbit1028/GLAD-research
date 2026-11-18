# Level B implementation: fixed prior bank and differentiable GIN adaptation

This implements Level B of [the mathematical design](ae_prior_mixture_bilevel.md)
through the repository's `BaseAnomalyModel` adapter, factory, and YAML configuration.
The [first experiment report](../reports/ae_prior_mixture_level_b_experiment.md)
contains matched controls, an untrained-AE attribution check, and coverage limits.
The candidate is `ae_prior_mixture_gin`. It learns from normal graphs only.
The primary score is the binary GIN's raw anomaly logit on the **soft** reconstructed
graph. Larger is more anomalous. No ellipsoids or geometry filtering are used.

## Exact implemented learning problem

The training normals are partitioned deterministically with seed 72000 into
70% fit, 15% meta, and the remainder selection. MUTAG split seed 0 therefore uses
70/15/15 normals internally, plus the existing 12 calibration normals and the
untouched-for-selection test set of 13 normals and 63 anomalies. The test set has
been inspected in earlier research; it is not a new independent confirmation set.
The original experiments used all 100 training normals for fitting, so their
scores are not a matched baseline for this protocol.

Fit the existing 64-component generator bank and component GIN on fit normals
only. Its feature statistics, temperature, parameters, and sampling distribution
are frozen. The ratio classifier still transmits gradients to its graph inputs.
Recompute the component log ratios on AE-transformed fit normals every outer step.
For soft graph inputs these are a continuous classifier-based surrogate; the
discrete generator bank does not define a literal density on soft matrices.
Differentiability must be distinguished from statistical ratio calibration.

Normal weights use 100 log-domain EM updates from uniform weights. Abnormal weights
use 60 projected gradient steps, step size 0.05, from four deterministic feasible
starts. The component cap is 0.25. Projection has the analytic active-set Jacobian,
including a zero derivative at a locally constant vertex. Combine restart solutions
with softmax of negative objective divided by 0.05. This is a **soft restart
combination**, not the old hard best-restart solution, and it has no global
optimality guarantee. Finite-step derivatives refer to these exact updates.

A fixed binary GIN initialization is pretrained on raw-normal mixture weights
for 800 Adam steps, learning rate 0.001, using a fixed pool of 64 graphs/component.
This initialization is independent of the AE. It is shared across the three
conditions for a classifier seed. The GIN has three layers of width 32, LayerNorm,
ReLU, trainable self-message coefficients, mean/max pooling, and a linear head.

For every AE update, start from this same initialization and reset momentum to
zero. Train the downstream GIN for 32 **functional momentum-SGD** steps:

\[
L_t(\theta,\phi)=\tfrac12\sum_k w_{Nk}(\phi)\overline{\mathrm{softplus}(f_\theta(S_k))}
+\tfrac12\sum_k w_{Ak}(\phi)\overline{\mathrm{softplus}(-f_\theta(S_k))}
+\tfrac{10^{-4}}2\|\theta\|^2,
\]
\[
v_{t+1}=0.9v_t+\nabla_\theta L_t,\qquad
\theta_{t+1}=\theta_t-0.01v_{t+1}.
\]

Each minibatch contains two draws per component. Graphs come from the fixed bank;
mixture weights remain attached. The same draw may contribute to both class
losses, representing genuine overlap of the two mixtures.

`autograd.grad(create_graph=True)` retains **all 32 updates**, including momentum,
and differentiates through both mixture optimizers. No direct or downstream
target logit is stopped. This is the complete derivative of the declared
finite-adaptation learner from a fixed pretrained initialization. It is not the
derivative of training a fresh GIN to convergence, nor of rebuilding the bank.
There is no detached warm start that changes between outer iterations.

## Graph reconstruction and loss

The current adapter supports one complete categorical one-hot node-feature group,
simple undirected graphs, fixed node count/correspondence, and at most 64 nodes.
Larger graphs are rejected rather than truncated. A two-layer width-16 equivariant
encoder uses node features plus adjacency-weighted neighbor sums. Symmetric pair
features `(h_u+h_v, h_u*h_v)` feed an edge decoder; node embeddings feed a category
decoder. Both decoder output layers start at zero.

The AE has an explicit identity residual:

\[
\widetilde A_{uv}=\sigma(6(2A_{uv}-1)+d^A_\phi(h_u,h_v)),\quad u\ne v,
\qquad \widetilde X_v=\mathrm{softmax}(6X_v+d^X_\phi(h_v)).
\]

The diagonal and padding are zero. This is a residual graph AE, not a pure
information bottleneck. It starts near identity, **not exactly at identity**.
The frozen-GIN and Level B controls use the same initialization. Dense weighted
GIN preserves the existing sparse GIN's message, self-loop, normalization,
and pooling semantics on hard graphs, with explicit masks.

Use weighted empirical squared MMD including diagonal terms, with an average of
Gaussian kernels at bandwidths 0.5, 1, and 2. All logits share one fixed scale:
the standard deviation of the no-AE reference GIN on fit normals, floored at 0.25.
This scale is shared across conditions within a classifier seed. There is no
per-population rescaling. Synthetic empirical weights are `w_N[k]/samples[k]`.

The outer objective is

\[
R=D(P_T,P_N)+D(P_T,P_0)+D(P_N,P_0)
+L_{rec}+L_{edit}+0.1L_{anchor}+L_{sep}.
\]

`L_rec` is adjacency BCE averaged over unordered valid node pairs plus categorical
cross entropy averaged over nodes, averaged over fit+meta graphs. `L_edit` is the
mean sum of squared positive violations of expected edits divided by per-graph
budgets: two edges and 0.5 node labels. These are penalties, **not hard constraints**;
report maxima and actual violations. `L_anchor` preserves reference raw-normal
logits with the same fixed scale. `L_sep` is independent-pilot balanced synthetic
BCE. Raw and synthetic target branches and mixture MMD weights stay attached.

Use 60 outer Adam updates at learning rate 0.003 with gradient clipping at 5.
Each update uses all 15 meta normals and four fresh pilots/component, with new
inner minibatch indices. Selection evaluates the same objective on 15 disjoint
selection normals and 16 independent fixed pilots/component every five updates,
including step 0. Reconstruction/anchor terms use selection normals only during
selection, under `no_grad`. Select the lowest normal-only objective. Refit the GIN
with the selected AE using exactly the same 32-step horizon and draws as selection.
The ordinary framework's final-epoch selection rule is explicitly replaced by
this internal normal-only protocol for this candidate.

## Controls and interpretable evaluation

All conditions share the fixed bank, internal split, component pools, classifier
initialization, adaptation schedule, and final evaluation pilots:

| Condition | AE optimization | Mixtures | Final GIN |
|---|---|---|---|
| `no_ae` | none; identity input | fitted on raw fit normals | fixed-initialization 32-step adaptation |
| `frozen_gin` | direct input gradient | raw-normal weights fixed | no-AE reference frozen |
| `level_b` | full finite-update gradient | recomputed through AE | differentiated 32-step adaptation |

Fresh evaluation uses 64 graphs/component, separate from every training,
temperature, meta, and selection stream. Pilot AUROC and ECDFs use mixture weights,
so every component is represented without a sampled discrete mixture choice.
Pilot logits are `f(S)` as in the training objective; real inference uses `f(T(G))`.
They are not the same input policy, and their agreement alone cannot validate
anomaly detection. Report these measurements together:

- Synthetic normal/abnormal weighted AUROC and relative real-normal ranks.
- Raw-real versus synthetic-normal MMD and transformed-real versus synthetic-normal
  MMD, including held-out normal groups, on the fixed common logit scale.
- Real test AUROC, normal-calibrated confusion counts, and a conditional bootstrap
  interval. Three classifier seeds assess some initialization variability, not
  dataset-split or prior-bank variability.
- Expected edge/category edits and hardened edits. Harden with adjacency threshold
  0.5 and node argmax; score hard graphs independently as an audit, not the primary
  trained inference policy. Hard-AUROC is threshold-free; no soft-calibrated
  confusion counts are claimed for the hard policy.
- Pilot connectivity, maximum degree, triangles, isolated nodes, and high-degree
  nodes, with component mean ranges. Matching one-dimensional logits does not
  demonstrate graph-distribution support or molecular validity.
- Full AE gradient norm versus the gradient with only trained GIN parameters
  stopped. Their difference isolates the GIN-training contribution; the latter
  still includes explicit mixture-weight MMD derivatives and is not Level A.

## Reproduction and verification

Run with the CUDA-enabled project environment and an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /home/mingeon/.conda/envs/agdiff/bin/python \
  -m unittest tests.test_ae_prior_mixture tests.test_prior_mixture tests.test_prior_mixture_gin -v
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /home/mingeon/.conda/envs/agdiff/bin/python \
  scripts/run_ae_prior_mixture.py --output outputs/diagnostics/ae-level-b-mutag-20261116 --stage bank
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /home/mingeon/.conda/envs/agdiff/bin/python \
  scripts/run_ae_prior_mixture.py --output outputs/diagnostics/ae-level-b-mutag-20261116 --seed 0
```

Repeat the last command with seeds 1 and 2 on available GPUs. Existing output
directories are protected against overwriting. Bank config/data hashes, the exact
data roles, source snapshots, model checkpoints, intermediate JSON logs, raw
score CSVs, and final results are retained. Checkpoint round trips verify score
parity. The ordinary factory/training adapter is also available through
`model=ae_prior_mixture_gin`.

The tests check sparse/dense and padding parity, AE permutation equivariance,
capped-simplex first/second derivatives, a full finite-difference rerun through
AE -> mixtures -> trained GIN -> outer loss, nonzero GIN parameter dependence,
normal-split separation, a tiny complete fit, and checkpoint inference.
