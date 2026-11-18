# Level C: adaptive fitted graph priors and a fully differentiated GIN pipeline

This implementation extends the [bilevel design](ae_prior_mixture_bilevel.md). The
prior parameters are functions of AE outputs, and **both** GINs are functions of
those parameters. The generator family stays fixed; its fitted distributions
change. This implementation supports a single categorical one-hot node-feature
group and node-aligned graphs with at most 64 nodes. It does not truncate graphs.

## Parameters and data roles

Write the decoded fit normals as
\(\widetilde G_i(\phi)=(X_i'(\phi),A_i'(\phi),M_i)\), where masks and node counts
remain unchanged. The residual AE returns symmetric soft edges, zero diagonal,
and categorical probabilities. Its identity skip starts close to identity,
not exactly at identity; an explicit untrained-AE control measures that effect.

The fitted prior settings are

\[
\bar d_\phi=\frac1{N_f}\sum_i\frac{\sum_{uv}A'_{iuv}}{n_i},\qquad
\pi_{\phi c}=\frac{\sum_{iu}M_{iu}X'_{iuc}+10^{-4}}
{\sum_i n_i+10^{-4}C}.
\]

Feature means and variances are node weighted. The standard deviation is
\(\sqrt{\mathrm{Var}(X')+10^{-6}}\), with a further 0.05 floor when used for GIN
normalization. These normalization tensors are passed through functional calls
and differentiated; they are not assigned to detached buffers during training.
Graph-size sampling uses the fixed empirical sizes of fit normals.

The training normals are partitioned once into 70% fit, 15% meta, and 15%
selection. Fit normals determine prior parameters and mixture weights. Meta
normals train the AE; reconstruction uses fit and meta normals. Selection normals
select the AE checkpoint. Calibration normals and test graphs are withheld until
after selection. All downstream training is replayed at the selected AE.

## Relaxed generator

There are 64 fixed regimes: two choices on each of six axes (block count,
contrast, degree heterogeneity, peripheral fraction, feature message mixing,
feature noise). Random assignments, permutations, sizes, random feature-network
weights, and random uniforms are independent of \(\phi\). A seed identifies
common random numbers, allowing controlled comparisons and finite differences.

The graph has two degree-corrected block layers with random correspondence,
followed by peripheral attachment. Smooth saturation converts the fitted mean
degree to probabilities. For an allowed edge in a layer,

\[
\widetilde a=\sigma\!\left(
\frac{\operatorname{logit}(p)+\log U-\log(1-U)}{\tau_A}\right),
\qquad A_{\mathrm{core}}=1-(1-A_1)(1-A_2).
\]

Structural zeros are masked exactly. Peripheral nodes use smooth existing
degree plus one as attractiveness. An independent fixed uniform sets an
expected attachment target, and probabilities are
\(p_{uv}=1-\exp(-m_u s_v/\sum_j s_j)\). This replaces discrete rounding and
preferential-attachment sampling. It does not exactly enforce the target count.

Two fixed random feature layers mix local features and weighted mean messages.
Their categorical output uses logits \(\log\pi_\phi+r_k(A,\xi)\) followed by a
Gumbel-softmax relaxation. Adding empirical log frequencies is a bias, **not a
guarantee of matching node-label marginals**. Both relaxation temperatures are
0.67. Hard audit graphs threshold the same soft edges at 0.5 and take categorical
argmax; there is no straight-through estimator.

This is a declared differentiable variant of the original graph prior. Its
gradients are not exact gradients of the earlier NumPy discrete sampler. Scores
on soft AE graphs and soft pilots define the primary experiment. Hardening is
reported separately because it changes the distribution and may change GIN
scores substantially.

## Nested finite training map

First, use raw fit normals to train fixed warm starts: 400 Adam updates for the
component GIN and 500 for the binary GIN. They are independent of the AE. Each
outer evaluation starts again from these weights, with fresh optimizer state.
The retained map is:

1. Fit \(\eta_\phi=(\bar d_\phi,\pi_\phi)\) and normalization to AE fit graphs.
2. Generate attached relaxed samples and perform four functional SGD updates of
   the component GIN, starting from its fixed warm start. Uniform component
   labels supply cross-entropy targets.
3. On separate component-stratified samples, perform four differentiable
   temperature updates. Use \(T=0.25+9.75\sigma(t)\), starting from \(T=1\),
   so calibration is smooth and bounded.
4. Evaluate \(\ell_{ik}=\log K+\log\operatorname{softmax}
   (c_{\omega_\phi}(T_\phi G_i)/T_\phi^{\rm cal})_k\).
   Fit normal weights with 40 log-domain EM steps. Fit abnormal weights with
   30 projected gradient steps from four fixed starts, cap 0.25, then combine
   restarts with a differentiable soft minimum of temperature 0.05.
   These use a component-classifier-based likelihood surrogate, rather than
   evaluated generator densities. Neither exact density ratios nor a global
   optimum of the weight objective are guaranteed.
5. Generate another independent attached sample stream. Perform eight
   momentum-SGD updates of the binary GIN from its fixed warm start:

\[
L_f=\tfrac12\sum_k w_{Nk}\overline{\operatorname{softplus}(f(S_k))}
+\tfrac12\sum_k w_{Ak}\overline{\operatorname{softplus}(-f(S_k))}
+\tfrac{10^{-4}}2\|\theta\|^2.
\]

Both networks have two GIN layers of width 16, LayerNorm, ReLU, and masked
mean/max pooling. Synthetic training is component stratified; mixture weights
enter the loss directly and remain attached. This avoids differentiation through
categorical component selection. Inner batches contain one sample per component;
temperature calibration contains two per component.

Let the complete finite map return
\(\theta_\phi,\omega_\phi,T_\phi^{\rm cal},w_N(\phi),w_A(\phi),\eta_\phi\).
For normal meta data, let \(P_0\) be raw-graph logits, \(P_T\) decoded-graph
logits, and \(P_N\) current synthetic-normal logits. The outer objective is

\[
L_{out}=D(P_T,P_N)+D(P_T,P_0)+D(P_N,P_0)
+L_{rec}+L_{edit}+0.1L_{anchor}+L_{sep}.
\]

Here \(D\) is biased empirical weighted MMD with Gaussian bandwidths 0.5, 1,
and 2 and one fixed shared logit scale from the raw reference. All three
distributions, their mixture weights, and the generated samples remain attached.
The reference teacher alone is detached. Reconstruction uses adjacency BCE and
categorical CE. Soft budget penalties target two expected edge edits and 0.5
expected node-label edits per graph; these are penalties, not hard constraints.
The anchor penalizes raw-normal logit drift from the fixed reference; separation
is the weighted synthetic BCE.

Thirty Adam outer updates at learning rate 0.003 use gradient clipping at five.
Every five updates, selection uses fixed independent pilots (eight per
component) and a fixed complete inner schedule. Outer optimization uses fresh
noise streams per update (two meta pilots per component).

Automatic differentiation includes

\[
\frac{dL}{d\phi}=
\partial_\phi L+
\partial_\eta L\,\frac{d\eta}{d\phi}+
\partial_\omega L\,\frac{d\omega}{d\phi}+
\partial_T L\,\frac{dT}{d\phi}+
\partial_w L\,\frac{dw}{d\phi}+
\partial_\theta L\,\frac{d\theta}{d\phi},
\]

with all indirect dependencies within these terms handled by the computation
graph. There is no detached adaptive warm start. The derivative is exact for
this finite relaxed computation within fixed active sets; it is not the
derivative of infinite training or of a guaranteed optimum.

## Matched comparisons and interpretation

The experiment runs four modes with identical architecture, seeds, warm starts,
inner horizons, normal partition, and audit noise:

| Mode | AE | Prior settings |
|---|---|---|
| No AE | Raw graph input | Fit to raw normals |
| Initial AE | Initialized, never optimized | Fit to initial AE outputs |
| Fixed prior | Optimized through the downstream pipeline | Frozen at initial AE settings |
| Level C | Optimized through the complete pipeline | Refit to current AE outputs |

Fixed prior and Level C have exactly the same forward computation at step zero.
The former still updates normalization, the component classifier, temperature,
mixtures and final GIN, but its generator settings cannot change. It is a
matched ablation for this experiment, not the previous Level B implementation.
The prior settings are fitted functions of the AE; there is no additional
independently optimized prior-parameter vector.

The serialized `eta` dictionary also contains fitted feature moments for audit
and reproducibility. The generator consumes only `mean_degree` and
`category_probs`; GIN normalization is computed separately from the current
transformed fit graphs in both learned modes. Consequently frozen moments in
the fixed-prior dictionary do not imply frozen GIN normalization.

Report real test AUROC, synthetic pilot AUROC, normal logit MMD, fitted parameter
changes, expected/hard graph edits, and hard pilot topology. Relative coverage
includes the ranks of real normals against **both** pilot populations and the
counts of normal and anomalous real graphs accepted by the synthetic-normal
95th percentile. A lower normal gap alone does not establish useful coverage.
Global degree and category parameters cannot independently specify connectivity,
maximum degree, triangles or chemical validity. Improvements must therefore be
checked against these structural limitations.

The checkpoint saves the selected AE, fitted priors, both GINs, normalization,
temperature, weights, sampler reference sizes/schema, selection diagnostics and
representation policy. The experiment also saves source/configuration hashes,
data roles, scores and status manifests.

## Entry points and checks

- Adapter: `src/models/candidates/ae_prior_mixture_level_c.py`.
- Template configuration: `configs/models/ae_prior_mixture_level_c.yaml`.
- Controlled experiment: `scripts/run_ae_prior_mixture_level_c.py`.
- Plots and aggregation: `scripts/summarize_ae_prior_mixture_level_c.py`.
- CUDA tests: sampler first/second derivatives; full nested finite differences
  with every inner stage replayed; separate prior-path finite differences;
  matching initial controls; graph invariants; coherent checkpoint replay.

Run the experiment with the CUDA-enabled environment, for example:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /home/mingeon/.conda/envs/agdiff/bin/python \
  scripts/run_ae_prior_mixture_level_c.py \
  --output outputs/diagnostics/ae-level-c-mutag-20261116 --seed 0
```

The framework entry point also accepts `model=ae_prior_mixture_level_c` and
records its internal normal-only selection policy.

The three-seed MUTAG comparison, relative-coverage audit and interpretation are
in [the experiment report](../reports/ae_prior_mixture_level_c_experiment.md).
