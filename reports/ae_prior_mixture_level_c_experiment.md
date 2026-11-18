# Level C adaptive-prior experiment — MUTAG, 2026-11-16

The implementation and mathematical specification are in
[the Level C documentation](../docs/ae_prior_mixture_level_c.md). This report
evaluates adaptive fitted priors with a general GIN classifier, including the
dependence of both GINs on the AE. Results are from the finite relaxed algorithm
described there; they are not a direct matched comparison with the earlier
Level B experiment, which used a different generator and training budget.

**Result: the full Level C gradient works, but this experiment does not show a
reliable detection benefit.** Allowing priors to adapt improves average AUROC
by only 0.0106 over the matched fixed-prior control, with a conditional paired
95% bootstrap interval of **[−0.0134, +0.0370]**. Structural coverage remains
poor, and one seed fails badly even while its normal-score discrepancy falls.

## Protocol

- MUTAG split seed 0: 100 training normals, 12 threshold-calibration normals,
  and 76 test graphs (13 normal, 63 anomalous). Training normals are divided
  into 70 fit, 15 meta, and 15 checkpoint-selection graphs.
- Three model-initialization seeds: 0, 1, 2. They share the same dataset split,
  sampling noise schedules and normal-only selection protocol. They are not
  independent datasets. This benchmark split was also inspected in preceding
  experiments, so this is an exploratory comparison.
- 64 priors, two GIN layers of width 16; fixed raw-normal warm starts with 400
  component-GIN and 500 binary-GIN Adam updates. Each retained finite pipeline
  has four component-GIN updates, four temperature updates, 40 normal-mixture
  steps, 30 abnormal-mixture steps, and eight binary-GIN updates.
- Thirty outer AE updates; selection every five updates, including the initial
  checkpoint. No test labels enter training, selection, or configuration changes
  during this experiment. The predeclared settings were retained for all runs.
- Fresh audit pilots: 64 per component, 4,096 total, evaluated with each fitted
  normal/abnormal mixture. Both pilot populations use the same stratified sample
  pool with different weights. Relative coverage therefore compares weighted
  populations, not separately sampled unweighted histograms.
- Primary scores use soft AE graphs and relaxed synthetic graphs. Hardening is
  a separate audit. Saved checkpoints reproduce both scores and generated
  pilot samples.

The four conditions are raw graphs without an AE; an initialized but untrained
AE; a trained AE with priors fixed at the initial-AE settings; and Level C, which
refits the prior parameters from the current AE outputs. The latter two have
identical forward computations at initialization. Both retain adaptive
normalization, component-classifier fitting, temperature, mixture weights and
binary-GIN fitting. Thus their difference isolates allowing the generator
parameters to change during AE learning.

## Detection and score alignment

Values below are mean ± sample standard deviation across the three
initializations. The MMD column compares selection-normal AE logits with fresh
synthetic-normal pilots using the fixed reference logit scale; smaller is better.

| Condition | Real test AUROC ↑ | Synthetic pilot AUROC ↑ | Normal-score squared MMD ↓ |
|---|---:|---:|---:|
| No AE | 0.547 ± 0.245 | 0.689 ± 0.117 | 0.733 ± 0.387 |
| Initial, untrained AE | 0.553 ± 0.233 | 0.690 ± 0.077 | 0.357 ± 0.102 |
| Trained AE, fixed prior | 0.519 ± 0.314 | 0.684 ± 0.081 | 0.309 ± 0.147 |
| **Level C: trained AE, adaptive prior** | **0.530 ± 0.307** | **0.689 ± 0.075** | **0.290 ± 0.145** |

| Seed | No AE | Initial AE | Fixed prior | Level C | Level C selected step |
|---|---:|---:|---:|---:|---:|
| 0 | 0.755 | 0.685 | 0.722 | 0.718 | 15 |
| 1 | 0.608 | 0.690 | 0.679 | 0.696 | 5 |
| 2 | 0.277 | 0.284 | 0.158 | 0.176 | 20 |

Level C reduces the score discrepancy by **6.2% versus fixed priors** and by
**60.5% versus no AE**. Much of the latter change already occurs with an
untrained AE (0.733 → 0.357). Its average real AUROC is lower than both the
no-AE and initial-AE controls. The conditional paired bootstrap intervals for
those differences are [−0.0798, +0.0525] and [−0.0619, +0.0159], respectively.
These intervals resample test observations while holding the fitted models
fixed; they are not confidence intervals over new datasets or training runs.

Seed 2 is especially informative. Normal-score MMD falls from **0.257 to
0.127** after Level C training, while real AUROC falls from **0.284 to 0.176**.
Thus the optimized score-alignment objective is not a reliable proxy for real
anomaly ranking. Its no-AE model already has AUROC 0.277, so the failure begins
with synthetic-to-real transfer and is then worsened by this outer objective.
No score reversal or test-based checkpoint substitution was applied.

![Matched controls, score distributions, prior movement and structural coverage](../outputs/diagnostics/ae-level-c-mutag-20261116/comparison.png)

The top panels use seed 0, identified in the title; the bottom panels include
all seeds. Blue is the final GIN's raw training-normal logit distribution, purple
dashed is its AE-transformed counterpart, green is synthetic normal and orange
is synthetic abnormal. Blue and purple are close, but both remain shifted from
the synthetic distributions. Red dotted real-anomaly curves expose why accepting
real normals alone is an insufficient coverage check.

## Relative prior coverage and operating thresholds

Using the 95th percentile of synthetic-normal logits as an acceptance cutoff:

| Level C seed | Real normals accepted | Real anomalies incorrectly accepted | P(abnormal-pilot score > real-normal score) | P(real-normal score > normal-pilot score) |
|---|---:|---:|---:|---:|
| 0 | 13/13 | 63/63 | 0.924 | 0.229 |
| 1 | 13/13 | 63/63 | 0.960 | 0.317 |
| 2 | 10/13 | 63/63 | 0.387 | 0.742 |

Probabilities are empirical weighted comparisons with half credit for ties.
A value near 0.5 in the last column would be consistent with similar score
distributions, but would not prove graph-distribution coverage. The pilot
threshold accepts **all real anomalies in every seed**. Seeds 0 and 1 place
real normals farther into the low-score tail than typical synthetic normals;
seed 2 reverses that relative position. The priors do not supply a stable
normal/abnormal reference for real data.

The separate threshold calibrated at the 99th percentile of the 12 held-out
real normals detects **12/63, 11/63, and 0/63 anomalies**, with **0/13, 0/13,
and 2/13 false positives**, respectively. These thresholds use no anomaly labels.

## What actually changed in the priors?

The fitted raw-normal mean degree is **2.22995**. The near-identity AE alone
moves it to **2.26544** before optimization. Level C selects:

| Seed | Selected degree parameter | Change from initial AE | Category-probability total variation from initial AE |
|---|---:|---:|---:|
| 0 | 2.26460 | −0.00085 | 0.00123 |
| 1 | 2.26171 | −0.00374 | 0.00038 |
| 2 | 2.25861 | −0.00683 | 0.00326 |

The degree changes are only **0.04–0.30%** of the starting value. Category
changes are also small. The fit is adaptive through the AE, but this does not
mean that optimization has substantially reshaped the generator.

To isolate the generator change, the posthoc audit compares selected versus
initial-AE priors at the **same final mixture weights and same random draws**.
Normal hard-pilot mean degree changes by approximately **−0.000007, −0.00200,
and −0.00454**. Connectivity is unchanged within numerical precision in all
three seeds. Hard-pilot graph changes occur in 3.1%, 2.0%, and 21.7% of the
weighted normal populations; these are small edge/label changes and do not
resolve the main structural discrepancies.

The real training graphs have mean expected AE edits of **0.453 edges** and
**0.291 node labels**. None of the train or test graphs changes after hardening.
Real-data hardening gives average AUROC 0.533, close to the soft result 0.530,
and does not fix the instability. This experiment primarily changes continuous
representations, mixture fitting and classifier decision rules.

## Why coverage remains poor

**The generator's structural family is mismatched.** For the selected Level C
normal mixture, hard-pilot structural averages across seeds are:

| Graph property | Real fit normals | Level C normal pilots |
|---|---:|---:|
| Connected fraction | 100% | 5.2% (individual seeds 4.7%, 10.7%, 0.24%) |
| Mean degree | 2.230 | 1.953 |
| Maximum degree | 3.000 | 4.962 |
| Triangles per graph | 0.000 | 2.191 |
| Isolated-node fraction | 0% | 17.6% |

For example, in seed 0 no evaluated component has mean connectivity above
10.9%, maximum degree below 4.59, or triangle count below 1.45. Reweighting these
empirical component means cannot reproduce the corresponding real means.
This is finite-pilot evidence of mismatch, not a proof that the population prior
has zero support near real graphs. Changing only a global degree target and
category bias does not directly control connectivity, degree limits, motifs or
attribute/topology dependence.

**Fitted parameters do not calibrate generated moments.** Level C soft normal
pilots have mean degrees **2.667, 2.926, 2.501**; the same draws after hardening
give **1.965, 2.047, 1.848**. Neither consistently equals the fitted prior degree
around 2.26. Soft node-weighted category-0 frequencies are **0.643, 0.600,
0.596**, versus **0.731** in real fit normals. Exponential saturation, relaxed
edge sampling, peripheral attachment and random feature mechanisms transform
the plug-in statistics. Log-frequency bias alone does not ensure the requested
category marginals. The soft/hard distribution gap is material: synthetic AUROC
drops from 0.689 to 0.618 after hardening.

**The likelihood surrogate contains limited component information.** On fresh
1,024-pilot samples, the component GIN has 64-way accuracy **3.22%, 1.95%,
2.44%**, versus chance 1.56%. Calibrated NLL is **4.098, 4.079, 4.084**, versus
uniform 4.159. Hardened-pilot NLL is **4.166, 4.222, 4.182**, all slightly worse
than uniform. Components overlap, so low accuracy alone does not establish
incorrect density ratios. It does show that this short training configuration
extracts little component-label information and transfers poorly to hardened
graphs. Meanwhile normal mixtures concentrate on different components:
component 3 at 99.3%, component 33 at 84.6%, and component 53 at 94.8%.

**Normal-only logit alignment does not constrain real anomaly ordering.** The
seed-2 counterexample above demonstrates this directly. The algorithm can move
normal logits toward the synthetic reference while retaining or worsening an
incorrect ranking of real anomalies. Successful backpropagation therefore does
not establish that the objective or prior family is adequate.

## Next change supported by these observations

Prior adaptation is technically feasible. The next experiment should make the
generator match graph statistics explicitly before relying on its logits:

1. Fit or unroll a differentiable calibration objective comparing **generated**
   normal graph statistics with real-normal statistics, rather than plugging
   real mean degree and category frequencies into an uncalibrated generator.
2. Add adaptable generator controls for degree distribution, connectivity,
   motifs and edge/label dependence. Keep abnormal synthesis structurally
   plausible so the GIN cannot rely on unrealistic generation artifacts.
3. Check ratio information and soft/hard consistency under longer predeclared
   inner horizons, then repeat the matched initial/fixed/adaptive comparison.
   The current four/eight-update map is a verified finite surrogate, not a
   fully converged nested optimization.

A candidate calibration loss is
\(\|\mathbb E_{q_{\eta,N}}s(G)-\mathbb E_{T_\phi D_f}s(G)\|^2\), where
\(s\) includes degree statistics, category frequencies, motif counts and a
smooth connectivity statistic. Both parameter fitting and sampling would stay
in the differentiated computation. These are proposed next changes; they were
not added or tuned after observing this experiment's test results.

## Verification

All **21 CUDA tests passed**, including all existing tests. The full nested
gradient had relative error **1.36 × 10⁻⁹** against central finite differences;
the isolated adaptive-prior contribution had error **9.25 × 10⁻¹⁰**. Every
inner stage was replayed for both finite-difference perturbations with fixed
noise. AE perturbations changed the component GIN, temperature and final GIN.
Sampler first- and second-order checks, symmetry, masking, categorical sums,
fixed-noise replay and coherent checkpoint tests passed. A standard `src.train`
run also trained, calibrated and saved the candidate through the framework.

The repository-wide whitespace check encounters pre-existing merge markers in
`README.md`. That file was not modified. The new Level C files passed scoped
syntax, whitespace and conflict-marker checks.

## Artifacts

Experiment root:
`outputs/diagnostics/ae-level-c-mutag-20261116`.
Each `seed{0,1,2}/{no_ae,initial_ae,fixed_prior,level_c}` directory contains the
checkpoint, resolved configuration, source snapshot/hashes, data roles, real
and pilot score CSVs, structural pilot CSV, result JSON and completion manifest.
The posthoc audit holds the final mixture weights, classifier and noise fixed
while comparing raw, initial-AE and selected prior parameters; it also evaluates
the component classifier on an independent 1,024-pilot sample.

Logs record intermediate objectives, gradient norms, prior mean degree,
checkpoint selection and elapsed time. Test evidence is in
`verification_tests.log` and `verification_framework.log`.
