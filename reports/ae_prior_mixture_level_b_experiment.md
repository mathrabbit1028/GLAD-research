# Level B on MUTAG: implementation, controls, and interpretation

**Run date: 2026-11-16.** Level B is implemented and its downstream GIN-training
gradient is active. The complete system improved over a matched no-AE control,
but this experiment does **not** establish that learned AE optimization is the
main source of improvement. An untrained near-identity AE already gives most of
the gain, and the prior bank still fails the structural and relative-coverage checks.

Implementation: [algorithm and exact settings](../docs/ae_prior_mixture_level_b.md),
[adapter](../src/models/candidates/ae_prior_mixture_gin.py),
[differentiable primitives](../src/models/candidates/ae_prior_mixture_impl/core.py),
[configuration](../configs/models/ae_prior_mixture_gin.yaml).

## Protocol and what was differentiated

Use the existing MUTAG split seed 0. Partition its 100 training normals into
70 fit, 15 AE-feedback, and 15 checkpoint-selection graphs. Keep the original
12 calibration normals for the 99th-percentile threshold and 13 normal + 63
anomalous test graphs for final evaluation. The test labels never enter training
or selection. This test set was used in earlier research, so this is not a newly
unseen confirmation dataset.

Fit one 64-component bank and component GIN on the 70 fit normals; freeze the
sampler, ratio classifier, temperature, and feature scaling. The ratio classifier
was trained for 100 epochs on 32,768 synthetic graphs and temperature-calibrated
on 8,192 independent graphs. Its calibrated 64-way accuracy was 4.55% versus
1.56% uniform chance. Overlapping components can make identification difficult;
this accuracy alone does not certify the quality of the ratio estimates.

For each classifier seed 0, 1, and 2, use the same initial binary GIN and synthetic
pools across controls. Pretrain that GIN for 800 steps independently of the AE,
then differentiate all 32 momentum-SGD adaptation steps from that fixed starting
point. Refit normal/abnormal weights through the AE at each outer step, using
100 EM updates and 60 capped-simplex projected updates. The abnormal solution
uses a soft combination of four restarts. Train the AE for 60 outer steps,
selecting solely on the disjoint selection normals. The selected GIN is refit
using the same adaptation horizon and selection draws.

Thus both paths are active:

\[
\phi\to T_\phi(G)\to f_{\theta(\phi)}(T_\phi(G)),\qquad
\phi\to T_\phi(\mathcal D_{fit})\to w_N,w_A\to\theta(\phi)\to f.
\]

This is the exact autograd derivative of the specified finite-update continuous
surrogate, conditional on the fixed bank, pretrained initialization, and draws,
away from nonsmooth boundaries. It is not a converged-inner-solver derivative
or a derivative through rebuilding the prior bank.

Nine main runs were completed: no AE, frozen GIN, and Level B, each at three
classifier seeds. After these runs, three additional **attribution controls**
repeated the full inner refit with the untrained near-identity AE. These controls
were not used to tune parameters or reselect checkpoints. Reproducing the saved
no-AE GIN weights first verified that their initializations matched.

Fresh evaluation pilots contain 64 graphs per component, 4,096 in total.
Mixture-weighted metrics compare normal and abnormal pilots using this common
stratified pool. These are independent of fitting, meta, and selection pilots.

## Results

Values below are mean ± sample standard deviation across the three classifier
seeds. They condition on a single fixed bank and dataset split.

| Method | Real test AUROC ↑ | Synthetic pilot AUROC ↑ | Selection-normal score gap, squared MMD ↓ |
|---|---:|---:|---:|
| No AE | 0.574 ± 0.055 | 0.697 ± 0.047 | 0.990 ± 0.197 |
| AE with frozen GIN | 0.549 ± 0.043 | 0.697 ± 0.047 | 0.948 ± 0.232 |
| Untrained near-identity AE + full inner refit | 0.664 ± 0.068 | 0.800 ± 0.084 | 0.476 ± 0.253 |
| **Trained Level B** | **0.687 ± 0.038** | **0.812 ± 0.029** | **0.463 ± 0.078** |

The MMD compares transformed real-normal logits with synthetic-normal logits,
using a fixed scale shared across conditions within each seed. It is a score-space
discrepancy, not a probability of graph coverage. These selection normals are
held out from gradients, but are used for checkpoint selection. The evaluation
pilots in this table are fresh. On the 13 test normals, the corresponding MMD
also decreased: **0.822 → 0.285** from no AE to Level B.

| Classifier seed | No AE AUROC | Frozen GIN AUROC | Initial AE AUROC | Trained Level B AUROC | Selected outer step |
|---|---:|---:|---:|---:|---:|
| 0 | 0.601 | 0.596 | 0.737 | 0.722 | 50 |
| 1 | 0.611 | 0.541 | 0.651 | 0.646 | 45 |
| 2 | 0.510 | 0.512 | 0.603 | 0.695 | 20 |

Level B improves over no AE in all three seeds. However, **outer AE training
improves over its untrained initialization only in seed 2**. Its mean additional
AUROC gain is 0.024. A paired, class-stratified graph bootstrap gives a 95%
interval **[-0.028, 0.070]** for that mean gain. For Level B versus no AE the gain
is 0.114 with conditional interval [0.006, 0.208]. These intervals hold the three
classifiers, bank, and split fixed; they do not cover all training or dataset
uncertainty.

The normal-score gap is 53.2% lower than no AE, but only 2.7% lower than the
untrained-AE control. It would therefore be misleading to attribute the entire
gap reduction to learned AE feedback.

![Matched controls and score distributions](../outputs/diagnostics/ae-level-b-mutag-20261116/comparison.png)

The top row uses seed 0; purple dashed is the raw training-normal curve, green is
the AE training-normal curve, blue/orange are normal/abnormal synthetic pilots.
The blue–purple gap remains visible. The bottom row includes all three seeds and
the initialization control.

## What happened and why it matters

**1. Differentiating GIN training changes the update substantially.** At the first
production update, the norm of the GIN-training contribution divided by the full
AE-gradient norm was 0.963, 0.995, and 0.986. These are vector-norm ratios, not
fractions of the objective. Stopping this path would materially change the
optimization. The selected GIN parameter distances from the no-AE reference were
2.324, 0.808, and 1.660 in Euclidean norm.

**2. A near-identity soft input already changes fitted mixtures.** At initialization,
the AE changes approximately 0.48 expected edges and 0.29 expected node labels per
training graph, even though thresholding/argmax leaves the graph unchanged.
Without any AE training, normal-mixture total variation changes by 0.049 and
abnormal-mixture total variation by 0.25 relative to the raw-normal fit. The GIN
then trains on different class distributions. This is the experimentally supported
reason to include the initialization control; the effect is not confined to the
direct AE input branch.

The ratio classifier was trained on hard graphs and is evaluated on soft AE
outputs. Autograd through this extension is mathematically available, but its
statistical density-ratio calibration on those soft inputs is unverified. The
results are consistent with substantial sensitivity to that representation change.
The discrete bank does not literally assign density to a soft adjacency/category
matrix: the optimized likelihood is the ratio network's continuous surrogate.
Correct differentiation of that surrogate does not make it an exact graph likelihood.

**3. Outer optimization is unstable and its safeguards create real tradeoffs.**
Recorded pre-clipping gradient norms reached about 506; the clip was 5.
Normal-only selection loss oscillated, and the best checkpoints were intermediate
steps, not the final step. For example, seed 0's selection objective fell from
5.121 to 1.388 while its fresh-pilot score MMD increased from 0.205 at initialization
to 0.489 at the selected checkpoint. The objective also protects the reference
GIN and synthetic separation, so minimizing it does not mean monotonically
minimizing the displayed MMD. Abrupt capped-solver changes and sensitivity of
unrolled GIN training are plausible contributors; these were not isolated by
separate optimizer ablations.

**4. Improvement is mainly in the retrained decision rule, with tiny direct
inference changes.** Level B used an average of 0.512 expected edge edits and
0.300 expected node-label edits per training graph. All hardened training and
test graphs were identical to their original graphs. Hard-graph AUROCs were
0.720, 0.661, and 0.696, close to the soft-policy results 0.722, 0.646, and 0.695.
Thus the final result is not explained by substantially rewriting discrete graph
structure. The edge penalty stayed within its budget on every observed training
graph; the node-edit penalty is not a hard constraint, and seed 0 reached 0.586
expected label edits on one graph against the 0.5 budget.

## Prior coverage is still inadequate

Relative pilot separation improved, but the model still regards almost all real
anomalies as normal relative to synthetic-normal scores:

| Level B seed | Real normals accepted below synthetic-normal 95th percentile | Real anomalies incorrectly accepted by that cutoff | Normal-vs-abnormal pilot AUROC |
|---|---:|---:|---:|
| 0 | 13 / 13 | 63 / 63 | 0.789 |
| 1 | 13 / 13 | 63 / 63 | 0.845 |
| 2 | 13 / 13 | 61 / 63 | 0.803 |

Real normals versus abnormal pilots have AUROC 0.973–0.993, but synthetic normals
versus real normals have AUROC only 0.101–0.274 when real normals are the positive
population. The real normal scores still lie unusually far toward the normal end
of the synthetic distribution; high acceptance alone is not evidence of coverage.

Structural checks explain why reweighting this fixed bank is insufficient:

| Statistic | Real fit normals | Level B synthetic normals, range across seeds |
|---|---:|---:|
| Connected graphs | 100% | 12.4–16.6% |
| Mean maximum degree | 3.00 | 5.19–5.51 |
| Mean triangle count | 0.00 | 3.10–3.15 |
| Isolated-node fraction | 0% | 11.3–11.6% |

Among these fresh component pilots, even the highest component connectivity rate
was 26.6%, and every component's mean maximum degree exceeded 5.14. These finite
pilot estimates strongly suggest a bank-capacity mismatch; they are not a proof
of zero generator support. The AE cannot change the bank in Level B.

Using the intended threshold calibrated from 12 real normals, Level B detected
only **9, 22, and 16 of 63 anomalies**, with **0, 2, and 1 false alarms** among 13
test normals. AUROC improvement therefore does not yet give satisfactory
thresholded anomaly detection.

## Verification and practical limits

All 15 CUDA tests passed: four Level B mathematical/integration tests plus the
11 existing prior-mixture tests. The small full-pipeline directional derivative
was -0.0000722612355594 by autograd and -0.0000722612354676 by central finite
differences, relative error 1.27e-9. Stopping GIN training gave
-0.0000804800179599, about 11.4% different. The check used double precision,
fixed draws, four components, and four inner steps; it is not a full-size
float32 numerical verification at every production update.

Sparse/dense hard-graph equivalence, padding, permutation equivariance,
capped-simplex first/second derivatives, a complete tiny fit, and checkpoint
score parity were checked. All nine main experiment checkpoints passed score
round trips. A separate standard `src.train` smoke run verified fitting,
normal calibration, saving, and correct selection-policy metadata.

This experiment changes the training protocol and classifier-training budget
relative to the earlier GIN runs: it uses 70 fit normals, a finite component pool,
and short functional adaptation. The earlier 100-normal, 100-epoch GIN result is
not a paired comparator. Three seeds and 13 test normals provide limited evidence.

## Recommended next changes

1. Keep the untrained-AE control and add an exactly identity-initialized AE.
   Use a common, declared relaxation for component-classifier training and AE
   inputs, then check ratio stability. This addresses the large effect of soft
   representation before claiming a benefit from learned feedback.
2. Stabilize the inner learner using larger per-component minibatches and a
   normal-only comparison of smaller adaptation steps or functional Adam. Measure
   gradient norms, active-set changes, and seed consistency. Retain the full
   GIN-training derivative rather than replacing it with a frozen classifier.
3. Repair prior structure and conditional node features, starting with the
   connectivity, degree, and motif mismatches above. Reweighting alone cannot
   change the available component families.
4. Repeat on independent normal splits and additional datasets after fixing the
   representation and bank issues. Do not select settings using this test AUROC.

## Saved artifacts and reproduction

All outputs are under
[`outputs/diagnostics/ae-level-b-mutag-20261116`](../outputs/diagnostics/ae-level-b-mutag-20261116).
This includes `bank.pt`, exact data-role indices, configs/source snapshots, a
checkpoint and raw score CSVs for each condition/seed, intermediate logs,
initialization-audit checkpoints, `summary.json`, and PNG/PDF plots.

Main commands are documented in the [implementation guide](../docs/ae_prior_mixture_level_b.md).
Attribution and summary commands:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 /home/mingeon/.conda/envs/agdiff/bin/python \
  scripts/audit_ae_prior_mixture_initialization.py \
  --root outputs/diagnostics/ae-level-b-mutag-20261116 --seed 0
# Repeat attribution for seeds 1 and 2. Use an interpreter with matplotlib/sklearn:
MPLCONFIGDIR=/tmp/level-b-mpl OMP_NUM_THREADS=1 python \
  scripts/summarize_ae_prior_mixture.py --root outputs/diagnostics/ae-level-b-mutag-20261116
```

Audit commands protect existing results; use a new experiment output directory
to reproduce all training. Statistical summaries and plotting run on CPU;
all neural training and scoring ran on CUDA.
