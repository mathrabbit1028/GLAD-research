# Causes of the synthetic and real normal logit gap

The synthetic normal prior does not reproduce the real normal graph distribution.
The frozen GIN responds to those differences, while its loss has no requirement
to align real-normal and synthetic-normal score distributions. This audit
measures the mismatch and tests selected score sensitivities. It does not
identify a unique additive decomposition of the gap.

## Audit setup

The saved classifier seed 0 was restored without fitting. All 8,192 blue-curve
graphs were regenerated with the original stream and verified against their
saved graph hash. The 100 real training graphs were also hash-verified. The
13 held-out test normals provide a descriptive reference, not a tuning set.
Raw logits reproduced the previous medians. Graph descriptors were measured
for these populations and for 256 independent graphs from each of the 64 prior
components. Four random replicates were scored for each graph perturbation.

The median logit is −3.648 for generated normals, −6.368 for training normals,
and −6.022 for test normals. Mean logits are −3.225, −5.886, and −6.114,
respectively. Lower values mean more normal according to the synthetic classifier.
The real training and test curves are similar; ordinary classifier memorization
of real training positives is not the explanation, because those graphs never
entered its binary loss. They did fit mixture weights and preprocessing.
The different curve sample counts affect precision and step size, not an
expected systematic location shift.

## Structural mismatch

| Statistic, averaged across graphs | Real training normals | Generated normals |
|---|---:|---:|
| Nodes per graph | 19.8800 | 19.8644 |
| Connected fraction | 1.0000 | 0.1350 |
| Maximum degree | 3.0000 | 5.3955 |
| Isolated node fraction | 0.0000 | 0.1169 |
| Triangles per graph | 0.0000 | 3.0391 |

Graph size is closely matched. Connectivity, degree patterns, and triangle
counts are not. These observations describe this MUTAG subset; they do not imply
that every triangle or higher-degree graph is intrinsically an invalid molecule.
The sampler targets aggregate mean degree and graph size through random block
layers and peripheral attachment. It does not reproduce the observed structural
constraints. Peripheral attachment explicitly allows zero edges.

Across the independent component pilots, connected fractions range from
1.56% to 17.58%, average maximum degrees from 5.19 to 6.81, and mean triangle
counts from 2.60 to 5.34. A mixture's expectation of a graph statistic is the
weighted average of component expectations. These pilot estimates therefore
suggest that weight optimization alone cannot match the observed normal
statistics; the generator family needs adjustment. The estimates are not a
proof of exact population bounds or zero support.

## Feature mismatch and score interventions

The first three node-category frequencies are 73.19%, 9.26%, and 17.15% in real
training graphs, versus 77.81%, 8.86%, and 13.17% in generated normals. More
importantly, topology and feature placement jointly influence the score.

| Population and perturbation | Mean logit, averaged across four replicates |
|---|---:|
| Original real training normals | −5.886 |
| Original generated normals | −3.225 |
| Generated normals with components connected | −4.199 |
| Real normals with labels shuffled within each graph | −3.989 |
| Generated normals with labels shuffled within each graph | −2.448 |
| Real topology with independent labels from real marginal frequencies | −4.120 |
| Generated topology with independent labels from real marginal frequencies | −0.200 |

Connecting components preserves node features but changes degree and message
passing as well as connectivity. It moves generated scores toward the real
normal scores by about 0.974 logit, without closing the mean gap of 2.661.
It does not isolate connectivity from all other structural effects.

Shuffling labels preserves every graph's edges and category counts but changes
which features occupy which structural positions. The real-normal mean rises
by 1.897, demonstrating sensitivity to feature placement. Giving both graph
populations the same independent label mechanism still leaves a mean score gap
of approximately 3.920. Thus, marginal label frequencies alone cannot explain
the original gap. Independent relabeling also destroys graph-specific label
counts and dependencies; these perturbations are diagnostic, not proposed
molecular generators or estimates of additive causal contributions.

The categorical sampler contains a separate calibration problem. It computes
`u = Phi(2*h + noise)` after tanh transformations, then bins `u` against the
empirical category CDF. This would produce the intended category proportions
if `u` were uniform. Applying a standard Gaussian CDF only guarantees that
property when its input is standard-normal. The implemented input is not
constrained to that distribution. Observed category-frequency errors confirm
that the intended proportions are not preserved here.

## Objective mismatch

The binary loss trains on generated normals and generated abnormalities only.
At an ideal balanced population optimum, its logit is
`log(q_abnormal(G) / q_normal(G))`. This is a relative discrimination score,
not an absolute normal-density or typicality measure. It can be strongly
negative for real graphs even when the normal generator fits them poorly.

The normal weights maximize an estimated likelihood objective within the fixed
prior bank. This picks the best available mixture under a learned surrogate;
it cannot guarantee that the real normal distribution lies within that family.
The resulting normal mixture has effective component count about 1.25, mostly
priors 16 and 48. Density-ratio estimation error may also contribute, but this
audit does not quantify that contribution.

## Recommended changes in order

1. **Repair structural and conditional-feature coverage.** Add generators
   conditioned on connectedness, the normal degree distribution, cycle or motif
   statistics, and feature-neighborhood relationships. For this molecular task,
   consider scaffold-based sampling and suitable validity constraints. Learn
   constraints from training data and domain knowledge, not test anomalies;
   avoid treating every pattern absent from 100 graphs as impossible.
2. **Calibrate categorical sampling.** Use an independently calibrated empirical
   CDF of the latent sampler output before applying the target category CDF,
   or use a categorical model conditioned on structure. Verify marginal and
   joint statistics on fresh simulations. Simply drawing independent labels
   from the correct frequencies is insufficient, as the interventions show.
3. **Refit and check mixtures against real normals.** Add normal-data discrepancy
   checks in graph statistics or an independent representation, with cross-fitting
   to prevent fitting and evaluating on the same normals. Do not ask reweighting
   to repair statistics outside the attainable component range.
4. **Anchor detector training to real normals and improve negatives.** A proposed
   next loss can use real and synthetic normals in its positive class, with
   separate balancing. This may improve detector transfer but does not fix the
   generator or guarantee overlapping curves. Generate informative abnormalities
   with controlled deviations around valid normal structure, rather than relying
   only on the least likely mixture.
5. **Measure the right comparisons.** Retain normal-versus-abnormal pilot
   discrimination, but add an independent real-normal-versus-generated-normal
   classifier two-sample test. Use inner normal-data splits or cross-fitting to
   refit the generator and evaluate on real normals it has not seen. Near-chance
   separation is desirable for this second comparison, though failure to detect
   a difference with a small sample is not proof of equality. Assess real anomaly
   transfer separately. This diagnostic follows the principle of
   [classifier two-sample tests](https://arxiv.org/abs/1610.06545).

Temperature scaling or moving a threshold cannot repair graph-distribution
mismatch. Mild late overfitting observed in the earlier run can be addressed
with independent validation, but is not an adequate explanation of the
structural differences measured here.

## Evidence

- [Measurements and perturbation repeats](../outputs/diagnostics/prior-mixture-gin-gap-audit/result.json)
- [Per-graph descriptors and logits](../outputs/diagnostics/prior-mixture-gin-gap-audit/graph_statistics.csv)
- [Audit log](../outputs/diagnostics/prior-mixture-gin-gap-audit.log)
- [Reproducible diagnostic](../scripts/diagnose_prior_mixture_gap.py)
- [Generator implementation](../src/models/candidates/prior_mixture_impl/sampling.py)
- [Binary GIN implementation](../src/models/candidates/prior_mixture_gin.py)

The audit changes no generator or classifier parameters. Recommendations above
remain proposed changes, not validated remedies.
