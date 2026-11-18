# Relative normal and abnormal pilot experiment

The learned mixtures are distinguishable on fresh synthetic graphs, but the
current ellipsoid-distance score does not distinguish them well. On MUTAG seed
0, the planned distance comparison gave AUROC 0.4929. A secondary comparison
using the estimated mixture likelihood ratio gave AUROC 0.9430. These are
synthetic-pilot results, not real-anomaly detection results.

## Experimental setup

The experiment retained the initial MUTAG split, 64 priors, 32,768 synthetic
ratio-training graphs, 8,192 synthetic calibration graphs, and 100 classifier
epochs. Only normal training graphs fitted the real-data geometry. Abnormal
weight fitting retained the 0.25 concentration cap but set the minimum shell
rate to zero. No sample rejection, radius training, final threshold calibration,
or test-label evaluation was performed.

After fitting, 8,192 new graphs were drawn independently from each of the normal,
abnormal, and uniform mixtures. Every graph was scored using the same frozen
normal geometry. There were two ellipsoids, with 58 and 42 normal training
members, in 16 dimensions. The normal mixture placed 88.79% on prior 16 and
11.20% on prior 48. The abnormal mixture placed 25% each on priors 44, 45, 46,
and 62. Indices are zero-based.

The resulting normal and abnormal mixture objectives were 1.8572 and -3.0387.
These are the estimated average log-likelihood-ratio objectives on real
training normals, not absolute graph log likelihoods. All 16 abnormal solver
starts reached the same objective; this does not certify a global minimum.

## Relative differences

The primary statistic is the fraction of independent normal/abnormal pilot
pairs for which the abnormal pilot has the higher score, with half credit for
ties. The score is distance to the union of normal ellipsoids. A value of 0.5
means no directional ranking advantage.

| Comparison | Pilot AUROC | 95% bootstrap interval |
|---|---:|---:|
| Abnormal versus normal using ellipsoid distance | 0.4929 | 0.4839–0.5024 |
| Uniform versus normal using ellipsoid distance | 0.4893 | 0.4808–0.4990 |
| Abnormal versus uniform using ellipsoid distance | 0.5039 | 0.4951–0.5131 |
| Abnormal versus normal using estimated likelihood ratio, secondary | 0.9430 | 0.9393–0.9462 |

The primary comparison provides no clear evidence that abnormal pilots are
farther from the real normal regions. Their median distance is slightly lower,
although they have a heavier upper tail. Uniform sampling gives a similar
score distribution.

The secondary diagnostic uses `log(estimated q_abnormal / estimated q_normal)`,
computed from the saved component-classifier probabilities. Its median is
-2.786 for normal pilots and 1.916 for abnormal pilots. Its high AUROC shows that
the frozen representation and component classifier retain information that
separates the two proposal populations. The later projection and ellipsoid
scoring do not preserve that separation as a useful radial ranking. This
experiment does not isolate projection, covariance estimation, or region
aggregation as the specific cause.

The likelihood-ratio analysis was added after observing the primary result and
is reported as secondary. It does not replace the primary result. All intervals
use 1,000 independent within-population bootstrap replicates and quantify pilot
sampling uncertainty conditional on this fit. They exclude uncertainty across
training runs, splits, and prior-bank designs.

## Absolute coverage

| Population | Median distance score | Inside normal regions | Inside boundary shell |
|---|---:|---:|---:|
| Normal pilots | 5.659 | 28 / 8,192 = 0.342% | 26.575% |
| Abnormal pilots | 5.139 | 5 / 8,192 = 0.061% | 32.178% |
| Uniform pilots | 5.317 | 14 / 8,192 = 0.171% | 31.506% |
| Real normal training graphs | 0.248 | 96 / 100 = 96% | 2% |

Inside means `score <= 1`; the shell is `1.05 <= score <= 4`. The real-normal
row is an in-sample reference fitted by the geometry, not a held-out coverage
estimate. Mean node counts were 19.93, 19.86, and 19.85 for the three pilot
populations, consistent with their shared training-derived size law.

The normal mixture has more probability mass inside the normal regions than the
abnormal mixture, but that mass is small for both. A strong distinction between
proposal origins therefore does not establish that generated normal graphs
adequately represent real normals.

## Numerical failure and sensitivity across refits

The first diagnostic attempt completed the 100 classifier epochs, then the
SLSQP projection inside abnormal-weight optimization failed with `Positive
directional derivative for linesearch`. The capped simplex was feasible; this
was a numerical solver failure, separate from the original shell infeasibility.

The fix projects directly onto the capped simplex by finding a scalar multiplier
for clipped weights. The general quadratic solver is retained only when an
additional shell inequality requires it. Gradient steps are scaled before
backtracking. Nine acceptance tests passed, including large-target projection
and full synthetic integration; an additional active-shell regression assertion
also passed. The failed run remains preserved.

The successful retry also revealed important sensitivity across refits. The
original experiment had two regions and component median pilot distances around
140–377. The first relative-diagnostic attempt had one region and medians around
12–21; the successful retry had two regions and medians around 3.84–7.98. These
were separate classifier/geometry fits with seed 0, not one reused checkpoint.
The framework requests deterministic algorithms with warnings rather than
strict enforcement, so bitwise-identical GPU training is not guaranteed. The
specific source and amplification of the geometric variation remain unresolved.

Geometry is fitted before abnormal weights, so disabling the abnormal shell
constraint cannot itself explain this change. Improved shell rates in the retry
must not be attributed to that constraint change. The within-fit relative
comparison is valid for the saved model, but the geometry needs stability
checks across fits before stronger conclusions.

## Artifacts and reproduction

The successful worker completed in approximately 540 seconds after the retry.
It saved intermediate geometry and proposal checkpoints, enabling further pilot
analysis without repeating classifier training.

- [Primary measurements](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry/result.json)
- [Secondary likelihood-ratio measurements](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry/secondary_ratio_analysis.json)
- [Raw unfiltered pilot scores](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry/pilot_scores.csv)
- [Progress log](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry.log)
- [Saved geometry](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry/geometry_checkpoint.pt)
- [Saved proposals](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry/proposal_checkpoint.pt)
- [Comparison plot](../outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry/pilot_comparison.png)
- [Failed first diagnostic](../outputs/diagnostics/prior-mixture-relative-mutag-seed0/manifest.json)

Run [run_prior_mixture_diagnostic.py](../scripts/run_prior_mixture_diagnostic.py)
with the original job directory as `--reference-run` and a fresh `--output`.
Use `--resume-geometry` with the saved geometry checkpoint to avoid repeating
its fitted stages. The driver checks data hashes, split, seed, and fitting
settings before reuse. All final pilot streams are independent of earlier
training and calibration simulations.
