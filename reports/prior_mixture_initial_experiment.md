# Prior mixture initial implementation and experiment

The candidate is implemented through the framework's template interface. Seven
acceptance tests pass, including an end-to-end synthetic integration test. The
first full-default MUTAG experiment stopped because its graph priors could not
meet the configured boundary-sampling requirement. There is no trained candidate
checkpoint or candidate test score from that run. Its constraints were not relaxed.

## Implementation and execution

The [adapter](../src/models/candidates/prior_mixture.py) implements prior sampling,
calibrated component classification, normal mixture fitting, constrained abnormal
mixture fitting, rejection sampling, and ellipsoid radius fitting. Its
[configuration](../configs/models/prior_mixture.yaml) is discoverable in the
terminal UI. Implementation choices and scope limits are recorded in the
[algorithm documentation](../docs/prior_mixture_ellipsoids.md#initial-implementation-choices).

The experiment used MUTAG, seed 0, the existing `agdiff` Python environment, and
an idle RTX A6000. Full documented budgets were retained: 64 priors, 512 training
simulations per prior, 128 synthetic calibration simulations per prior, 100
ratio-classifier epochs, and 128 shell-pilot simulations per prior. The configured
100 radius epochs were not reached. The split contained 100 normal training
graphs, 12 normal calibration graphs, and 76 test graphs (13 normal, 63 anomalous).

```bash
/home/mingeon/.conda/envs/agdiff/bin/python scripts/run_benchmarks.py \
  --profile experiments/prior_mixture_mutag.json \
  --output outputs/experiments/prior-mixture-mutag-initial
```

The framework preserved its source snapshot, resolved configuration, split,
manifest, and [intermediate log](../outputs/experiments/prior-mixture-mutag-initial/prior_mixture_mutag_seed0/run.log).
The worker ran for approximately 592 seconds before stopping. Progress was
checked during generation, classifier training, and shell sampling. No test
scores were used to select parameters or training duration.

## Observed candidate failure

The fitted geometry retained two ellipsoids with 41 and 59 training members,
using 16 embedding dimensions. The normal mixture assigned 87.60% of its mass to
component 16 and 12.40% to component 48. Its effective component count,
$1/\sum_k w_k^2$, was 1.28. Selecting the best-fitting priors did not make their
generated graphs sufficiently close to the normal neighborhoods.

Only 13 of 8,192 pilot graphs fell into the required shell
$1.05\leq s_0\leq4$: an overall acceptance rate of 0.159%. Under the component
weight cap of 0.25, the maximum feasible estimated shell rate was **0.78125%**,
below the required **5%**. The feasibility check raised:

```text
Abnormal mixture infeasible: maximum capped shell rate=0.0078125, required=0.05
```

Across components, median generated scores ranged from 139.63 to 377.00, far
outside the shell. This establishes a mismatch between the current generator
bank and fitted real-normal geometry. It does not isolate how much comes from
the generator, representation, or ellipsoid tightness, and does not prove that
the priors assign mathematically zero density to real graphs.

Reweighting alone cannot meet the constraint with these estimated component
acceptance rates. Broadening the shell would change the experiment and could
produce trivial outliers. No positive/negative rejection-training sets, learned
radii, or candidate anomaly metrics were produced by this MUTAG run.

The component classifier also generalized weakly. Final training accuracy was
8.02%; independent synthetic calibration accuracy was 4.86%, versus 1.56% uniform
guessing. Temperature calibration selected 1.348, with cross entropy 3.760 and
binned calibration error 0.0123. Accuracy alone cannot distinguish intrinsic
prior overlap from estimation error; low calibration error also does not
certify useful likelihood ratios on real graphs.

## Reference experiment and calibration problem

The existing OCGIN baseline ran separately on GPU 2 using the same split and its
configured 500 epochs. Its [result](../outputs/experiments/prior-mixture-mutag-ocgin-reference/ocgin_mutag_seed0/results/ocgin_MUTAG_seed0.json)
was:

| Measure | OCGIN reference |
|---|---:|
| AUROC | 85.47% |
| Average precision | 96.81% |
| F1 at the calibrated threshold | 3.13% |
| True positives / false negatives | 1 / 62 |
| True negatives / false positives | 13 / 0 |

The test set contains 82.89% anomalies, so average precision must be interpreted
against that high positive prevalence. There is no candidate ranking comparison
because the candidate did not finish training.

One normal calibration graph scored 74.38; the next highest scored 7.70. With
only 12 calibration graphs, the 99th-percentile rule yielded a threshold of
67.04. That operating point detected just one anomaly despite good ranking
performance. The saved [calibration diagnostic](../outputs/experiments/prior-mixture-mutag-ocgin-reference/ocgin_mutag_seed0/calibration_diagnostics.json)
records all 12 scores. Small differences between re-scored values and the saved
threshold reflect float32 GPU arithmetic.

This is a calibration-sensitivity observation from one seed, not a reason to
optimize thresholds against these test labels. It demonstrates why ranking and
threshold-based metrics should both be reported.

## Validation and interpretation

The [seven acceptance tests](../tests/test_prior_mixture.py) cover identifiable
mixture recovery, the literal abnormal minimum, constrained feasibility and
infeasibility, generated feature validity, ellipsoid-union scoring,
configuration validation, GPU permutation/batch consistency, and checkpoint
restoration. The end-to-end synthetic test exercises accepted positive and
negative generation and actual radius updates.

That integration test uses synthetic normals, four priors, three ratio epochs,
two embedding dimensions, and explicitly reduced acceptance constraints. It
verifies all code stages; it is not evidence that the default MUTAG design works.
Both the default failure and the controlled test remain visible.

An initial checkpoint assertion used float64's default tolerance even though
the encoder uses float32. An approximately $1.5\times10^{-6}$ score difference
was addressed by checking restored weights exactly and using a declared
float32-appropriate tolerance for scores. The environment also emitted a
pandas/numexpr version warning without preventing execution. No packages were
changed.

The next research question is normal-domain coverage before abnormal-weight
optimization. Training-only investigations should examine better structural
and categorical priors, embedding coverage independent of relative likelihood,
and whether lower dimension or different covariance estimation gives a more
reliable boundary. Merely lowering the acceptance requirement would not
resolve the observed distribution mismatch.

Machine-readable measurements are in
[prior_mixture_initial_experiment.json](prior_mixture_initial_experiment.json).
