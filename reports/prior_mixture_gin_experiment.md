# GIN classification and prior coverage experiment

Replacing ellipsoid scoring with a binary GIN resolves the synthetic-pilot
separation failure. Across three classifier seeds, fresh-pilot AUROC is
0.9344 ± 0.0010 (sample standard deviation), compared with 0.5021 for the frozen
ellipsoid score on the same graphs. However, real anomaly detection remains
weak and sensitive to classifier training: AUROC ranges from 0.4139 to 0.6825.
The two generated populations differ, but the generated abnormalities do not
provide a reliable decision boundary for real MUTAG anomalies.

## Controlled setup

The experiment reuses the exact normal and abnormal mixture weights from the
previous successful diagnostic. The normal weights are approximately 88.79%
on prior 16 and 11.20% on prior 48; the abnormal weights are 25% each on priors
44, 45, 46, and 62. Indices are zero-based. Abnormal fitting used the 0.25 cap
and no shell constraint. No prior or component classifier was refitted.

An independent three-layer GIN with width 32, LayerNorm, and mean/max graph
pooling was initialized for each classifier seed 0, 1, and 2. Each trained for
100 epochs on the same 8,192 normal and 8,192 abnormal synthetic graphs.
A separate 2,048 graphs per class monitored validation. All examples were
unfiltered, and the final epoch was used. No ellipsoid score or sample filter
entered training or the GIN anomaly score.

After fitting, 8,192 new graphs from each of the normal, abnormal, and uniform
mixtures were evaluated. The same fresh graphs were used for all three seeds
and for the frozen ellipsoid and mixture-ratio references. Dataset hashes,
source-checkpoint hashes, prior specifications, and synthetic graph hashes were
verified. Both the mixture fit and the real-data split retain seed 0; only the
binary classifier seed changes.

The real split remains 100 training normals, 12 calibration normals, and a test
set of 13 normals plus 63 anomalies. Mixture fitting and feature standardization
use only training normals. The binary network trains only on synthetic labels.
No real test graph or label is used for fitting or selection.

## Separation and real anomaly detection

| Classifier seed | Fresh pilot AUROC and 95% interval | Real anomaly AUROC and 95% interval | Real test F1 |
|---|---:|---:|---:|
| 0 | 0.9347 (0.9309–0.9381) | 0.6007 (0.4603–0.7424) | 0.4390 |
| 1 | 0.9351 (0.9317–0.9386) | 0.4139 (0.2783–0.5458) | 0.3421 |
| 2 | 0.9332 (0.9295–0.9367) | 0.6825 (0.5397–0.8181) | 0.4051 |

On the same fresh pilots, the frozen ellipsoid AUROC is 0.5021
(95% interval 0.4932–0.5108), and the frozen component-derived mixture ratio
reaches 0.9385 (0.9349–0.9416). These are reference scores from the earlier
proposal checkpoint; the ellipsoid reference is the preliminary diagnostic
geometry, not a fully trained detector. Its difference from the previously
reported 0.4929 comes from drawing new pilots.

Intervals use 1,000 bootstrap replicates within populations. They condition on
one prior fit and one classifier and do not measure uncertainty across splits
or prior banks. The three classifier runs share observations, so their pilot
counts are not pooled. Mean real anomaly AUROC is 0.5657, with sample standard
deviation 0.1377 across classifier seeds.

The framework detection threshold uses the 99th percentile of the 12 real
calibration normals. Confusion counts are:

| Seed | True positives | False negatives | False positives | True negatives |
|---|---:|---:|---:|---:|
| 0 | 18 | 45 | 1 | 12 |
| 1 | 13 | 50 | 0 | 13 |
| 2 | 16 | 47 | 0 | 13 |

The threshold cannot explain the weak AUROC, which does not depend on a selected
cutoff. High average precision should also be interpreted against the high
anomaly prevalence, 63 of 76 test graphs.

## Coverage from a synthetic normal cutoff

For this diagnostic, each classifier uses a cutoff at the 95th percentile of
its synthetic validation normal scores. This cutoff is fixed without inspecting
real calibration or test scores. The table reports the fraction **accepted as
normal**; accepting real anomalies is a failure.

| Population | Classifier seed 0 | Classifier seed 1 | Classifier seed 2 |
|---|---:|---:|---:|
| Normal pilots | 7,744/8,192 (94.53%) | 7,796/8,192 (95.17%) | 7,801/8,192 (95.23%) |
| Abnormal pilots | 2,644/8,192 (32.28%) | 2,859/8,192 (34.90%) | 2,781/8,192 (33.95%) |
| Uniform pilots | 5,333/8,192 (65.10%) | 5,474/8,192 (66.82%) | 5,391/8,192 (65.81%) |
| Real test normals | 13/13 (100.00%) | 13/13 (100.00%) | 13/13 (100.00%) |
| Real test anomalies | 61/63 (96.83%) | 56/63 (88.89%) | 60/63 (95.24%) |

At the binary decision boundary (logit zero), all 13 test normals are also
accepted in all runs. Normal-pilot acceptance is 85.88–86.84%, compared with
13.78–15.20% for abnormal pilots. However, 52–55 of the 63 real anomalies are
accepted as normal at this boundary.

The relative comparison is strong: AUROC separating real test normals from
abnormal pilots is 0.9953–0.9969. Yet this does not carry over to real anomalies.
For classifier seed 0, median logits are −3.648 for synthetic normals, +3.139
for synthetic abnormalities, −6.022 for real test normals, and −5.938 for real
test anomalies. Real anomalies largely occupy the classifier's normal side.

Thus, the normal-boundary compatibility check succeeds on this small held-out
set, but it is overly permissive for real anomalies. Accepting 13 of 13 normals
has a 95% Wilson interval of approximately 77.2–100%; it is not proof of full
coverage. A discriminative classifier can assign normal scores outside the
support of both generators, so acceptance does not establish graph-density
coverage. The experimental conclusion is strong separation between proposals
and a substantial gap in transfer to real anomalies.

## Problems observed and implementation checks

The main unresolved problem is the abnormal training distribution. Minimizing
normal-data likelihood produces a contrast to the normal prior, but does not
ensure that it resembles real anomalies. These results demonstrate the gap in
the learned score; they do not isolate which generator mechanisms cause it.
The present generators also lack molecular-validity constraints.

A second problem is classifier sensitivity on real graphs. Fresh-pilot AUROC
changes by less than 0.002 across seeds, while real anomaly AUROC changes by
about 0.269. The priors, synthetic graphs, and real split were fixed, so prior
refitting cannot explain this variation. The runs vary initialization and
minibatch order, with the framework's existing warning-only GPU determinism.

There is also mild late overfitting. Validation loss is lowest among logged
checkpoints around epochs 30–40, then increases by epoch 100 even while training
loss falls. The planned final-epoch selection was retained. This observation
motivates a future validation-based stopping study; no checkpoint was selected
using real test performance.

All three runs completed without numerical failures in roughly 299–317 seconds
each on separate GPUs. Eleven acceptance tests passed, including a complete
geometry-free fit, encoder updates, checkpoint restoration, graph permutation
invariance, and independent sampling streams. Raw-score AUROCs were recomputed
and verified against the saved results. The implementation is registered through
the framework template as `prior_mixture_gin`; the standard `fit` path can learn
new mixtures without fitting ellipsoids.

## Artifacts

- [Algorithm and reproduction instructions](../docs/prior_mixture_gin.md)
- [Combined measurements and verification](../outputs/diagnostics/prior-mixture-gin-mutag-summary/summary.json)
- [Comparison figure](../outputs/diagnostics/prior-mixture-gin-mutag-summary/gin_comparison.png)
- [PDF figure](../outputs/diagnostics/prior-mixture-gin-mutag-summary/gin_comparison.pdf)
- [Seed 0 results](../outputs/diagnostics/prior-mixture-gin-mutag-seed0/result.json)
- [Seed 1 results](../outputs/diagnostics/prior-mixture-gin-mutag-seed1/result.json)
- [Seed 2 results](../outputs/diagnostics/prior-mixture-gin-mutag-seed2/result.json)
- [Seed 0 progress log](../outputs/diagnostics/prior-mixture-gin-mutag-seed0.log)
- [Seed 1 progress log](../outputs/diagnostics/prior-mixture-gin-mutag-seed1.log)
- [Seed 2 progress log](../outputs/diagnostics/prior-mixture-gin-mutag-seed2.log)

Each run directory also contains the classifier checkpoint, raw pilot and real
scores, resolved configuration, original split, source snapshot, and manifest.
The figure uses seed 0 for its real-score panel as a fixed representative, not
the best-performing seed.
