# Prior mixture classification with GIN

This variant replaces the ellipsoid decision rule with an independently trained
binary graph classifier. It retains the prior bank and likelihood-based normal
and abnormal mixture weights. All generated graphs enter classifier training;
there is no ellipsoid fitting, boundary-shell constraint, or rejection sampling.
The candidate is registered as `prior_mixture_gin` through the framework's model
adapter and configuration templates.

The [MUTAG experiment report](../reports/prior_mixture_gin_experiment.md) records
the controlled three-seed comparison, coverage diagnostics, and real anomaly
detection results.

## Model

Let the generator bank be \(p_1,\ldots,p_K\), and let
\(q_w(G)=\sum_k w_k p_k(G)\). The bank is the existing independent,
Section-4-inspired implementation described in
[the original design](prior_mixture_ellipsoids.md). The generator is not an
exact reproduction of GraphPFN and does not enforce molecular validity.

A component-identity GIN, trained on uniform prior draws and temperature-scaled
on independent synthetic data, estimates \(r_k(G)=p_k(G)/\bar p(G)\), where
\(\bar p=K^{-1}\sum_k p_k\). For real training normals \(D_N\), optimize

\[
 L(w)=|D_N|^{-1}\sum_{G\in D_N}\log\left(\sum_k w_k\hat r_k(G)\right).
\]

The normal weights maximize this surrogate objective on the simplex. The
abnormal weights minimize it with \(w_k\le0.25\) for diversity. The capped
minimization is nonconvex and uses multiple starts; the result is not a
certified global optimum. The shell-rate constraint is disabled. Component
classification errors remain a source of uncertainty in both mixtures.

Draw balanced datasets independently from \(q_N\) and \(q_A\). Train a new GIN
from random initialization using binary cross entropy:

\[
 \mathcal L(\theta)=-\tfrac12\mathbb E_{q_N}\log(1-\sigma(f_\theta(G)))
                   -\tfrac12\mathbb E_{q_A}\log\sigma(f_\theta(G)).
\]

The classifier has three GIN layers, width 32, node LayerNorm, ReLU, mean and
maximum graph pooling, and a linear binary head. Node feature standardization
uses only real training normals. The binary classifier does not reuse the
component classifier's encoder. Its target is the generating mixture, not a
real anomaly label; the two mixtures can overlap.

The anomaly score is the raw binary logit \(f_\theta(G)\). It permits a flexible
boundary in graph space without imposing ellipsoidal regions. Under an ideal
population optimum with balanced classes, this logit estimates
\(\log(q_A(G)/q_N(G))\). Finite training and approximation error prevent treating
it as an exact density ratio. Raw logits avoid probability saturation ties.

## Training and evaluation protocol

The default classifier uses 8,192 samples per class, 100 epochs, batch size 64,
Adam with learning rate 0.001, and gradient clipping at 5. An independent 2,048
samples per class track validation loss and AUROC. Following the shared
framework protocol, the final epoch is used; validation does not select a model.
No held-out real graph is used to fit mixtures, preprocessing, or the classifier.

The standard adapter `fit` learns the priors and classifier from real normal
training data. The controlled experiment driver can instead reuse a previously
saved cap-only mixture fit. It verifies the dataset hash and prior specification
and restores the exact real-data split. Reusing the same mixtures is necessary
for a comparison focused on the classifier change.

The controlled MUTAG experiment varies classifier seeds 0, 1, and 2 while fixing
the mixture fit, split seed 0, and synthetic datasets. These runs measure
classifier-training variation, not variation across prior fits or data splits.
Independent streams are used for classifier training, synthetic validation, and
fresh evaluation pilots. Stream seeds and graph hashes are saved.

The shared evaluator sets the final detection threshold at the 99th percentile
of the 12 held-out calibration normals. It then evaluates the unchanged test set
of 13 normals and 63 anomalies. Test labels are used only for reporting.

## Verifying relative prior coverage

The primary comparison is AUROC on 8,192 fresh, unfiltered pilots per mixture:
\(P(f(G_A)>f(G_N))\), with half credit for ties. This measures whether the
normal and abnormal proposals differ in a direction the classifier can learn.
For a paired reference, the previous ellipsoid and component-derived mixture
ratio also score these same fresh graphs; neither reference score is an input
to GIN training or inference.

Coverage of real normals is a separate question. Report:

1. The fraction of each population classified normal at binary logit zero
   (equal synthetic class priors).
2. The fraction below a cutoff fixed at the 95th percentile of **synthetic**
   validation normals, including held-out real normals. This checks whether a
   boundary learned from synthetic normals transfers to real normals.
3. Real-normal score positions relative to synthetic normals and abnormal pilots,
   and real anomaly detection AUROC on the untouched test set.

A threshold calibrated on real normals can mask a synthetic-to-real mismatch;
therefore both the synthetic cutoff and the framework's real-normal cutoff are
reported. High pilot AUROC alone does not verify real-normal prior support, and
acceptance by a discriminative boundary is not a graph-density coverage test.
A classifier may be confident outside both training distributions.

Bootstrap intervals resample within the compared populations. They condition on
the fitted proposals and classifier. The three runs' range is reported separately;
identical evaluation graphs across runs are not pooled as independent samples.
The 13 test normals limit the precision of real-normal acceptance estimates.

## Reproduction and artifacts

The registered configuration is
[`configs/models/prior_mixture_gin.yaml`](../configs/models/prior_mixture_gin.yaml).
It includes the shared proposal settings; its inherited geometry settings are
inactive. Standard framework training selects `model=prior_mixture_gin`.

For the controlled replacement experiment:

```bash
CUDA_VISIBLE_DEVICES=4 /home/mingeon/.conda/envs/agdiff/bin/python \
  scripts/run_prior_mixture_gin.py \
  --reference-run outputs/diagnostics/prior-mixture-relative-mutag-seed0-retry \
  --output outputs/diagnostics/prior-mixture-gin-mutag-seed0 \
  --classifier-seed 0 --pilots 8192 --bootstrap 1000
```

Use a fresh output directory. Repeat with classifier seeds 1 and 2 to assess
classifier variation. Each run saves source/config snapshots, hashes, the split,
a classifier checkpoint, per-graph synthetic and real scores, diagnostics, and
results. The GIN adapter also supports framework checkpoint restoration.
