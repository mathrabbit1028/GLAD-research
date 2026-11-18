# Graph anomaly detection with prior mixtures and multiple ellipsoids

The [GIN variant](prior_mixture_gin.md) replaces the ellipsoid classifier and
sample filters with a general graph classifier. This document preserves the
original ellipsoid design and its experimental history.

**Status: research design with an initial candidate implementation.** The method learns two mixtures over a
bank of graph generators: one that explains observed normal graphs well, and
one that explains them poorly. Samples from these mixtures supply synthetic
normal graphs and pseudo-anomalies. A classifier then learns the sizes of
multiple ellipsoidal normal regions in a graph embedding space.

The central hypothesis is that dataset-specific selection of graph priors can
produce more useful training examples than a fixed perturbation rule, while
multiple ellipsoids can preserve distinct normal modes. This is a research
hypothesis, not an established performance claim.

Two distinctions determine the implementation. The generators need not have
tractable likelihoods, so mixture fitting uses estimated likelihood ratios.
Also, minimizing normal-data likelihood defines a contrasting distribution;
it does not identify the unknown distribution of real anomalies. The design
therefore includes a literal max/min baseline and a constrained variant with
sample filtering.

## Problem and notation

Assume a collection of independent, attributed graphs
$G=(V,E,X)$ and a training set
$\mathcal D_{\mathrm{tr}}=\{G_i\}_{i=1}^{N}$ containing normal graphs only.
Here, normal means belonging to the accepted population, not Gaussian.
The task is graph-level anomaly detection, with one score per graph and larger
scores indicating greater abnormality.

| Symbol | Meaning |
|---|---|
| $K$ | Number of prior components |
| $q_k(G)$ | Distribution induced by graph generator $k$ |
| $w\in\Delta_K$ | Nonnegative mixture weights summing to one |
| $q_w(G)=\sum_k w_kq_k(G)$ | Mixture distribution over whole graphs |
| $w_N,w_A$ | Weights for the normal and contrasting mixtures |
| $h_\psi(G)\in\mathbb R^d$ | Permutation-invariant graph embedding |
| $J$ | Number of ellipsoids, independent of $K$ |
| $\mu_j,\Sigma_j,R_j^2$ | Center, positive-definite shape, and squared radius |
| $\mathcal D_{\mathrm{cal}}$ | Held-out normal graphs used only for the final threshold |

A component is selected independently for each generated graph. The likelihood
is therefore a product of mixtures, not a mixture of whole-dataset likelihoods.
Neither the prior index nor the graph's original class label is a detector
input. If the supplied training collection is unlabeled and contaminated,
the normal-only assumption must be changed explicitly; ordinary maximum
likelihood can otherwise absorb anomalies as normal modes.

## Connection to the graph prior in Section 4

[GraphPFN Section 4](https://arxiv.org/html/2509.21489v4#S4) constructs graph
structure by combining degree-corrected stochastic block models at two levels,
using a random node correspondence and an edge union. A preferential-attachment
stage adds peripheral nodes with random initial degrees. Attributes come from
a random neural structural causal model with mixed MLP and GNN computations;
degree and PageRank can also enter its inputs. Hyperparameters are sampled to
obtain diverse graphs.

Use this construction as the sampling mechanism for each component. GraphPFN
addresses node-level prediction; the proposed extension treats each generated
graph as one example and assigns its normal/pseudo-anomaly training label
separately. The source's generated node targets are not graph anomaly labels.
The prior bank, mixture optimization, rejection rules, and ellipsoidal detector
below are proposed additions.

## Constructing a bank of distinct priors

Let $\theta$ describe a generator configuration and $\xi$ its random choices.
For component $k$, specify a hyperprior $p_k(\theta)$ and sample

$$
\theta\sim p_k,\qquad \xi\sim p(\xi\mid\theta),\qquad
G=\operatorname{Generate}(\theta,\xi).
$$

This induces $q_k$. The integral over hidden configurations is a definition of
the distribution, not an assumed ability to evaluate its density.

Components must differ in their hyperparameter distributions. Giving identical
generators different random seeds does not create distinct priors. Use a
stratified bank of regimes, with variation within each regime, rather than one
fixed graph per component. Store each regime's complete parameter ranges,
sampling laws, feature transformation, validity rules, and version.

For a first design, allocate components across these proposed regimes:

| Axis | Proposed contrasting regimes |
|---|---|
| Community organization | Few or many communities; balanced or unequal sizes |
| Connectivity | Weak, moderate, or strong within-community preference |
| Degree variation | Narrow or heavy-tailed degree profiles |
| Peripheral structure | Small or large peripheral fraction |
| Attribute dependence | Weak or strong dependence on graph structure |
| Attribute variation | Low or high noise, nonlinear effects, and correlation |

Cross the axes selectively instead of taking their full Cartesian product.
Include broad components spanning adjacent regimes. Merge components that are
empirically indistinguishable on independent synthetic samples; their individual
weights would be unstable even when their combined mixture is well determined.
Freeze the bank before optimizing either weight vector.

### Shared graph and feature constraints

Use simple, undirected, unweighted graphs for the initial version, matching a
node-feature and adjacency-only detector. Generate the same feature schema as
the real data. Fit all transformation parameters on training normals only.

- Draw graph size from the same training-derived distribution for every
  component. This prevents prior identity from being determined solely by
  size. Conditioning on size intentionally limits synthetic size anomalies.
- For continuous attributes, map generated variables into training-derived
  scales using a shared transformation. Avoid independently standardizing
  every graph, which can erase meaningful attribute anomalies.
- For categorical attributes, use a declared discretization or categorical
  decoder with valid categories and one-hot groups. Continuous SCM outputs
  cannot be substituted directly for categorical features.
- For featureless datasets, derive the same features from real and generated
  graphs: the repository uses $[1,\log(1+\mathrm{degree})]$ per node. Disable
  unused attribute-generation regimes in this case.
- Reject invalid graphs using the same rule for both mixtures. Include any
  deterministic projection or rejection in the definition of $q_k$, and
  train the ratio estimator on the resulting distribution.

The generic generator does not establish chemical or other domain validity.
Domains requiring such validity need an explicit validator or a domain-specific
generator. Empty edge sets, isolated nodes, and singleton graphs need defined
behavior; zero-node graphs are excluded. Features unavailable at deployment
cannot be used during fitting.

## The ideal mixture objectives

For fixed components with well-defined densities or probability masses on a
common graph space, define average log likelihood

$$
\mathcal L(w)=\frac1N\sum_{i=1}^{N}
\log\left(\sum_{k=1}^{K}w_kq_k(G_i)\right),\qquad
\Delta_K=\{w:w_k\geq0,\ \sum_kw_k=1\}.
$$

The literal interpretation of the proposed idea is

$$
w_N\in\arg\max_{w\in\Delta_K}\mathcal L(w),\qquad
w_A^{\mathrm{literal}}\in\arg\min_{w\in\Delta_K}\mathcal L(w).
$$

This formulation requires the bank to cover the observations. If every
component assigns zero density to a training graph, its likelihood cannot be
repaired by reweighting. The same support problem is more difficult to detect
when densities are estimated.

### Why the literal abnormal mixture collapses

For positive component likelihoods, $\mathcal L$ is concave in $w$. If $e_k$
is the simplex vertex choosing only component $k$, then

$$
\mathcal L(w)=\mathcal L\left(\sum_kw_ke_k\right)
\geq\sum_kw_k\mathcal L(e_k)\geq\min_k\mathcal L(e_k).
$$

Consequently, a global minimum is attained by at least one single component:

$$
k_*\in\arg\min_k\frac1N\sum_i\log q_k(G_i),\qquad
w_A^{\mathrm{literal}}=e_{k_*}.
$$

Ties can yield other minimizers. Zero component likelihoods can make the
objective $-\infty$, making the selection still less informative. A low
likelihood may arise from an irrelevant feature scale or unrealistic topology,
so the literal solution is an essential ablation, not sufficient evidence of
a useful anomaly generator. In particular, $w_A$ is neither $1-w_N$ nor a
component-wise inversion of $w_N$.

## Estimating likelihood ratios from simulations

The full graph sampler has hidden random variables and generation histories.
A likelihood conditional on one sampled history is not the marginal likelihood
of an observed graph. The proposed implementation avoids evaluating that
marginal by learning relative densities from simulated component labels.
Classifier-based ratio estimation is supported by
[Cranmer, Pavez, and Louppe](https://arxiv.org/abs/1506.02169v2);
the multiclass construction below follows directly from Bayes' rule.

Choose known sampling probabilities $\pi_k>0$, initially $\pi_k=1/K$, and define

$$
m(G)=\sum_k\pi_kq_k(G).
$$

Generate independent pairs $k\sim\pi$, $G\sim q_k$. Train a graph classifier
$c_\psi(k\mid G)$ to predict the component with multiclass cross entropy. At
the population optimum,

$$
c^*(k\mid G)=\frac{\pi_kq_k(G)}{m(G)},\qquad
r_k(G)=\frac{q_k(G)}{m(G)}=\frac{c^*(k\mid G)}{\pi_k}.
$$

It follows that

$$
\mathcal L(w)=\frac1N\sum_i\log m(G_i)
+\frac1N\sum_i\log\left(\sum_kw_kr_k(G_i)\right).
$$

The first term is independent of $w$. Thus exact ratios suffice to recover
both weight optima without knowing the absolute likelihood. In practice use

$$
R_{ik}=\widehat r_k(G_i)=\frac{c_\psi(k\mid G_i)}{\pi_k},\qquad
\widehat{\mathcal L}(w)=\frac1N\sum_i\log\left(\sum_kw_kR_{ik}\right).
$$

Train and calibrate the component classifier using separate synthetic
simulation sets, then freeze it before fitting weights. Its actual sampling
class frequencies must equal the $\pi$ used in the ratio conversion. Use
log-softmax and log-sum-exp for numerical stability; arbitrary clipping changes
the objective and must be disclosed if needed.

### What the estimated objective means

A practical classifier has the form $c(k\mid h_\psi(G))$. With a fixed embedding,
Bayes' rule identifies ratios of the **embedding distributions**. These equal
graph likelihood ratios only if the embedding retains the information needed
to distinguish the components. Otherwise, report the result as approximate
mixture likelihood fitting in representation space, not exact graph maximum
likelihood. Freeze preprocessing and the embedding as well as the classifier;
moving them while optimizing $w$ changes the objective.

Use a permutation-invariant graph encoder with graph pooling. A GIN encoder
with pooled node representations is a reasonable first architecture. Do not
feed simulator seeds, component IDs, generation metadata, or arbitrary node
ordering into it. Train on prior classification for the initial version; a
separately specified self-supervised loss on real training normals is an
optional representation ablation.

Check held-out synthetic cross entropy, probability calibration, and the
empirical identity $\mathbb E_{G\sim m}[\widehat r_k(G)]\approx1$. Also compare
real and generated training-domain descriptors such as density, degree
quantiles, clustering, and attribute statistics. Posterior probabilities
always sum to one, even outside the bank's support: these diagnostics can flag
misspecification but cannot certify coverage. Uniform component predictions
can mean indistinguishable priors, not evidence of normality.

## Fitting the two weight vectors

### Normal weights

Maximize $\widehat{\mathcal L}$ on the simplex. For the unregularized objective,
an expectation-maximization update is

$$
\gamma_{ik}^{(t)}=
\frac{w_k^{(t)}R_{ik}}{\sum_\ell w_\ell^{(t)}R_{i\ell}},\qquad
w_k^{(t+1)}=\frac1N\sum_i\gamma_{ik}^{(t)}.
$$

Initialize with strictly positive weights. Stop by a declared objective-change
tolerance or iteration budget. Concavity makes maximization a convex
optimization problem, although indistinguishable components can prevent a
unique weight vector. Projected optimization is an alternative. An optional
penalty $-\lambda_N\operatorname{KL}(w\Vert\pi)$ discourages unstable sparse
weights, but changes pure maximum likelihood and requires a different update.
The initial design uses the unregularized version.

### Abnormal weights

First compute the literal baseline by evaluating all $K$ vertices using
$R_{ik}$. For the main variant, restrict concentration and require the mixture
to generate some graphs near the preliminary normal boundary:

$$
\mathcal W_A=
\left\{w\in\Delta_K:\ w_k\leq u,\quad
\sum_kw_ka_k\geq a_{\min}\right\},\qquad
w_A\approx\arg\min_{w\in\mathcal W_A}\widehat{\mathcal L}(w).
$$

Here $1/K\leq u\leq1$, and $a_k$ is an independently estimated probability
that a graph from component $k$ falls in the preliminary shell defined below.
For example, $u=0.25$ requires at least four active components. This restricts
weight concentration, but does not guarantee diversity if those components
generate similar graphs. The shell constraint discourages spending all mass
on components whose samples will be discarded as trivial outliers.

This is a concave minimization problem over a convex polytope and is generally
nonconvex as an optimization task. Use feasible projected gradient descent
with multiple starts and a fixed budget, retaining the lowest feasible
objective. The gradient for either weight problem is

$$
\frac{\partial\widehat{\mathcal L}}{\partial w_k}
=\frac1N\sum_i\frac{R_{ik}}{\sum_\ell w_\ell R_{i\ell}}.
$$

Projection is a small quadratic program over the stated constraints; check
feasibility first with a linear program. Record constraint residuals and the
spread of final objectives across starts. Do not claim a global minimum for
this solver. If the constraints are infeasible, revise the bank or explicitly
change the experimental configuration; do not silently remove constraints.
Pilot uncertainty in $a_k$ and the stricter final acceptance rule can still
produce low acceptance at generation time.

## Establishing multiple normal modes

Use the frozen encoder's embeddings of real training normals. Standardize
embedding coordinates using training statistics and remove constant directions.
If needed, fit a lower-dimensional projection on training normals only. Apply
the same fixed transform to all subsequent graphs, and denote the resulting
embedding by $z(G)\in\mathbb R^d$.

Cluster these real embeddings into $J$ groups, initially with seeded k-means.
Choose $J$ before final evaluation and merge groups below a declared minimum
occupancy. Do not equate a prior component with a normal mode: one component
may cover several modes, and several components may explain one mode.

For each group, estimate its mean $\mu_j$ and sample covariance $S_j$. Use
shrinkage and a positive floor,

$$
\Sigma_j=(1-\rho)S_j+\rho\,\frac{\operatorname{tr}(S_j)}d I
+\epsilon_\Sigma I,\qquad 0\leq\rho\leq1,\quad\epsilon_\Sigma>0.
$$

Define the squared Mahalanobis distance

$$
d_j^2(G)=(z(G)-\mu_j)^\top\Sigma_j^{-1}(z(G)-\mu_j).
$$

Set an initial squared radius $R_{j,0}^2$ to a floored 95th percentile of
$d_j^2$ among that group's real training members. This empirical radius does
not assume Gaussian embeddings. The preliminary union score is

$$
s_0(G)=\min_{1\leq j\leq J}\frac{d_j^2(G)}{R_{j,0}^2}.
$$

All centers, covariance matrices, and initial radii are fitted before the
abnormal shell pilot. This ordering avoids a circular dependency between
abnormal generation and the detector it trains. Use pooled or diagonal
covariances when group sample sizes are too small; merge singleton groups.
If fewer than two distinct embeddings remain, report insufficient geometric
support instead of constructing artificial normal modes.

## Generating the classifier training data

Draw a component $k\sim\operatorname{Categorical}(w_N)$ or
$k\sim\operatorname{Categorical}(w_A)$ and then draw $G\sim q_k$.
Generate fresh samples independently of those used to train or calibrate the
ratio estimator and to estimate shell acceptance.

The main variant uses these fixed acceptance rules:

$$
\mathcal A_N=\{G:s_0(G)\leq1\},
$$

$$
\mathcal A_A=\left\{G:1+\delta\leq s_0(G)\leq1+\Delta,
\quad t(G)\geq\gamma\right\},\qquad 0<\delta<\Delta,
$$

where the estimated contrast score is

$$
t(G)=\log\frac{\sum_kw_{A,k}\widehat r_k(G)}
{\sum_kw_{N,k}\widehat r_k(G)}.
$$

For the earlier abnormal optimization, estimate
$a_k=\Pr_{q_k}(1+\delta\leq s_0(G)\leq1+\Delta)$ using a separate pilot bank.
The pilot excludes $t$, which depends on the as-yet-unfitted abnormal weights.

Accepted normal samples receive label $y=0$ and accepted abnormal samples
receive label $y=1$. The gap between the two acceptance sets avoids assigning
conflicting synthetic labels at the same preliminary boundary. The outer
limit rejects very easy outliers. The contrast rule favors graphs more
consistent with the abnormal proposal than the normal proposal. None of these
rules makes a pseudo-anomaly a verified real anomaly; rare normal structures
can still receive an incorrect synthetic label.

Rejection changes the distributions actually used for training:

$$
\widetilde q_N(G)=\frac{q_{w_N}(G)\mathbf1_{\mathcal A_N}(G)}{Z_N},
\qquad
\widetilde q_A(G)=\frac{q_{w_A}(G)\mathbf1_{\mathcal A_A}(G)}{Z_A},
$$

where $Z_N,Z_A$ are acceptance probabilities. Thus the fitted weights describe
the proposal mixtures, not the component frequencies after rejection. Record
both. Keep all real training normals as positives, including those outside
the preliminary regions, so filtering synthetic positives does not erase
observed normal tails.

Set a finite proposal budget and report acceptance by component. If either
accepted class is empty, or accepted negatives lose all useful variation,
stop with an explicit generation failure. Any fallback to unfiltered graphs
or a different shell is a separate, recorded variant. Rejection can also
change graph-size frequencies despite shared size sampling; inspect this and
compare a size-stratified sampling ablation.

## Training an ellipsoidal classifier

The initial implementation freezes the encoder, centers, and covariance
matrices. It trains one radius per region using real normals and both
synthetic classes. This makes the classifier's geometric interpretation exact
and prevents a trainable encoder from collapsing normal representations.

Parameterize radii through $R_j^2=R_{j,0}^2\exp(v_j)$ and constrain
$v_j\in[\log c_{\min},\log c_{\max}]$ with positive finite bounds. Define

$$
s(G)=\min_j\frac{d_j^2(G)}{R_j^2},\qquad
p_A(G)=\operatorname{sigmoid}\bigl(\beta[s(G)-1]\bigr),\quad\beta>0.
$$

Minimize the class-balanced objective

$$
\begin{aligned}
\mathcal L_{\mathrm{det}}={}&
\tfrac12\left[
(1-\eta)\,\mathbb E_{G\sim\mathcal D_{\mathrm{tr}}}
\operatorname{BCE}(p_A(G),0)
+\eta\,\mathbb E_{G\sim\widetilde q_N}
\operatorname{BCE}(p_A(G),0)\right]\\
&+\tfrac12\,\mathbb E_{G\sim\widetilde q_A}
\operatorname{BCE}(p_A(G),1)
+\frac{\lambda_R}{J}\sum_jv_j^2.
\end{aligned}
$$

Here $0\leq\eta<1$ controls synthetic-positive influence. The radius penalty
anchors the regions to real normal geometry, while bounded radii prevent
arbitrary expansion or collapse. Use BCE with logits for stability and a fixed
positive $\beta$ initially. The hard minimum is differentiable almost
everywhere; ties can use a deterministic subgradient. Each negative must be
outside every ellipsoid to be classified as abnormal.

At the nominal boundary, the normal set in embedding space is exactly

$$
\mathcal E=\bigcup_{j=1}^{J}
\{z:(z-\mu_j)^\top\Sigma_j^{-1}(z-\mu_j)\leq R_j^2\}.
$$

Ellipsoids may overlap; their union need not be connected. This can preserve
a low-normality gap between distinct modes that a single large ellipsoid
would include. The claim concerns embedding space: the corresponding set of
graphs need not have an ellipsoidal geometry.

The sigmoid output is a synthetic-training discrimination score, not a
calibrated probability of real anomaly. A separately trained unrestricted MLP
would not preserve the union-of-ellipsoids decision rule. Joint encoder or
covariance learning is a later ablation requiring renewed covariance fitting,
anti-collapse controls, and a fresh final calibration.

With $J=1$, radius learning only multiplies every score by the same positive
constant. It cannot change AUROC or average precision, and quantile calibration
cancels that scale change in the final decisions. With $J>1$, learning relative
radii can change rankings and region assignments. Even then, this initial
classifier cannot discover new embedding directions or relocate a missing
mode; those capabilities require a richer, separately evaluated training stage.

## Inference and threshold calibration

Return $s(G)$ as the anomaly score. It is deterministic, independent of other
graphs in the inference batch, and requires no graph generation.

After every model parameter has been frozen, set

$$
\tau=Q_{0.99}\{s(G):G\in\mathcal D_{\mathrm{cal}}\},\qquad
\widehat y(G)=\mathbf1[s(G)>\tau].
$$

This follows the repository's normal-only calibration protocol. The deployed
normal region remains a union of ellipsoids, now with squared radii
$\tau R_j^2$. An empirical 99th percentile does not by itself give an exact
finite-sample 1% false-positive guarantee, particularly with few calibration
graphs. Do not use calibration graphs to select $K$, $J$, the prior bank,
shell widths, or training epochs.

## End to end procedure

```text
Input: normal training graphs Dtr, normal calibration graphs Dcal,
       fixed configuration and random seeds

1. Fit feature transformations and the shared size law using Dtr.
2. Specify and freeze K valid prior components q[1:K].
3. Generate independent synthetic ratio-training and ratio-calibration sets.
4. Train the component classifier; calibrate its probabilities; freeze it.
5. Cache R[i,k] = c(k | G_i) / pi[k] for G_i in Dtr.
6. Fit wN by maximizing the cached mixture objective.
7. Fit the embedding transform, normal clusters, centers, covariances,
   and initial radii using only Dtr. Freeze the preliminary score s0.
8. Draw an independent pilot from every component; estimate shell rates a[k].
9. Compute the literal abnormal vertex. For the main variant, check
   feasibility and minimize the objective over WA with multiple starts.
10. Draw fresh graphs from q_wN and q_wA; apply the fixed acceptance rules.
    Fail explicitly if the accepted training sets are unusable.
11. Train bounded ellipsoid radii on Dtr and both accepted synthetic sets.
12. Freeze the detector. Calibrate the final threshold using Dcal only.
13. Score each test graph independently; use test labels only for evaluation.

Output: fitted prior descriptions, wN, wA, frozen encoder and transforms,
        ellipsoid parameters, threshold, and generation diagnostics
```

## Initial experimental configuration

These values define a starting experiment, not tuned recommendations or
GraphPFN defaults. Reduce the bank or use fewer modes when data cannot
support the proposed resolution.

| Parameter | Initial value or rule |
|---|---|
| Prior count $K$ | 64 distinct regimes; merge indistinguishable components |
| Ratio training samples | 512 graphs per component |
| Ratio calibration samples | 128 fresh graphs per component |
| Shell pilot samples | 128 additional graphs per component |
| Ratio classifier training | 100 fixed epochs; temperature fitted on synthetic calibration only |
| Embedding dimension $d$ | At most 16; reduce for training-set rank and occupancy |
| Region count $J$ | Start at 4; merge groups smaller than $\max(10,2d)$ |
| Covariance shrinkage $\rho$ | 0.1, with $\epsilon_\Sigma=10^{-4}$ after standardization |
| Normal weight solver | Positive uniform start, 1000 iterations, objective tolerance $10^{-6}$ |
| Abnormal concentration cap $u$ | 0.25, requiring at least four surviving components |
| Abnormal shell requirement $a_{\min}$ | 0.05, subject to explicit feasibility check |
| Abnormal optimization | 16 feasible starts, 500 projected steps per start with backtracking |
| Shell bounds | $\delta=0.05$, $\Delta=3$, hence $1.05\leq s_0\leq4$ |
| Contrast threshold $\gamma$ | $\log 2$ |
| Accepted synthetic training sets | Target $N$ positives and $N$ negatives; at most $100N$ proposals per class |
| Real versus synthetic positives | $\eta=0.5$ |
| Radius training | 100 fixed epochs, Adam, learning rate $10^{-3}$ |
| Radius bounds and regularization | $c_{\min}=0.5$, $c_{\max}=2$, $\lambda_R=0.01$ |
| Sigmoid slope $\beta$ | 5 |
| Final operating threshold | Repository's 99th percentile of held-out normal scores |

If fewer than four distinguishable components survive, the stated capped
variant is not applicable as configured. If clustering reduces to one group,
report a single-ellipsoid result rather than claiming multiple supported
normal modes. With very small training sets, use a declared diagonal-covariance
fallback or report insufficient data.

## Evaluation and failure analysis

Use the shared splits and evaluation rules in [protocol.md](protocol.md).
Compare AUROC and anomaly-positive average precision, plus F1 at the frozen
normal-calibrated threshold. Report results across the same seeds as the
baselines, including graph-generation cost, training time, inference time,
memory, and generation failures. Synthetic classification accuracy alone does
not validate anomaly detection on real graphs.

| Comparison | Question it tests |
|---|---|
| Uniform versus fitted normal weights | Does mixture fitting improve generated normal coverage? |
| Literal vertex versus constrained abnormal weights | Does preventing concentration produce more useful negatives? |
| Unfiltered versus shell-filtered proposals | Does boundary filtering help beyond rejecting easy outliers? |
| Shell only versus shell plus contrast rule | Do fitted abnormal weights add information beyond geometry? |
| Shell-filtered uniform priors versus shell-filtered fitted priors | Is the gain due to mixture selection or just the rejection rule? |
| Real normals only versus real plus synthetic normals | Do generated positives help or introduce domain bias? |
| $J=1$ versus several ellipsoids | Does the dataset benefit from multiple normal modes? |
| Spherical, diagonal, and full covariance | Does directional geometry justify its estimation cost? |
| Fixed initial radii versus trained radii | Does synthetic supervision improve the actual detector? |
| Ellipsoidal head versus unrestricted MLP | What is gained or lost by constraining the decision regions? |

Watch for five specific failure modes:

1. **Prior mismatch.** No mixture covers the real normal population. Inspect
   descriptor coverage and synthetic-to-real discrepancies before interpreting
   weights as meaningful.
2. **Ratio-estimation error.** Miscalibrated component predictions distort both
   optima. Check independent simulation diagnostics and repeat with new seeds.
3. **Shortcut discrimination.** Feature validity, graph size, or rendering
   conventions reveal the synthetic class. Match schemas and audit accepted
   sample distributions, not only proposal distributions.
4. **False pseudo-anomalies.** A missing normal mode is treated as abnormal.
   Preserve real positives, inspect rejection rates by normal cluster, and
   compare radius changes and cluster occupancy.
5. **Circular synthetic supervision.** Labels produced by the preliminary
   geometry merely reproduce that geometry. The fixed-radius and uniformly
   sampled shell ablations are necessary to establish added value.

Normal-only observations cannot identify every possible anomaly distribution.
The resulting detector depends on the bank's inductive assumptions and on
which distinctions the embedding preserves. Strong real-data performance must
be established empirically, especially for anomalies inside represented
normal modes.

## Computational cost and repository integration

After graph encoding, caching $R$ costs $O(NK)$ memory, and each mixture
objective or gradient pass costs $O(NK)$. Large datasets can stream row blocks.
Simulation and component-classifier training are likely to dominate fitting;
their cost depends on graph size and the actual sampler, which must not be
assumed to be linear in the number of edges.

For full covariance ellipsoids, factorization costs $O(Jd^3)$ once, and scoring
adds $O(Jd^2)$ per graph after encoding. Use Cholesky solves rather than explicit
matrix inversion. Diagonal covariance reduces the geometric scoring cost to
$O(Jd)$.

The implementation belongs in the candidate area and implements
the existing [BaseAnomalyModel contract](../src/models/base_model.py):

- `fit` uses only the supplied training subset, fits the prior and geometry
  stages, and never accesses calibration or test graphs through the backing
  dataset. Framework calibration occurs after `fit`.
- `anomaly_score` returns a finite tensor of shape `[batch.num_graphs]` with
  the score $s(G)$. Use evaluation mode and graph-local pooling.
- Checkpoint state includes encoder weights, all feature/embedding transforms,
  centers, covariance factors, learned radii, and preprocessing schema. Save
  prior specifications, sampling probabilities, classifier calibration,
  mixture weights, seeds, and acceptance diagnostics for reproducibility.
- Keep the framework-managed threshold in the normal checkpoint flow. Store
  fitted state as tensors and plain containers compatible with the existing
  loading contract.

Implementation acceptance checks should cover permutation-invariant scoring,
score consistency across inference batches, checkpoint round trips, valid
generated schemas, solver feasibility, and recovery of known mixture weights
on a toy problem with identifiable components and exact ratios. Benchmark
runs should start only after these checks and the generation diagnostics pass.

## Initial implementation choices

The adapter in [prior_mixture.py](../src/models/candidates/prior_mixture.py)
starts from the framework's candidate template. Its editable settings are in
[prior_mixture.yaml](../configs/models/prior_mixture.yaml), with a lightweight
validator for terminal UI planning. Supporting sampler, optimization, and
geometry modules live in `src/models/candidates/prior_mixture_impl/`.

The sampler is an independent implementation of the described construction,
not a copy of GraphPFN's released sampler. It uses Bernoulli degree-corrected
block layers, capped edge probabilities, lognormal degree effects, and
preferential attachment with positive initial attractiveness so isolated nodes
remain eligible. A random bijection combines the layers. The attribute SCM
uses random tanh networks, mixed local and graph computations, optional degree
and PageRank inputs, and mean, max, min, normalized, or attention aggregation.

The 64-component bank uses six binary regimes with randomness within each:
community count, affinity contrast, degree heterogeneity, peripheral fraction,
attribute dependence, and attribute noise. For degree-only features, the last
two axes instead vary density and attachment attractiveness. Smaller banks
select evenly spaced regime IDs. Distinct regimes can still overlap strongly;
the implementation reports classifier diagnostics rather than automatically
merging priors from an unvalidated statistical equivalence test.

The initial feature adapter supports continuous features, one complete one-hot
categorical group, or the framework's degree features. Ambiguous binary schemas
are rejected. Mixed categorical groups require a future explicit schema adapter.
Categorical outputs use a fixed normal-CDF mapping and training category
frequencies. No chemical validity is claimed for generated molecular graphs.
The dense sampler rejects training graphs above `prior.max_nodes` (2048 by
default); it never truncates them.

Temperature calibration uses independent synthetic calibration graphs. Geometry
uses a training-only standardized SVD projection and merges underpopulated
k-means groups. The abnormal optimizer projects directly onto the capped simplex
using a scalar multiplier. It uses SLSQP only when the additional shell
inequality is violated by that projection; a linear program checks feasibility.
Gradient steps are scaled before backtracking to bound projection inputs.
Radius fitting uses cached distances and the full class-balanced loss.

Every stage emits `PME` JSON log records, including solver results, shell rates,
proposal and acceptance counts, actual region occupancy, and radius scales.
Accepted samples are generated with a finite budget. A nonempty set below the
requested target is used with an explicit shortfall diagnostic; an empty class
or fewer than two distinct abnormal distance vectors stops training. This
shortfall policy does not change the acceptance thresholds.

The reproducible initial experiment profile is
[prior_mixture_mutag.json](../experiments/prior_mixture_mutag.json): MUTAG,
seed 0, the full default budgets, and one GPU. The profile names the installed
`agdiff` Python environment and GPU 1; change those execution settings for
another machine. The shared runner snapshots source/configuration, logs stages,
calibrates on held-out normals, and evaluates held-out graphs.

Run acceptance checks with the experiment Python using
`python -m unittest discover -s tests -p test_prior_mixture.py -v`, and run the
profile with `python scripts/run_benchmarks.py --profile experiments/prior_mixture_mutag.json`.
The GPU-specific scoring check is skipped when no GPU is visible; full
acceptance requires a GPU run. The numerical mixture and schema checks run
without training a detector.

The [initial experiment report](../reports/prior_mixture_initial_experiment.md)
records seven passing acceptance tests and the full-default MUTAG run's
failure at the abnormal-mixture feasibility check. No candidate test metric
is available from that run; the report includes the observed prior-coverage
problem and a separate OCGIN reference experiment.

### Relative proposal diagnostic

The diagnostic in [run_prior_mixture_diagnostic.py](../scripts/run_prior_mixture_diagnostic.py)
separates mixture usefulness from the absolute shell requirement. It reuses the
framework configuration, training split, sampler, and proposal-fitting stage,
but sets `mixture.min_shell_rate=0` while retaining the 0.25 weight cap. It does
not train radii or evaluate real test anomalies. The ordinary detector keeps
its original shell constraint.

After freezing both weight vectors and the normal geometry, sample 8,192 fresh,
unfiltered graphs from each of the normal, abnormal, and uniform mixtures.
Use independent random streams for these pilots, separate from ratio training,
synthetic calibration, and the earlier per-component pilot. Evaluate all three
mixtures with the same preliminary score $s_0$.

Report score quantiles, the normal-region fraction $\Pr(s_0\leq1)$, and the shell
fraction separately from relative discrimination. The primary relative measure
is $\Pr(s_0(G_A)>s_0(G_N))+\tfrac12\Pr(s_0(G_A)=s_0(G_N))$. This is a synthetic
pilot AUROC: 0.5 indicates no directional ranking advantage, and larger values
indicate that abnormal pilots tend to lie farther from the real normal regions.
The normal-versus-uniform and uniform-versus-abnormal comparisons locate the
contribution of each fitted mixture.

A 1,000-replicate independent bootstrap within each pilot population supplies
a 95% interval conditional on the fitted model. It does not capture uncertainty
across dataset splits, ratio-estimator training runs, or generator designs.
Relative separation can be useful even when both populations remain far from
real normals; it is not proof of adequate absolute coverage or real-anomaly
detection performance.

The driver saves a frozen-geometry checkpoint before abnormal optimization, a
proposal checkpoint before drawing final pilots, raw per-graph scores and
component IDs, a source snapshot, configuration, and a machine-readable result.
Use a fresh output directory for each experiment. The `--resume-geometry` option
reuses a frozen geometry checkpoint after checking its data hash, split, seed,
and fitted-stage settings.

The [relative pilot experiment report](../reports/prior_mixture_relative_experiment.md)
records the completed MUTAG diagnostic: ellipsoid-distance pilot AUROC 0.493,
secondary likelihood-ratio pilot AUROC 0.943, low absolute normal-region coverage,
and sensitivity of the geometry across refits. These are synthetic-pilot
measurements, not real-anomaly detection scores.
