# Autoencoder feedback through prior mixtures and GIN training

**Status:** mathematical design. The fixed-bank Level B variant now has a
[finite-update implementation](ae_prior_mixture_level_b.md); Level C remains a
proposal. Implementation does not establish anomaly-detection effectiveness.

The proposed optimization is possible, provided that the gradient includes how
AE-transformed data changes mixture fitting **and the trained GIN weights**.
A frozen-GIN input gradient is only an approximation to this optimization.
The original prior-mixture implementation cannot propagate the complete gradient unchanged:
it uses detached arrays, discrete sampling, SciPy optimizers, and inference
methods that disable gradients.

This document specifies a differentiable bilevel formulation, a practical first
implementation with a fixed prior bank, and the extra changes required to let
the prior bank and component classifier depend on the AE as well. It extends
[the GIN mixture design](prior_mixture_gin.md); it does not reintroduce ellipsoids.

## 1. Meaning of the AE output and the matching objective

Let a graph be \(G=(A,X,n)\), with symmetric adjacency \(A\), no self-loops,
and node features \(X\). Let

\[
T_\phi(G)=D_{\phi_D}(E_{\phi_E}(G))
             =(\widetilde A_\phi,\widetilde X_\phi,n),
\qquad \phi=(\phi_E,\phi_D).
\]

Here “AE output” means the **decoded graph**, not just its latent vector.
The first design preserves node count and correspondence with the input nodes.
Changing node count, discovering node correspondences, or operating entirely in
latent-vector space would require a different generator and reconstruction loss.

The desired feedback compares three logit distributions, all evaluated with the
same final classifier for a given AE parameter value:

- Original training-normal graphs: \(P_0^\phi\).
- AE-transformed training-normal graphs: \(P_T^\phi\).
- Synthetic normal graphs from the learned normal mixture: \(P_N^\phi\).

The original purple curve corresponds to the first population; the original
blue curve corresponds to the third. The new AE introduces the second population.
Matching only the AE outputs to synthetic outputs does not necessarily close the
original blue–purple gap. The proposed loss explicitly includes that gap.

All three distributions depend on \(\phi\): even the logits of unchanged raw
graphs depend on \(\phi\) through the classifier trained downstream of the AE.
Except for a deliberately fixed reference teacher, none of these target branches
is detached in the full bilevel formulation.

## 2. Data roles and notation

Use real normal graphs only for fitting and selecting the method. Within the
existing training-normal set, make disjoint subsets:

\[
\mathcal D_{\rm fit},\quad \mathcal D_{\rm meta},\quad
\mathcal D_{\rm select}.
\]

For a first MUTAG study, a prespecified 70/15/15 split of the existing 100 training
normals is possible, though statistically noisy. Cross-fitting is a later option.
The existing 12 calibration normals remain reserved for final threshold
calibration; the existing test set is not used for loss construction, early
stopping, or hyperparameter selection. Controls must use the same reduced fit
split. Report the additional split as a protocol change.

| Symbol | Meaning |
|---|---|
| \(\phi\) | Encoder and decoder parameters |
| \(\eta(\phi)\) | Prior-bank settings fitted to transformed fit normals |
| \(p_{k,\eta}\), \(g_k(\epsilon;\eta)\) | Prior component and its sampler |
| \(\omega(\phi)\), \(T_c(\phi)>0\) | Component-classifier weights and temperature |
| \(w_N(\phi),w_A(\phi)\) | Normal and abnormal mixture weights |
| \(\theta(\phi)\) | Final binary GIN parameters after its training procedure |
| \(f_\theta(G)\) | Binary GIN logit; larger means more anomalous |
| \(\Xi\) | Fixed random draws used in one unrolled computation |
| \(\operatorname{sg}\) | Stop-gradient; used only where explicitly stated |

Separate random streams are required for component-classifier training,
temperature fitting, final-GIN training, meta-loss pilots, and final evaluation.
For differentiation, condition on the draws \(\Xi\); refresh them between outer
updates. Reuse the same draws for finite-difference gradient checks.

## 3. The complete dependence on the AE

The inner pipeline is a learning operator, not a fixed neural network:

\[
\begin{aligned}
\widetilde{\mathcal D}_\phi &=T_\phi(\mathcal D_{\rm fit}),\\
\eta(\phi)&=H(\widetilde{\mathcal D}_\phi),\\
(\omega(\phi),T_c(\phi))&=\mathcal C(\eta(\phi);\Xi_C),\\
(w_N(\phi),w_A(\phi))&=\mathcal W(
        \widetilde{\mathcal D}_\phi,\eta(\phi),\omega(\phi),T_c(\phi)),\\
\theta(\phi)&=\mathcal A(\eta(\phi),w_N(\phi),w_A(\phi);\Xi_f).
\end{aligned}
\]

\(\mathcal A\) includes the actual classifier training updates. Its output is
not independent of the AE merely because classifier training occurs in another
function or uses a separate optimizer.

For a transformed graph, the total derivative is

\[
\frac{d f_{\theta(\phi)}(T_\phi(G))}{d\phi}
=
\underbrace{\frac{\partial f}{\partial T}\frac{\partial T_\phi(G)}{\partial\phi}}
_{\text{direct input path}}
+
\underbrace{\frac{\partial f}{\partial\theta}\frac{d\theta(\phi)}{d\phi}}
_{\text{classifier-training path}}.
\tag{1}
\]

For an unchanged raw graph,

\[
\frac{d f_{\theta(\phi)}(G)}{d\phi}
=\frac{\partial f}{\partial\theta}\frac{d\theta(\phi)}{d\phi}.
\tag{2}
\]

For a synthetic graph, there is additionally a generator path through
\(g_k(\epsilon;\eta(\phi))\). Synthetic-mixture expectations also have derivatives
through their weights. Leaving any of these paths detached changes the gradient
being computed and must be reported as an approximation.

## 4. Differentiable graph representation

A node-aligned decoder can output edge logits \(B\) and category logits \(C\):

\[
\widetilde A_{uv}=\sigma((B_{uv}+B_{vu})/2),\ u\ne v,
\qquad \widetilde A_{uu}=0,
\qquad \widetilde X_{vc}=\operatorname{softmax}(C_v)_c.
\]

For continuous node attributes, use a continuous output head instead. Preserve
feature schema explicitly; do not let automatic schema detection interpret
probability vectors as a new continuous-feature dataset. For featureless graphs,
recompute the degree-based feature channel from \(\widetilde A\) so that structure
and features remain consistent.

Use a GIN forward that accepts real edge weights:

\[
H^{(\ell+1)}_v=\operatorname{ReLU}\!\left(
\operatorname{LN}\!\left(
\operatorname{MLP}_\ell\!\left((1+\epsilon_\ell)H_v^{(\ell)}
+\sum_{u\ne v}\widetilde A_{vu}H_u^{(\ell)}\right)\right)\right).
\]

Mean/max graph pooling and a linear head can remain as in the current model.
Max pooling is differentiable almost everywhere and uses subgradients at ties.
Node masks must exclude padding from messages, normalization, and pooling.
The hard-adjacency special case must reproduce the current sparse GIN.
[PyG DenseGINConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DenseGINConv.html)
provides a dense-adjacency interface, but masking and self-loop semantics still
need verification against this implementation.

Hard thresholding, `argmax`, and converting a learned adjacency to integer
`edge_index` do not transmit ordinary pathwise gradients. Choose explicitly:

| Representation | Gradient interpretation |
|---|---|
| Soft adjacency and feature probabilities | Exact autograd derivative of the declared continuous surrogate |
| Logistic/Gumbel relaxations with fixed noise | Pathwise derivative of a relaxed stochastic objective |
| Hard forward with straight-through backward | Biased surrogate gradient; not the derivative of the hard forward computation |
| Truly discrete sampling with score-function estimators | Possible under regularity conditions, usually high variance; not automatic backprop through discrete choices |

A straight-through value has the form
\(Y_{ST}=\operatorname{sg}(Y_{hard}-Y_{soft})+Y_{soft}\).
[Gumbel-Softmax](https://arxiv.org/abs/1611.01144) supplies a categorical relaxation;
it does not make exact discrete samples ordinarily differentiable. Soft-graph
classification is generally not equal to expected classification over hard
samples. Measure that discrepancy rather than assuming equivalence.

## 5. Inner mixture fitting

For transformed fit graph \(\widetilde G_i\), define the component likelihood-ratio
surrogate

\[
\ell_{ik}(\phi)=\log K+
\log\operatorname{softmax}\!\left(c_{\omega(\phi)}(\widetilde G_i)/T_c(\phi)\right)_k,
\]

\[
J(w;\phi)=\frac1{|\mathcal D_{\rm fit}|}\sum_i
\operatorname{logsumexp}_k(\log w_k+\ell_{ik}(\phi)).
\]

The inner objectives retain the current direction and concentration cap:

\[
w_N^*(\phi)\in\arg\max_{w\in\Delta_K}J(w;\phi),\qquad
w_A^*(\phi)\in\arg\min_{w\in\Delta_K,\ w_k\le c}J(w;\phi),
\quad c=0.25.
\]

No shell constraint or ellipsoid is used. These are estimated relative-likelihood
objectives. The abnormal problem is generally nonconvex; its optimum need not be
unique or smoothly dependent on \(\phi\).

For implementable unrolling, replace convergence-based loops by a fixed number
of differentiable updates. Normal EM can use

\[
\gamma_{ik}^{(s)}=
\operatorname{softmax}_k(\log w_k^{(s)}+\ell_{ik}),\qquad
w_k^{(s+1)}=\frac1N\sum_i\gamma_{ik}^{(s)}.
\tag{3}
\]

Initialize uniformly and use stable log-domain arithmetic. An explicit floor or
Dirichlet pseudocount is a modified objective and must be recorded if introduced.
Finite-step derivatives describe this solver, not an assumed converged optimum.

For abnormal fitting, fixed-step projected descent is

\[
w_A^{(s+1)}=\Pi_{\Delta_K\cap[0,c]^K}
\left(w_A^{(s)}-\rho_s\nabla_w J(w_A^{(s)};\phi)\right).
\tag{4}
\]

The capped-simplex projection satisfies \(w_i=\operatorname{clip}(v_i-\lambda,0,c)\).
On a region with free coordinates \(F=\{i:0<w_i<c\}\), its Jacobian on the free
block is \(I-\mathbf1\mathbf1^\top/|F|\); other rows and columns are zero.
At active-set transitions the projection is not classically differentiable.
At a locally constant capped vertex, a zero weight derivative can be correct.
Use a verified custom backward or differentiable constrained solver; differentiating
through the current NumPy bisection is not possible.

A unique winning restart is piecewise differentiable, but hard winner changes
are nonsmooth. A declared smooth alternative combines restart outputs as
\(\bar w_A=\sum_r\alpha_rw_{A,r}\), with
\(\alpha_r=\operatorname{softmax}_r(-J(w_{A,r};\phi)/\tau_r)\).
It preserves feasibility by convexity of the constraint set, but changes the
selected abnormal solution. It neither guarantees a global minimum nor removes
all active-set difficulties.

## 6. GIN training without a discrete mixture-index gradient

Draw a small minibatch from **each** prior component:
\(S^{tr}_{kj}(\phi)=g_k(\epsilon^{tr}_{kj};\eta(\phi))\),
\(j=1,\ldots,m_k\). Instead of sampling a discrete component label with `choice`,
use the weighted expectation

\[
\begin{aligned}
\mathcal L_{cls}(\theta;\phi)
={}&\frac12\sum_k\frac{w_{N,k}(\phi)}{m_k}
              \sum_j\operatorname{softplus}(f_\theta(S^{tr}_{kj}(\phi)))\\
 &+\frac12\sum_k\frac{w_{A,k}(\phi)}{m_k}
              \sum_j\operatorname{softplus}(-f_\theta(S^{tr}_{kj}(\phi)))
 +\frac{\lambda_\theta}{2}\|\theta\|_2^2.
\end{aligned}
\tag{5}
\]

This estimates the same balanced mixture-classification objective as mixture
sampling, with different Monte Carlo variance. Sharing a component draw between
its normal and abnormal terms is valid: mixture components may contribute to
both populations. Component enumeration removes the discrete *mixture-index*
obstacle, not the discrete graph-generation obstacle.

Unroll classifier updates from an initialization independent of the current AE:

\[
\theta_{t+1}(\phi)=\theta_t(\phi)-\alpha_t
                       \nabla_\theta\mathcal L_{cls}^{(t)}(\theta_t(\phi);\phi),
\qquad t=0,\ldots,T_f-1.
\tag{6}
\]

This explicitly defines \(\theta(\phi)=\theta_{T_f}(\phi)\).
For \(Z_t=d\theta_t/d\phi\),

\[
Z_{t+1}=Z_t-\alpha_t\left(
\nabla^2_{\theta\theta}\mathcal L_{cls}^{(t)}Z_t+
\nabla^2_{\theta\phi}\mathcal L_{cls}^{(t)}\right),\qquad Z_0=0.
\tag{7}
\]

Reverse-mode autodiff obtains the needed products without materializing \(Z_t\).
The mixed derivative in (7) contains the mixture-weight and generator dependencies.
If those are detached and the classifier still trains on synthetic graphs only,
\(\mathcal L_{cls}\) may have **no remaining dependence on the AE**. Merely unrolling
GIN training would then fail to implement the intended feedback.

Functional Adam is also possible, but its moments and parameter updates must
remain in the unrolled graph. Ordinary in-place optimizer steps do not provide
this higher-order derivative. Differentiating learning dynamics is the relevant
approach, as studied by
[Franceschi et al.](https://proceedings.mlr.press/v70/franceschi17a.html).

## 7. A distribution loss that also differentiates mixture weights

Use separate meta pilots \(S^{meta}_{kj}(\phi)\), not the classifier-training
samples. For \(G_i\in\mathcal D_{\rm meta}\), define

\[
u_i=f_{\theta(\phi)}(G_i),\quad
z_i=f_{\theta(\phi)}(T_\phi(G_i)),\quad
v_{kj}=f_{\theta(\phi)}(S^{meta}_{kj}(\phi)).
\]

Let

\[
P_0^\phi=\frac1B\sum_i\delta_{u_i},\quad
P_T^\phi=\frac1B\sum_i\delta_{z_i},\quad
P_N^\phi=\sum_k\frac{w_{N,k}(\phi)}{m_k^{meta}}\sum_j\delta_{v_{kj}}.
\]

A convenient differentiable discrepancy is weighted squared kernel MMD. For
weighted empirical measures \(P=\sum_i a_i\delta_{x_i}\),
\(Q=\sum_j b_j\delta_{y_j}\), with each weight vector summing to one,

\[
D(P,Q)=\sum_{i,i'}a_i a_{i'}k(x_i,x_{i'})
       +\sum_{j,j'}b_j b_{j'}k(y_j,y_{j'})
       -2\sum_{i,j}a_i b_j k(x_i,y_j).
\tag{8}
\]

Use a mixture of Gaussian kernels with fixed bandwidths. Fix one shared logit
scale from a baseline teacher on fit normals; do not normalize each population
separately or optimize the bandwidth to make discrepancies disappear. Equation
(8) is the empirical RKHS distance including diagonals; it is not the unbiased
population MMD estimator. Gradients flow through the logits **and** \(a,b\).

For a real/transformed versus mixture comparison, the synthetic–synthetic term
contains \(w_{N,k}w_{N,l}\), and the cross term contains \(w_{N,k}\). Detaching these
weights would omit part of the objective's derivative even if all logits remained
attached. Kernel matching is supported by the two-sample-testing framework of
[Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html).

The proposed full alignment objective is

\[
\mathcal L_{align}(\phi)=
D(P_T^\phi,P_N^\phi)
+\lambda_{raw}D(P_T^\phi,P_0^\phi)
+\lambda_{gap}D(P_N^\phi,P_0^\phi).
\tag{9}
\]

The last term directly addresses the original blue–purple gap. It has an AE
gradient through the trained GIN, weights, and generators even though it contains
no explicit \(T_\phi(G)\) logit branch. This is a central reason to retain the
classifier-training derivative.

One-dimensional Wasserstein loss is an alternative. For equal-size, equal-weight
samples, \(W_2^2=B^{-1}\sum_i(x_{(i)}-y_{(i)})^2\); sorting is piecewise
differentiable. Nonuniform trainable mixture weights need weighted transport or
a differentiable approximation. Sorting samples from detached discrete component
choices does not recover the missing weight gradient. MMD avoids that issue.

## 8. Preventing a low-loss but useless solution

Logit matching alone is insufficient. The system could make the GIN nearly
constant, make the two synthetic mixtures indistinguishable, or change normal
graphs into generator artifacts. An AE trained only on normals could also
transform real anomalies into apparently normal graphs.

Use normalized reconstruction losses on node-aligned graphs:

\[
\mathcal L_{rec}=
\mathbb E_G\left[
\frac{\sum_{u<v}\operatorname{BCE}(\widetilde A_{uv},A_{uv})}
     {\max(1,n(n-1)/2)}
+\lambda_X\frac1n\sum_v\operatorname{CE}(X_v,\widetilde X_v)
\right].
\tag{10}
\]

Continuous features use scaled squared error. Edge/nonedge balancing can be
added explicitly. A decoder must be permutation-equivariant and retain node
correspondence; otherwise (10) is not a valid graph reconstruction comparison.

Also monitor or constrain expected edits:

\[
e_A(G)=\sum_{u<v}\{A_{uv}(1-\widetilde A_{uv})
                       +(1-A_{uv})\widetilde A_{uv}\},\qquad
e_X(G)=\sum_v(1-\widetilde X_{v,c_v}).
\tag{11}
\]

Penalize squared violations of prespecified per-graph edit budgets. Prefer small
edits because our earlier audit showed the prior bank itself was structurally
mismatched; encouraging the AE to imitate those defects would be a bad outcome.
For additional structural preservation, use normalized soft degree, triangle,
and label-edge statistics. A mean-degree constraint alone is inadequate.

Train a fixed reference classifier \(f_{ref}\) before outer adaptation and retain
its original-normal predictions as a stability anchor:

\[
\mathcal L_{anchor}=\frac1B\sum_i
\left[f_{\theta(\phi)}(G_i)-\operatorname{sg}(f_{ref}(G_i))\right]^2/s_0^2.
\tag{12}
\]

The teacher does not certify anomaly correctness. This anchor only discourages
arbitrary classifier drift and collapse on the normal reference. Use independent
synthetic meta samples to compute \(\mathcal L_{sep}\), the balanced BCE in (5)
after final-GIN training, without its parameter penalty. Monitor synthetic AUROC
and logit spread as additional collapse checks; easy synthetic separation is not
evidence of real anomaly performance.

One complete outer objective is

\[
\boxed{
\mathcal R(\phi)=\mathcal L_{align}
 +\lambda_{rec}\mathcal L_{rec}
 +\lambda_{edit}\mathcal L_{edit}
 +\lambda_{anchor}\mathcal L_{anchor}
 +\lambda_{sep}\mathcal L_{sep}.
}
\tag{13}
\]

The outer optimization is

\[
\phi^*\in\arg\min_\phi\mathbb E_{\Xi}[\mathcal R(\phi;\Xi)],
\qquad
\phi_{r+1}=\phi_r-\beta_r\frac{d\mathcal R(\phi_r;\Xi_r)}{d\phi}.
\]

Writing all fitted inner quantities as
\(y(\phi)=(\eta,\omega,T_c,w_N,w_A,\theta)\), the total gradient is

\[
\frac{d\mathcal R}{d\phi}
=\left.\frac{\partial\mathcal R}{\partial\phi}\right|_y
+\left(\frac{dy}{d\phi}\right)^\top\nabla_y\mathcal R.
\]

This notation includes the explicit weight derivatives in MMD and the derivative
of the final classifier's trained parameters. The stochastic gradient conditions
on the sampled randomness; interpreting it as a gradient of the expectation
requires the usual interchange-of-expectation-and-differentiation conditions.

Reconstruction may use fit and meta normals; selection normals never receive
gradients. All coefficients and edit budgets must be prespecified or selected
using the inner normal-only selection split. These safeguards reduce specific
failure modes; they do not prove that anomalies survive the AE or that the
normal prior has adequate support. If alignment needs large edits, retain the
mismatch as a failure rather than forcing the graphs to conform.

## 9. Practical implementation levels

### Level A: fixed GIN, direct AE gradient only

Fit the existing pipeline, freeze all its parameters and pilot targets, and
optimize the AE through a differentiable GIN forward. This is an inexpensive
control, **not the user's full optimization**. Equations (2) and the training
term of (1) are omitted. The last term in (9) is constant within such an update.

If two frozen target distributions differ, a third distribution cannot match
both exactly. For weighted squared one-dimensional Wasserstein losses, the
unconstrained best third distribution has quantile function

\[
Q_*^{-1}(p)=\frac{\lambda_0P_0^{-1}(p)+\lambda_NP_N^{-1}(p)}
                  {\lambda_0+\lambda_N},
\]

and minimum loss
\(\lambda_0\lambda_N W_2^2(P_0,P_N)/(\lambda_0+\lambda_N)\).
This bound applies to the frozen-target Wasserstein problem, not to MMD or the
full changing-classifier system. The full system can change both distributions
but introduces the collapse risks above.

### Level B: fixed bank, differentiable mixtures and GIN training

**Recommended first implementation.** Fit the prior bank, component classifier,
temperature, and input feature scaling once on fit normals and freeze them.
They remain differentiable with respect to their graph inputs. Recompute the
component ratios on \(T_\phi(G)\), unroll both mixture optimizers, and unroll
weighted final-GIN training using fixed-bank component samples.

Then

\[
\phi\to T_\phi(G)\to \ell_{ik}(\phi)
\to(w_N(\phi),w_A(\phi))\to\theta(\phi)\to\mathcal R
\]

is an active gradient path. The direct input path also remains active.
This fully includes the GIN-weight dependence for the **declared fixed-bank,
finite-update surrogate**. It avoids needing gradients through NumPy graph
sampling because sampled graphs are independent of \(\phi\) in this variant.

It does not include how rebuilding the prior bank or retraining the component
classifier would change with the AE. Therefore it is not the total derivative
of every stage of the original changing-data pipeline. A periodic bank refresh
can be added between optimization blocks, but it remains outside those blocks'
gradient and must be labeled as such.

### Level C: differentiable full pipeline with relaxed graph sampling

This is the closest realization of the complete requested dependency. Make
\(\eta(\phi)\), component-classifier training, temperature fitting, mixture fitting,
and final-GIN training differentiable, in addition to the AE input branch.
A concrete replacement contract is:

1. Preserve node counts and fixed feature schema. Compute mean degree,
   category frequencies, feature means and variances from soft reconstructed
   graphs using tensor arithmetic. Smooth zero frequencies and variance floors
   explicitly. These fitted settings are \(\eta(\phi)\).
2. Draw component identities, block memberships, correspondence permutations,
   and other random hyperparameters from distributions independent of \(\phi\).
   Condition on those draws for pathwise differentiation.
3. Construct block-edge probabilities using the differentiable mean degree and
   fixed affinities. For \(p\in(0,1)\), use relaxed Bernoulli edges
   \(a=\sigma((\operatorname{logit}p+\log u-\log(1-u))/\tau_A)\),
   \(u\sim U(0,1)\). Share draws across symmetric entries and zero the diagonal.
   Use \(1-(1-a_1)(1-a_2)\) for relaxed layer union.
4. Replace degree rounding and discrete preferential-attachment choices that
   depend on \(\phi\) with an explicit smooth mechanism. For example, for a new
   node, set edge probabilities proportional to existing soft degree plus a
   positive offset, with a smooth target expected attachment count, then draw
   relaxed Bernoulli edges. This changes the current fixed-count sampler and
   must be documented as a new generator variant.
5. Use weighted message aggregation in the random feature mechanism. Replace
   adjacency-dependent hard neighbor selection with weighted aggregation or
   smooth extrema. Produce categorical probabilities with a differentiable
   conditional head and relaxed categorical sampling. If marginal calibration
   is fitted, unroll its fixed number of updates too. Neither Gumbel sampling
   nor adding empirical log-frequency biases guarantees correct marginals.
6. Train the component classifier on uniformly weighted relaxed component
   batches using functional updates. Unroll temperature optimization on separate
   synthetic draws, parameterizing \(T_c=\exp(t_c)\); replace the SciPy scalar
   solve by a specified fixed-step tensor optimizer.
7. Differentiate feature standardization fitted on transformed normals and every
   downstream operation through (3)–(13).

Level C is exact autodiff for the chosen finite-step **relaxed** algorithm,
conditional on its draws, away from its nonsmooth boundaries. It is not an exact
gradient of the original discrete simulator. Report generator changes, relaxation
temperatures, and the discrepancy between soft and hardened graph behavior.
A score-function alternative could preserve discrete sampling, but would need
explicit probability models and variance reduction across a long learning
pipeline; it is not the recommended first implementation.

## 10. Outer optimization and implementation pseudocode

Start with reconstruction pretraining on fit normals. For each outer iteration,
construct the inner computations from a declared initialization, calculate the
meta loss after inner training, and update \(\phi\) with its total derivative.

```text
pretrain AE on fit normals using reconstruction and edit constraints
fit and freeze a reference teacher and a shared logit scale

for outer_step = 1 ... R:
    draw independent inner-training and meta-pilot randomness
    transformed_fit = AE_phi(fit_normals)             # do not detach

    if Level B:
        eta, component_model, temperature = fixed_reference_objects
        component_samples = fixed_bank_draws          # independent of phi
    if Level C:
        eta = fit_smooth_bank_settings(transformed_fit)
        component_samples = relaxed_sampler(eta, inner_randomness)
        component_model = unroll_component_training(component_samples)
        temperature = unroll_temperature_fit(independent_calibration_draws)

    ratios = component_model(transformed_fit, temperature)
    w_normal = unroll_normal_EM(ratios)
    w_abnormal = unroll_capped_abnormal_solver(ratios)

    theta = independent_GIN_initialization
    for classifier_step = 1 ... T_f:
        loss = weighted_component_BCE(theta, w_normal, w_abnormal, samples)
        gradient = autograd.grad(loss, theta, create_graph=True)
        theta = functional_update(theta, gradient)    # retain update graph

    raw_logits = GIN(theta, meta_normals)
    ae_logits = GIN(theta, AE_phi(meta_normals))
    synthetic_logits = GIN(theta, independent_meta_component_draws)
    outer_loss = weighted_MMD_terms + reconstruction + safeguards
    outer_gradient = autograd.grad(outer_loss, phi)    # includes theta(phi)
    update_only_AE_parameters(phi, outer_gradient)

    evaluate fresh diagnostics on selection normals without gradients
    retain only coherent AE + downstream-pipeline checkpoints
```

If initialization is warm-started from an earlier outer iteration and detached,
the computed derivative conditions on that initialization. If only the final
\(T_f\) steps of a longer training run are retained, this is a truncated
hypergradient, not the derivative of training from scratch or exact convergence.
Increasing the retained horizon is an ablation, not an assumed equivalence.

A small feasibility configuration can use 4 priors, 1–2 GIN layers, 2–4 graphs
per component per inner minibatch, and 2–5 unrolled steps per optimizer. Check
gradients there first. Then expand to 64 priors, the established width-32 GIN,
and longer horizons. Full reverse differentiation through the previous
100-epoch fits would be expensive; gradient checkpointing, truncated unrolling,
and later implicit differentiation are options. The small configuration is a
numerical feasibility test, not a benchmark replacement.

## 11. Optional implicit differentiation

For a smooth inner minimization with a locally isolated stationary point,
\(\nabla_\theta\mathcal L_{cls}(\theta^*,\phi)=0\), let
\(H=\nabla^2_{\theta\theta}\mathcal L_{cls}\) and
\(C=\nabla^2_{\theta\phi}\mathcal L_{cls}\). If \(H\) is invertible,

\[
\frac{d\theta^*}{d\phi}=-H^{-1}C,
\qquad
\nabla_\phi\mathcal R=
\partial_\phi\mathcal R-C^\top H^{-\top}\nabla_\theta\mathcal R.
\tag{14}
\]

Use linear solves/Hessian-vector products instead of forming the inverse.
Here \(C\) must already include the dependence of mixture weights and generators
on \(\phi\). For a nested constrained pipeline, differentiate all relevant
stationarity/KKT systems or compose their sensitivities; applying (14) to GIN
alone while treating upstream fits as constants misses those terms.

Neural Hessians can be singular or indefinite, mixtures may have multiple
solutions, and capped active sets can change. Damping, regularization, and fixed
active-set assumptions alter the calculation and need validation. Equation (14)
is not a general justification for applying implicit differentiation to the
current multistart abnormal optimizer unchanged. Finite unrolling is the more
transparent initial design.

## 12. Repository changes required

| Current location | Barrier | Proposed change |
|---|---|---|
| `prior_mixture_gin.py`: scoring methods | `@torch.no_grad()` and NumPy score output | Add a tensor-valued gradient-enabled forward for meta learning; retain inference methods for evaluation |
| `prior_mixture.py`: cleaning and ratio fitting | Detached CPU graphs; `cpu().numpy()` ratios | Separate differentiable training representation from detached serialization/evaluation paths |
| `ComponentClassifier` | Integer edges; no learned edge-weight input | Implement verified dense/weighted GIN for both component and final classifiers |
| `prior_mixture_impl/optimization.py` | NumPy/SciPy optimization and restart selection | Fixed-step tensor solvers with verified higher-order gradients |
| `prior_mixture_impl/sampling.py` | NumPy draws, categorical `searchsorted`, hard adjacency and neighbor selection | Keep fixed in Level B; replace with declared relaxed sampler in Level C |
| `fit_from_proposals` | Hard mixture choices; conventional optimizer updates | Component-weighted loss and functional inner training |
| Data-dependent preprocessing | Frozen buffers copied outside gradient graph | Explicit functional state; fixed in Level B, differentiable in Level C |

Freezing parameters with `requires_grad_(False)` does not by itself prevent an
input gradient. In contrast, `no_grad()` disables recording of operations, and
`eval()` only changes evaluation behavior. These distinctions are documented in
[PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd).
For the GIN-training hypergradient, its functional inner parameters must support
the derivatives needed to construct training updates; do not freeze that branch.

The new candidate should be separate, for example `ae_prior_mixture_gin`, and
continue to expose the framework's fit, anomaly-score, and checkpoint contract.
This document does not modify the current candidates or saved experiments.

## 13. Final inference and validation

After choosing \(\phi\), rerun the declared downstream pipeline at that fixed AE
state using the prespecified fitting protocol. Do not deploy a classifier trained
at an earlier AE state. Save the AE, bank settings, component classifier,
temperature, mixtures, final GIN, feature transforms, and representation policy
as one coherent checkpoint. If the final fit uses a longer inner horizon than
the outer optimization, report and measure that training/deployment mismatch.

The primary inference score is

\[
s(G)=f_{\theta_{final}}(T_{\phi_{final}}(G)).
\]

Use the same soft/relaxed/hard policy at calibration and testing. For a stochastic
policy, prespecify the number of draws and average logits identically in both
phases. Calibrate the final threshold on the untouched normal calibration set.
Reconstruction error may be evaluated as a separate anomaly score or a
prespecified combined-score ablation; do not silently add it to this algorithm.

Required checks before interpreting results:

1. **Gradient correctness.** On a tiny deterministic problem, compare the total
   AE gradient with central finite differences while rerunning every inner fit
   at \(\phi\pm\varepsilon v\) with identical randomness. Check (7) independently.
   Use double precision away from ties and active-set transitions. Verify that
   changing AE outputs changes \(\theta_{T_f}\), and that differentiating (9)'s
   raw–synthetic term produces the expected indirect gradient.
2. **Gradient ablations.** Compare Level A, Level B, and the additional Level C
   paths. Report the norm and direction of the direct and indirect terms;
   neither term is guaranteed to dominate. Stop-gradient targets are an explicit
   control, not the default full method.
3. **Representation checks.** Verify symmetry, zero diagonal, masks, categorical
   normalization, permutation equivariance, and parity with sparse GIN on hard
   graphs. Compare soft outputs with hardened or sampled counterparts.
4. **Fresh-distribution checks.** Plot raw, transformed, and synthetic normal
   logits on independent pilots, including held-out selection normals. Evaluate
   graph-statistic discrepancy and an independent real-versus-synthetic normal
   two-sample diagnostic. Matching one scalar logit distribution does not prove
   graph-distribution coverage.
5. **Degeneracy checks.** Monitor reconstruction, edit budgets, lost normal modes,
   synthetic class separation, and teacher drift. Reject constant-score or
   severely distorted solutions even when alignment loss is low.
6. **Anomaly preservation.** After all selection, evaluate real anomalies through
   the AE and compare with the no-AE baseline. Observe whether anomalies become
   more normal after transformation. Synthetic pilot success cannot establish
   that this failure mode is absent.

The main claim to test is whether the **total-gradient** feedback improves
normal prior compatibility and real anomaly detection under a small graph-edit
budget. A smaller blue–purple logit gap by itself is not sufficient evidence.
