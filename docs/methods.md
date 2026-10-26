# Method implementations

The benchmark area implements the nine entries in page 1 of the
[source report](../reports/normal_only_graph_anomaly_detection_report.pdf).
The matrix is in [`../configs/suite/default.yaml`](../configs/suite/default.yaml);
shared data/calibration rules are in [protocol.md](protocol.md).
Candidates are separate in `configs/models/` and `src/models/candidates/`;
all methods share the terminal UI and experiment engine described in the project README.
The recent candidates on page 2 are outside the default matrix.

No method tests, training or evaluation were performed for the setup or
reorganization. Configured budgets and local implementations do not establish
reproduction of the published ranking. MUSE is independently written; UniFORM
is an experimental paper-based adaptation with unverified fidelity.

| Config | Implementation basis | Fixed training budget |
|---|---|---:|
| `lgkde` | Pinned official core/generator and local vectorized MMD adapter | 100 epochs |
| `signet` | Pinned UB-GOLD SIGNET core | 500 epochs |
| `muse` | Independent implementation of the paper and public author-code design | 200 reconstruction + 500 classifier epochs |
| `uniform` | Experimental graph-level adaptation of the AAAI paper | 400 epochs |
| `cvtgad` | Pinned UB-GOLD CVTGAD core with device buffer fix | 1000 epochs |
| `gladc` | Pinned UB-GOLD GLADC core | 100 epochs |
| `ocgin` | OCGTL authors' alternative GIN implementation via UB-GOLD | 500 epochs |
| `ocgtl` | Pinned UB-GOLD OCGTL core | 500 epochs |
| `glocalkd` | Existing local adapter of the official encoder/distillation design | 150 epochs |

Source references, implementation status and current config/adapter hashes are
in [`../reports/methods.json`](../reports/methods.json). Vendor licenses, commits
and file hashes live under `src/models/vendor/{lgkde,ubgold}/`. Benchmark adapters live in
`src/models/benchmarks/`. OCGTL-derived
code retains AGPL-3.0; the shared MIT notice does not override it.

The AGDiff candidate is documented separately in [agdiff.md](agdiff.md); it is
not one of the nine comparison baselines.

[GRAFT-AD](graft_ad.md) is an experimental candidate using an independently
implemented GraphPFN-inspired prior and controlled pseudo-anomalies. Its adapter
and configuration are registered; neural training/evaluation remain deferred,
with no performance claims.

## Method adaptations


- **LGKDE:** `mixed_anomaly_ratio=0` is enforced. Synthetic graphs come from
  training normals only. All training graphs are used as density references
  in chunks. The vendored generator and vectorized MMD changes are described
  in `src/models/vendor/lgkde/UPSTREAM.txt`.
- **SIGNET / CVTGAD:** queries are scored individually against eight fixed
  training-normal references. No other test graph is used as contrastive
  context. Training loss statistics are checkpointed. SIGNET uses its
  endpoint-averaged feature branch and adds self loops to edgeless graphs.
  See `src/models/vendor/ubgold/UPSTREAM.txt` for the retained adaptations.
- **GLADC / GLocalKD:** fixed dataset padding and their existing reconstruction
  or distillation objectives are retained. These models, CVTGAD structural
  encodings and LGKDE spectral perturbations can be costly on large graphs.
- **OCGIN:** the OCGTL authors' implementation is the disclosed alternative to
  the OCGIN authors' repository. The mean embedding of training normals sets a
  fixed center, followed by squared-distance minimization. The center is saved
  and restored; hardcoded CUDA placement is removed. The reused GIN retains
  its AGPL-3.0 notice.
- **MUSE:** GIN encodings feed feature and adjacency decoders. Error means and
  sample standard deviations form a four-dimensional representation; a second
  MLP autoencoder learns that representation on training normals. Scoring uses
  its residual norm divided by training standard deviations. Single-value and
  zero-variance cases have explicit numerical floors. Fixed epochs replace
  anomaly-labelled validation search. This is independently written code, not
  a vendored or parity-verified copy of the official repository.
- **UniFORM:** no official repository was verified. The local implementation
  combines sampled node/edge communities and random walks, overlap-connected
  sample messages, energy readouts, EMA check/support views, nearest-neighbor
  pseudo-tasks, and Langevin phantom features. Community descriptors use node
  feature means/stds, mean degree and the unique-node fraction. Sampling budgets,
  overlap connectivity, augmentations, EMA rate, pseudo-task sizes and loss
  weights are explicit local choices where the paper does not provide a
  complete executable specification. Scores use deterministic graph-local
  sampling. This experimental graph-level adaptation requires independent
  review before being presented as a faithful UniFORM reproduction.

## Primary references


- [LGKDE author repository](https://github.com/MathAdventurer/LGKDE)
- [SIGNET author repository](https://github.com/yixinliu233/SIGNET)
- [MUSE paper](https://arxiv.org/html/2410.20366v1) and
  [author repository](https://github.com/kswoo97/GLAD_MUSE)
- [UniFORM paper](https://ojs.aaai.org/index.php/AAAI/article/view/33369/35524)
- [CVTGAD author repository](https://github.com/jindongli-Ai/CVTGAD)
- [GLADC author repository](https://github.com/XuexiongLuoMQ/GLADC)
- [OCGIN author repository](https://github.com/LingxiaoShawn/GLOD-Issues) and
  [OCGTL authors' alternative implementation](https://github.com/boschresearch/GraphLevel-AnomalyDetection)
- [GLocalKD author repository](https://github.com/RongrongMa/GLocalKD)
- [Pinned shared implementation](https://github.com/UB-GOLD/UB-GOLD/tree/c940457bc3be46fbd26cdaea12386a5c0a82f697)
