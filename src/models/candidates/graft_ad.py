"""GRAFT-AD adapter: normal-only prior fitting and surrogate-negative training."""
from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from ..base_model import BaseAnomalyModel
from .graft_ad_config import validate_config
from .graft_ad_impl.detector import GraphDetector
from .graft_ad_impl.features import FeatureSchema
from .graft_ad_impl.generation import generate_pool
from .graft_ad_impl.graph import GraphSample, canonical_edges, descriptors
from .graft_ad_impl.prior import NormalGraphPrior
from .graft_ad_impl.verifier import DescriptorVerifier


def _checkpoint(value):
    """Only tensors and built-in containers/scalars enter weights_only checkpoints."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _checkpoint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint(item) for item in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise TypeError(f"Unsupported GRAFT-AD checkpoint value: {type(value).__name__}")


class GRAFTADAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        validate_config(config)
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("GRAFT-AD input_dim must be a positive integer")
        self.input_dim, self.device = input_dim, torch.device(device)
        arch = config["architecture"]
        self.network = GraphDetector(input_dim, arch["hidden_dim"], arch["gin_layers"], arch["dropout"]).to(self.device)
        self.fitted = False
        self.artifacts = {}

    def _samples(self, batch):
        """Copy only graph inputs; never read graph labels or mutate the loader."""
        samples = []
        for graph in batch.to_data_list():
            n = graph.num_nodes
            if n is None or n < 1:
                raise ValueError("GRAFT-AD requires nonempty graphs")
            if (graph.x is None or graph.x.ndim != 2 or graph.x.shape != (n, self.input_dim)):
                raise ValueError(f"GRAFT-AD requires {self.input_dim} node features per node")
            if (graph.edge_index is None or graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2
                    or graph.edge_index.dtype != torch.long):
                raise ValueError("GRAFT-AD requires int64 edge_index with shape [2, edges]")
            x = graph.x.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
            if not np.isfinite(x).all():
                raise ValueError("GRAFT-AD input features must be finite")
            edges = canonical_edges(graph.edge_index.detach().cpu().numpy(), n)
            samples.append(GraphSample(x, edges))
        if not samples:
            raise ValueError("GRAFT-AD requires at least one graph in a batch")
        return samples

    def _batch(self, samples):
        graphs = []
        for graph in samples:
            edges = np.concatenate((graph.edges, graph.edges[::-1]), axis=1).copy()
            graphs.append(Data(x=torch.from_numpy(graph.x), edge_index=torch.from_numpy(edges), num_nodes=graph.n))
        return Batch.from_data_list(graphs).to(self.device)

    def fit(self, train_loader, config, device):
        validate_config(config)
        self.fitted, self.artifacts = False, {}
        self.device = torch.device(device)
        self.network.to(self.device)
        if getattr(train_loader, "drop_last", False):
            raise ValueError("GRAFT-AD needs all supplied training normals; disable drop_last")
        normals = []
        for batch in train_loader:
            normals.extend(self._samples(batch))
        if len(normals) < 2:
            raise ValueError("GRAFT-AD requires at least two normal training graphs")
        # The experiment's global Torch seed supplies the per-run generator seed.
        seed = (int(torch.initial_seed()) + config["prior"]["seed_offset"]) % 2**32
        rng = np.random.default_rng(seed)
        verifier_settings = config["verifier"]
        def describe(graph):
            return descriptors(graph, verifier_settings["clustering_samples"], verifier_settings["edge_samples"])
        normal_descriptors = np.stack([describe(graph) for graph in normals])
        verifier = DescriptorVerifier(verifier_settings).fit(normal_descriptors)
        schema = FeatureSchema(config["features"], config["prior"]).fit(normals, rng)
        with torch.no_grad():
            self.network.feature_mean.copy_(torch.from_numpy(schema.mean).to(self.device))
            self.network.feature_scale.copy_(torch.from_numpy(schema.scale).to(self.device))
        print(f"graft_ad seed={seed} training_graphs={len(normals)} degree_features={schema.degree_mode} "
              f"gate_near={verifier.near:.6f} gate_far={verifier.far:.6f}", flush=True)
        prior = NormalGraphPrior(config["prior"], schema).fit(normals, normal_descriptors, verifier, describe, rng)
        pool, generation = generate_pool(prior, verifier, normal_descriptors, describe, config["generation"], rng)
        self.artifacts = _checkpoint({
            "seed": seed, "schema": schema.state_dict(), "verifier": verifier.state_dict(),
            "prior_library": prior.library, "prior_match_report": prior.match_report,
            "generation": generation,
            "provenance": {"implementation": "independent local GraphPFN-inspired prior",
                           "reference": "https://arxiv.org/html/2509.21489v4",
                           "uses_predictor_weights": False},
            "settings": config,
        })
        settings = config["training"]
        optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[settings["optimizer"]]
        optimizer = optimizer_class(self.network.parameters(), lr=settings["learning_rate"],
                                    weight_decay=settings["weight_decay"])
        batch_size = getattr(train_loader, "batch_size", None) or 1
        families = list(pool)
        for epoch in range(1, settings["epochs"] + 1):
            self.network.train()
            total, count = 0., 0
            order = rng.permutation(len(normals))
            for start in range(0, len(normals), batch_size):
                real = [normals[int(i)] for i in order[start:start + batch_size]]
                items, family_ids = [], []
                for _ in range(len(real) * settings["negatives_per_normal"]):
                    index = int(rng.integers(len(families)))
                    family = families[index]
                    items.append(pool[family][int(rng.integers(len(pool[family])))])
                    family_ids.append(index)
                fake = [graph for graph, _ in items]
                weights = torch.tensor([record["weight"] for _, record in items], device=self.device)
                indices = torch.tensor(family_ids, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(self._batch(real + fake))
                positive = F.softplus(logits[:len(real)]).mean()
                negative_logits = logits[len(real):]
                negative_losses = F.softplus(-negative_logits)
                # Equal contribution per represented family, regardless of its pool size.
                negative = torch.stack([
                    (negative_losses[indices == family] * weights[indices == family]).sum()
                    / weights[indices == family].sum().clamp_min(1e-8)
                    for family in sorted(set(family_ids))
                ]).mean()
                loss = positive + settings["negative_weight"] * negative
                if settings["ranking_weight"]:
                    anchors = [normals[record["anchor"]] for _, record in items]
                    anchor_logits = self.network(self._batch(anchors))
                    rank = (weights * F.relu(settings["ranking_margin"] + anchor_logits - negative_logits)).sum()
                    rank = rank / weights.sum().clamp_min(1e-8)
                    loss = loss + settings["ranking_weight"] * rank
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"GRAFT-AD non-finite training loss at epoch {epoch}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), settings["gradient_clip_norm"])
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"GRAFT-AD non-finite gradients at epoch {epoch}")
                optimizer.step()
                total += float(loss.detach()) * len(real)
                count += len(real)
            if epoch == 1 or epoch % 10 == 0 or epoch == settings["epochs"]:
                print(f"graft_ad epoch={epoch}/{settings['epochs']} loss={total / count:.6f}", flush=True)
        self.network.eval()
        self.fitted = True

    @torch.no_grad()
    def anomaly_score(self, batch):
        if not self.fitted:
            raise RuntimeError("GRAFT-AD must be fitted or restored before scoring")
        self.network.eval()
        scores = self.network(self._batch(self._samples(batch))).reshape(-1)
        if scores.numel() != batch.num_graphs or not torch.isfinite(scores).all():
            raise FloatingPointError("GRAFT-AD must return one finite anomaly score per graph")
        return scores

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError("Cannot checkpoint an unfitted GRAFT-AD detector")
        return _checkpoint({"version": 1, "input_dim": self.input_dim,
                            "network": self.network.state_dict(), "artifacts": self.artifacts})

    def load_state_dict(self, state):
        self.fitted = False
        if state.get("version") != 1 or state.get("input_dim") != self.input_dim:
            raise ValueError("Incompatible GRAFT-AD checkpoint version or input_dim")
        self.network.load_state_dict(state["network"])
        self.artifacts = _checkpoint(state["artifacts"])
        self.network.to(self.device).eval()
        self.fitted = True
