"""Independent Section-4-inspired sampler; not a GraphPFN code reproduction.

Every component uses two degree-corrected block layers, a random correspondence,
peripheral attachment, and a random mixed MLP/GNN attribute mechanism.
"""
import math
import numpy as np
import torch
from torch_geometric.data import Data


class PriorBank:
    def __init__(self, graphs, settings, seed):
        self.rng = np.random.default_rng(seed)
        self.sizes = np.array([g.num_nodes for g in graphs])
        if self.sizes.min() < 1 or self.sizes.max() > settings.max_nodes:
            raise ValueError('Training size exceeds the declared dense sampler limit; no graphs truncated')
        self.mean_degree = float(np.mean([g.edge_index.shape[1] / g.num_nodes for g in graphs]))
        x = torch.cat([g.x for g in graphs]).numpy().astype(np.float64)
        self.width = x.shape[1]
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-6)
        degree_features = self.width == 2 and all(
            torch.allclose(g.x[:, 0], torch.ones(g.num_nodes)) and
            torch.allclose(g.x[:, 1], torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().log1p(), atol=1e-5)
            for g in graphs)
        onehot = bool(np.all((x == 0) | (x == 1)) and np.all(x.sum(1) == 1))
        self.mode = settings.feature_mode
        if self.mode == 'auto':
            self.mode = 'degree' if degree_features else 'categorical' if onehot else 'continuous'
            if self.mode == 'continuous' and np.all((x == 0) | (x == 1)):
                raise ValueError('Ambiguous binary feature groups: declare a supported feature schema explicitly')
        if self.mode == 'degree' and not degree_features:
            raise ValueError('degree mode requires the repository degree feature schema')
        if self.mode == 'categorical' and not onehot:
            raise ValueError('categorical mode currently supports one complete one-hot group')
        self.category_cdf = np.cumsum(x.mean(0)) if self.mode == 'categorical' else None
        # Six binary axes. Degree-only data uses six structural axes, avoiding
        # duplicate components distinguished solely by unused attributes.
        indices = np.linspace(0, 63, settings.components).round().astype(int)
        self.specs = [dict(bits=[(int(i) >> b) & 1 for b in range(6)]) for i in indices]
        self.specification = dict(version=1, feature_mode=self.mode,
                                  mean_degree=self.mean_degree, regimes=self.specs,
                                  size_law='empirical_training', seed=int(seed),
                                  feature_mean=self.mean.tolist(), feature_scale=self.scale.tolist(),
                                  sizes=self.sizes.tolist(),
                                  category_cdf=None if self.category_cdf is None else self.category_cdf.tolist())

    def _block_layer(self, n, groups, density, contrast, spread, compartments=None):
        rng = self.rng
        groups = min(groups, n)
        probabilities = rng.dirichlet(np.ones(groups))
        labels = rng.choice(groups, n, p=probabilities)
        degree = rng.lognormal(-spread ** 2 / 2, spread, n)
        affinity = np.where(labels[:, None] == labels[None, :], contrast, 1.)
        if compartments is not None:
            affinity *= compartments[:, None] == compartments[None, :]
        np.fill_diagonal(affinity, 0)
        affinity *= degree[:, None] * degree[None, :]
        mean = affinity.sum() / max(1, n * (n - 1))
        probability = np.minimum(0.95, affinity * density / max(mean, 1e-12))
        edges = np.triu(rng.random((n, n)) < probability, 1)
        return edges | edges.T

    def sample(self, component):
        rng = self.rng
        bits = self.specs[component]['bits']
        n = int(rng.choice(self.sizes))
        peripheral = rng.uniform(*( (0.05, 0.2) if bits[3] == 0 else (0.25, 0.5)))
        core = max(1, n - int(n * peripheral))
        groups = int(rng.integers(*( (1, 4) if bits[0] == 0 else (4, 9))))
        contrast = rng.uniform(*( (1.1, 3.) if bits[1] == 0 else (4., 12.)))
        spread = rng.uniform(*( (0.05, 0.3) if bits[2] == 0 else (0.6, 1.2)))
        degree_scale = rng.uniform(0.6, 1.5)
        if self.mode == 'degree':
            degree_scale *= (0.65 if bits[4] == 0 else 1.5)
        density = min(0.8, self.mean_degree * degree_scale / max(1, core - 1))
        compartments = rng.integers(0, int(rng.integers(1, 4)), core)
        first = self._block_layer(core, groups, density / 2, contrast, spread, compartments)
        second = self._block_layer(core, groups, density / 2, contrast, spread)
        correspondence = rng.permutation(core)
        adjacency = np.zeros((n, n), dtype=bool)
        adjacency[:core, :core] = first | second[correspondence][:, correspondence]
        for node in range(core, n):
            maximum = min(node, max(1, int(math.ceil(self.mean_degree))))
            degree = int(rng.integers(0, maximum + 1))
            attractiveness = 0.25 if self.mode == 'degree' and bits[5] else 1.
            weights = adjacency[:node, :node].sum(1).astype(float) + attractiveness
            targets = rng.choice(node, degree, replace=False, p=weights / weights.sum())
            adjacency[node, targets] = True
            adjacency[targets, node] = True
        features = self._features(adjacency, bits)
        edges = torch.from_numpy(np.vstack(np.nonzero(adjacency)).astype(np.int64))
        return Data(x=torch.tensor(features, dtype=torch.float32), edge_index=edges, num_nodes=n)

    def _features(self, adjacency, bits):
        rng = self.rng
        n = len(adjacency)
        degree = adjacency.sum(1)
        if self.mode == 'degree':
            return np.stack([np.ones(n), np.log1p(degree)], 1)
        hidden = 16
        features = rng.normal(size=(n, hidden))
        if rng.random() < 0.5:
            features[:, 0] = np.log1p(degree)
        normalized = adjacency / np.maximum(degree[:, None], 1)
        if rng.random() < 0.5:
            rank = np.full(n, 1 / n)
            for _ in range(20):
                rank = 0.15 / n + 0.85 * (normalized.T @ rank + rank[degree == 0].sum() / n)
            features[:, 1] = n * rank
        mix = rng.uniform(*( (0., 0.3) if bits[4] == 0 else (0.7, 1.)))
        noise = rng.uniform(*( (0.05, 0.2) if bits[5] == 0 else (0.4, 0.9)))
        aggregation = int(rng.integers(0, 5))
        for _ in range(int(rng.integers(2, 5))):
            local = features @ (rng.normal(size=(hidden, hidden)) / math.sqrt(hidden))
            if aggregation == 0:
                messages = normalized @ features
            elif aggregation in (1, 2):
                messages = np.zeros_like(features)
                for node in range(n):
                    neighbors = features[adjacency[node]]
                    if len(neighbors):
                        messages[node] = neighbors.max(0) if aggregation == 1 else neighbors.min(0)
            elif aggregation == 3:
                denom = np.sqrt(np.maximum(degree, 1))
                messages = (adjacency / denom[:, None] / denom[None, :]) @ features
            else:
                attention = features @ features.T / math.sqrt(hidden)
                mask = adjacency | np.eye(n, dtype=bool)
                attention = np.where(mask, attention, -1e9)
                attention = np.exp(attention - attention.max(1, keepdims=True)) * mask
                messages = (attention / attention.sum(1, keepdims=True)) @ features
            graph_values = messages @ (rng.normal(size=(hidden, hidden)) / math.sqrt(hidden))
            selected = rng.random(hidden) < mix
            features = np.tanh(np.where(selected, graph_values, local) + rng.normal(0, noise, (n, hidden)))
        if self.mode == 'categorical':
            from scipy.special import ndtr
            # Fixed population-scale transform, not per-graph normalization.
            values = ndtr(features[:, int(rng.integers(hidden))] * 2 + rng.normal(0, noise, n))
            categories = np.searchsorted(self.category_cdf, values).clip(0, self.width - 1)
            return np.eye(self.width)[categories]
        output = features @ (rng.normal(size=(hidden, self.width)) / math.sqrt(hidden))
        return (output + rng.normal(0, noise, (n, self.width))) * self.scale + self.mean

    def labelled(self, per_component, progress=None):
        graphs, labels = [], []
        for component in range(len(self.specs)):
            graphs.extend(self.sample(component) for _ in range(per_component))
            labels.extend([component] * per_component)
            if progress and ((component + 1) % 8 == 0 or component + 1 == len(self.specs)):
                progress(component + 1, len(graphs))
        return graphs, torch.tensor(labels, dtype=torch.long)
