"""Experimental paper-based UniFORM adaptation for normal-only graph AD.

Source: https://doi.org/10.1609/aaai.v39i12.33369, Eqs.3-16.
No official code was verified. Community descriptors, sampling budgets,
augmentations and loss weights are explicit local choices; see docs/methods.md.
"""
import copy
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..base_model import BaseAnomalyModel


class SampleEnergyGNN(nn.Module):
    """Overlap-connected sample graph, cosine gating, layer-weighted energies."""
    def __init__(self, descriptor_dim, hidden_dim, layers):
        super().__init__()
        self.input = nn.Linear(descriptor_dim, hidden_dim)
        self.similarity = nn.Linear(descriptor_dim, hidden_dim, bias=False)
        self.updates = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.readouts = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers + 1))
        self.layer_weights = nn.Parameter(torch.full((layers + 1,), 1 / (layers + 1)))

    def forward(self, features, connections):
        similarity = F.normalize(self.similarity(features), dim=-1)
        gates = (similarity @ similarity.T + 1) / 2
        deg = connections.sum(1) + 1
        weights = connections * gates / torch.sqrt(deg[:, None] * deg[None, :])
        hidden, sample_layers, graph_layers = self.input(features), [], []
        for layer in range(len(self.readouts)):
            transformed = self.readouts[layer](hidden)
            sample_layers.append(torch.sigmoid(transformed))
            graph_layers.append(torch.sigmoid(transformed.mean(0, keepdim=True)))
            if layer < len(self.updates):
                hidden = F.relu(self.updates[layer](hidden + weights @ hidden))
        samples = sum(w * h for w, h in zip(self.layer_weights, sample_layers))
        graph = sum(w * h for w, h in zip(self.layer_weights, graph_layers))
        # Last token is the whole-graph readout; other tokens are node/edge/walk communities.
        embeddings = torch.cat([samples, graph], dim=0)
        return embeddings, -torch.logsumexp(embeddings, dim=-1)


class UniFORMAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        self.config, self.device = config, device
        a = config.architecture
        # Mean/std of node features + log mean-degree + sampled unique-node fraction.
        self.network = SampleEnergyGNN(2 * input_dim + 2, a.hidden_dim, a.num_layers).to(device)
        self.momentum = copy.deepcopy(self.network).requires_grad_(False)

    def _pool(self, graph, deterministic=False):
        a = self.config.architecture
        graph = graph.cpu()
        if not graph.num_nodes:
            raise ValueError("UniFORM requires nonempty graphs")
        generator = None
        if deterministic:
            digest = hashlib.sha256(graph.edge_index.contiguous().numpy().tobytes()
                                    + graph.x.contiguous().numpy().tobytes()).digest()
            generator = torch.Generator().manual_seed(
                (int.from_bytes(digest[:8], 'little') + a.sampling_seed) % (2**63 - 1))
        def choice(values):
            return values[int(torch.randint(len(values), (), generator=generator))]
        neighbors = [[] for _ in range(graph.num_nodes)]
        for u, v in graph.edge_index.T.tolist():
            neighbors[u].append(v)
        def community(seed_nodes, walk=False):
            visited = list(seed_nodes[:a.community_size])
            while len(visited) < a.community_size:
                anchor = visited[-1] if walk else choice(visited)
                if walk and float(torch.rand((), generator=generator)) < a.restart_probability:
                    anchor = seed_nodes[0]
                options = neighbors[anchor] or [anchor]
                if walk:
                    # Information-content proxy; fixed graph-local degree, no fitted statistics.
                    weights = torch.tensor([1 + math.log1p(len(neighbors[j])) for j in options])
                    visited.append(options[int(torch.multinomial(weights, 1, generator=generator))])
                else:
                    visited.append(choice(options))
            return visited
        samples = []
        order = torch.randperm(graph.num_nodes, generator=generator).tolist()
        for node in order[:a.node_samples]:
            samples.append(community([node]))
        undirected = [(u, v) for u, v in graph.edge_index.T.tolist() if u < v]
        if undirected:
            order = torch.randperm(len(undirected), generator=generator).tolist()
            for j in order[:a.edge_samples]:
                samples.append(community(list(undirected[j])))
        for _ in range(a.walk_samples):
            node = int(torch.randint(graph.num_nodes, (), generator=generator))
            samples.append(community([node], walk=True))
        if not samples:
            raise ValueError("UniFORM sampling budgets must produce a nonempty pool")
        x = graph.x.float()
        features = []
        for nodes in samples:
            values = x[nodes]
            mean_degree = sum(len(neighbors[j]) for j in nodes) / len(nodes)
            features.append(torch.cat([values.mean(0), values.std(0, unbiased=False),
                                       x.new_tensor([math.log1p(mean_degree), len(set(nodes))/len(nodes)])]))
        # Sample-sample connectivity is shared membership within this graph only.
        memberships = [set(nodes) for nodes in samples]
        connections = torch.tensor([[float(i != j and bool(left & right))
                                     for j, right in enumerate(memberships)]
                                    for i, left in enumerate(memberships)])
        return torch.stack(features).to(self.device), connections.to(self.device)

    def _view(self, features, connections):
        s = self.config.training
        features = features * (torch.rand_like(features) >= s.feature_mask_probability)
        upper = torch.triu((torch.rand_like(connections) >= s.context_edge_drop_probability).float(), 1)
        return features, connections * (upper + upper.T)

    def _phantoms(self, features, connections):
        s = self.config.training
        sample = features.detach().clone()
        for _ in range(s.langevin_steps):
            sample.requires_grad_(True)
            _, energy = self.network(sample, connections)
            gradient, = torch.autograd.grad(energy.sum(), sample)
            sample = (sample - s.langevin_step_size * gradient
                      + math.sqrt(2 * s.langevin_step_size) * torch.randn_like(sample)).detach()
        return sample

    def _inter_loss(self, query, check, support, phantom):
        s = self.config.training
        count = len(query)
        if count < 2:
            raise ValueError("UniFORM needs at least two community tokens")
        ways = torch.randperm(count, device=self.device)[:min(s.n_way, count)]
        similarity = F.normalize(check[ways], dim=1) @ F.normalize(support, dim=1).T
        similarity[torch.arange(len(ways), device=self.device), ways] = -torch.inf
        shots = min(s.k_shot, count - 1)
        selected = similarity.topk(shots, dim=1).indices
        positive = support[selected]
        all_support = positive.reshape(-1, positive.size(-1))
        anchors = query[ways]
        denominator = torch.logsumexp(anchors @ torch.cat([all_support, phantom.detach()]).T, dim=1)
        return (denominator[:, None] - (anchors[:, None, :] * positive).sum(-1)).mean()

    def fit(self, train_loader, config, device):
        s = config.training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=s.learning_rate,
                                     weight_decay=s.weight_decay)
        for epoch in range(1, s.epochs + 1):
            self.network.train()
            self.momentum.eval()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                # Per-graph backward bounds memory; averaging retains the minibatch objective.
                graphs = batch.to_data_list()
                total = 0.
                for graph in graphs:
                    features, connections = self._pool(graph)
                    query, normal_energy = self.network(*self._view(features, connections))
                    with torch.no_grad():
                        check, _ = self.momentum(*self._view(features, connections))
                        support, _ = self.momentum(*self._view(features, connections))
                    phantom_features = self._phantoms(features, connections)
                    phantom, phantom_energy = self.network(phantom_features, connections)
                    inter = self._inter_loss(query, check, support, phantom)
                    intra = normal_energy.mean() - phantom_energy.mean()
                    regularization = normal_energy.square().mean() + phantom_energy.square().mean()
                    loss = (s.inter_weight * inter + s.intra_weight * intra
                            + s.regularization_weight * regularization)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"UniFORM: non-finite loss at epoch {epoch}")
                    (loss / len(graphs)).backward()
                    total += float(loss.detach()) / len(graphs)
                nn.utils.clip_grad_norm_(self.network.parameters(), s.gradient_clip_norm)
                optimizer.step()
                with torch.no_grad():
                    for target, source in zip(self.momentum.parameters(), self.network.parameters()):
                        target.mul_(s.ema_decay).add_(source, alpha=1-s.ema_decay)
                losses.append(total)
            if epoch == 1 or epoch % 10 == 0 or epoch == s.epochs:
                print(f"epoch={epoch}/{s.epochs} loss={sum(losses)/len(losses):.6f}", flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        self.network.eval()
        scores = []
        for graph in batch.to_data_list():
            _, energy = self.network(*self._pool(graph, deterministic=True))
            scores.append(energy[-1])
        return torch.stack(scores)

    def state_dict(self):
        return {"network": self.network.state_dict(), "momentum": self.momentum.state_dict()}

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.momentum.load_state_dict(state["momentum"])
        self.network.eval()
        self.momentum.eval()
