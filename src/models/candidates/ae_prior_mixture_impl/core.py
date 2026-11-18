"""Continuous graph AE, weighted GIN, and differentiable finite inner solvers."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from ..prior_mixture import ComponentClassifier


def dense_graphs(graphs, device, max_nodes=64):
    batch = graphs if isinstance(graphs, Batch) else Batch.from_data_list(graphs)
    if int((batch.ptr[1:] - batch.ptr[:-1]).max()) > max_nodes:
        raise ValueError('Graph exceeds dense Level B node limit; no graph is truncated')
    batch = batch.to(device)
    x, mask = to_dense_batch(batch.x, batch.batch)
    a = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=x.shape[1])
    return x, a, mask


def take(graphs, indices):
    return tuple(t[indices] for t in graphs)


class DenseGIN(ComponentClassifier):
    """Same weights, self-message, normalization and pooling as sparse GIN."""
    def forward(self, graphs):
        x, a, mask = graphs
        x = ((x - self.feature_mean) / self.feature_scale) * mask.unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv.nn((1 + conv.eps) * x + torch.bmm(a, x))))
            x = x * mask.unsqueeze(-1)
        mean = x.sum(1) / mask.sum(1, keepdim=True)
        maximum = x.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
        embedding = torch.cat([mean, maximum], 1)
        return self.head(embedding).squeeze(-1)


class ResidualGraphAE(nn.Module):
    """Equivariant node-aligned encoder/decoder with an explicit identity skip.

    A residual AE, not a pure information bottleneck. Categorical features only.
    Soft probabilities are used throughout training and primary inference.
    """
    def __init__(self, width, hidden, identity_logit):
        super().__init__()
        self.identity_logit = identity_logit
        self.encode1 = nn.Linear(width, hidden)
        self.encode2 = nn.Linear(hidden, hidden)
        self.node = nn.Linear(hidden, width)
        self.edge = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.node.weight)
        nn.init.zeros_(self.node.bias)
        nn.init.zeros_(self.edge[-1].weight)
        nn.init.zeros_(self.edge[-1].bias)

    def forward(self, graphs, hard=False):
        x, a, mask = graphs
        h = torch.tanh(self.encode1(x + torch.bmm(a, x)))
        h = torch.tanh(self.encode2(h + torch.bmm(a, h))) * mask.unsqueeze(-1)
        left, right = h.unsqueeze(2), h.unsqueeze(1)
        edge_delta = self.edge(torch.cat([left + right, left * right], -1)).squeeze(-1)
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        valid = valid & ~torch.eye(a.shape[1], device=a.device, dtype=torch.bool).unsqueeze(0)
        ap = torch.sigmoid(self.identity_logit * (2 * a - 1) + edge_delta) * valid
        xp = F.softmax(self.identity_logit * x + self.node(h), -1) * mask.unsqueeze(-1)
        if hard:
            ap = (ap >= .5).to(ap.dtype) * valid
            xp = F.one_hot(xp.argmax(-1), x.shape[-1]).to(x.dtype) * mask.unsqueeze(-1)
        return xp, ap, mask


class CappedSimplex(torch.autograd.Function):
    """Euclidean projection with the analytic active-set Jacobian."""
    @staticmethod
    def forward(ctx, values, cap):
        # Float64 bisection keeps the simplex sum accurate even at a capped vertex.
        v = values.double()
        low, high = v.min(-1, keepdim=True).values - cap, v.max(-1, keepdim=True).values
        for _ in range(64):
            middle = (low + high) / 2
            above = (v - middle).clamp(0, cap).sum(-1, keepdim=True) > 1
            low, high = torch.where(above, middle, low), torch.where(above, high, middle)
        projected = (v - (low + high) / 2).clamp(0, cap).to(values.dtype)
        free = (projected > 1e-7) & (projected < cap - 1e-7)
        ctx.save_for_backward(free)
        return projected

    @staticmethod
    def backward(ctx, gradient):
        (free,) = ctx.saved_tensors
        count = free.sum(-1, keepdim=True).clamp_min(1)
        mean = (gradient * free).sum(-1, keepdim=True) / count
        return (gradient - mean) * free, None


def fit_mixtures(log_ratios, cfg):
    """Log-domain EM and multi-start projected descent with a soft restart mix.

    The abnormal softmin changes hard restart selection. It is feasible, and
    differentiable within active sets, but is not a global optimum certificate.
    """
    k = log_ratios.shape[1]
    logw = torch.full((k,), -math.log(k), dtype=log_ratios.dtype, device=log_ratios.device)
    for _ in range(cfg.normal_steps):
        responsibilities = F.log_softmax(log_ratios + logw, -1)
        logw = torch.logsumexp(responsibilities, 0) - math.log(len(log_ratios))
    normal = logw.softmax(0)
    # Fixed starts independent of AE; avoid a hard argmin/top-k initialization.
    index = torch.arange(k, device=normal.device, dtype=normal.dtype)
    starts = torch.stack([torch.ones_like(index) if r == 0 else
                          1 + .8 * torch.cos(index * (r + 1)) for r in range(cfg.restarts)])
    w = CappedSimplex.apply(starts / starts.sum(-1, keepdim=True), cfg.cap)
    # Row scaling cancels exactly from the likelihood gradient and weight fitting.
    ratios = (log_ratios - log_ratios.max(-1, keepdim=True).values).exp()
    for _ in range(cfg.abnormal_steps):
        denominators = ratios @ w.T
        gradient = (ratios.unsqueeze(1) / denominators.unsqueeze(-1).clamp_min(1e-30)).mean(0)
        w = CappedSimplex.apply(w - cfg.step_size * gradient, cfg.cap)
    objectives = (ratios @ w.T).clamp_min(1e-30).log().mean(0)
    combination = F.softmax(-objectives / cfg.restart_temperature, 0)
    abnormal = (combination.unsqueeze(-1) * w).sum(0)
    return normal, abnormal


def balanced_loss(logits, normal, abnormal):
    # logits: component-major [K, samples_per_component].
    return .5 * ((F.softplus(logits).mean(1) * normal).sum() +
                 (F.softplus(-logits).mean(1) * abnormal).sum())


def unroll(network, initial, batches, normal, abnormal, cfg, higher_order=True):
    """Momentum-SGD from an AE-independent initialization; all updates functional."""
    params = {name: value.detach().clone().requires_grad_(True) for name, value in initial.items()}
    velocities = {name: torch.zeros_like(value) for name, value in params.items()}
    for graphs in batches:
        logits = functional_call(network, params, (graphs,)).reshape(len(normal), -1)
        loss = balanced_loss(logits, normal, abnormal)
        loss = loss + .5 * cfg.weight_decay * sum(p.square().sum() for p in params.values())
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=higher_order)
        velocities = {name: cfg.momentum * velocities[name] + g for name, g in zip(params, gradients)}
        params = {name: p - cfg.learning_rate * velocities[name] for name, p in params.items()}
        if not higher_order:
            params = {name: p.detach().requires_grad_(True) for name, p in params.items()}
            velocities = {name: v.detach() for name, v in velocities.items()}
    return params


def mmd(x, y, scale, wx=None, wy=None):
    x, y = x.flatten() / scale, y.flatten() / scale
    wx = torch.ones_like(x) / len(x) if wx is None else wx
    wy = torch.ones_like(y) / len(y) if wy is None else wy
    def kernel(a, b):
        delta = (a[:, None] - b[None, :]).square()
        return sum(torch.exp(-delta / (2 * sigma ** 2)) for sigma in [.5, 1., 2.]) / 3
    return wx @ kernel(x, x) @ wx + wy @ kernel(y, y) @ wy - 2 * wx @ kernel(x, y) @ wy


def reconstruction(original, transformed, edge_budget, node_budget):
    x, a, mask = original
    xp, ap, _ = transformed
    valid = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu(1)
    pairs = valid.sum((1, 2)).clamp_min(1)
    n = mask.sum(1)
    rec_a = (F.binary_cross_entropy(ap.clamp(1e-7, 1 - 1e-7), a, reduction='none') * valid).sum((1, 2)) / pairs
    rec_x = -(x * xp.clamp_min(1e-7).log()).sum((1, 2)) / n
    edits_a = ((a * (1 - ap) + (1 - a) * ap) * valid).sum((1, 2))
    edits_x = ((1 - (x * xp).sum(-1)) * mask).sum(1)
    penalty = (F.relu(edits_a / edge_budget - 1).square() +
               F.relu(edits_x / node_budget - 1).square()).mean()
    return (rec_a + rec_x).mean(), penalty, edits_a, edits_x
