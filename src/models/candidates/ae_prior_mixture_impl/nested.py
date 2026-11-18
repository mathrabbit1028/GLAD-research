"""Functional finite training maps, including differentiable normalization."""
import math
import torch
from torch.func import functional_call
from torch.nn import functional as F
from .core import balanced_loss


def detached(values):
    return {k: v.detach().clone() for k, v in values.items()}


def train_map(network, initial, buffers, batches, cfg, weights=None, higher_order=True):
    params = {k: v.detach().clone().requires_grad_(True) for k, v in initial.items()}
    velocity = {k: torch.zeros_like(v) for k, v in params.items()}
    for graphs in batches:
        logits = functional_call(network, (params, buffers), (graphs,))
        if weights is None:
            labels = torch.arange(logits.shape[-1], device=logits.device).repeat_interleave(
                len(logits) // logits.shape[-1])
            loss = F.cross_entropy(logits, labels)
        else:
            loss = balanced_loss(logits.reshape(len(weights[0]), -1), *weights)
        loss = loss + .5 * cfg.weight_decay * sum(p.square().sum() for p in params.values())
        grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=higher_order)
        velocity = {k: cfg.momentum * velocity[k] + g for k, g in zip(params, grads)}
        params = {k: p - cfg.learning_rate * velocity[k] for k, p in params.items()}
        if not higher_order:
            params = {k: p.detach().requires_grad_(True) for k, p in params.items()}
            velocity = detached(velocity)
    return params


def fit_temperature(logits, cfg, initial=1., higher_order=True):
    """Bounded smooth temperature; finite SGD, with no detached classifier branch."""
    fraction = (initial - .25) / 9.75
    raw = logits.new_tensor(math.log(fraction / (1-fraction)), requires_grad=True)
    labels = torch.arange(logits.shape[-1], device=logits.device).repeat_interleave(
        len(logits) // logits.shape[-1])
    for _ in range(cfg.steps):
        temperature = .25 + 9.75 * raw.sigmoid()
        loss = F.cross_entropy(logits / temperature, labels)
        gradient, = torch.autograd.grad(loss, raw, create_graph=higher_order)
        raw = raw - cfg.learning_rate * gradient
        if not higher_order:
            raw = raw.detach().requires_grad_(True)
    return .25 + 9.75 * raw.sigmoid()
