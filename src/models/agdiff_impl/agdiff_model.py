"""AGDiff components following Sections 4.2–4.4 of the paper."""
from __future__ import annotations

import torch
import math
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import negative_sampling


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GINBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in range(layers):
            self.layers.append(GINConv(MLP(input_dim if layer == 0 else hidden_dim,
                                           hidden_dim, hidden_dim)))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None,
                batch: Tensor | None = None) -> Tensor:
        pooled = []
        for layer in self.layers:
            if edge_weight is None:
                x = F.relu(layer(x, edge_index))
            else:
                # Weighted GIN implements Eq. (6)'s continuous sigmoid adjacency.
                messages = x[edge_index[0]] * edge_weight.unsqueeze(-1)
                aggregated = torch.zeros_like(x).index_add(0, edge_index[1], messages)
                x = F.relu(layer.nn((1 + layer.eps) * x + aggregated))
            if batch is not None:
                pooled.append(global_add_pool(x, batch))
        return torch.cat(pooled, dim=-1) if batch is not None else x


class GraphVAE(nn.Module):
    """Node-wise variational encoder and feature/inner-product edge decoder."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 layers: int = 3, kl_coefficient: float = 1e-3) -> None:
        super().__init__()
        self.encoder = GINBackbone(input_dim, hidden_dim, layers)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLP(latent_dim, hidden_dim, input_dim)
        self.kl_coefficient = kl_coefficient

    def encode(self, data: Batch) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(data.x, data.edge_index)
        return self.mu(hidden), self.logvar(hidden).clamp(-10.0, 10.0)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, data: Batch) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(data)
        return self.sample(mu, logvar), mu, logvar

    def loss(self, data: Batch) -> Tensor:
        z, mu, logvar = self(data)
        # Squared Frobenius error per node (sum over attributes), Eq. (7).
        attr_loss = (self.decoder(z) - data.x).square().sum(-1).mean()
        edge_losses = []
        # Each graph has its own adjacency matrix; cross-graph pairs are excluded.
        for graph_id in range(data.num_graphs):
            nodes = (data.batch == graph_id).nonzero(as_tuple=True)[0]
            n = nodes.numel()
            local = torch.full((data.num_nodes,), -1, dtype=torch.long,
                               device=data.x.device)
            local[nodes] = torch.arange(n, device=data.x.device)
            edge_mask = (data.batch[data.edge_index[0]] == graph_id)
            edges = local[data.edge_index[:, edge_mask]]
            target = z.new_zeros((n, n))
            target[edges[0], edges[1]] = 1.0
            valid = ~torch.eye(n, device=z.device, dtype=torch.bool)
            if valid.any():
                logits = z[nodes] @ z[nodes].T
                positive = valid & (target > 0)
                negative = valid & (target == 0)
                terms = []
                if positive.any():
                    terms.append(F.softplus(-logits[positive]).mean())
                if negative.any():
                    terms.append(F.softplus(logits[negative]).mean())
                edge_losses.append(torch.stack(terms).sum())
        edge_loss = torch.stack(edge_losses).mean() if edge_losses else z.new_zeros(())
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return attr_loss + edge_loss + self.kl_coefficient * kl


class ConditionalDiffusion(nn.Module):
    def __init__(self, latent_dim: int, steps: int = 1000) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("diffusion_steps must be positive")
        self.steps = steps
        self.condition = MLP(latent_dim, latent_dim, latent_dim)
        self.time_dim = 32
        self.noise_predictor = MLP(latent_dim * 2 + self.time_dim,
                                   max(64, latent_dim * 2), latent_dim)
        beta = torch.linspace(1e-4, 2e-2, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        # DDPM posterior variance, with zero variance at the final step.
        previous = torch.cat([torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]])
        self.register_buffer("posterior_variance", beta * (1 - previous) / (1 - alpha_bar))
        self.register_buffer("posterior_coef1", beta * previous.sqrt() / (1 - alpha_bar))
        self.register_buffer("posterior_coef2", (1 - previous) * alpha.sqrt() / (1 - alpha_bar))

    def _condition(self, z0: Tensor) -> Tensor:
        return torch.sigmoid(self.condition(z0 + torch.randn_like(z0)))

    def _predict(self, z: Tensor, condition: Tensor, step: int | Tensor) -> Tensor:
        if isinstance(step, int):
            t = z.new_full((z.size(0), 1), step)
        else:
            t = step.to(z.dtype).unsqueeze(-1)
        frequencies = torch.exp(torch.arange(self.time_dim // 2, device=z.device,
                                             dtype=z.dtype) * (-math.log(10000) / (self.time_dim // 2 - 1)))
        angles = t * frequencies
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.noise_predictor(torch.cat([z, condition, embedding], dim=-1))

    def noise_loss(self, z0: Tensor, condition: Tensor | None = None) -> Tensor:
        steps = torch.randint(self.steps, (z0.size(0),), device=z0.device)
        noise = torch.randn_like(z0)
        alpha_bar = self.alpha_bar[steps].unsqueeze(-1)
        zt = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise
        if condition is None:
            condition = self._condition(z0)
        return F.mse_loss(self._predict(zt, condition, steps), noise)

    def sample(self, z0: Tensor, condition: Tensor | None = None,
               checkpoint_reverse: bool = True) -> Tensor:
        """Reverse denoising from a forward-corrupted normal node embedding."""
        if condition is None:
            condition = self._condition(z0)
        z = self.alpha_bar[-1].sqrt() * z0 + (1 - self.alpha_bar[-1]).sqrt() * torch.randn_like(z0)

        def reverse_chunk(current: Tensor, context: Tensor, high: int, low: int) -> Tensor:
            for step in range(high, low - 1, -1):
                prediction = self._predict(current, context, step)
                clean = ((current - (1 - self.alpha_bar[step]).sqrt() * prediction)
                         / self.alpha_bar[step].sqrt())
                mean = self.posterior_coef1[step] * clean + self.posterior_coef2[step] * current
                current = (mean + self.posterior_variance[step].sqrt() * torch.randn_like(current)
                           if step else mean)
            return current

        # Recompute each short segment during backward instead of retaining all
        # 1,000 reverse steps and their large node-by-latent activations.
        chunk_size = (16 if checkpoint_reverse and torch.is_grad_enabled()
                      and self.steps >= 100 else self.steps)
        for high in range(self.steps - 1, -1, -chunk_size):
            low = max(0, high - chunk_size + 1)
            if chunk_size < self.steps:
                z = checkpoint(lambda current, context, hi=high, lo=low:
                               reverse_chunk(current, context, hi, lo),
                               z, condition, use_reentrant=False)
            else:
                z = reverse_chunk(z, condition, high, low)
        return z


class GraphDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 3,
                 spectral_normalization: bool = False) -> None:
        super().__init__()
        self.backbone = GINBackbone(input_dim, hidden_dim, layers)
        self.head = MLP(hidden_dim * layers, hidden_dim, 1)
        if spectral_normalization:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.utils.parametrizations.spectral_norm(module)

    def forward(self, data: Batch, edge_weight: Tensor | None = None) -> Tensor:
        if edge_weight is None:
            edge_weight = getattr(data, "edge_weight", None)
        x = self.backbone(data.x, data.edge_index, edge_weight, data.batch)
        return self.head(x).squeeze(-1)


class AGDiff(nn.Module):
    """AGDiff model with latent diffusion and differentiable soft graph decoding."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16,
                 diffusion_steps: int = 1000, gin_layers: int = 3,
                 kl_coefficient: float = 1e-3,
                 detector_spectral_norm: bool = False) -> None:
        super().__init__()
        self.vae = GraphVAE(input_dim, hidden_dim, latent_dim, gin_layers, kl_coefficient)
        self.diffusion = ConditionalDiffusion(latent_dim, diffusion_steps)
        self.detector = GraphDetector(input_dim, hidden_dim, gin_layers,
                                      detector_spectral_norm)

    def pretrain_loss(self, data: Batch) -> Tensor:
        return self.vae.loss(data)

    def joint_loss(self, data: Batch, lambda_diff: float) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.vae.encode(data)
        z0 = self.vae.sample(mu, logvar)
        condition = self.diffusion._condition(z0)
        diffusion_loss = self.diffusion.noise_loss(z0, condition)
        pseudo = self.pseudo_graphs(data, z0, condition)
        normal_logits = self.detector(data)
        pseudo_logits = self.detector(pseudo)
        logits = torch.cat([normal_logits, pseudo_logits])
        labels = torch.cat([torch.ones_like(normal_logits), torch.zeros_like(pseudo_logits)])
        classification_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return classification_loss + lambda_diff * diffusion_loss, classification_loss, diffusion_loss

    def pseudo_graphs(self, data: Batch, z0: Tensor,
                      condition: Tensor | None = None) -> Batch:
        sampled_z = self.diffusion.sample(z0, condition)
        return self.decode_latents(data, sampled_z)

    def decode_latents(self, data: Batch, z: Tensor) -> Batch:
        """Decode node latents and their inner products into a soft graph."""
        x = self.vae.decoder(z)
        edges: list[Tensor] = []
        weights: list[Tensor] = []
        offset = 0
        for count in torch.bincount(data.batch).tolist():
            graph_z = z[offset:offset + count]
            if count > 512:
                # Approximate dense decoding with observed edges and sampled non-edges
                # to keep memory bounded on the largest TU graphs.
                local_edges = data.edge_index[:,
                    (data.edge_index[0] >= offset) & (data.edge_index[0] < offset + count)
                ] - offset
                positive = local_edges[:, local_edges[0] < local_edges[1]]
                negative = negative_sampling(
                    local_edges, num_nodes=count,
                    num_neg_samples=max(positive.size(1) * 2, 1),
                    method="sparse", force_undirected=True,
                )
                negative = negative[:, negative[0] < negative[1]]
                candidates = torch.cat([positive, negative], dim=1)
                if candidates.numel():
                    probability = torch.sigmoid((graph_z[candidates[0]] *
                                                 graph_z[candidates[1]]).sum(-1))
                    hard = torch.zeros_like(probability)
                    keep = min(positive.size(1), probability.numel())
                    if keep:
                        hard[probability.topk(keep).indices] = 1.0
                    weight = hard + probability - probability.detach()
                    forward = candidates + offset
                    edges.append(torch.cat([forward, forward.flip(0)], dim=1))
                    weights.append(torch.cat([weight, weight]))
                offset += count
                continue
            probabilities = torch.sigmoid(graph_z @ graph_z.T)
            adjacency = ~torch.eye(count, dtype=torch.bool, device=z.device)
            edge_index = adjacency.nonzero(as_tuple=False).T.contiguous()
            edges.append(edge_index + offset)
            weights.append(probabilities[adjacency])
            offset += count
        pseudo = Batch(x=x, edge_index=torch.cat(edges, dim=1),
                       edge_weight=torch.cat(weights), batch=data.batch, ptr=data.ptr)
        return pseudo

    @torch.no_grad()
    def anomaly_score(self, data: Batch) -> Tensor:
        return -self.detector(data)
