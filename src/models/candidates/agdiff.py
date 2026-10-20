"""AGDiff candidate adapter for the shared normal-only experiment framework."""
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from ...utils.config import Config
from ..base_model import BaseAnomalyModel
from .agdiff_impl.agdiff_model import AGDiff
from .agdiff_config import _positive_integer, validate_config


class AGDiffAdapter(BaseAnomalyModel):
    def __init__(self, input_dim: int, config: Config, device: torch.device) -> None:
        _positive_integer(input_dim, "input_dim")
        validate_config(config)
        self.input_dim, self.device = input_dim, torch.device(device)
        arch, settings = config.architecture, config.training
        self.network = AGDiff(
            input_dim=input_dim,
            hidden_dim=arch.hidden_dim,
            latent_dim=arch.latent_dim,
            diffusion_steps=settings.diffusion_steps,
            gin_layers=arch.gin_layers,
            kl_coefficient=settings.kl_coefficient,
            detector_spectral_norm=arch.get("detector_spectral_norm", False),
            reconstruction_block_size=arch.get("reconstruction_block_size", 512),
            dense_decode_max_nodes=arch.get("dense_decode_max_nodes", 512),
            reverse_checkpointing=settings.get("reverse_checkpointing", True),
            reverse_chunk_size=settings.get("reverse_chunk_size", 16),
        ).to(self.device)

    def _prepare(self, batch: Batch) -> Batch:
        """Use graph inputs only; do not mutate the loader batch or retain labels."""
        graphs = []
        for graph in batch.to_data_list():
            if graph.num_nodes < 1:
                raise ValueError("AGDiff requires nonempty graphs")
            if (graph.x is None or graph.x.ndim != 2
                    or graph.x.size(0) != graph.num_nodes or graph.x.size(1) != self.input_dim):
                raise ValueError(f"AGDiff expects {self.input_dim} node features from the shared feature pipeline")
            if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
                raise ValueError("AGDiff expects edge_index with shape [2, number_of_edges]")
            x = graph.x.detach().to(device=self.device, dtype=torch.float32).clone()
            if not torch.isfinite(x).all():
                raise ValueError("AGDiff node features must be finite")
            graphs.append(Data(x=x, edge_index=graph.edge_index.detach().to(self.device).clone(),
                               num_nodes=graph.num_nodes))
        if not graphs:
            raise ValueError("AGDiff requires at least one graph in a batch")
        return Batch.from_data_list(graphs)

    def fit(self, train_loader: DataLoader, config: Config, device: torch.device) -> None:
        validate_config(config)
        if not len(train_loader.dataset):
            raise ValueError("AGDiff requires normal training graphs")
        self.device = torch.device(device)
        self.network.to(self.device)
        settings = config.training
        optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[settings.get("optimizer", "Adam")]
        for phase, epochs in (("pretrain", settings.pretrain_epochs), ("joint", settings.joint_epochs)):
            self.network.zero_grad(set_to_none=True)
            self.network.requires_grad_(False)
            components = ((self.network.vae,) if phase == "pretrain" else
                          (self.network.diffusion, self.network.detector))
            for component in components:
                component.requires_grad_(True)
            parameters = [p for component in components for p in component.parameters()]
            optimizer = optimizer_class(parameters, lr=settings.learning_rate,
                                        weight_decay=settings.get("weight_decay", 0.))
            for epoch in range(1, epochs + 1):
                self.network.train()
                if phase == "joint":
                    # Frozen parameters still permit gradients through the decoder
                    # to the diffusion-generated latents in the classification loss.
                    self.network.vae.eval()
                else:
                    self.network.diffusion.eval()
                    self.network.detector.eval()
                total, classification, diffusion, graphs_seen = 0., 0., 0., 0
                for batch in train_loader:
                    data = self._prepare(batch)
                    optimizer.zero_grad(set_to_none=True)
                    if phase == "pretrain":
                        loss = self.network.pretrain_loss(data)
                    else:
                        loss, loss_cls, loss_diff = self.network.joint_loss(data, settings.lambda_diff)
                        classification += float(loss_cls.detach()) * data.num_graphs
                        diffusion += float(loss_diff.detach()) * data.num_graphs
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite AGDiff {phase} loss at epoch {epoch}")
                    loss.backward()
                    for component in components:
                        norm = torch.nn.utils.clip_grad_norm_(component.parameters(), settings.gradient_clip_norm)
                        if not torch.isfinite(norm):
                            raise FloatingPointError(f"Non-finite AGDiff {phase} gradients at epoch {epoch}")
                    optimizer.step()
                    total += float(loss.detach()) * data.num_graphs
                    graphs_seen += data.num_graphs
                if not graphs_seen:
                    raise ValueError("AGDiff training loader yielded no batches; disable drop_last or reduce the batch size")
                if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                    suffix = (f" classification={classification/graphs_seen:.6f} diffusion={diffusion/graphs_seen:.6f}"
                              if phase == "joint" else "")
                    print(f"{phase} epoch={epoch}/{epochs} loss={total/graphs_seen:.6f}{suffix}", flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch: Batch) -> Tensor:
        self.network.eval()
        data = self._prepare(batch)
        scores = self.network.anomaly_score(data).reshape(-1)
        if scores.numel() != data.num_graphs or not torch.isfinite(scores).all():
            raise FloatingPointError("AGDiff must return one finite anomaly score per graph")
        return scores

    def state_dict(self) -> Mapping[str, Any]:
        # Preserve plain network keys for existing framework checkpoints.
        return self.network.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.network.load_state_dict(state)
        self.network.zero_grad(set_to_none=True)
        self.network.vae.requires_grad_(False)
        self.network.eval()
