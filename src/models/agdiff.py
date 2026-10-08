"""Training adapter for the paper-aligned AGDiff implementation."""
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ..utils.config import Config
from .base_model import BaseAnomalyModel
from .agdiff_impl.agdiff_model import AGDiff


class AGDiffAdapter(BaseAnomalyModel):
    def __init__(self, input_dim: int, config: Config, device: torch.device) -> None:
        self.network = AGDiff(
            input_dim=input_dim,
            hidden_dim=config.architecture.hidden_dim,
            latent_dim=config.architecture.latent_dim,
            diffusion_steps=config.training.diffusion_steps,
            gin_layers=config.architecture.gin_layers,
            kl_coefficient=config.training.kl_coefficient,
            detector_spectral_norm=config.architecture.get("detector_spectral_norm", False),
        ).to(device)

    def fit(self, train_loader: DataLoader, config: Config, device: torch.device) -> None:
        settings = config.training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=settings.learning_rate)
        for phase, epochs in (("pretrain", settings.pretrain_epochs),
                              ("joint", settings.joint_epochs)):
            for epoch in range(1, epochs + 1):
                self.network.train()
                total = 0.0
                batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if phase == "pretrain":
                        loss = self.network.pretrain_loss(batch)
                    else:
                        loss, _, _ = self.network.joint_loss(batch, settings.lambda_diff)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {phase} loss at epoch {epoch}")
                    loss.backward()
                    for component in (self.network.vae, self.network.diffusion,
                                      self.network.detector):
                        torch.nn.utils.clip_grad_norm_(
                            component.parameters(), settings.gradient_clip_norm)
                    optimizer.step()
                    total += float(loss.detach())
                    batches += 1
                if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                    print(f"{phase} epoch={epoch}/{epochs} loss={total / batches:.6f}",
                          flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch: Batch) -> Tensor:
        self.network.eval()
        return self.network.anomaly_score(batch)

    def state_dict(self) -> Mapping[str, Any]:
        return self.network.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.network.load_state_dict(state)
        self.network.eval()
