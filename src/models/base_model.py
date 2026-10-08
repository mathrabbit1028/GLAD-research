"""Contract shared by graph-level anomaly detectors."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from ..utils.config import Config


class BaseAnomalyModel(ABC):
    @abstractmethod
    def fit(self, train_loader: DataLoader, config: Config, device: torch.device) -> None:
        """Fit using normal training graphs only."""

    @abstractmethod
    def anomaly_score(self, batch: Batch) -> Tensor:
        """Return a 1D tensor; larger values mean more anomalous."""

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return checkpointable parameters."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore checkpointed parameters."""
