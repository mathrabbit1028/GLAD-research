"""TU loaders shared by training and evaluation."""
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from ..utils.config import Config
from .dataset import IndexedGraphs, load_tu_dataset, make_split


class TUDataModule:
    def __init__(self, config: Config, seed: int, split: dict[str, int | list[int]] | None = None) -> None:
        """Load a TU dataset and create or restore its deterministic split."""
        self.config = config
        self.seed = seed
        self.dataset = load_tu_dataset(config.data_root, config.name)
        self.split = split or make_split(
            self.dataset, seed, config.normal_train_fraction, config.test_protocol
        )
        self.train = IndexedGraphs(self.dataset, self.split["train_indices"])
        self.test = IndexedGraphs(self.dataset, self.split["test_indices"])

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Return normal training graphs in reproducible batch order."""
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(self.train, batch_size=self.config.batch_size,
                          shuffle=shuffle, generator=generator)

    def test_dataloader(self) -> DataLoader:
        """Return held-out normal and anomalous graphs without shuffling."""
        return DataLoader(self.test, batch_size=self.config.batch_size, shuffle=False)
