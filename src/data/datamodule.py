"""TU loaders shared by training and evaluation."""
import torch
from torch_geometric.loader import DataLoader
from ..utils.config import Config
from .dataset import GraphFeatures, IndexedGraphs, load_tu_dataset, make_split


class TUDataModule:
    def __init__(self, config: Config, seed: int, split: dict[str, int | list[int]] | None = None) -> None:
        """Load a TU dataset and create or restore its deterministic split."""
        self.config = config
        self.seed = seed
        self.dataset = GraphFeatures(load_tu_dataset(config.data_root, config.name),
                                     config.get("feature_policy", "tu_or_degree"))
        self.split = split or make_split(
            self.dataset, seed, config.normal_train_fraction, config.test_protocol,
            config.get("normal_validation_fraction", 0.), config.get("normal_label", "majority")
        )
        self.train = IndexedGraphs(self.dataset, self.split["train_indices"])
        self.test = IndexedGraphs(self.dataset, self.split["test_indices"])
        self.validation = IndexedGraphs(self.dataset, self.split.get("validation_indices", []))
        # Structural metadata only; no graph features/labels are fitted on held-out data.
        # Use the graph-indicator file instead of materializing every dense graph.
        if config.max_nodes == "auto":
            from collections import Counter
            from pathlib import Path
            indicator = Path(config.data_root) / config.name / "raw" / f"{config.name}_graph_indicator.txt"
            if indicator.exists():
                with indicator.open() as handle:
                    self.max_nodes = max(Counter(line.strip() for line in handle if line.strip()).values())
            else:
                self.max_nodes = max(self.dataset[i].num_nodes for i in range(len(self.dataset)))
        else:
            self.max_nodes = int(config.max_nodes)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Return normal training graphs in reproducible batch order."""
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(self.train, batch_size=self.config.batch_size,
                          shuffle=shuffle, generator=generator)

    def test_dataloader(self) -> DataLoader:
        """Return held-out normal and anomalous graphs without shuffling."""
        return DataLoader(self.test, batch_size=self.config.batch_size, shuffle=False)

    def validation_dataloader(self) -> DataLoader:
        """Held-out normals used exclusively for threshold calibration."""
        return DataLoader(self.validation, batch_size=self.config.batch_size, shuffle=False)
