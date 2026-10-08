"""TU graph loading and a deterministic one-class split."""
from pathlib import Path
from collections.abc import Sequence
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset


def load_tu_dataset(root: str | Path, name: str) -> TUDataset:
    """Load a locally processed TU dataset without downloading data."""
    root = Path(root).resolve()
    if not (root / name / "processed" / "data.pt").is_file():
        raise FileNotFoundError(f"TU dataset is missing: {root / name}")
    return TUDataset(str(root), name=name, use_node_attr=True)


def make_split(dataset: TUDataset, seed: int, train_fraction: float, protocol: str) -> dict[str, int | list[int]]:
    """Choose normal training graphs and held-out graphs with a fixed seed."""
    labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
    classes = sorted(set(labels))
    if len(classes) != 2:
        raise ValueError(f"Expected binary graph labels, got {classes}")
    counts = {label: labels.count(label) for label in classes}
    # Majority class is normal; use a stable tie break.
    normal_label = max(classes, key=lambda label: (counts[label], -label))
    normal = [i for i, label in enumerate(labels) if label == normal_label]
    anomalies = [i for i, label in enumerate(labels) if label != normal_label]
    rng = torch.Generator().manual_seed(seed)
    normal = [normal[i] for i in torch.randperm(len(normal), generator=rng).tolist()]
    anomalies = [anomalies[i] for i in torch.randperm(len(anomalies), generator=rng).tolist()]
    n_train = max(1, min(len(normal) - 1, int(train_fraction * len(normal))))
    train, test_normal = normal[:n_train], normal[n_train:]
    if protocol == "remaining_normals_plus_equal_or_fewer_anomalies":
        test_anomalies = anomalies[:min(len(test_normal), len(anomalies))]
    elif protocol == "remaining_normals_plus_all_anomalies":
        test_anomalies = anomalies
    else:
        raise ValueError(f"Unknown test protocol: {protocol}")
    test = test_normal + test_anomalies
    test = [test[i] for i in torch.randperm(len(test), generator=rng).tolist()]
    if not test_normal or not test_anomalies:
        raise ValueError("Split needs both normal and anomaly test graphs")
    return {"normal_label": normal_label, "train_indices": train, "test_indices": test}


class IndexedGraphs(torch.utils.data.Dataset):
    def __init__(self, dataset: TUDataset, indices: Sequence[int]) -> None:
        """Expose only the selected graph indices from a TU dataset."""
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        """Return the number of selected graphs."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Data:
        """Return the selected graph at a local subset index."""
        return self.dataset[self.indices[index]]
