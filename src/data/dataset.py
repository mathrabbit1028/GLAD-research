"""TU graph loading and a deterministic one-class split."""
from pathlib import Path
from collections.abc import Sequence
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


def load_tu_dataset(root: str | Path, name: str) -> TUDataset:
    """Load a locally processed TU dataset without downloading data."""
    root = Path(root).resolve()
    if not (root / name / "processed" / "data.pt").is_file():
        raise FileNotFoundError(f"TU dataset is missing: {root / name}")
    return TUDataset(str(root), name=name, use_node_attr=True)


def make_split(dataset, seed: int, train_fraction: float, protocol: str,
               validation_fraction: float = 0.0, normal_label="majority") -> dict:
    """Choose normal training graphs and held-out graphs with a fixed seed."""
    labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(f"Expected at least two graph classes, got {classes}")
    if not (0 < train_fraction < 1 and 0 <= validation_fraction < 1 - train_fraction):
        raise ValueError("Normal train/validation fractions must leave a positive test fraction")
    counts = {label: labels.count(label) for label in classes}
    # Majority class is normal; use a stable tie break.
    normal_label = (max(classes, key=lambda label: (counts[label], -label))
                    if normal_label == "majority" else int(normal_label))
    if normal_label not in classes:
        raise ValueError(f"Normal class {normal_label} absent from {classes}")
    normal = [i for i, label in enumerate(labels) if label == normal_label]
    anomalies = [i for i, label in enumerate(labels) if label != normal_label]
    rng = torch.Generator().manual_seed(seed)
    normal = [normal[i] for i in torch.randperm(len(normal), generator=rng).tolist()]
    anomalies = [anomalies[i] for i in torch.randperm(len(anomalies), generator=rng).tolist()]
    n_validation = max(1, int(validation_fraction * len(normal))) if validation_fraction else 0
    n_train = max(1, min(len(normal) - n_validation - 1, int(train_fraction * len(normal))))
    if len(normal) < n_train + n_validation + 1:
        raise ValueError("Not enough normal graphs for disjoint train/validation/test sets")
    train = normal[:n_train]
    validation = normal[n_train:n_train + n_validation]
    test_normal = normal[n_train + n_validation:]
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
    return {"normal_label": normal_label, "train_indices": train,
            "validation_indices": validation, "test_indices": test}


class GraphFeatures(torch.utils.data.Dataset):
    """Preserve TU features; give featureless graphs a fixed, label-free encoding."""

    def __init__(self, dataset, policy="tu_or_degree"):
        if policy not in {"tu_or_degree", "tu"}:
            raise ValueError(f"Unknown feature policy: {policy}")
        self.dataset, self.policy = dataset, policy
        self.num_features = dataset.num_features or (2 if policy == "tu_or_degree" else 0)
        if not self.num_features:
            raise ValueError("Featureless dataset requires feature_policy=tu_or_degree")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        graph = self.dataset[index].clone()
        if graph.x is None or graph.x.size(-1) == 0:
            deg = degree(graph.edge_index[0], graph.num_nodes, dtype=torch.float32)
            graph.x = torch.stack([torch.ones_like(deg), torch.log1p(deg)], dim=1)
        else:
            graph.x = graph.x.float()
        return graph


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
