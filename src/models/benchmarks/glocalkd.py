"""Official GLocalKD encoder structure adapted to padded PyG graph batches."""
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from ...utils.config import Config
from ..base_model import BaseAnomalyModel


class GraphConv(nn.Module):
    """GLocalKD GraphConv: normalized (A + I)XW with optional input dropout."""

    def __init__(self, input_dim: int, output_dim: int, add_self: bool = True,
                 normalize_embedding: bool = True, dropout: float = 0.0,
                 bias: bool = True, device: torch.device | None = None) -> None:
        super().__init__()
        self.add_self = add_self
        self.normalize_embedding = normalize_embedding
        self.dropout = nn.Dropout(dropout) if dropout > 0.001 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim, device=device))
        self.bias = nn.Parameter(torch.empty(output_dim, device=device)) if bias else None
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        x = self.dropout(x)
        aggregated = torch.matmul(adjacency, x)
        if self.add_self:
            aggregated = aggregated + x
        output = torch.matmul(aggregated, self.weight)
        if self.bias is not None:
            output = output + self.bias
        if self.normalize_embedding:
            output = F.normalize(output, p=2, dim=2)
        return output


class DistillationEncoder(nn.Module):
    """Official layered GCN; graph representations use final-layer max pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 layers: int, max_nodes: int, dropout: float,
                 device: torch.device | None = None) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("GLocalKD requires at least two graph convolution layers")
        self.max_nodes = max_nodes
        # Official build_conv_layers applies dropout only to intermediate layers.
        self.conv_first = GraphConv(input_dim, hidden_dim, device=device)
        self.conv_block = nn.ModuleList(
            GraphConv(hidden_dim, hidden_dim, dropout=dropout, device=device)
            for _ in range(layers - 2)
        )
        self.conv_last = GraphConv(hidden_dim, embedding_dim, device=device)
        self.activation = nn.ReLU()

    def _batch_norm(self, x: Tensor) -> Tensor:
        # The reference creates BatchNorm1d(max_nodes) inside each forward pass.
        return nn.BatchNorm1d(x.size(1), device=x.device)(x)

    def _padded_graph(self, batch: Batch) -> tuple[Tensor, Tensor]:
        x, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=self.max_nodes)
        adjacency = to_dense_adj(
            batch.edge_index, batch=batch.batch, max_num_nodes=self.max_nodes,
            batch_size=batch.num_graphs,
        ).clamp_max_(1.0)
        return x, adjacency

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        x, adjacency = self._padded_graph(batch)
        return self.forward_dense(x, adjacency)

    def forward_dense(self, x: Tensor, adjacency: Tensor) -> tuple[Tensor, Tensor]:
        """Run the reference encoder on already padded dense tensors."""
        x = self.conv_first(x, adjacency)
        x = self.activation(x)
        x = self._batch_norm(x)
        for layer in self.conv_block:
            x = layer(x, adjacency)
            x = self.activation(x)
            x = self._batch_norm(x)
        node_embedding = self.conv_last(x, adjacency)
        graph_embedding = node_embedding.max(dim=1).values
        return node_embedding, graph_embedding


class GLocalKD(nn.Module):
    """Frozen teacher and trainable student with node/global feature matching."""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 embedding_dim: int = 256, layers: int = 3,
                 max_nodes: int = 1, student_dropout: float = 0.1,
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.teacher = DistillationEncoder(input_dim, hidden_dim, embedding_dim,
                                           layers, max_nodes, dropout=0.0, device=device)
        self.student = DistillationEncoder(input_dim, hidden_dim, embedding_dim,
                                           layers, max_nodes, dropout=student_dropout,
                                           device=device)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True) -> "GLocalKD":
        super().train(mode)
        self.teacher.eval()
        return self

    def distillation_loss(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        student_nodes, student_graphs = self.student(batch)
        with torch.no_grad():
            teacher_nodes, teacher_graphs = self.teacher(batch)
        local_per_graph = F.mse_loss(student_nodes, teacher_nodes, reduction="none").mean(2).mean(1)
        global_per_graph = F.mse_loss(student_graphs, teacher_graphs, reduction="none").mean(1)
        loss = global_per_graph.mean() + local_per_graph.mean()
        return loss, local_per_graph, global_per_graph

    @torch.no_grad()
    def anomaly_score(self, batch: Batch) -> Tensor:
        # Reference evaluation uses batch_size=1 and a fresh BatchNorm per layer.
        # Process one graph at a time so both normalization and kernels match.
        self.eval()
        scores = []
        for graph in batch.to_data_list():
            single = Batch.from_data_list([graph])
            _, local, global_ = self.distillation_loss(single)
            scores.append(global_ + local)
        return torch.cat(scores)


class GLocalKDAdapter(BaseAnomalyModel):
    def __init__(self, input_dim: int, config: Config, device: torch.device) -> None:
        architecture = config.architecture
        self.network = GLocalKD(
            input_dim=input_dim,
            hidden_dim=architecture.hidden_dim,
            embedding_dim=architecture.embedding_dim,
            layers=architecture.gcn_layers,
            max_nodes=architecture.max_nodes,
            student_dropout=architecture.student_dropout,
            device=device,
        ).to(device)

    def fit(self, train_loader: DataLoader, config: Config,
            device: torch.device) -> None:
        settings = config.training
        optimizer = torch.optim.Adam(self.network.student.parameters(),
                                     lr=settings.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings.scheduler_step_size,
            gamma=settings.scheduler_gamma)
        for epoch in range(1, settings.epochs + 1):
            self.network.train()
            total_loss = 0.0
            batches = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = self.network.distillation_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite GLocalKD loss at epoch {epoch}")
                # Match the reference's loss-scaled gradient, not just its MSE.
                loss.backward(loss.clone().detach())
                torch.nn.utils.clip_grad_norm_(self.network.student.parameters(),
                                               settings.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.detach())
                batches += 1
            if epoch == 1 or epoch % 10 == 0 or epoch == settings.epochs:
                print(f"epoch={epoch}/{settings.epochs} loss={total_loss / batches:.6f}",
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
