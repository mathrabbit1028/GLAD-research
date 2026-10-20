"""MUSE: GIN reconstruction -> four error statistics -> one-class MLP AE.

Independent implementation of arXiv:2410.20366v1, Sections 5 and D.3-D.4.
Public author code: https://github.com/kswoo97/GLAD_MUSE.
Fixed final epochs replace the author's anomaly-labelled validation search.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import dropout_edge, to_dense_adj
from ..base_model import BaseAnomalyModel


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, output_dim))


class Reconstruction(nn.Module):
    def __init__(self, input_dim, architecture):
        super().__init__()
        a = architecture
        self.dropout, self.edge_drop = a.dropout, a.edge_drop_probability
        self.convs = nn.ModuleList([GINConv(mlp(input_dim if i == 0 else a.hidden_dim,
                                               a.hidden_dim, a.hidden_dim), train_eps=False)
                                    for i in range(a.num_layers)])
        width = a.hidden_dim * a.num_layers
        self.features = mlp(width, a.hidden_dim, input_dim)
        self.edges = mlp(width, a.hidden_dim, a.hidden_dim)

    def forward(self, batch):
        edges = batch.edge_index
        if self.training:
            edges, _ = dropout_edge(edges, p=self.edge_drop, force_undirected=True)
        x, layers = batch.x, []
        for conv in self.convs:
            x = F.relu(F.dropout(conv(x, edges), self.dropout, training=self.training))
            layers.append(x)
        z = torch.cat(layers, dim=-1)
        return self.features(z), self.edges(z)


class MUSEAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        self.device, self.config = device, config
        self.network = Reconstruction(input_dim, config.architecture).to(device)
        a = config.architecture
        layers = [nn.Linear(4, a.classifier_hidden_dim), nn.ReLU()]
        for _ in range(a.classifier_layers - 2):
            layers.extend([nn.Linear(a.classifier_hidden_dim, a.classifier_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(a.classifier_hidden_dim, 4))
        self.classifier = nn.Sequential(*layers).to(device)
        self.error_std = None

    def _batch(self, graphs):
        # Strip graph labels and other dataset metadata at the adapter boundary.
        return Batch.from_data_list([Data(x=g.x, edge_index=g.edge_index, num_nodes=g.num_nodes)
                                     for g in graphs]).to(self.device)

    def _loss_and_statistics(self, graphs, training):
        batch = self._batch(graphs)
        features, embeddings = self.network(batch)
        losses, statistics = [], []
        for i, graph in enumerate(graphs):
            start, stop = int(batch.ptr[i]), int(batch.ptr[i + 1])
            n = stop - start
            if n < 1:
                raise ValueError("MUSE requires nonempty graphs")
            target = to_dense_adj(graph.edge_index.to(self.device), max_num_nodes=n)[0].clamp_max(1)
            z = embeddings[start:stop]
            logits = z @ z.T
            feature_errors = 1 - F.cosine_similarity(batch.x[start:stop], features[start:stop],
                                                    dim=1, eps=1e-6)
            if training:
                positives = target.sum()
                # Empty/complete graphs retain finite BCE with a positive weight floor.
                ratio = ((n*n - positives) / positives.clamp_min(1)).clamp_min(1e-6)
                weight = ratio.pow(self.config.architecture.positive_weight_exponent)
                edge_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
                losses.append((feature_errors.mean() + edge_loss) / 2)
            else:
                errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten()
                # Official code uses sample std; single-node graphs need correction=0.
                statistics.append(torch.stack([errors.mean(), errors.std(unbiased=errors.numel() > 1),
                                                feature_errors.mean(),
                                                feature_errors.std(unbiased=n > 1)]))
        return torch.stack(losses).mean() if training else torch.stack(statistics)

    def fit(self, train_loader, config, device):
        s = config.training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=s.learning_rate,
                                     weight_decay=s.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s.scheduler_period)
        for epoch in range(1, s.epochs + 1):
            self.network.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss_and_statistics(batch.to_data_list(), training=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"MUSE reconstruction: non-finite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == s.epochs:
                print(f"reconstruction epoch={epoch}/{s.epochs} loss={sum(losses)/len(losses):.6f}", flush=True)
        self.network.eval()
        with torch.no_grad():
            representations = torch.cat([self._loss_and_statistics(batch.to_data_list(), False)
                                         for batch in train_loader])
            self.error_std = representations.std(dim=0, unbiased=False).clamp_min(s.std_epsilon)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=s.classifier_learning_rate,
                                     weight_decay=s.classifier_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s.scheduler_period)
        for epoch in range(1, s.classifier_epochs + 1):
            self.classifier.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (self.classifier(representations) - representations).norm(dim=1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"MUSE classifier: non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch == 1 or epoch % 50 == 0 or epoch == s.classifier_epochs:
                print(f"classifier epoch={epoch}/{s.classifier_epochs} loss={float(loss.detach()):.6f}", flush=True)
        self.classifier.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        if self.error_std is None:
            raise RuntimeError("MUSE requires fitted training-normal error statistics")
        self.network.eval()
        self.classifier.eval()
        errors = self._loss_and_statistics(batch.to_data_list(), training=False)
        # Strictly monotone anomaly-oriented version of paper Eq.14's normality score.
        return ((errors - self.classifier(errors)) / self.error_std).norm(dim=1)

    def state_dict(self):
        return {"network": self.network.state_dict(), "classifier": self.classifier.state_dict(),
                "error_std": self.error_std}

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.classifier.load_state_dict(state["classifier"])
        self.error_std = state["error_std"].to(self.device)
        self.network.eval()
        self.classifier.eval()
