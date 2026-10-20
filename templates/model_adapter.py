"""Starter candidate: replace this graph-feature autoencoder with your method.

The starter pools node features and ignores topology beyond that pooling.
It demonstrates the framework contract; it is not a published benchmark.
"""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from src.models.base_model import BaseAnomalyModel


class CandidateAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        self.device = device
        hidden = config.architecture.hidden_dim
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, input_dim)).to(device)

    def _features(self, batch):
        batch = batch.to(self.device)
        return global_mean_pool(batch.x, batch.batch)

    def fit(self, train_loader, config, device):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.training.learning_rate)
        for epoch in range(1, config.training.epochs + 1):
            self.network.train()
            total, count = 0., 0
            for batch in train_loader:
                features = self._features(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = (self.network(features) - features).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                count += 1
            if epoch == 1 or epoch % 10 == 0 or epoch == config.training.epochs:
                print(f'epoch={epoch}/{config.training.epochs} loss={total/count:.6f}', flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        self.network.eval()
        features = self._features(batch)
        return (self.network(features) - features).square().mean(dim=1)

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, state):
        self.network.load_state_dict(state)
        self.network.eval()
