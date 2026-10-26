"""GIN detector with per-node LayerNorm and graph-local mean/max pooling."""
import torch
from torch import nn
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool


class GraphDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, gin_layers, dropout):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.layers = nn.ModuleList()
        for layer in range(gin_layers):
            mlp = nn.Sequential(nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim), nn.ReLU())
            self.layers.append(GINConv(mlp, train_eps=True))
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim * gin_layers, hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, batch):
        x = (batch.x - self.feature_mean) / self.feature_scale
        readouts = []
        for layer in self.layers:
            x = layer(x, batch.edge_index)
            readouts.extend((global_mean_pool(x, batch.batch, size=batch.num_graphs),
                             global_max_pool(x, batch.batch, size=batch.num_graphs)))
        return self.head(torch.cat(readouts, dim=1)).flatten()
