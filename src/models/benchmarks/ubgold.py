"""Four pinned UB-GOLD cores under the shared normal-only benchmark protocol."""
from types import SimpleNamespace
import copy
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_scipy_sparse_matrix, degree
from ..base_model import BaseAnomalyModel
from ..vendor.ubgold.nn.ocgtl import ocgtl
from ..vendor.ubgold.nn.signet import SIGNET
from ..vendor.ubgold.nn.cvtgad import CVTGAD
from ..vendor.ubgold.nn.gladc import NetGe


def structural_features(graph, rw_dim, dg_dim):
    """UB-GOLD RW return probabilities + clipped one-hot degrees, CPU sparse."""
    graph = graph.clone().cpu()
    adjacency = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
    deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).numpy()
    inverse = np.divide(1., deg, out=np.zeros_like(deg), where=deg > 0)
    transition = adjacency @ sp.diags(inverse)
    power = transition
    columns = []
    for _ in range(rw_dim):
        columns.append(torch.from_numpy(power.diagonal().copy()).float())
        power = power @ transition
    one_hot = F.one_hot(torch.from_numpy(deg.clip(0, dg_dim - 1)).long(), dg_dim).float()
    graph.x_s = torch.cat([torch.stack(columns, -1), one_hot], -1)
    return graph


class UBGOLDAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        self.config, self.device, self.name = config, device, config.name
        arch = dict(config.architecture)
        args = SimpleNamespace(**arch, gpu=device, device=str(device))
        if self.name == 'ocgtl':
            self.network = ocgtl(input_dim, args)
        elif self.name == 'signet':
            self.network = SIGNET(input_dim, 0, args, device)
        elif self.name == 'cvtgad':
            self.network = CVTGAD(arch['hidden_dim'], arch['num_layer'], input_dim,
                                   arch['rw_dim'] + arch['dg_dim'], args)
        elif self.name == 'gladc':
            self.network = NetGe(input_dim, arch['hidden_dim'], arch['output_dim'], arch['dropout'], 1)
        else:
            raise ValueError(f'Unsupported method: {self.name}')
        self.network.to(device)
        self.references, self.stats = [], None
        self.noise_encoder = copy.deepcopy(self.network.shared_encoder) if self.name == 'gladc' else None

    def _prepare(self, graph):
        # Labels and dataset metadata never enter the model or its checkpoints.
        graph = Data(x=graph.x.detach().cpu().clone(), edge_index=graph.edge_index.cpu().clone(),
                     num_nodes=graph.num_nodes)
        if self.name == 'signet' and graph.edge_index.numel() == 0:
            nodes = torch.arange(graph.num_nodes)
            graph.edge_index = torch.stack([nodes, nodes])
        if self.name == 'cvtgad':
            a = self.config.architecture
            graph = structural_features(graph, a.rw_dim, a.dg_dim)
        return graph

    def _batch(self, graphs):
        return Batch.from_data_list(graphs).to(self.device)

    def _dense(self, batch):
        size = self.config.architecture.max_nodes
        if int((batch.ptr[1:] - batch.ptr[:-1]).max()) > size:
            raise ValueError('GLADC max_nodes would truncate a graph; increase its config')
        x, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=size)
        adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=size).clamp_max(1)
        return x, adj

    def _components(self, batch, training=False):
        if self.name == 'ocgtl':
            return [self.network.loss_func(self.network(batch), eval=not training)]
        if self.name == 'signet':
            first, second, _, _ = self.network(batch)
            return [self.network.loss_nce(first, second)]
        if self.name == 'cvtgad':
            g_f, g_s, n_f, n_s = self.network(batch.x, batch.x_s, batch.edge_index,
                                             batch.batch, batch.num_graphs)
            return [self.network.calc_loss_g(g_f, g_s), self.network.calc_loss_n(n_f, n_s, batch.batch)]
        x, adj = self._dense(batch)
        encoded, pooled = self.network.shared_encoder(x, adj)
        x_fake, s_fake, reconstructed, pooled_fake = self.network(encoded, adj)
        node = (encoded - reconstructed).square().mean((1, 2))
        graph = (pooled - pooled_fake).square().mean(1)
        if not training:
            return [node + graph]
        with torch.no_grad():
            for (name, param), noisy in zip(self.network.shared_encoder.named_parameters(),
                                           self.noise_encoder.parameters()):
                if name.startswith('proj_head'):
                    noisy.copy_(param)
                else:
                    noisy.copy_(param + torch.randn_like(param) * param.std())
            self.noise_encoder.train()
            _, noisy_pooled = self.noise_encoder(x, adj)
        structure, attributes = self.network.loss_func(
            adj + torch.eye(adj.size(-1), device=adj.device), s_fake, x, x_fake)
        return [structure + attributes + node.mean() + graph.mean()
                + self.network.loss_cal(noisy_pooled, pooled)]

    def fit(self, train_loader, config, device):
        graphs = [self._prepare(train_loader.dataset[i]) for i in range(len(train_loader.dataset))]
        if len(graphs) < 2:
            raise ValueError('These contrastive benchmarks require at least two training graphs')
        # Fixed training context gives a score independent of test batch composition.
        self.references = [g.clone() for g in graphs[:config.training.reference_count]]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=config.training.learning_rate,
                                     weight_decay=config.training.weight_decay)
        batch_size = max(2, train_loader.batch_size)
        settings = config.training
        for epoch in range(1, settings.epochs + 1):
            self.network.train()
            batches = list(torch.randperm(len(graphs)).split(batch_size))
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = torch.cat([batches[-2], batches[-1]])
                batches.pop()
            weights = None
            if settings.adaptive and self.stats is not None:
                weights = self.stats[1].pow(settings.alpha)
                weights = weights / weights.mean().clamp_min(1e-12)
            collected, total = [], 0.
            for indices in batches:
                batch = self._batch([graphs[i] for i in indices.tolist()])
                optimizer.zero_grad(set_to_none=True)
                components = self._components(batch, training=True)
                means = torch.stack([value.mean() for value in components])
                loss = means.sum() if weights is None else (weights * means).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{self.name}: non-finite loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                collected.append(torch.stack([v.detach().reshape(-1) for v in components]))
                total += loss.detach().item()
            if settings.adaptive:
                values = torch.cat(collected, dim=1)
                self.stats = torch.stack([values.mean(1), values.std(1, unbiased=False).clamp_min(1e-6)])
            if epoch == 1 or epoch % 10 == 0 or epoch == settings.epochs:
                print(f'epoch={epoch}/{settings.epochs} loss={total/len(batches):.6f}', flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        self.network.eval()
        graphs = [self._prepare(g) for g in batch.to_data_list()]
        if self.name in {'ocgtl', 'gladc'}:
            return self._components(self._batch(graphs))[0].reshape(-1)
        if not self.references:
            raise RuntimeError('Contrastive scoring requires fitted training references')
        scores = []
        for graph in graphs:
            # The query never interacts with another held-out graph.
            context = self._batch([*self.references, graph])
            values = torch.stack([v.reshape(-1)[-1] for v in self._components(context)])
            if self.config.training.adaptive:
                values = (values - self.stats[0]) / self.stats[1]
            scores.append(values.sum())
        return torch.stack(scores)

    def state_dict(self):
        return {'network': self.network.state_dict(),
                'references': [{k: v.cpu() if torch.is_tensor(v) else v for k, v in g.to_dict().items()}
                               for g in self.references],
                'stats': self.stats.cpu() if self.stats is not None else None}

    def load_state_dict(self, state):
        self.network.load_state_dict(state['network'])
        self.references = [Data(**g) for g in state['references']]
        self.stats = state['stats'].to(self.device) if state['stats'] is not None else None
        self.network.eval()
