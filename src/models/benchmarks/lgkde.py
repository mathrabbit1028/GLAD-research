"""Official LGKDE model/generator integrated with the shared one-class protocol."""
import dgl
import torch
from torch_geometric.data import Batch

from ..base_model import BaseAnomalyModel
from ..vendor.lgkde.model import LGKDE
from ..vendor.lgkde.ng import NegativeSampleGenerator
from ..vendor.lgkde.losses import LogRatioLossAD
from .lgkde_distance import VectorizedDGMMD


def to_dgl(graph):
    graph = graph.cpu()
    result = dgl.graph(tuple(graph.edge_index), num_nodes=graph.num_nodes)
    result = dgl.add_self_loop(dgl.remove_self_loop(result))
    result.ndata['attr'] = graph.x.clone()
    return result


class LGKDEAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        self.device = device
        self.config = config
        self.network = LGKDE(in_dim=input_dim, **dict(config.architecture)).to(device)
        # Same parameters/state keys; only replace the graph-pair calculation.
        self.network.dgmmd.__class__ = VectorizedDGMMD
        self.generator = NegativeSampleGenerator(**dict(config.negative_sampling))
        self.references = []

    def fit(self, train_loader, config, device):
        if config.training.get('mixed_anomaly_ratio', 0.) != 0.:
            raise ValueError('The normal-only LGKDE adapter requires mixed_anomaly_ratio=0')
        self.references = [to_dgl(train_loader.dataset[i]) for i in range(len(train_loader.dataset))]
        settings = config.training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=settings.learning_rate,
                                     weight_decay=settings.weight_decay)
        criterion = LogRatioLossAD(epsilon=settings.epsilon)
        for epoch in range(1, settings.epochs + 1):
            self.network.train()
            losses = []
            for batch in train_loader:
                graphs = [to_dgl(g) for g in batch.to_data_list()]
                normal = dgl.batch(graphs).to(device)
                optimizer.zero_grad(set_to_none=True)
                positive = self.network.get_reference_scores(normal)
                negative_scores = []
                for increase in (False, True):
                    negative = dgl.batch([
                        self.generator.generate_negative_graph(g, increase_edges=increase)
                        for g in graphs
                    ]).to(device)
                    negative_scores.append(self.network.get_query_scores(negative, normal))
                loss = sum(criterion(positive, score) for score in negative_scores) / 2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite LGKDE loss at epoch {epoch}')
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())
            if epoch == 1 or epoch % 10 == 0 or epoch == settings.epochs:
                print(f'epoch={epoch}/{settings.epochs} loss={sum(losses)/len(losses):.6f}', flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch: Batch):
        if not self.references:
            raise RuntimeError('LGKDE needs fitted training references')
        self.network.eval()
        query = dgl.batch([to_dgl(g) for g in batch.to_data_list()]).to(self.device)
        # KDE is a mean over references. Weighted chunk means exactly retain
        # all training graphs while bounding the dense node-distance matrices.
        size = self.config.training.reference_batch_size
        density = torch.zeros(batch.num_graphs, device=self.device)
        for start in range(0, len(self.references), size):
            subset = self.references[start:start + size]
            reference = dgl.batch(subset).to(self.device)
            density += self.network.get_query_scores(query, reference) * len(subset)
        return -density / len(self.references)

    def state_dict(self):
        return {'network': self.network.state_dict(), 'references': [
            {'edges': torch.stack(g.edges()), 'x': g.ndata['attr'].cpu()}
            for g in self.references
        ]}

    def load_state_dict(self, state):
        self.network.load_state_dict(state['network'])
        self.references = []
        for item in state['references']:
            graph = dgl.graph(tuple(item['edges'].cpu()), num_nodes=len(item['x']))
            graph.ndata['attr'] = item['x'].cpu()
            self.references.append(graph)
        self.network.eval()
