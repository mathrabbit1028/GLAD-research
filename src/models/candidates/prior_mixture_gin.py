"""Template adapter: classify unfiltered prior samples with an independent GIN."""
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from src.models.base_model import BaseAnomalyModel
from .prior_mixture import ComponentClassifier, CandidateAdapter as ProposalAdapter
from .prior_mixture_impl.sampling import PriorBank
from .prior_mixture_impl.diagnostics import separation_auc


def graph_digest(graphs):
    digest = hashlib.sha256()
    for graph in graphs:
        for value in (graph.x, graph.edge_index):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, str(array.dtype))).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def sample_mixture(graphs, settings, weights, count, seed):
    """Independent component and generator streams; no score-based rejection."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (settings.components,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid mixture weights')
    sampler = PriorBank(graphs, settings, seed)
    components = np.random.default_rng(seed + 1000000).choice(len(weights), count, p=weights / weights.sum())
    return [sampler.sample(int(k)) for k in components], components


class CandidateAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        from .prior_mixture_gin_config import validate_config
        validate_config(config)
        self.config, self.device = config, torch.device(device)
        self.network = ComponentClassifier(input_dim, config.architecture.hidden_dim,
                                            config.architecture.layers, 1).to(device)
        self.diagnostics = {}
        self.proposals = None
        self.started = time.monotonic()

    def _log(self, stage, **values):
        print('PMG ' + json.dumps(dict(stage=stage, seconds=round(time.monotonic()-self.started, 2),
                                      **values), allow_nan=False), flush=True)

    @torch.no_grad()
    def score_graphs(self, graphs):
        self.network.eval()
        chunks = []
        for start in range(0, len(graphs), self.config.training.batch_size):
            chunks.append(self.anomaly_score(Batch.from_data_list(graphs[start:start+self.config.training.batch_size])).cpu())
        return torch.cat(chunks).numpy()

    def fit(self, train_loader, config, device):
        """Fit mixtures on real normals, then learn only synthetic class labels."""
        proposals = ProposalAdapter(self.network.feature_mean.numel(), config.proposal, device)
        _, graphs, _, _ = proposals.fit_proposals(train_loader, config.proposal, device, use_geometry=False)
        self.fit_from_proposals(graphs, proposals.weights_normal.cpu().numpy(),
                                proposals.weights_abnormal.cpu().numpy(), proposals.prior_specification)
        self.diagnostics['proposal_fit'] = proposals.diagnostics

    def fit_from_proposals(self, graphs, normal, abnormal, specification):
        """Reuse frozen weights for controlled experiments; never reuse their encoder."""
        cfg = self.config
        self.proposals = dict(normal=np.asarray(normal).tolist(), abnormal=np.asarray(abnormal).tolist(),
                              specification=specification)
        features = torch.cat([graph.x for graph in graphs]).to(self.device)
        self.network.feature_mean.copy_(features.mean(0))
        self.network.feature_scale.copy_(features.std(0, unbiased=False).clamp_min(.1))
        datasets = {}
        for offset, (name, count) in enumerate([('train', cfg.sampling.train_per_class),
                                                ('validation', cfg.sampling.validation_per_class)]):
            parts, hashes = [], []
            for label, weights in enumerate((normal, abnormal)):
                generated, _ = sample_mixture(graphs, cfg.proposal.prior, weights, count,
                                              cfg.sampling.seed + offset*1000 + label)
                parts.extend(generated)
                hashes.append(graph_digest(generated))
                self._log('sampling', population=name, label=label, graphs=count)
            datasets[name] = (parts, torch.cat([torch.zeros(count), torch.ones(count)]))
            self.diagnostics[name+'_data'] = dict(per_class=count, sha256=hashes)
        synthetic, labels = datasets['train']
        validation, validation_labels = datasets['validation']
        labels = labels.to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.training.learning_rate)
        history = []
        for epoch in range(1, cfg.training.epochs+1):
            self.network.train()
            order = torch.randperm(len(synthetic)).tolist()
            total, correct = 0., 0
            for start in range(0, len(order), cfg.training.batch_size):
                ids = order[start:start+cfg.training.batch_size]
                batch = Batch.from_data_list([synthetic[i] for i in ids]).to(self.device)
                logits = self.network(batch)[0].flatten()
                target = labels[ids]
                loss = F.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite binary GIN loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.training.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ids)
                correct += int(((logits >= 0) == target.bool()).sum())
            if epoch == 1 or epoch % cfg.training.log_every == 0 or epoch == cfg.training.epochs:
                values = self.score_graphs(validation)
                validation_loss = float(F.binary_cross_entropy_with_logits(torch.tensor(values), validation_labels))
                count = cfg.sampling.validation_per_class
                record = dict(epoch=epoch, train_loss=total/len(synthetic), train_accuracy=correct/len(synthetic),
                              validation_loss=validation_loss,
                              validation_auc=separation_auc(values[:count], values[count:]))
                history.append(record)
                self._log('binary_training', **record)
        self.diagnostics.update(history=history, selection='final_epoch',
                                training_graph_digest=graph_digest(graphs),
                                seconds=time.monotonic()-self.started)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        self.network.eval()
        # Raw logits avoid saturated sigmoid ties and retain likelihood ordering.
        values = self.network(batch.to(self.device))[0].flatten()
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite binary GIN score')
        return values

    def state_dict(self):
        return dict(version=1, network=self.network.state_dict(), diagnostics=self.diagnostics,
                    proposals=self.proposals)

    def load_state_dict(self, state):
        if state['version'] != 1:
            raise ValueError('Unsupported prior-mixture GIN state')
        self.network.load_state_dict(state['network'])
        self.diagnostics, self.proposals = state['diagnostics'], state['proposals']
        self.network.eval()
