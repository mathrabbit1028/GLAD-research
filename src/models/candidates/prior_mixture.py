"""Template-based candidate implementing prior mixtures and multiple ellipsoids."""
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from src.models.base_model import BaseAnomalyModel
from .prior_mixture_config import validate_config
from .prior_mixture_impl.sampling import PriorBank
from .prior_mixture_impl.optimization import normal_weights, abnormal_weights
from .prior_mixture_impl.geometry import fit_geometry, distances, scores


class ComponentClassifier(nn.Module):
    def __init__(self, input_dim, hidden, layers, components):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_scale', torch.ones(input_dim))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            width = input_dim if layer == 0 else hidden
            self.convs.append(GINConv(nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                                   nn.Linear(hidden, hidden)), train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden * 2, components)

    def forward(self, batch):
        x = (batch.x.float() - self.feature_mean) / self.feature_scale
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, batch.edge_index)))
        embedding = torch.cat([global_mean_pool(x, batch.batch), global_max_pool(x, batch.batch)], 1)
        return self.head(embedding), embedding


class CandidateAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        validate_config(config)
        self.device, self.config = torch.device(device), config
        self.network = ComponentClassifier(input_dim, config.architecture.hidden_dim,
                                           config.architecture.layers, config.prior.components).to(device)
        self.temperature = 1.
        self.geometry = None
        self.diagnostics = {}
        self.prior_specification = None
        self.weights_normal = self.weights_abnormal = None
        self.started = time.monotonic()

    def _log(self, stage, **values):
        print('PME ' + json.dumps(dict(stage=stage, seconds=round(time.monotonic() - self.started, 2),
                                      **values), allow_nan=False), flush=True)

    @torch.no_grad()
    def _encode(self, graphs):
        self.network.eval()
        logits, embeddings = [], []
        size = self.config.training.batch_size
        for start in range(0, len(graphs), size):
            batch = Batch.from_data_list(graphs[start:start + size]).to(self.device)
            output, embedding = self.network(batch)
            logits.append(output)
            embeddings.append(embedding)
        return torch.cat(logits), torch.cat(embeddings)

    def _log_ratios(self, logits):
        return F.log_softmax(logits.double() / self.temperature, dim=1) + math.log(self.config.prior.components)

    def _train_ratio(self, graphs, labels, calibration_graphs, calibration_labels):
        cfg = self.config.training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate)
        labels = labels.to(self.device)
        for epoch in range(1, cfg.ratio_epochs + 1):
            self.network.train()
            order = torch.randperm(len(graphs)).tolist()
            total, correct = 0., 0
            for start in range(0, len(order), cfg.batch_size):
                ids = order[start:start + cfg.batch_size]
                batch = Batch.from_data_list([graphs[i] for i in ids]).to(self.device)
                target = labels[ids]
                logits, _ = self.network(batch)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite component-classifier loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(ids)
                correct += int((logits.argmax(1) == target).sum())
            if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.ratio_epochs:
                self._log('ratio_training', epoch=epoch, epochs=cfg.ratio_epochs,
                          cross_entropy=total / len(graphs), accuracy=correct / len(graphs))
        logits, _ = self._encode(calibration_graphs)
        target = calibration_labels.to(self.device)
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(lambda t: float(F.cross_entropy(logits / math.exp(t), target)),
                                 bounds=(-3., 3.), method='bounded')
        if not result.success:
            raise RuntimeError('Synthetic temperature calibration failed')
        self.temperature = math.exp(float(result.x))
        probabilities = F.softmax(logits / self.temperature, 1)
        confidence, predictions = probabilities.max(1)
        ece = 0.
        for lower in torch.linspace(0, 0.9, 10, device=self.device):
            mask = (confidence > lower) & (confidence <= lower + 0.1)
            if mask.any():
                ece += float(mask.float().mean() *
                             (confidence[mask].mean() - (predictions[mask] == target[mask]).float().mean()).abs())
        self.diagnostics['ratio'] = dict(temperature=self.temperature,
                                        cross_entropy=float(F.cross_entropy(logits / self.temperature, target)),
                                        accuracy=float((predictions == target).float().mean()), ece=ece,
                                        ratio_mean=(probabilities.mean(0) * probabilities.shape[1]).cpu().tolist())
        self._log('ratio_calibration', **self.diagnostics['ratio'])
        self.network.requires_grad_(False)

    @staticmethod
    def _clean_training(train_loader):
        graphs = []
        for batch in train_loader:
            for graph in batch.to_data_list():
                if graph.num_nodes < 1 or not torch.isfinite(graph.x).all():
                    raise ValueError('Training graph has no nodes or non-finite features')
                edge = graph.edge_index.detach().cpu()
                edge = edge[:, edge[0] != edge[1]]
                # Both training and inference require simple undirected inputs.
                unique = torch.unique(edge, dim=1)
                reverse = torch.unique(edge.flip(0), dim=1)
                if edge.shape[1] != unique.shape[1] or not torch.equal(unique, reverse):
                    raise ValueError('Prior mixture expects simple undirected input graphs')
                if graph.edge_index.shape[1] != edge.shape[1]:
                    raise ValueError('Prior mixture expects inputs without self loops')
                graphs.append(Data(x=graph.x.detach().cpu().float().clone(), edge_index=edge.clone(),
                                   num_nodes=graph.num_nodes))
        if len(graphs) < 2:
            raise ValueError('At least two normal training graphs are required')
        return graphs

    def fit_proposals(self, train_loader, config, device, on_geometry=None, use_geometry=True):
        """Fit the frozen encoder, geometry, and two proposal distributions.

        This stage can be measured independently without training an anomaly
        classifier. The caller controls the shell constraint through config.
        """
        self.started = time.monotonic()
        self.network.requires_grad_(True)
        seed = int(torch.initial_seed() % (2 ** 32))
        rng = np.random.default_rng(seed)
        graphs = self._clean_training(train_loader)
        self._log('training_data', graphs=len(graphs), nodes_min=min(g.num_nodes for g in graphs),
                  nodes_max=max(g.num_nodes for g in graphs), seed=seed)
        bank = PriorBank(graphs, config.prior, seed)
        self.prior_specification = bank.specification
        features = torch.cat([g.x for g in graphs]).to(self.device)
        self.network.feature_mean.copy_(features.mean(0))
        self.network.feature_scale.copy_(features.std(0, unbiased=False).clamp_min(0.1))
        self._log('sampling_ratio_data', feature_mode=bank.mode, components=config.prior.components,
                  samples=config.prior.components * config.prior.train_per_component)
        synthetic, labels = bank.labelled(config.prior.train_per_component,
            lambda component, count: self._log('sampling_ratio_training', components_done=component, graphs=count))
        validation, validation_labels = bank.labelled(config.prior.calibration_per_component,
            lambda component, count: self._log('sampling_ratio_calibration', components_done=component, graphs=count))
        self._log('ratio_data_ready', training=len(synthetic), calibration=len(validation))
        self._train_ratio(synthetic, labels, validation, validation_labels)
        del synthetic, validation
        logits, embeddings = self._encode(graphs)
        log_ratios = self._log_ratios(logits).cpu().numpy()
        w_normal, normal_info = normal_weights(log_ratios, config.mixture.normal_steps, config.mixture.tolerance)
        self.weights_normal = torch.tensor(w_normal, dtype=torch.float64, device=self.device)
        self.diagnostics['normal_mixture'] = dict(**normal_info, weights=w_normal.tolist(),
                                                effective_components=float(1 / np.square(w_normal).sum()))
        self._log('normal_mixture', **self.diagnostics['normal_mixture'])
        if not use_geometry:
            if config.mixture.min_shell_rate != 0:
                raise ValueError('Geometry-free proposals require min_shell_rate=0')
            w_abnormal, info = abnormal_weights(log_ratios, np.zeros(config.prior.components),
                                               config.mixture, rng)
            self.weights_abnormal = torch.tensor(w_abnormal, dtype=torch.float64, device=self.device)
            self.diagnostics['abnormal_mixture'] = dict(**info, weights=w_abnormal.tolist())
            self._log('abnormal_mixture', **self.diagnostics['abnormal_mixture'])
            return bank, graphs, embeddings, rng
        state, counts = fit_geometry(embeddings, config.geometry, seed)
        self.geometry = {k: v.to(self.device) for k, v in state.items()}
        self.diagnostics['geometry'] = dict(regions=len(counts), occupancy=counts,
                                            dimension=state['projection'].shape[1])
        self._log('geometry_ready', **self.diagnostics['geometry'])
        if on_geometry is not None:
            on_geometry(self, bank, log_ratios)
        self.fit_abnormal_proposal(bank, log_ratios, rng, config)
        return bank, graphs, embeddings, rng

    def fit_abnormal_proposal(self, bank, log_ratios, rng, config):
        """Complete proposal fitting from an already frozen normal geometry."""
        shell_rates = []
        pilot_score_quantiles = []
        for component in range(config.prior.components):
            pilot = [bank.sample(component) for _ in range(config.prior.pilot_per_component)]
            _, pilot_embeddings = self._encode(pilot)
            preliminary = scores(distances(pilot_embeddings, self.geometry), self.geometry, preliminary=True)
            inside = ((preliminary >= 1 + config.generation.shell_inner) &
                      (preliminary <= 1 + config.generation.shell_outer))
            shell_rates.append(float(inside.double().mean()))
            pilot_score_quantiles.append(torch.quantile(preliminary, torch.tensor([0.1, 0.5, 0.9],
                                        device=self.device, dtype=torch.float64)).cpu().tolist())
            if (component + 1) % 8 == 0 or component + 1 == config.prior.components:
                self._log('shell_pilot', completed=component + 1, components=config.prior.components,
                          mean_shell_rate=float(np.mean(shell_rates)))
        self.diagnostics['pilot'] = dict(shell_rates=shell_rates, score_quantiles=pilot_score_quantiles)
        self._log('shell_pilot_complete', **self.diagnostics['pilot'])
        w_abnormal, abnormal_info = abnormal_weights(log_ratios, shell_rates, config.mixture, rng,
                    progress=lambda restart, value: self._log('abnormal_restart', restart=restart, objective=value))
        self.weights_abnormal = torch.tensor(w_abnormal, dtype=torch.float64, device=self.device)
        self.diagnostics['abnormal_mixture'] = dict(**abnormal_info, weights=w_abnormal.tolist())
        self._log('abnormal_mixture', **self.diagnostics['abnormal_mixture'])

    def fit(self, train_loader, config, device):
        bank, graphs, embeddings, rng = self.fit_proposals(train_loader, config, device)
        w_normal = self.weights_normal.detach().cpu().numpy()
        w_abnormal = self.weights_abnormal.detach().cpu().numpy()
        positive = self._accepted_distances(bank, w_normal, len(graphs), False, rng)
        negative = self._accepted_distances(bank, w_abnormal, len(graphs), True, rng)
        real = distances(embeddings, self.geometry).detach()
        self._train_radii(real, positive, negative)
        self._log('fit_complete', normal_regions=len(self.geometry['centers']), diagnostics=self.diagnostics)

    def _accepted_distances(self, bank, weights, n, abnormal, rng):
        cfg = self.config.generation
        target = max(1, int(math.ceil(n * cfg.target_ratio)))
        budget = target * cfg.proposal_multiplier
        count, proposed = 0, 0
        accepted = []
        proposal_counts = np.zeros(len(weights), dtype=int)
        acceptance_counts = np.zeros(len(weights), dtype=int)
        size_sums = [0, 0]
        while proposed < budget and count < target:
            size = min(self.config.training.batch_size, budget - proposed)
            ids = rng.choice(len(weights), size=size, p=weights / weights.sum())
            samples = [bank.sample(int(k)) for k in ids]
            logits, embeddings = self._encode(samples)
            ds = distances(embeddings, self.geometry)
            preliminary = scores(ds, self.geometry, preliminary=True)
            if abnormal:
                log_ratios = self._log_ratios(logits)
                contrast = (torch.logsumexp(log_ratios + self.weights_abnormal.log(), 1) -
                            torch.logsumexp(log_ratios + self.weights_normal.log(), 1))
                keep = ((preliminary >= 1 + cfg.shell_inner) & (preliminary <= 1 + cfg.shell_outer) &
                        (contrast >= cfg.contrast))
            else:
                keep = preliminary <= 1
            chosen = torch.where(keep)[0][:target - count]
            accepted.append(ds[chosen].detach())
            np.add.at(proposal_counts, ids, 1)
            np.add.at(acceptance_counts, ids[chosen.cpu().numpy()], 1)
            size_sums[0] += sum(g.num_nodes for g in samples)
            size_sums[1] += sum(samples[i].num_nodes for i in chosen.cpu().tolist())
            count += len(chosen)
            proposed += size
            if proposed == size or proposed % (self.config.training.batch_size * 10) == 0 or count == target:
                self._log('generation', kind='abnormal' if abnormal else 'normal',
                          proposed=proposed, accepted=count, target=target, budget=budget)
        summary = dict(proposed=proposed, accepted=count, target=target, acceptance=count / proposed,
                       component_proposals=proposal_counts.tolist(), component_accepted=acceptance_counts.tolist(),
                       proposal_mean_nodes=size_sums[0] / proposed,
                       accepted_mean_nodes=size_sums[1] / max(count, 1))
        self.diagnostics['abnormal_generation' if abnormal else 'normal_generation'] = summary
        self._log('generation_complete', kind='abnormal' if abnormal else 'normal', **summary)
        if count == 0:
            raise RuntimeError(f'No accepted {"abnormal" if abnormal else "normal"} graphs within proposal budget')
        if count < target:
            self._log('generation_shortfall', kind='abnormal' if abnormal else 'normal',
                      message='Using available nonempty accepted set; target not reached')
        result = torch.cat(accepted)
        if abnormal and len(torch.unique(result, dim=0)) < 2:
            raise RuntimeError('Accepted abnormal examples have insufficient geometric variation')
        return result

    def _train_radii(self, real, positive, negative):
        cfg = self.config.geometry
        log_scales = nn.Parameter(self.geometry['log_scales'].clone())
        optimizer = torch.optim.Adam([log_scales], lr=self.config.training.learning_rate)
        eta = cfg.synthetic_positive_fraction
        initial_real = scores(real, self.geometry).detach()
        for epoch in range(1, self.config.training.radius_epochs + 1):
            radii = self.geometry['initial_radii'] * log_scales.exp()
            def loss_for(ds, label):
                value = (ds / radii).min(1).values
                logits = cfg.beta * (value - 1)
                return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label))
            loss = (0.5 * ((1 - eta) * loss_for(real, 0.) + eta * loss_for(positive, 0.)) +
                    0.5 * loss_for(negative, 1.) + cfg.radius_penalty * log_scales.square().mean())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite radius loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_scales.clamp_(math.log(cfg.radius_min), math.log(cfg.radius_max))
            if epoch == 1 or epoch % self.config.training.log_every == 0 or epoch == self.config.training.radius_epochs:
                self._log('radius_training', epoch=epoch, loss=float(loss.detach()),
                          radius_scales=log_scales.detach().exp().cpu().tolist())
        self.geometry['log_scales'] = log_scales.detach()
        self.diagnostics['radius_training'] = dict(scales=log_scales.detach().exp().cpu().tolist(),
                        initial_real_coverage=float((initial_real <= 1).double().mean()),
                        final_real_coverage=float((scores(real, self.geometry) <= 1).double().mean()),
                        final_loss=float(loss.detach()))
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        if self.geometry is None:
            raise RuntimeError('Fit or restore the model before scoring')
        self.network.eval()
        _, embeddings = self.network(batch.to(self.device))
        result = scores(distances(embeddings, self.geometry), self.geometry)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Non-finite anomaly scores')
        return result

    def state_dict(self):
        return dict(version=1, network=self.network.state_dict(), temperature=self.temperature,
                    geometry=self.geometry, normal_weights=self.weights_normal,
                    abnormal_weights=self.weights_abnormal, diagnostics=self.diagnostics,
                    prior_specification=self.prior_specification)

    def load_state_dict(self, state):
        if state['version'] != 1:
            raise ValueError('Unsupported prior mixture checkpoint version')
        self.network.load_state_dict(state['network'])
        self.temperature = float(state['temperature'])
        self.geometry = {k: v.to(self.device) for k, v in state['geometry'].items()}
        self.weights_normal = state['normal_weights'].to(self.device)
        self.weights_abnormal = (None if state['abnormal_weights'] is None
                                 else state['abnormal_weights'].to(self.device))
        self.diagnostics = state['diagnostics']
        self.prior_specification = state['prior_specification']
        self.network.requires_grad_(False)
        self.network.eval()
