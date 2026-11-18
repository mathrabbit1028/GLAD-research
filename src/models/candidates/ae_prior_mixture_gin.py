"""Template adapter for Level B: AE -> mixtures -> trained binary GIN.

The fixed bank and ratio classifier are fitted only on the fit-normal subset.
The complete finite momentum-SGD adaptation is differentiated from a fixed,
AE-independent pretrained initialization. No optimizer state crosses outer steps.
"""
import copy
import json
import math
import time
import numpy as np
import torch
from torch.func import functional_call
from torch_geometric.loader import DataLoader
from src.models.base_model import BaseAnomalyModel
from src.utils.seed import set_seed
from .prior_mixture import CandidateAdapter as ProposalAdapter
from .prior_mixture_gin import graph_digest
from .prior_mixture_impl.sampling import PriorBank
from .ae_prior_mixture_impl.core import (
    DenseGIN, ResidualGraphAE, dense_graphs, take, fit_mixtures,
    balanced_loss, unroll, mmd, reconstruction,
)


def normal_partition(graphs, cfg):
    order = np.random.default_rng(cfg.split.seed).permutation(len(graphs))
    nf, nm = int(len(graphs) * cfg.split.fit_fraction), int(len(graphs) * cfg.split.meta_fraction)
    if min(nf, nm, len(graphs) - nf - nm) < 2:
        raise ValueError('Level B requires at least two normals in each internal subset')
    return dict(fit=order[:nf].tolist(), meta=order[nf:nf+nm].tolist(), select=order[nf+nm:].tolist())


def prepare_bank(graphs, cfg, device):
    """Reusable across controlled runs; no meta/selection normals reach fitting."""
    if max(g.num_nodes for g in graphs) > cfg.limits.max_nodes:
        raise ValueError('Graph exceeds dense Level B node limit; no graph is truncated')
    for graph in graphs:
        x = graph.x
        if not bool((((x == 0) | (x == 1)).all()) and (x.sum(-1) == 1).all()):
            raise ValueError('Level B requires one categorical one-hot node-feature group')
    roles = normal_partition(graphs, cfg)
    fit = [graphs[i] for i in roles['fit']]
    proposal = ProposalAdapter(fit[0].x.shape[-1], cfg.proposal, device)
    proposal.fit_proposals(DataLoader(fit, batch_size=cfg.proposal.training.batch_size, shuffle=False),
                           cfg.proposal, device, use_geometry=False)
    if proposal.prior_specification['feature_mode'] != 'categorical':
        raise ValueError('This Level B implementation supports categorical node features only')
    return dict(version=1, roles=roles, training_digest=graph_digest(graphs),
                ratio_state={k: v.detach().cpu() for k, v in proposal.network.state_dict().items()},
                temperature=proposal.temperature, specification=proposal.prior_specification,
                diagnostics=proposal.diagnostics)


class CandidateAdapter(BaseAnomalyModel):
    selection_policy = 'internal_normal_split'

    def __init__(self, input_dim, config, device):
        from .ae_prior_mixture_config import validate_config
        validate_config(config)
        if torch.device(device).type != 'cuda':
            raise ValueError('Level B model training and evaluation require CUDA')
        self.config, self.device = config, torch.device(device)
        self.network = DenseGIN(input_dim, config.architecture.hidden_dim, config.architecture.layers, 1).to(device)
        self.ae = ResidualGraphAE(input_dim, config.ae.hidden_dim, config.ae.identity_logit).to(device)
        self.ratio = DenseGIN(input_dim, config.proposal.architecture.hidden_dim,
                              config.proposal.architecture.layers, config.proposal.prior.components).to(device)
        self.diagnostics = {}
        self.started = time.monotonic()
        self.weights = None
        self.scale = 1.

    def _log(self, stage, **values):
        print('LEVEL_B ' + json.dumps(dict(stage=stage, mode=self.config.mode,
              seconds=round(time.monotonic()-self.started, 2), **values), allow_nan=False), flush=True)

    def _dense(self, graphs):
        return dense_graphs(graphs, self.device, self.config.limits.max_nodes)

    def component_pool(self, graphs, count, seed):
        bank = PriorBank(graphs, self.config.proposal.prior, seed)
        samples, _ = bank.labelled(count)
        return self._dense(samples)

    def _batches(self, pool, steps, per_component, seed):
        k = self.config.proposal.prior.components
        size = len(pool[0]) // k
        rng = np.random.default_rng(seed)
        result = []
        for _ in range(steps):
            ids = np.arange(k)[:, None] * size + rng.integers(size, size=(k, per_component))
            result.append(take(pool, torch.tensor(ids.flatten(), device=self.device)))
        return result

    def _mixtures(self, transformed=True):
        graphs = self.ae(self.fit_data) if transformed else self.fit_data
        logs = torch.log_softmax(self.ratio(graphs) / self.temperature, -1) + math.log(self.config.proposal.prior.components)
        return fit_mixtures(logs, self.config.mixture)

    def _inner(self, weights, batches, higher_order):
        return unroll(self.network, self.initial, batches, *weights, self.config.inner, higher_order)

    def _objective(self, real, pilots, params, weights, teacher, rec_data=None):
        cfg = self.config.loss
        w_n, w_a = weights
        transformed = self.ae(real)
        raw = functional_call(self.network, params, (real,))
        ae = functional_call(self.network, params, (transformed,))
        synth = functional_call(self.network, params, (pilots,))
        sample_weight = w_n.repeat_interleave(len(synth) // len(w_n)) / (len(synth) // len(w_n))
        tn = mmd(ae, synth, self.scale, wy=sample_weight)
        tr = mmd(ae, raw, self.scale)
        nr = mmd(raw, synth, self.scale, wy=sample_weight)
        rec, edit, edges, nodes = reconstruction(
            real if rec_data is None else rec_data,
            transformed if rec_data is None else self.ae(rec_data), cfg.edge_budget, cfg.node_budget)
        anchor = ((raw - teacher) / self.scale).square().mean()
        sep = balanced_loss(synth.reshape(len(w_n), -1), w_n, w_a)
        total = tn + cfg.raw * tr + cfg.gap * nr + cfg.reconstruction * rec + cfg.edit * edit + cfg.anchor * anchor + cfg.separation * sep
        values = dict(total=total, transformed_synthetic_mmd=tn, transformed_raw_mmd=tr,
                      raw_synthetic_mmd=nr, reconstruction=rec, edit_penalty=edit,
                      anchor=anchor, separation_bce=sep, expected_edge_edits=edges.mean(),
                      expected_node_edits=nodes.mean())
        return total, values

    @staticmethod
    def _numbers(values):
        return {k: float(v.detach()) for k, v in values.items()}

    def fit(self, train_loader, config, device):
        graphs = ProposalAdapter._clean_training(train_loader)
        bank = prepare_bank(graphs, config, device)
        self.fit_prepared(graphs, bank, seed=int(torch.initial_seed() % (2**32)))

    def fit_prepared(self, graphs, bank, seed=0):
        cfg = self.config
        if bank['training_digest'] != graph_digest(graphs) or bank['roles'] != normal_partition(graphs, cfg):
            raise ValueError('Cached bank data or normal-only partition differs')
        self.roles = bank['roles']
        self.bank = bank
        fit = [graphs[i] for i in self.roles['fit']]
        self.fit_data = self._dense(fit)
        self.meta_data = self._dense([graphs[i] for i in self.roles['meta']])
        self.select_data = self._dense([graphs[i] for i in self.roles['select']])
        self.rec_data = self._dense([graphs[i] for i in self.roles['fit'] + self.roles['meta']])
        self.ratio.load_state_dict(bank['ratio_state'])
        self.ratio.requires_grad_(False)
        self.temperature = bank['temperature']
        self.network.feature_mean.copy_(self.ratio.feature_mean)
        self.network.feature_scale.copy_(self.ratio.feature_scale)
        self.train_pool = self.component_pool(fit, cfg.sampling.train_per_component, cfg.sampling.seed)
        self.select_pool = self.component_pool(fit, cfg.sampling.select_per_component, cfg.sampling.seed+1)
        with torch.no_grad():
            self.raw_weights = tuple(w.detach() for w in self._mixtures(False))
        # The same AE-independent warm start is recomputed identically in controls.
        set_seed(seed)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.inner.warmup_lr)
        warm_batches = self._batches(self.train_pool, cfg.inner.warmup_steps, cfg.inner.per_component,
                                     cfg.sampling.seed+2)
        for step, batch in enumerate(warm_batches, 1):
            loss = balanced_loss(self.network(batch).reshape(len(self.raw_weights[0]), -1), *self.raw_weights)
            loss = loss + .5 * cfg.inner.weight_decay * sum(p.square().sum() for p in self.network.parameters())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == cfg.inner.warmup_steps:
                self._log('fixed_initialization_training', step=step, loss=float(loss.detach()))
        del warm_batches
        self.initial = {k: v.detach().clone() for k, v in self.network.named_parameters()}
        fixed_batches = self._batches(self.train_pool, cfg.inner.steps, cfg.inner.per_component, cfg.sampling.seed+3)
        self.reference = {k: v.detach() for k, v in self._inner(self.raw_weights, fixed_batches, False).items()}
        with torch.no_grad():
            reference_fit = functional_call(self.network, self.reference, (self.fit_data,))
            self.scale = max(float(reference_fit.std(unbiased=False)), .25)
            teacher_meta = functional_call(self.network, self.reference, (self.meta_data,))
            teacher_select = functional_call(self.network, self.reference, (self.select_data,))
        self.diagnostics.update(roles=self.roles, scale=self.scale, seed=seed,
                                inner_steps=cfg.inner.steps, conditional_on_fixed_bank=True,
                                initialization='fixed AE-independent raw-normal pretrained GIN; momentum reset',
                                raw_weights=[w.cpu().tolist() for w in self.raw_weights])
        if cfg.mode == 'no_ae':
            self.network.load_state_dict({**self.network.state_dict(), **self.reference})
            self.weights = self.raw_weights
            self.diagnostics.update(selected_step=0, history=[])
            return

        outer = torch.optim.Adam(self.ae.parameters(), lr=cfg.outer.learning_rate)
        best, best_state, best_step = float('inf'), None, None
        history = []
        for step in range(cfg.outer.steps+1):
            if step % cfg.outer.select_every == 0 or step == cfg.outer.steps:
                with torch.no_grad():
                    weights = self.raw_weights if cfg.mode == 'frozen_gin' else self._mixtures()
                params = self.reference if cfg.mode == 'frozen_gin' else self._inner(weights, fixed_batches, False)
                with torch.no_grad():
                    _, values = self._objective(self.select_data, self.select_pool, params, weights, teacher_select)
                record = dict(step=step, split='select', **self._numbers(values))
                if not math.isfinite(record['total']):
                    raise FloatingPointError('Nonfinite selection objective')
                history.append(record)
                if record['total'] < best:
                    best, best_state, best_step = record['total'], copy.deepcopy(self.ae.state_dict()), step
                self._log('normal_only_selection', **record, best_step=best_step)
                del params, weights
            if step == cfg.outer.steps:
                break
            # Independent pilots, refreshed each outer update; fixed during its derivative.
            pilots = self.component_pool(fit, cfg.sampling.meta_per_component, cfg.sampling.seed+1000+step)
            batches = self._batches(self.train_pool, cfg.inner.steps, cfg.inner.per_component,
                                    cfg.sampling.seed+2000+step)
            weights = self.raw_weights if cfg.mode == 'frozen_gin' else self._mixtures()
            params = self.reference if cfg.mode == 'frozen_gin' else self._inner(weights, batches, True)
            loss, values = self._objective(self.meta_data, pilots, params, weights, teacher_meta, self.rec_data)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite Level B objective')
            if step == 0 and cfg.mode == 'level_b':
                # Ablate only the GIN-training path; retain mixture MMD derivatives.
                stopped = {k: v.detach() for k, v in params.items()}
                direct, _ = self._objective(self.meta_data, pilots, stopped, weights, teacher_meta, self.rec_data)
                variables = tuple(self.ae.parameters())
                total_grad = torch.autograd.grad(loss, variables, retain_graph=True)
                direct_grad = torch.autograd.grad(direct, variables, retain_graph=True)
                total_vec = torch.cat([g.flatten() for g in total_grad])
                direct_vec = torch.cat([g.flatten() for g in direct_grad])
                indirect_vec = total_vec - direct_vec
                self.diagnostics['gradient_paths'] = dict(total_norm=float(total_vec.norm()),
                    gin_stopped_norm=float(direct_vec.norm()), gin_training_norm=float(indirect_vec.norm()),
                    training_to_total_ratio=float(indirect_vec.norm()/total_vec.norm().clamp_min(1e-20)),
                    total_vs_gin_stopped_cosine=float(torch.nn.functional.cosine_similarity(total_vec, direct_vec, dim=0)))
                self._log('gradient_paths', **self.diagnostics['gradient_paths'])
                del total_grad, direct_grad, total_vec, direct_vec, indirect_vec, direct, stopped
            outer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.ae.parameters(), cfg.outer.gradient_clip, error_if_nonfinite=True)
            outer.step()
            if step == 0 or (step+1) % cfg.outer.select_every == 0:
                record = dict(step=step+1, split='meta', gradient_norm=float(norm), **self._numbers(values))
                history.append(record)
                self._log('outer_update', **record)
            del loss, values, params, weights, pilots, batches
        self.ae.load_state_dict(best_state)
        with torch.no_grad():
            self.weights = self.raw_weights if cfg.mode == 'frozen_gin' else tuple(w.detach() for w in self._mixtures())
        # Refit with the selected AE and exactly the selection training horizon/draws.
        final = self.reference if cfg.mode == 'frozen_gin' else self._inner(self.weights, fixed_batches, False)
        self.network.load_state_dict({**self.network.state_dict(), **final})
        self.diagnostics.update(selected_step=best_step, selection_objective=best, history=history,
                                final_weights=[w.cpu().tolist() for w in self.weights],
                                gin_parameter_change_from_raw=float(torch.sqrt(sum(
                                    (final[k].detach()-self.reference[k]).square().sum() for k in final))))
        self._log('selected_checkpoint', step=best_step, objective=best,
                  gin_parameter_change=self.diagnostics['gin_parameter_change_from_raw'])

    @torch.no_grad()
    def score_graphs(self, graphs, transform=True, hard=False):
        values = []
        for start in range(0, len(graphs), 128):
            dense = self._dense(graphs[start:start+128])
            if transform and self.config.mode != 'no_ae':
                dense = self.ae(dense, hard=hard)
            values.append(self.network(dense).cpu())
        result = torch.cat(values).numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite Level B logits')
        return result

    @torch.no_grad()
    def anomaly_score(self, batch):
        dense = self._dense(batch)
        if self.config.mode != 'no_ae':
            dense = self.ae(dense)
        scores = self.network(dense)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite Level B logits')
        return scores

    def state_dict(self):
        return dict(version=1, mode=self.config.mode, network=self.network.state_dict(),
                    ae=self.ae.state_dict(), weights=[w.detach().cpu() for w in self.weights],
                    scale=self.scale, diagnostics=self.diagnostics)

    def load_state_dict(self, state):
        if state['version'] != 1 or state['mode'] != self.config.mode:
            raise ValueError('Checkpoint version or inference policy differs')
        self.network.load_state_dict(state['network'])
        self.ae.load_state_dict(state['ae'])
        self.weights = tuple(w.to(self.device) for w in state['weights'])
        self.scale, self.diagnostics = state['scale'], state['diagnostics']
