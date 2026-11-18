"""Level C: differentiate fitted priors, ratio GIN, calibration, weights and GIN.

All retained inner updates start from raw-normal, AE-independent warm starts.
This is a finite relaxed training map, not an implicit optimum of a discrete model.
"""
import copy
import json
import math
import time
import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F
from src.models.base_model import BaseAnomalyModel
from src.utils.seed import set_seed
from .prior_mixture import CandidateAdapter as ProposalAdapter
from .ae_prior_mixture_gin import normal_partition
from .ae_prior_mixture_impl.core import (
    DenseGIN, ResidualGraphAE, dense_graphs, fit_mixtures, balanced_loss, mmd, reconstruction,
)
from .ae_prior_mixture_impl.nested import detached, train_map, fit_temperature
from .ae_prior_mixture_impl.relaxed_prior import fit_prior, RelaxedPriorBank


def detach_state(state):
    return dict(eta=detached(state['eta']), buffers=detached(state['buffers']),
                ratio=detached(state['ratio']), temperature=state['temperature'].detach(),
                weights=tuple(w.detach() for w in state['weights']), params=detached(state['params']))


class CandidateAdapter(BaseAnomalyModel):
    selection_policy = 'internal_normal_split'

    def __init__(self, input_dim, config, device):
        from .ae_prior_mixture_level_c_config import validate_config
        validate_config(config)
        if torch.device(device).type != 'cuda':
            raise ValueError('Level C model training and evaluation require CUDA')
        self.config, self.device = config, torch.device(device)
        c = config.architecture
        self.network = DenseGIN(input_dim, c.hidden_dim, c.layers, 1).to(device)
        self.ratio = DenseGIN(input_dim, c.hidden_dim, c.layers, config.prior.components).to(device)
        self.ae = ResidualGraphAE(input_dim, config.ae.hidden_dim, config.ae.identity_logit).to(device)
        self.diagnostics, self.scale = {}, 1.
        self.started = time.monotonic()

    def _log(self, stage, **values):
        print('LEVEL_C ' + json.dumps(dict(stage=stage, mode=self.config.mode,
              seconds=round(time.monotonic()-self.started, 2), **values), allow_nan=False), flush=True)

    def _dense(self, graphs):
        values = dense_graphs(graphs, self.device, self.config.limits.max_nodes)
        dtype = next(self.network.parameters()).dtype
        return tuple(v.to(dtype=dtype) if v.is_floating_point() else v for v in values)

    def _new_bank(self, reference):
        c = self.config
        return RelaxedPriorBank(reference, components=c.prior.components, seed=c.sampling.seed,
                                edge_temperature=c.prior.edge_temperature,
                                node_temperature=c.prior.node_temperature)

    @staticmethod
    def _buffers(stats):
        return dict(feature_mean=stats['feature_mean'], feature_scale=stats['feature_scale'].clamp_min(.05))

    def _warm(self, network, cfg, eta, weights, seed):
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.warmup_lr)
        for step in range(cfg.warmup_steps):
            # A fixed bank of independent noise seeds; none depend on AE weights.
            graphs = self.bank.sample(eta, cfg.per_component, seed+step % 64)
            logits = network(graphs)
            if weights is None:
                labels = torch.arange(self.config.prior.components, device=self.device).repeat_interleave(cfg.per_component)
                loss = F.cross_entropy(logits, labels)
            else:
                loss = balanced_loss(logits.reshape(self.config.prior.components, -1), *weights)
            loss = loss + .5*cfg.weight_decay*sum(p.square().sum() for p in network.parameters())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or (step+1) % 100 == 0 or step+1 == cfg.warmup_steps:
                self._log('fixed_ratio_warmup' if weights is None else 'fixed_binary_warmup',
                          step=step+1, loss=float(loss.detach()))

    def setup(self, graphs, seed=0, dtype=torch.float32):
        """Fit reference warm starts using fit normals only, independent of AE."""
        c = self.config
        for g in graphs:
            if not bool((((g.x == 0) | (g.x == 1)).all()) and (g.x.sum(-1) == 1).all()):
                raise ValueError('Level C requires one categorical one-hot node-feature group')
        self.network.to(dtype=dtype)
        self.ratio.to(dtype=dtype)
        self.ae.to(dtype=dtype)
        self.roles = normal_partition(graphs, c)
        self.fit_data = self._dense([graphs[i] for i in self.roles['fit']])
        self.meta_data = self._dense([graphs[i] for i in self.roles['meta']])
        self.select_data = self._dense([graphs[i] for i in self.roles['select']])
        self.rec_data = self._dense([graphs[i] for i in self.roles['fit']+self.roles['meta']])
        self.bank = self._new_bank(self.fit_data)
        self.raw_eta = detached(fit_prior(self.fit_data))
        with torch.no_grad():
            self.initial_eta = detached(fit_prior(self.ae(self.fit_data)))
        buffers = self._buffers(self.raw_eta)
        self.network.load_state_dict({**self.network.state_dict(), **buffers})
        self.ratio.load_state_dict({**self.ratio.state_dict(), **buffers})
        set_seed(seed)
        self._warm(self.ratio, c.ratio, self.raw_eta, None, c.sampling.seed+100)
        self.ratio_initial = detached(dict(self.ratio.named_parameters()))
        self.initial_temperature = 1.
        # Exactly the ratio/calibration schedule later used for checkpoint selection.
        partial = self._ratio_stage(self.fit_data, self.raw_eta, buffers, False, c.sampling.seed+10000)
        self.raw_weights = tuple(w.detach() for w in partial['weights'])
        self._warm(self.network, c.inner, self.raw_eta, self.raw_weights, c.sampling.seed+200)
        self.initial = detached(dict(self.network.named_parameters()))
        self.reference = detach_state(self.pipeline(False, c.sampling.seed+10000, raw=True))
        with torch.no_grad():
            ref = self.reference
            fit_logits = functional_call(self.network, (ref['params'], ref['buffers']), (self.fit_data,))
            self.scale = max(float(fit_logits.std(unbiased=False)), .25)
            self.teacher_meta = functional_call(self.network, (ref['params'], ref['buffers']), (self.meta_data,))
            self.teacher_select = functional_call(self.network, (ref['params'], ref['buffers']), (self.select_data,))
        self.diagnostics.update(roles=self.roles, seed=seed, scale=self.scale,
            initialization='fixed raw-normal pretrained ratio and binary GIN; optimizers reset in every retained unroll',
            derivative='exact AD of all finite relaxed stages, conditional on base noise',
            horizons=dict(ratio=c.ratio.steps, temperature=c.temperature.steps, binary=c.inner.steps),
            raw_weights=[w.cpu().tolist() for w in self.raw_weights],
            prior_variant='two relaxed block layers; smooth expected peripheral attachment; conditional categorical random GNN')

    def _ratio_stage(self, transformed, eta, buffers, higher_order, seed):
        c = self.config
        batches = [self.bank.sample(eta, c.ratio.per_component, seed+i) for i in range(c.ratio.steps)]
        ratio = train_map(self.ratio, self.ratio_initial, buffers, batches, c.ratio, higher_order=higher_order)
        calibration = self.bank.sample(eta, c.ratio.calibration_per_component, seed+100)
        cal_logits = functional_call(self.ratio, (ratio, buffers), (calibration,))
        temperature = fit_temperature(cal_logits, c.temperature, self.initial_temperature, higher_order)
        logs = F.log_softmax(functional_call(self.ratio, (ratio, buffers), (transformed,))/temperature, -1)
        weights = fit_mixtures(logs+math.log(c.prior.components), c.mixture)
        return dict(ratio=ratio, temperature=temperature, weights=weights)

    def pipeline(self, higher_order=True, seed=91000, freeze_prior=False, raw=False):
        c = self.config
        transformed = self.fit_data if raw else self.ae(self.fit_data)
        stats = fit_prior(transformed)
        eta = self.initial_eta if freeze_prior and not raw else stats
        buffers = self._buffers(stats)
        if not higher_order:
            transformed = tuple(v.detach() for v in transformed)
            eta, buffers = detached(eta), detached(buffers)
        state = self._ratio_stage(transformed, eta, buffers, higher_order, seed)
        batches = [self.bank.sample(eta, c.inner.per_component, seed+200+i) for i in range(c.inner.steps)]
        weights = state['weights'] if higher_order else tuple(w.detach() for w in state['weights'])
        params = train_map(self.network, self.initial, buffers, batches, c.inner,
                           weights=weights, higher_order=higher_order)
        state.update(eta=eta, buffers=buffers, params=params)
        return state

    def _objective(self, real, pilots, state, teacher, rec_data=None):
        c = self.config.loss
        wn, wa = state['weights']
        transformed = self.ae(real)
        def score(g):
            return functional_call(self.network, (state['params'], state['buffers']), (g,))
        raw, ae, syn = score(real), score(transformed), score(pilots)
        repeats = len(syn)//len(wn)
        sw = wn.repeat_interleave(repeats)/repeats
        tn, tr, nr = mmd(ae, syn, self.scale, wy=sw), mmd(ae, raw, self.scale), mmd(raw, syn, self.scale, wy=sw)
        original = real if rec_data is None else rec_data
        changed = transformed if rec_data is None else self.ae(rec_data)
        rec, edit, edges, nodes = reconstruction(original, changed, c.edge_budget, c.node_budget)
        anchor = ((raw-teacher)/self.scale).square().mean()
        sep = balanced_loss(syn.reshape(len(wn), -1), wn, wa)
        total = tn+c.raw*tr+c.gap*nr+c.reconstruction*rec+c.edit*edit+c.anchor*anchor+c.separation*sep
        return total, dict(total=total, transformed_synthetic_mmd=tn, transformed_raw_mmd=tr,
            raw_synthetic_mmd=nr, reconstruction=rec, edit_penalty=edit, anchor=anchor,
            separation_bce=sep, expected_edge_edits=edges.mean(), expected_node_edits=nodes.mean())

    @staticmethod
    def _numbers(values):
        return {k: float(v.detach()) for k, v in values.items()}

    def fit(self, train_loader, config, device):
        self.fit_graphs(ProposalAdapter._clean_training(train_loader), int(torch.initial_seed() % 2**32))

    def fit_graphs(self, graphs, seed=0):
        self.setup(graphs, seed)
        c = self.config
        fixed_seed = c.sampling.seed+10000
        if c.mode == 'no_ae':
            final, selected, history = self.reference, 0, []
        elif c.mode == 'initial_ae':
            final, selected, history = detach_state(self.pipeline(False, fixed_seed)), 0, []
        else:
            optimizer = torch.optim.Adam(self.ae.parameters(), lr=c.outer.learning_rate)
            best, best_ae, selected, history = float('inf'), None, None, []
            for step in range(c.outer.steps+1):
                if step % c.outer.select_every == 0 or step == c.outer.steps:
                    state = detach_state(self.pipeline(False, fixed_seed, freeze_prior=c.mode == 'fixed_prior'))
                    with torch.no_grad():
                        pilots = self.bank.sample(state['eta'], c.sampling.select_per_component, c.sampling.seed+20000)
                        _, values = self._objective(self.select_data, pilots, state, self.teacher_select)
                    record = dict(step=step, split='select', **self._numbers(values))
                    if not math.isfinite(record['total']):
                        raise FloatingPointError('Nonfinite Level C selection objective')
                    history.append(record)
                    if record['total'] < best:
                        best, best_ae, selected = record['total'], copy.deepcopy(self.ae.state_dict()), step
                    self._log('normal_only_selection', **record, best_step=selected,
                              prior_mean_degree=float(state['eta']['mean_degree']))
                    del state, pilots, values
                if step == c.outer.steps:
                    break
                state = self.pipeline(True, c.sampling.seed+30000+step*500, freeze_prior=c.mode == 'fixed_prior')
                pilots = self.bank.sample(state['eta'], c.sampling.meta_per_component, c.sampling.seed+50000+step)
                loss, values = self._objective(self.meta_data, pilots, state, self.teacher_meta, self.rec_data)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite Level C outer objective')
                if step == 0:
                    variables = tuple(self.ae.parameters())
                    total_grad = torch.autograd.grad(loss, variables, retain_graph=True)
                    stopped = dict(state, params=detached(state['params']))
                    loss_stop, _ = self._objective(self.meta_data, pilots, stopped, self.teacher_meta, self.rec_data)
                    stop_grad = torch.autograd.grad(loss_stop, variables, retain_graph=True)
                    full = torch.cat([g.flatten() for g in total_grad])
                    direct = torch.cat([g.flatten() for g in stop_grad])
                    self.diagnostics['gradient_paths'] = dict(total_norm=float(full.norm()),
                        binary_gin_stopped_norm=float(direct.norm()), binary_training_norm=float((full-direct).norm()))
                    self._log('gradient_paths', **self.diagnostics['gradient_paths'])
                    del total_grad, stop_grad, stopped, loss_stop, full, direct
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.ae.parameters(), c.outer.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                if step == 0 or (step+1) % c.outer.select_every == 0:
                    record = dict(step=step+1, split='meta', gradient_norm=float(norm), **self._numbers(values))
                    history.append(record)
                    self._log('outer_update', **record)
                del state, pilots, loss, values
            self.ae.load_state_dict(best_ae)
            final = detach_state(self.pipeline(False, fixed_seed, freeze_prior=c.mode == 'fixed_prior'))
            self.diagnostics['selection_objective'] = best
        self._install(final)
        self.diagnostics.update(selected_step=selected, history=history,
            final_weights=[w.cpu().tolist() for w in self.weights],
            gin_parameter_change_from_raw=float(torch.sqrt(sum((final['params'][k]-self.reference['params'][k]).square().sum() for k in final['params']))),
            ratio_parameter_change_from_raw=float(torch.sqrt(sum((final['ratio'][k]-self.reference['ratio'][k]).square().sum() for k in final['ratio']))),
            final_temperature=float(self.temperature), raw_temperature=float(self.reference['temperature']))
        self._log('selected_checkpoint', step=selected, prior_mean_degree=float(self.eta['mean_degree']),
                  temperature=float(self.temperature))

    def _install(self, state):
        self.eta, self.weights, self.temperature = detached(state['eta']), tuple(w.detach() for w in state['weights']), state['temperature'].detach()
        self.network.load_state_dict({**self.network.state_dict(), **state['params'], **state['buffers']})
        self.ratio.load_state_dict({**self.ratio.state_dict(), **state['ratio'], **state['buffers']})

    @torch.no_grad()
    def score_graphs(self, graphs, transform=True, hard=False):
        scores = []
        for start in range(0, len(graphs), 128):
            dense = self._dense(graphs[start:start+128])
            if transform and self.config.mode != 'no_ae':
                dense = self.ae(dense, hard=hard)
            scores.append(self.network(dense).cpu())
        result = torch.cat(scores).numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite Level C logits')
        return result

    @torch.no_grad()
    def anomaly_score(self, batch):
        dense = self._dense(batch)
        if self.config.mode != 'no_ae':
            dense = self.ae(dense)
        return self.network(dense)

    def state_dict(self):
        return dict(version=1, mode=self.config.mode, network=self.network.state_dict(), ratio=self.ratio.state_dict(),
            ae=self.ae.state_dict(), eta=detached(self.eta), raw_eta=detached(self.raw_eta), initial_eta=detached(self.initial_eta),
            temperature=self.temperature, weights=[w.detach().cpu() for w in self.weights],
            raw_weights=[w.detach().cpu() for w in self.raw_weights], scale=self.scale, diagnostics=self.diagnostics,
            fit_reference=[v.detach().cpu() for v in self.fit_data], roles=self.roles)

    def load_state_dict(self, state):
        if state['version'] != 1 or state['mode'] != self.config.mode:
            raise ValueError('Level C checkpoint version or inference policy differs')
        for name in ('network', 'ratio', 'ae'):
            getattr(self, name).load_state_dict(state[name])
        for name in ('eta', 'raw_eta', 'initial_eta'):
            setattr(self, name, {k: v.to(self.device) for k, v in state[name].items()})
        self.weights = tuple(w.to(self.device) for w in state['weights'])
        self.raw_weights = tuple(w.to(self.device) for w in state['raw_weights'])
        self.temperature = state['temperature'].to(self.device)
        self.scale, self.diagnostics, self.roles = state['scale'], state['diagnostics'], state['roles']
        self.fit_data = tuple(v.to(self.device) for v in state['fit_reference'])
        self.bank = self._new_bank(self.fit_data)
