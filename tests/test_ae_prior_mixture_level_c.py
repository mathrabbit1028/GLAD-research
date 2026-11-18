"""Numerical verification of the full finite Level C training derivative."""
import io
import json
import unittest
from contextlib import redirect_stdout

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.models.factory import create_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def graph(n, offset=0):
    edge = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
    return Data(x=torch.nn.functional.one_hot((torch.arange(n) + offset) % 3, 3).float(),
                edge_index=torch.cat([edge, edge.flip(0)], 1), num_nodes=n)


def tiny_config():
    cfg = load_config(['model=ae_prior_mixture_level_c']).model
    cfg.architecture.hidden_dim, cfg.architecture.layers = 5, 1
    cfg.ae.hidden_dim, cfg.ae.identity_logit = 5, 3.
    cfg.prior.components = 4
    # A cap of .25 would force the abnormal weights to be uniformly .25.
    cfg.mixture.cap = .6
    cfg.mixture.normal_steps, cfg.mixture.abnormal_steps, cfg.mixture.restarts = 3, 4, 2
    cfg.ratio.warmup_steps, cfg.ratio.steps = 2, 2
    cfg.ratio.per_component, cfg.ratio.calibration_per_component = 2, 2
    cfg.temperature.steps = 2
    cfg.inner.warmup_steps, cfg.inner.steps, cfg.inner.per_component = 2, 2, 2
    cfg.outer.steps, cfg.outer.select_every = 1, 1
    cfg.sampling.meta_per_component, cfg.sampling.select_per_component = 2, 2
    return cfg


@unittest.skipUnless(torch.cuda.is_available(), 'Level C verification requires GPU')
class LevelCTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        set_seed(17)
        self.device = torch.device('cuda:0')
        self.graphs = [graph(3 + i % 5, i % 3) for i in range(20)]

    def setup_model(self, dtype=torch.float64):
        cfg = tiny_config()
        model = create_model(3, cfg, self.device)
        with redirect_stdout(io.StringIO()):
            model.setup(self.graphs, 17, dtype=dtype)
        return cfg, model

    def test_full_pipeline_hypergradient_finite_difference(self):
        _, model = self.setup_model()
        parameter = model.ae.edge[-1].bias
        original = parameter.detach().clone()

        def objective(higher_order, freeze_prior=False):
            state = model.pipeline(higher_order=higher_order, seed=91000,
                                   freeze_prior=freeze_prior)
            pilots = model.bank.sample(state['eta'], 2, 92000)
            loss, _ = model._objective(model.meta_data, pilots, state, model.teacher_meta)
            return loss, state

        loss, baseline = objective(True)
        analytic = torch.autograd.grad(loss, parameter)[0].item()
        frozen_loss, _ = objective(True, freeze_prior=True)
        frozen = torch.autograd.grad(frozen_loss, parameter)[0].item()

        def displaced(delta, freeze_prior=False):
            with torch.no_grad():
                parameter.copy_(original + delta)
            value, state = objective(False, freeze_prior=freeze_prior)
            return float(value.detach()), state

        epsilon = 1e-4
        plus, changed = displaced(epsilon)
        minus, _ = displaced(-epsilon)
        frozen_plus, _ = displaced(epsilon, freeze_prior=True)
        frozen_minus, _ = displaced(-epsilon, freeze_prior=True)
        with torch.no_grad():
            parameter.copy_(original)
        finite = (plus - minus) / (2 * epsilon)
        frozen_finite = (frozen_plus - frozen_minus) / (2 * epsilon)
        relative = abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-10)
        prior_analytic, prior_finite = analytic - frozen, finite - frozen_finite
        prior_relative = abs(prior_analytic - prior_finite) / max(abs(prior_analytic), abs(prior_finite), 1e-10)
        def distance(key):
            return float(torch.sqrt(sum((baseline[key][name].detach() - value.detach()).square().sum()
                                        for name, value in changed[key].items())))
        theta_change, ratio_change = distance('params'), distance('ratio')
        print('LEVEL_C_HYPERGRADIENT ' + json.dumps(dict(
            analytic=analytic, finite_difference=finite, relative_error=relative,
            prior_stopped=frozen, prior_stopped_finite_difference=frozen_finite,
            prior_path=prior_analytic, prior_path_finite_difference=prior_finite,
            prior_path_relative_error=prior_relative,
            theta_change=theta_change, ratio_change=ratio_change,
            temperature_change=float((baseline['temperature'].detach() - changed['temperature'].detach()).abs()))))
        self.assertLess(relative, .005)
        self.assertLess(prior_relative, .005)
        self.assertGreater(abs(analytic - frozen), 1e-9)
        self.assertGreater(theta_change, 1e-12)
        self.assertGreater(ratio_change, 1e-12)

    def test_initial_prior_control_is_matched(self):
        _, model = self.setup_model()
        adaptive = model.pipeline(higher_order=False, seed=91000)
        frozen = model.pipeline(higher_order=False, seed=91000, freeze_prior=True)
        for key in ['eta', 'buffers', 'ratio', 'temperature', 'weights', 'params']:
            torch.testing.assert_close(adaptive[key], frozen[key], rtol=1e-10, atol=1e-12)
        # This comparison must also match independently generated outer pilots.
        torch.testing.assert_close(model.bank.sample(adaptive['eta'], 2, 92000),
                                   model.bank.sample(frozen['eta'], 2, 92000),
                                   rtol=1e-10, atol=1e-12)

    def test_tiny_fit_and_coherent_checkpoint(self):
        cfg = tiny_config()
        model = create_model(3, cfg, self.device)
        with redirect_stdout(io.StringIO()):
            model.fit(DataLoader(self.graphs, batch_size=8, shuffle=False), cfg, self.device)
        self.assertEqual([len(model.roles[k]) for k in ['fit', 'meta', 'select']], [14, 3, 3])
        self.assertEqual(len(set(sum(model.roles.values(), []))), 20)
        expected = model.anomaly_score(Batch.from_data_list(self.graphs[:4]))
        payload = model.state_dict()
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored = create_model(3, cfg, self.device)
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location=self.device))
        torch.testing.assert_close(expected, restored.anomaly_score(Batch.from_data_list(self.graphs[:4])))
        self.assertTrue(bool(torch.isfinite(expected).all()))
        for left, right in zip(model.weights, restored.weights):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(model.eta, restored.eta)
        torch.testing.assert_close(model.temperature, restored.temperature)
        torch.testing.assert_close(model.ratio.state_dict(), restored.ratio.state_dict())
        torch.testing.assert_close(model.bank.sample(model.eta, 2, 93000),
                                   restored.bank.sample(restored.eta, 2, 93000))


if __name__ == '__main__':
    unittest.main()
