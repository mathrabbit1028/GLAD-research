"""Acceptance checks for the new candidate, including CUDA scoring contracts."""
import io
import unittest
from contextlib import redirect_stdout
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from src.utils.config import load_config
from src.models.factory import create_model
from src.models.candidates.prior_mixture_impl.optimization import normal_weights, abnormal_weights, objective, project_capped_simplex
from src.models.candidates.prior_mixture_impl.geometry import fit_geometry, distances, scores
from src.models.candidates.prior_mixture_impl.sampling import PriorBank
from src.models.candidates.prior_mixture_impl.diagnostics import separation_auc, compare_scores, summarize_scores


def graph(n, category=0):
    edge = torch.tensor([[i, i+1] for i in range(n-1)], dtype=torch.long).reshape(-1, 2).T
    edge = torch.cat([edge, edge.flip(0)], 1)
    x = torch.nn.functional.one_hot((torch.arange(n) + category) % 3, 3).float()
    return Data(x=x, edge_index=edge, num_nodes=n)


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(['model=prior_mixture']).model

    def test_relative_pilot_statistics(self):
        self.assertEqual(separation_auc([0, 1], [2, 3]), 1.)
        self.assertEqual(separation_auc([2, 3], [0, 1]), 0.)
        self.assertEqual(separation_auc([1, 1], [1, 1]), .5)
        self.assertEqual(separation_auc([0, 2], [1, 2]), .625)
        result = compare_scores([1, 2, 3], [1, 2, 3], bootstrap=20)
        self.assertEqual(result['auc'], .5)
        summary = summarize_scores([.5, 1., 1.05, 4., 4.1], .05, 3.)
        self.assertEqual(summary['inside_count'], 2)
        self.assertEqual(summary['shell_count'], 2)
        with self.assertRaises(ValueError):
            separation_auc([], [1])

    def test_identifiable_mixture_and_vertex(self):
        ratios = np.array([[1.6, .4]] * 65 + [[.4, 1.6]] * 35)
        logs = np.log(ratios)
        weights, _ = normal_weights(logs, 10000, 1e-13)
        np.testing.assert_allclose(weights, [.75, .25], atol=2e-6)
        self.cfg.mixture.abnormal_mode = 'literal'
        abnormal, info = abnormal_weights(logs, [1, 1], self.cfg.mixture, np.random.default_rng(0))
        self.assertEqual(info['literal_component'], 1)
        self.assertLessEqual(objective(logs, abnormal), objective(logs, np.array([.3, .7])))

    def test_constraints_and_infeasibility(self):
        logs = np.log(np.array([[.2, .4, 1.4, 2.], [2., 1.4, .4, .2]]))
        cfg = self.cfg.mixture
        cfg.restarts, cfg.abnormal_steps = 2, 10
        weights, _ = abnormal_weights(logs, [0, .1, .2, .3], cfg, np.random.default_rng(1))
        np.testing.assert_allclose(weights, np.full(4, .25), atol=1e-7)
        cfg.cap = .5
        weights, _ = abnormal_weights(logs, [0, .1, .2, .3], cfg, np.random.default_rng(1))
        self.assertAlmostEqual(float(weights.sum()), 1., places=7)
        self.assertLessEqual(float(weights.max()), .50000001)
        self.assertGreaterEqual(float(weights @ [0, .1, .2, .3]), cfg.min_shell_rate - 1e-7)
        cfg.min_shell_rate = .24
        weights, _ = abnormal_weights(logs, [0, .1, .2, .3], cfg, np.random.default_rng(1))
        self.assertGreaterEqual(float(weights @ [0, .1, .2, .3]), .24 - 1e-7)
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            abnormal_weights(logs, [0, 0, 0, 0], cfg, np.random.default_rng(1))

    def test_capped_projection_with_large_targets(self):
        np.testing.assert_allclose(project_capped_simplex([1000, 999, 998, 997, -1000], .25),
                                   [.25, .25, .25, .25, 0.], atol=1e-8)
        target = np.array([3., 2., 1., 0., -4.])
        projected = project_capped_simplex(target, .4)
        self.assertAlmostEqual(float(projected.sum()), 1.)
        self.assertLessEqual(float(projected.max()), .4)
        np.testing.assert_allclose(projected, project_capped_simplex(target+1e5, .4), atol=1e-8)

    def test_sampler_schema_and_independence(self):
        source = [graph(1), graph(5), graph(8)]
        bank = PriorBank(source, self.cfg.prior, 3)
        self.assertEqual(bank.mode, 'categorical')
        for k in [0, 15, 31, 63]:
            for _ in range(3):
                generated = bank.sample(k)
                self.assertTrue(torch.equal(generated.x.sum(1), torch.ones(generated.num_nodes)))
                self.assertTrue(torch.all((generated.x == 0) | (generated.x == 1)))
                adjacency = torch.zeros(generated.num_nodes, generated.num_nodes)
                adjacency[generated.edge_index[0], generated.edge_index[1]] = 1
                self.assertTrue(torch.equal(adjacency, adjacency.T))
                self.assertEqual(float(adjacency.diag().sum()), 0.)
        first, second = PriorBank(source, self.cfg.prior, 2), PriorBank(source, self.cfg.prior, 2)
        self.assertTrue(torch.equal(first.sample(4).x, second.sample(4).x))

    def test_ellipsoid_union_and_single_radius_scaling(self):
        state = dict(initial_radii=torch.tensor([1., 1.]), log_scales=torch.zeros(2))
        ds = torch.tensor([[.5, 100.], [100., .5], [2., 2.]])
        torch.testing.assert_close(scores(ds, state), torch.tensor([.5, .5, 2.]))
        state = dict(initial_radii=torch.tensor([2.]), log_scales=torch.tensor([0.7]))
        torch.testing.assert_close(scores(ds[:, :1], state), scores(ds[:, :1], state, True) / np.exp(.7))

    def test_invalid_config(self):
        from src.models.candidates.prior_mixture_config import validate_config
        self.cfg.mixture.cap = .001
        with self.assertRaises(ValueError):
            validate_config(self.cfg)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA scoring check requires a GPU')
    def test_scoring_permutation_batch_and_checkpoint(self):
        device = torch.device('cuda:0')
        model = create_model(3, self.cfg, device)
        reference = [graph(3+i, i) for i in range(20)]
        _, embedding = model._encode(reference)
        state, _ = fit_geometry(embedding, self.cfg.geometry, 0)
        model.geometry = {k: v.to(device) for k, v in state.items()}
        model.weights_normal = torch.full((64,), 1/64, device=device)
        model.weights_abnormal = model.weights_normal.clone()
        test = graph(7)
        single = model.anomaly_score(Batch.from_data_list([test]))
        combined = model.anomaly_score(Batch.from_data_list([test, graph(2)]))[:1]
        torch.testing.assert_close(single, combined, rtol=2e-4, atol=1e-6)
        permutation = torch.randperm(test.num_nodes)
        reverse = torch.argsort(permutation)
        permuted = Data(x=test.x[permutation], edge_index=reverse[test.edge_index], num_nodes=test.num_nodes)
        torch.testing.assert_close(single, model.anomaly_score(Batch.from_data_list([permuted])), rtol=2e-4, atol=1e-6)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = create_model(3, self.cfg, device)
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
        for key, value in model.network.state_dict().items():
            torch.testing.assert_close(value, restored.network.state_dict()[key], rtol=0, atol=0)
        # The encoder uses float32 CUDA reductions even though geometry is double.
        torch.testing.assert_close(single, restored.anomaly_score(Batch.from_data_list([test])),
                                   rtol=2e-5, atol=2e-6)
        degenerate = Data(x=torch.tensor([[1., 0., 0.]]), edge_index=torch.empty((2, 0), dtype=torch.long))
        self.assertTrue(torch.isfinite(restored.anomaly_score(Batch.from_data_list([degenerate]))).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'Full integration requires a GPU')
    def test_full_pipeline_on_synthetic_normals(self):
        # This verifies the later stages independently of MUTAG prior coverage.
        # It is not a benchmark or a replacement for the default experiment.
        from src.utils.seed import set_seed
        set_seed(42)
        cfg = self.cfg
        cfg.prior.components = 4
        cfg.prior.train_per_component = 64
        cfg.prior.calibration_per_component = 32
        cfg.prior.pilot_per_component = 64
        cfg.training.ratio_epochs = 3
        cfg.training.radius_epochs = 20
        cfg.mixture.restarts = 2
        cfg.mixture.abnormal_steps = 10
        cfg.mixture.min_shell_rate = 0.
        cfg.geometry.dimension = 2
        cfg.geometry.regions = 2
        cfg.generation.contrast = 0.
        seed_bank = PriorBank([graph(5+i, i) for i in range(8)], cfg.prior, 71)
        normals = [seed_bank.sample(i % 4) for i in range(64)]
        device = torch.device('cuda:0')
        model = create_model(3, cfg, device)
        with redirect_stdout(io.StringIO()):
            model.fit(DataLoader(normals, batch_size=16), cfg, device)
        self.assertGreater(model.diagnostics['normal_generation']['accepted'], 0)
        self.assertGreater(model.diagnostics['abnormal_generation']['accepted'], 1)
        self.assertTrue(torch.all(model.geometry['log_scales'].exp() >= cfg.geometry.radius_min))
        self.assertTrue(torch.all(model.geometry['log_scales'].exp() <= cfg.geometry.radius_max))
        self.assertTrue(any(abs(x - 1) > 1e-4 for x in model.diagnostics['radius_training']['scales']))
        batch = Batch.from_data_list(normals[:3])
        expected = model.anomaly_score(batch)
        restored = create_model(3, cfg, device)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
        torch.testing.assert_close(expected, restored.anomaly_score(batch), rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
