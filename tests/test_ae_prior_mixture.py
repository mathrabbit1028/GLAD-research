"""Verify the actual training derivative and graph semantics, on CUDA."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
import torch
from torch.func import functional_call
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.models.factory import create_model
from src.models.candidates.prior_mixture import ComponentClassifier
from src.models.candidates.ae_prior_mixture_impl.core import (
    DenseGIN, ResidualGraphAE, dense_graphs, CappedSimplex, fit_mixtures,
    unroll, mmd, balanced_loss,
)


def graph(n, offset=0):
    edge = torch.stack([torch.arange(n-1), torch.arange(1, n)])
    return Data(x=torch.nn.functional.one_hot((torch.arange(n)+offset)%3, 3).float(),
                edge_index=torch.cat([edge, edge.flip(0)], 1), num_nodes=n)


@unittest.skipUnless(torch.cuda.is_available(), 'Level B verification requires GPU')
class LevelBTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        set_seed(17)
        self.device = torch.device('cuda:0')

    def test_dense_sparse_parity_and_equivariance(self):
        sparse = ComponentClassifier(3, 8, 3, 1).to(self.device)
        dense = DenseGIN(3, 8, 3, 1).to(self.device)
        dense.load_state_dict(sparse.state_dict())
        graphs = [graph(3), graph(7), graph(5)]
        batch = Batch.from_data_list(graphs).to(self.device)
        packed = dense_graphs(graphs, self.device)
        torch.testing.assert_close(sparse(batch)[0].flatten(), dense(packed), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(dense(packed)[:1], dense(dense_graphs(graphs[:1], self.device)), atol=2e-6, rtol=2e-5)
        ae = ResidualGraphAE(3, 5, 6.).to(self.device)
        with torch.no_grad():
            ae.node.weight.normal_(std=.1)
            ae.edge[-1].weight.normal_(std=.1)
        p = torch.tensor([2, 0, 4, 1, 3])
        original = graphs[2]
        permuted = Data(x=original.x[p], edge_index=torch.argsort(p)[original.edge_index], num_nodes=5)
        before = ae(dense_graphs([original], self.device))
        after = ae(dense_graphs([permuted], self.device))
        torch.testing.assert_close(before[0][:, p], after[0])
        torch.testing.assert_close(before[1][:, p][:, :, p], after[1])
        torch.testing.assert_close(dense(before), dense(after), atol=2e-6, rtol=2e-5)
        with self.assertRaisesRegex(ValueError, 'no graph is truncated'):
            dense_graphs(graphs, self.device, max_nodes=4)

    def test_projection_derivatives(self):
        x = torch.tensor([.7, .22, .18, .10, -.2], device=self.device, dtype=torch.double, requires_grad=True)
        fn = lambda z: CappedSimplex.apply(z, .4)
        self.assertTrue(torch.autograd.gradcheck(fn, (x,), eps=1e-5, atol=1e-5))
        self.assertTrue(torch.autograd.gradgradcheck(fn, (x,), eps=1e-5, atol=1e-5))
        value = fn(x)
        self.assertAlmostEqual(float(value.sum()), 1., places=10)
        self.assertTrue(bool(((value >= 0) & (value <= .4)).all()))
        vertex = CappedSimplex.apply(torch.tensor([10., 9., -10., -11.], device=self.device, requires_grad=True), .5)
        self.assertEqual(float(torch.autograd.grad(vertex.square().sum(), vertex.grad_fn.next_functions[0][0].variable)[0].norm()), 0.)

    def test_full_hypergradient_finite_difference(self):
        cfg = load_config(['model=ae_prior_mixture_gin']).model
        cfg.mixture.normal_steps, cfg.mixture.abnormal_steps, cfg.mixture.restarts = 3, 5, 2
        cfg.mixture.cap = .6
        cfg.inner.learning_rate = .04
        cfg.inner.momentum = .5
        ae = ResidualGraphAE(3, 5, 3.).to(self.device).double()
        ratio = DenseGIN(3, 5, 2, 4).to(self.device).double().requires_grad_(False)
        net = DenseGIN(3, 5, 2, 1).to(self.device).double()
        # Softmax columns must differ enough to exercise fitted weights.
        with torch.no_grad():
            ratio.head.weight.mul_(3)
        def pack(gs):
            x, a, mask = dense_graphs(gs, self.device)
            return x.double(), a.double(), mask
        fit = pack([graph(4), graph(6, 1), graph(5, 2)])
        pilots = pack([graph(3+i%4, i//2) for i in range(8)])
        initial = dict(net.named_parameters())
        original = copy.deepcopy(ae.state_dict())
        direction = {k: torch.randn_like(v) for k, v in ae.named_parameters()}
        length = torch.sqrt(sum(v.square().sum() for v in direction.values()))
        direction = {k: v/length for k, v in direction.items()}
        def objective(higher, stopped=False):
            transformed = ae(fit)
            logs = torch.log_softmax(ratio(transformed) / 3, -1)
            weights = fit_mixtures(logs, cfg.mixture)
            params = unroll(net, initial, [pilots]*4, *weights, cfg.inner, higher)
            if stopped:
                params = {k: v.detach() for k, v in params.items()}
            raw = functional_call(net, params, (fit,))
            out = functional_call(net, params, (transformed,))
            syn = functional_call(net, params, (pilots,))
            weights_samples = weights[0].repeat_interleave(2)/2
            loss = mmd(out, syn, 1., wy=weights_samples) + mmd(raw, syn, 1., wy=weights_samples)
            # A raw-graph anchor exercises theta(phi) without an AE-input branch.
            loss = loss + .2 * balanced_loss(syn.reshape(4, 2), *weights) + 5 * raw.square().mean()
            return loss, params
        loss, before = objective(True)
        grads = torch.autograd.grad(loss, tuple(ae.parameters()))
        analytic = sum((g*direction[k]).sum() for k, g in zip(direction, grads)).item()
        stopped_loss, _ = objective(True, True)
        stopped_grads = torch.autograd.grad(stopped_loss, tuple(ae.parameters()))
        stopped = sum((g*direction[k]).sum() for k, g in zip(direction, stopped_grads)).item()
        def displaced(epsilon):
            ae.load_state_dict({k: original[k] + epsilon*direction[k] for k in original})
            value, params = objective(False)
            return float(value.detach()), params
        epsilon = 1e-4
        plus, changed = displaced(epsilon)
        minus, _ = displaced(-epsilon)
        finite = (plus-minus)/(2*epsilon)
        relative = abs(analytic-finite)/max(abs(analytic), abs(finite), 1e-10)
        theta_change = float(torch.sqrt(sum((before[k]-changed[k]).square().sum() for k in before)))
        print('HYPERGRADIENT '+json.dumps(dict(analytic=analytic, finite_difference=finite,
              relative_error=relative, gin_stopped=stopped, theta_change=theta_change)))
        self.assertLess(relative, .005)
        self.assertGreater(abs(analytic-stopped), 1e-7)
        self.assertGreater(theta_change, 1e-10)

    def test_tiny_fit_and_checkpoint(self):
        cfg = load_config(['model=ae_prior_mixture_gin']).model
        cfg.architecture.hidden_dim, cfg.architecture.layers = 8, 2
        cfg.proposal.architecture.hidden_dim, cfg.proposal.architecture.layers = 8, 2
        cfg.proposal.prior.components = 4
        cfg.proposal.prior.train_per_component, cfg.proposal.prior.calibration_per_component = 8, 4
        cfg.proposal.training.ratio_epochs = 2
        cfg.proposal.mixture.abnormal_steps, cfg.proposal.mixture.restarts = 5, 2
        cfg.mixture.normal_steps, cfg.mixture.abnormal_steps, cfg.mixture.restarts = 5, 5, 2
        cfg.inner.warmup_steps, cfg.inner.steps = 3, 2
        cfg.outer.steps, cfg.outer.select_every = 2, 1
        cfg.sampling.train_per_component, cfg.sampling.select_per_component = 4, 2
        cfg.sampling.meta_per_component = 2
        graphs = [graph(3+i%5, i%3) for i in range(20)]
        model = create_model(3, cfg, self.device)
        with redirect_stdout(io.StringIO()):
            model.fit(DataLoader(graphs, batch_size=8, shuffle=False), cfg, self.device)
        self.assertGreater(model.diagnostics['gradient_paths']['gin_training_norm'], 0.)
        self.assertEqual([len(model.roles[k]) for k in ('fit', 'meta', 'select')], [14, 3, 3])
        self.assertEqual(len(set(sum(model.roles.values(), []))), 20)
        expected = model.anomaly_score(Batch.from_data_list(graphs[:4]))
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        restored = create_model(3, cfg, self.device)
        restored.load_state_dict(torch.load(buf, weights_only=True))
        torch.testing.assert_close(expected, restored.anomaly_score(Batch.from_data_list(graphs[:4])))
        self.assertTrue(bool(torch.isfinite(expected).all()))


if __name__ == '__main__':
    unittest.main()
