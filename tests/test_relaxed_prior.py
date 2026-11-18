"""Numerical and graph-semantics checks for the Level C relaxed generator."""
import unittest

import torch
from torch.nn import functional as F

from src.models.candidates.ae_prior_mixture_impl.relaxed_prior import (
    fit_prior, RelaxedPriorBank,
)


def reference(device, dtype=torch.double):
    mask = torch.arange(7, device=device)[None, :] < torch.tensor([4, 7, 5], device=device)[:, None]
    x = F.one_hot(torch.arange(21, device=device).reshape(3, 7) % 3, 3).to(dtype) * mask.unsqueeze(-1)
    a = torch.zeros(3, 7, 7, device=device, dtype=dtype)
    for index, size in enumerate([4, 7, 5]):
        row = torch.arange(size - 1, device=device)
        a[index, row, row + 1] = 1
        a[index, row + 1, row] = 1
    return x, a, mask


@unittest.skipUnless(torch.cuda.is_available(), 'Level C verification requires GPU')
class RelaxedPriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = torch.device('cuda:0')
        self.reference = reference(self.device)

    def test_semantics_fixed_noise_and_hard_pairing(self):
        eta = fit_prior(self.reference)
        bank = RelaxedPriorBank(self.reference, components=4, seed=314)
        first = bank.sample(eta, 5, 2718)
        replay = RelaxedPriorBank(self.reference, components=4, seed=314).sample(eta, 5, 2718)
        for before, after in zip(first, replay):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        x, a, mask = first
        torch.testing.assert_close(a, a.transpose(-1, -2), rtol=0, atol=0)
        self.assertEqual(float(a.diagonal(dim1=-2, dim2=-1).abs().max()), 0.)
        self.assertEqual(float(x[~mask].abs().max()), 0.)
        self.assertEqual(float(a[~(mask.unsqueeze(1) & mask.unsqueeze(2))].abs().max()), 0.)
        torch.testing.assert_close(x.sum(-1), mask.double())
        self.assertTrue(bool(((a >= 0) & (a <= 1)).all()))
        self.assertTrue(set(mask.sum(1).tolist()).issubset({4, 5, 7}))
        hard_x, hard_a, hard_mask = bank.sample(eta, 5, 2718, hard=True)
        torch.testing.assert_close(hard_mask, mask)
        torch.testing.assert_close(hard_a, (a >= .5).double())
        torch.testing.assert_close(hard_x, F.one_hot(x.argmax(-1), 3).double() * mask.unsqueeze(-1))
        changed = dict(eta, mean_degree=eta['mean_degree'] * 1.2)
        _, changed_a, _ = bank.sample(changed, 5, 2718)
        self.assertGreater(float(changed_a.sum()), float(a.sum()))
        changed = dict(eta, category_probs=eta['category_probs'] * torch.tensor([2., 1., 1.], device=self.device))
        changed['category_probs'] = changed['category_probs'] / changed['category_probs'].sum()
        changed_x, _, _ = bank.sample(changed, 5, 2718)
        self.assertGreater(float(changed_x[:, :, 0].sum()), float(x[:, :, 0].sum()))

    def test_fit_and_sampler_first_second_derivatives(self):
        x, adjacency, mask = self.reference
        valid = (mask.unsqueeze(1) & mask.unsqueeze(2)) & ~torch.eye(7, dtype=torch.bool, device=self.device)[None]
        bank = RelaxedPriorBank(self.reference, components=4, seed=99, hidden=5)
        direction = torch.tensor([.7, -.2, .4], device=self.device, dtype=torch.double)
        def objective(value):
            soft_x = (4 * x + value * direction).softmax(-1) * mask.unsqueeze(-1)
            soft_a = (4 * (2 * adjacency - 1) + value).sigmoid() * valid
            eta = fit_prior((soft_x, soft_a, mask))
            sampled_x, sampled_a, sampled_mask = bank.sample(eta, 2, 1234)
            return sampled_a.square().mean() + (sampled_x * direction).sum() / sampled_mask.sum()
        value = torch.tensor(.17, device=self.device, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(objective, (value,), eps=1e-5, atol=1e-6, rtol=1e-4))
        self.assertTrue(torch.autograd.gradgradcheck(objective, (value,), eps=1e-5, atol=1e-6, rtol=1e-4))
        gradient = torch.autograd.grad(objective(value), value)[0]
        self.assertGreater(abs(float(gradient)), 1e-5)
        print('RELAXED_PRIOR_DERIVATIVE', float(gradient))

    def test_fitted_moments_ignore_padding_and_are_permutation_invariant(self):
        before = fit_prior(self.reference)
        x, a, mask = self.reference
        permutation = torch.tensor([4, 1, 5, 0, 3, 6, 2], device=self.device)
        after = fit_prior((x[:, permutation], a[:, permutation][:, :, permutation], mask[:, permutation]))
        for name in before:
            torch.testing.assert_close(before[name], after[name])
        self.assertAlmostEqual(float(before['mean_degree']), (6/4 + 12/7 + 8/5)/3)
        self.assertAlmostEqual(float(before['category_probs'].sum()), 1.)
        self.assertTrue(bool((before['feature_scale'] > 0).all()))


if __name__ == '__main__':
    unittest.main()
