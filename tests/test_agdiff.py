"""Core structural and gradient checks for the AGDiff reconstruction."""
import unittest

import torch
from torch_geometric.data import Batch, Data

from src.models.agdiff_impl.agdiff_model import AGDiff


class AGDiffChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.batch = Batch.from_data_list([
            Data(x=torch.eye(3),
                 edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])),
            Data(x=torch.eye(3)[:2], edge_index=torch.tensor([[0, 1], [1, 0]])),
        ])
        self.model = AGDiff(3, hidden_dim=8, latent_dim=4, diffusion_steps=8)

    def test_pretraining_and_joint_gradients(self):
        pretrain_loss = self.model.pretrain_loss(self.batch)
        pretrain_loss.backward()
        self.assertTrue(torch.isfinite(pretrain_loss))
        self.assertGreater(self.model.vae.decoder.net[0].weight.grad.abs().sum().item(), 0)

        self.model.zero_grad(set_to_none=True)
        loss, classification_loss, diffusion_loss = self.model.joint_loss(self.batch, 1.0)
        loss.backward()
        for value in (loss, classification_loss, diffusion_loss):
            self.assertTrue(torch.isfinite(value))
        for component in (self.model.vae, self.model.diffusion, self.model.detector):
            self.assertGreater(sum(parameter.grad.abs().sum().item()
                                   for parameter in component.parameters()
                                   if parameter.grad is not None), 0)

    def test_soft_graph_decode_stays_within_each_graph(self):
        mu, logvar = self.model.vae.encode(self.batch)
        latent = self.model.vae.sample(mu, logvar)
        pseudo = self.model.pseudo_graphs(self.batch, latent)
        source_graph = pseudo.batch[pseudo.edge_index[0]]
        target_graph = pseudo.batch[pseudo.edge_index[1]]
        self.assertTrue(torch.equal(source_graph, target_graph))
        self.assertFalse(torch.any(pseudo.edge_index[0] == pseudo.edge_index[1]))
        self.assertTrue(torch.all((pseudo.edge_weight >= 0) & (pseudo.edge_weight <= 1)))
        self.model.detector(pseudo).sum().backward()
        self.assertGreater(sum(parameter.grad.abs().sum().item()
                               for parameter in self.model.diffusion.parameters()
                               if parameter.grad is not None), 0)

    def test_singleton_graph_loss_is_finite(self):
        single = Batch.from_data_list([Data(
            x=torch.ones(1, 3), edge_index=torch.empty(2, 0, dtype=torch.long))])
        self.assertTrue(torch.isfinite(self.model.pretrain_loss(single)))
        self.assertTrue(torch.isfinite(self.model.joint_loss(single, 1.0)[0]))
