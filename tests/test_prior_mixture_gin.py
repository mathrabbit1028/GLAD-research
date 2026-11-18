"""Integration checks for the classifier replacement and independent sampling."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from src.utils.config import load_config
from src.models.factory import create_model
from src.models.candidates.prior_mixture_gin import sample_mixture, graph_digest


def graph(n):
    edge = torch.stack([torch.arange(n-1),torch.arange(1,n)])
    return Data(x=torch.nn.functional.one_hot(torch.arange(n)%3,3).float(),
                edge_index=torch.cat([edge,edge.flip(0)],1),num_nodes=n)


class GinMixtureTests(unittest.TestCase):
    def test_streams_and_validation(self):
        cfg=load_config(['model=prior_mixture_gin']).model
        graphs=[graph(n) for n in range(3,10)]
        weights=np.full(64,1/64)
        first,_=sample_mixture(graphs,cfg.proposal.prior,weights,12,100)
        second,_=sample_mixture(graphs,cfg.proposal.prior,weights,12,100)
        third,_=sample_mixture(graphs,cfg.proposal.prior,weights,12,101)
        self.assertEqual(graph_digest(first),graph_digest(second))
        self.assertNotEqual(graph_digest(first),graph_digest(third))
        with self.assertRaises(ValueError):
            sample_mixture(graphs,cfg.proposal.prior,-weights,12,100)
        from src.models.candidates.prior_mixture_gin_config import validate_config
        cfg.proposal.mixture.min_shell_rate=.05
        with self.assertRaisesRegex(ValueError,'no geometry'):
            validate_config(cfg)

    @unittest.skipUnless(torch.cuda.is_available(),'Integration requires GPU')
    def test_geometry_free_fit_and_checkpoint(self):
        from src.utils.seed import set_seed
        set_seed(43)
        cfg=load_config(['model=prior_mixture_gin']).model
        cfg.proposal.prior.components=4
        cfg.proposal.prior.train_per_component=16
        cfg.proposal.prior.calibration_per_component=8
        cfg.proposal.training.ratio_epochs=2
        cfg.proposal.mixture.restarts=2
        cfg.proposal.mixture.abnormal_steps=10
        cfg.sampling.train_per_class=32
        cfg.sampling.validation_per_class=16
        cfg.training.epochs=3
        model=create_model(3,cfg,torch.device('cuda:0'))
        before=model.network.convs[0].nn[0].weight.detach().clone()
        graphs=[graph(3+i%6) for i in range(20)]
        with patch('src.models.candidates.prior_mixture.fit_geometry',side_effect=AssertionError('Geometry called')):
            with redirect_stdout(io.StringIO()):
                model.fit(DataLoader(graphs,batch_size=8),cfg,torch.device('cuda:0'))
        self.assertFalse(torch.equal(before,model.network.convs[0].nn[0].weight))
        self.assertEqual(model.diagnostics['selection'],'final_epoch')
        expected=model.anomaly_score(Batch.from_data_list(graphs[:3]))
        buffer=io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        restored=create_model(3,cfg,torch.device('cuda:0'))
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(expected,restored.anomaly_score(Batch.from_data_list(graphs[:3])),rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(expected[:1],restored.anomaly_score(Batch.from_data_list(graphs[:1])),rtol=2e-5,atol=2e-6)
        original=graphs[2]
        p=torch.randperm(original.num_nodes)
        inverse=torch.argsort(p)
        permuted=Data(x=original.x[p],edge_index=inverse[original.edge_index],num_nodes=original.num_nodes)
        torch.testing.assert_close(expected[2:],restored.anomaly_score(Batch.from_data_list([permuted])),rtol=2e-5,atol=2e-6)
        self.assertTrue(torch.isfinite(expected).all())


if __name__=='__main__':
    unittest.main()
