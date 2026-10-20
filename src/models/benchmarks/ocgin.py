"""OCGIN with the OCGTL authors' GIN and fixed normal-training center.

Reference: boschresearch/GraphLevel-AnomalyDetection, models/Models.py
and models/Losses.py (OCC_loss). The reused myGIN retains its AGPL notice
under vendor/ubgold/licenses/. CUDA literals are replaced by the selected device.
"""
from types import SimpleNamespace
import torch
from torch_geometric.data import Data, Batch
from ..base_model import BaseAnomalyModel
from ..vendor.ubgold.nn.encoder import myGIN


class OCGINAdapter(BaseAnomalyModel):
    def __init__(self, input_dim, config, device):
        self.device = device
        self.network = myGIN(input_dim, config.architecture.hidden_dim,
                             SimpleNamespace(**config.architecture)).to(device)
        self.network.reset_parameters()
        self.center = None

    def _clean(self, batch):
        return Batch.from_data_list([Data(x=g.x, edge_index=g.edge_index, num_nodes=g.num_nodes)
                                     for g in batch.to_data_list()]).to(self.device)

    def fit(self, train_loader, config, device):
        self.network.eval()
        with torch.no_grad():
            total, count = None, 0
            for batch in train_loader:
                z = self.network(self._clean(batch))
                total = z.sum(0) if total is None else total + z.sum(0)
                count += len(z)
            if not count:
                raise ValueError("OCGIN requires normal training graphs")
            self.center = (total / count).detach()
        settings = config.training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=settings.learning_rate,
                                     weight_decay=settings.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings.scheduler_step_size,
                                                    gamma=settings.scheduler_gamma)
        for epoch in range(1, settings.epochs + 1):
            self.network.train()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                scores = (self.network(self._clean(batch)) - self.center).square().sum(-1)
                loss = scores.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"OCGIN: non-finite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == settings.epochs:
                print(f"epoch={epoch}/{settings.epochs} loss={sum(losses)/len(losses):.6f}", flush=True)
        self.network.eval()

    @torch.no_grad()
    def anomaly_score(self, batch):
        if self.center is None:
            raise RuntimeError("OCGIN requires a fitted normal-training center")
        self.network.eval()
        return (self.network(self._clean(batch)) - self.center).square().sum(-1)

    def state_dict(self):
        return {"network": self.network.state_dict(), "center": self.center}

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.center = state["center"].to(self.device)
        self.network.eval()
