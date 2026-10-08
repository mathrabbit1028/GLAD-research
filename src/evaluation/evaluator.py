"""Shared scoring, threshold calibration, and held-out evaluation."""
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from ..models.base_model import BaseAnomalyModel
from .metrics import classification_metrics


class Evaluator:
    def __init__(self, device: torch.device) -> None:
        """Keep the scoring device for calibration and held-out evaluation."""
        self.device = device

    def collect(
        self, model: BaseAnomalyModel, loader: DataLoader, normal_label: int | None = None,
        measure_latency: bool = False,
    ) -> tuple[list[int], list[float], float | None, float | None]:
        """Collect labels and scores, optionally measuring latency and CUDA memory."""
        labels, scores = [], []
        elapsed, count = 0.0, 0
        if self.device.type == "cuda" and measure_latency:
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch in loader:
            batch = batch.to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                values = model.anomaly_score(batch).reshape(-1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if measure_latency:
                elapsed += time.perf_counter() - start
                count += values.numel()
            scores.extend(values.detach().cpu().tolist())
            if normal_label is not None:
                labels.extend((batch.y.reshape(-1).cpu() != normal_label).int().tolist())
        memory = (torch.cuda.max_memory_allocated(self.device) / 1024**2
                  if self.device.type == "cuda" and measure_latency else None)
        return labels, scores, (1000 * elapsed / count if count else None), memory

    def calibrate(self, model: BaseAnomalyModel, normal_loader: DataLoader, quantile: float) -> float:
        """Set a score threshold from normal training graphs only."""
        if not 0 < quantile < 1:
            raise ValueError("threshold_quantile must be between zero and one")
        _, scores, _, _ = self.collect(model, normal_loader)
        return float(np.quantile(scores, quantile))

    def evaluate(
        self, model: BaseAnomalyModel, test_loader: DataLoader, normal_label: int, threshold: float,
    ) -> tuple[dict[str, float | int | None], list[int], list[float]]:
        """Evaluate held-out graphs and return metrics with per-graph values."""
        labels, scores, latency, memory = self.collect(
            model, test_loader, normal_label, measure_latency=True
        )
        metrics = classification_metrics(labels, scores, threshold)
        metrics.update({"latency_ms_per_graph": latency,
                        "peak_cuda_memory_mb": memory,
                        "threshold": threshold,
                        "n_test": len(labels),
                        "n_anomaly": sum(labels)})
        return metrics, labels, scores
