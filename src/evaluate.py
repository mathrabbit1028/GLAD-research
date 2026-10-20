"""Evaluate a saved run on its exact held-out TU graph split."""
import argparse
import csv
import json
from pathlib import Path
import torch
from .utils.config import _wrap
from .data.datamodule import TUDataModule
from .evaluation.evaluator import Evaluator
from .models.factory import create_model
from .utils.seed import set_seed
from .utils.device import require_cuda, require_cuda_available


def main() -> None:
    """Restore a checkpoint and write held-out metrics and graph scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    require_cuda_available()
    root = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = _wrap(checkpoint["config"])
    device = require_cuda(cfg.device, require_dgl=cfg.model.name == 'lgkde')
    cfg.device = str(device)
    set_seed(int(cfg.seed), bool(cfg.reproducibility.deterministic))
    data_cfg = _wrap(dict(cfg.dataset))
    data_cfg.data_root = str(root / data_cfg.data_root)
    data = TUDataModule(data_cfg, int(cfg.seed), split=checkpoint["split"])
    model = create_model(checkpoint["input_dim"], cfg.model, device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, labels, scores = Evaluator(device).evaluate(
        model, data.test_dataloader(), data.split["normal_label"], checkpoint["threshold"]
    )
    result_dir = root / cfg.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.model.name}_{cfg.dataset.name}_seed{cfg.seed}"
    record = {"algorithm": cfg.model.name, "dataset": cfg.dataset.name,
              "seed": int(cfg.seed), **metrics, "checkpoint": str(Path(args.checkpoint).resolve())}
    (result_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    with (result_dir / f"{stem}_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["graph_index", "label", "score"])
        writer.writerows(zip(data.split["test_indices"], labels, scores))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
