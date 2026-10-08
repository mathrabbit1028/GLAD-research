"""YAML-driven training; run from the project root with python -m src.train."""
import argparse
from pathlib import Path
import torch
from .data.datamodule import TUDataModule
from .evaluation.evaluator import Evaluator
from .models.factory import create_model
from .utils.config import load_config, to_plain, _wrap
from .utils.seed import set_seed


def main() -> None:
    """Parse YAML overrides, fit the selected model, and save a checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. dataset=mutag seed=0")
    args = parser.parse_args()
    cfg = load_config(args.overrides)
    root = Path(__file__).resolve().parents[1]
    set_seed(int(cfg.seed), bool(cfg.reproducibility.deterministic))
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                          else "cpu" if cfg.device == "auto" else cfg.device)
    data_cfg = _wrap(dict(cfg.dataset))
    data_cfg.data_root = str(root / data_cfg.data_root)
    data = TUDataModule(data_cfg, int(cfg.seed))
    model = create_model(data.dataset.num_features, cfg.model, device)
    model.fit(data.train_dataloader(), cfg.model, device)
    threshold = Evaluator(device).calibrate(
        model, data.train_dataloader(shuffle=False), float(cfg.evaluation.threshold_quantile)
    )
    path = root / cfg.output_dir / "checkpoints" / f"{cfg.model.name}_{cfg.dataset.name}_seed{cfg.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": to_plain(cfg),
                "split": data.split, "threshold": threshold,
                "input_dim": data.dataset.num_features}, path)
    print(f"checkpoint={path} threshold={threshold:.6f}", flush=True)


if __name__ == "__main__":
    main()
