"""YAML-driven training; run from the project root with python -m src.train."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from .data.datamodule import TUDataModule
from .evaluation.evaluator import Evaluator
from .models.factory import create_model
from .utils.config import load_config, to_plain, _wrap
from .utils.seed import set_seed
from .utils.device import require_cuda


def main() -> None:
    """Parse YAML overrides, fit the selected model, and save a checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. dataset=mutag seed=0")
    parser.add_argument("--config-file", type=Path, help="Frozen resolved job config from the experiment runner")
    args = parser.parse_args()
    if args.config_file and args.overrides:
        parser.error("Use either --config-file or key=value overrides")
    cfg = (_wrap(yaml.safe_load(args.config_file.read_text())) if args.config_file
           else load_config(args.overrides))
    device = require_cuda(cfg.device, require_dgl=cfg.model.name == 'lgkde')
    cfg.device = str(device)
    root = Path(__file__).resolve().parents[1]
    set_seed(int(cfg.seed), bool(cfg.reproducibility.deterministic))
    data_cfg = _wrap(dict(cfg.dataset))
    data_cfg.data_root = str(root / data_cfg.data_root)
    data = TUDataModule(data_cfg, int(cfg.seed))
    cfg.dataset.max_nodes = data.max_nodes
    if cfg.model.get("architecture", {}).get("max_nodes") == "auto":
        cfg.model.architecture.max_nodes = data.max_nodes
    if cfg.evaluation.get("model_selection", "final_epoch") != "final_epoch":
        raise ValueError("Only fixed final-epoch model selection is implemented")
    source = cfg.evaluation.get("threshold_source", "training_normals")
    if source == "validation_normals" and not len(data.validation):
        raise ValueError("validation_normals calibration requires a normal validation split")
    if source not in {"training_normals", "validation_normals"}:
        raise ValueError(f"Unknown threshold source: {source}")
    output = root / cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
    (output / "split.json").write_text(json.dumps(data.split, indent=2) + "\n")
    model = create_model(data.dataset.num_features, cfg.model, device)
    model.fit(data.train_dataloader(), cfg.model, device)
    calibration = data.train_dataloader(shuffle=False)
    if source == "validation_normals":
        calibration = data.validation_dataloader()
    threshold = Evaluator(device).calibrate(model, calibration, float(cfg.evaluation.threshold_quantile))
    path = root / cfg.output_dir / "checkpoints" / f"{cfg.model.name}_{cfg.dataset.name}_seed{cfg.seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": to_plain(cfg),
                "split": data.split, "threshold": threshold,
                "input_dim": data.dataset.num_features}, path)
    print(f"checkpoint={path} threshold={threshold:.6f}", flush=True)


if __name__ == "__main__":
    main()
