"""Build PyG processed files from TU raw files already present on disk."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from torch_geometric.datasets import TUDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config, group_path
DEFAULT_DATA_ROOT = ROOT / "data" / "TU"


def configured_datasets(include_extra: bool = False) -> dict[str, str]:
    datasets: dict[str, str] = {}
    selected = [group_path(ROOT / "configs", "dataset", name)
                for name in load_config().suite.datasets]
    selected += [path for path in sorted((ROOT / 'configs/dataset').glob('*.yaml')) if path not in selected]
    if include_extra:
        selected += sorted((ROOT / "configs/extra/dataset").glob("*.yaml"))
    for config_path in selected:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        datasets[str(config["name"])] = config_path.stem
    return datasets


def prepare(name: str, data_root: Path, force: bool) -> int:
    raw_dir = data_root / name / "raw"
    required = [raw_dir / f"{name}_A.txt",
                raw_dir / f"{name}_graph_indicator.txt",
                raw_dir / f"{name}_graph_labels.txt"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Raw TU files for {name} are missing; this command does not download data:\n{paths}"
        )

    processed_dir = data_root / name / "processed"
    if force and processed_dir.exists():
        shutil.rmtree(processed_dir)

    dataset = TUDataset(root=str(data_root), name=name, use_node_attr=True)
    output = processed_dir / "data.pt"
    if not output.is_file():
        raise RuntimeError(f"PyG did not create the expected processed file: {output}")
    return len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-extra", action="store_true",
                        help="also include optional datasets under configs/extra/")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="TU dataset root (default: data/TU)")
    parser.add_argument("--dataset", action="append", dest="selected",
                        help="TU dataset name; repeat to select several (default: the 12 suite datasets)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild processed files even when they already exist")
    args = parser.parse_args()
    datasets = configured_datasets(args.include_extra)
    unknown = set(args.selected or []) - set(datasets)
    if unknown:
        parser.error(f"Unknown datasets {sorted(unknown)}; choices: {sorted(datasets)}")

    data_root = args.root.resolve()
    selected = args.selected or (list(datasets) if args.include_extra else
                                 [yaml.safe_load(group_path(ROOT / 'configs', 'dataset', key).read_text())['name']
                                  for key in load_config().suite.datasets])
    for name in selected:
        count = prepare(name, data_root, args.force)
        print(f"{name}: {count} graphs -> {data_root / name / 'processed' / 'data.pt'}")


if __name__ == "__main__":
    main()
