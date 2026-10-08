"""Load grouped YAML configs and command-line key=value overrides."""
import copy
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import yaml


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        """Expose a configuration key through attribute access."""
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign an attribute to the matching configuration key."""
        self[name] = value


def _wrap(value: Any) -> Any:
    """Recursively turn YAML mappings into attribute-accessible configs."""
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def to_plain(value: Any) -> Any:
    """Convert nested Config objects to standard checkpoint-safe containers."""
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def _read(path: Path) -> dict[str, Any]:
    """Read one YAML mapping, treating an empty file as an empty mapping."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(overrides: Iterable[str] = (), config_dir: str | Path | None = None) -> Config:
    """Compose model and dataset YAML files, then apply key=value overrides."""
    directory = Path(config_dir or Path(__file__).resolve().parents[2] / "configs")
    main = _read(directory / "config.yaml")
    defaults = {next(iter(item)): next(iter(item.values())) for item in main.pop("defaults") if "_self_" not in item}
    selected = dict(defaults)
    values = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected key=value override: {override}")
        key, raw = override.split("=", 1)
        if key in defaults:
            selected[key] = raw
        else:
            values.append((key, yaml.safe_load(raw)))
    config = _wrap(main)
    for group, name in selected.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"Invalid {group} config name: {name}")
        config[group] = _wrap(_read(directory / group / f"{name}.yaml"))
    for key, value in values:
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"Unknown config key: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"Unknown config key: {key}")
        target[parts[-1]] = _wrap(value)
    def resolve(value: Any) -> Any:
        """Replace full-value references such as ${dataset.latent_dim}."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            result = config
            for part in value[2:-1].split("."):
                result = result[part]
            return copy.deepcopy(result)
        if isinstance(value, dict):
            return Config({key: resolve(item) for key, item in value.items()})
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value
    return resolve(config)
