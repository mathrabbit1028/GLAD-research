"""Load grouped YAML configs and command-line key=value overrides."""
import copy
from importlib import import_module
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


def validate_model_config(config: Config) -> None:
    """Call an optional lightweight validator without importing the model adapter."""
    validator = config.get("validator")
    if validator is None:
        return
    if not isinstance(validator, str) or not validator.startswith("src.models."):
        raise ValueError(f"Invalid model config validator: {validator!r}")
    module_name, _, function_name = validator.rpartition(".")
    try:
        function = getattr(import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load model config validator: {validator}") from error
    if not callable(function):
        raise ValueError(f"Model config validator must be callable: {validator}")
    function(config)


def _read(path: Path) -> dict[str, Any]:
    """Read one YAML mapping, treating an empty file as an empty mapping."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def group_path(directory: Path, group: str, name: str) -> Path:
    """Find an active config, or an explicitly selected optional model/dataset."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Invalid {group} config name: {name}")
    candidates = [directory / group / f"{name}.yaml"]
    if group == "model":
        candidates = [directory / location / f"{name}.yaml"
                      for location in ["benchmarks", "models"]]
    elif group == "dataset":
        candidates.append(directory / "extra" / group / f"{name}.yaml")
    found = [path for path in candidates if path.is_file()]
    if len(found) > 1:
        raise ValueError(f"Duplicate {group} config name {name}: {found}")
    path = found[0] if found else candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {group} config: {name}")
    return path


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
        if key in defaults or key == "suite":
            selected[key] = raw
        else:
            values.append((key, yaml.safe_load(raw)))
    config = _wrap(main)
    for group, name in selected.items():
        config[group] = _wrap(_read(group_path(directory, group, name)))
    def apply_value(key: str, value: Any) -> None:
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"Unknown config key: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(f"Unknown config key: {key}")
        target[parts[-1]] = _wrap(value)

    # Protocol overrides change shared defaults; direct effective-field overrides win.
    for key, value in values:
        if key.startswith("protocol."):
            apply_value(key, value)
    if "protocol" in config:
        for group in ["dataset", "evaluation"]:
            config[group].update(config.protocol.get(group, {}))
    for key, value in values:
        if not key.startswith("protocol."):
            apply_value(key, value)
    pattern = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")
    def reference(key: str, stack: tuple[str, ...]) -> Any:
        if key in stack:
            raise ValueError(f"Circular config reference: {' -> '.join((*stack, key))}")
        result = config
        for part in key.split("."):
            result = result[part]
        return resolve(copy.deepcopy(result), (*stack, key))

    def resolve(value: Any, stack: tuple[str, ...] = ()) -> Any:
        """Resolve typed references and embedded references in output paths."""
        if isinstance(value, str):
            match = pattern.fullmatch(value)
            if match:
                return reference(match.group(1), stack)
            return pattern.sub(lambda match: str(reference(match.group(1), stack)), value)
        if isinstance(value, dict):
            return Config({key: resolve(item, stack) for key, item in value.items()})
        if isinstance(value, list):
            return [resolve(item, stack) for item in value]
        return value
    return resolve(config)
