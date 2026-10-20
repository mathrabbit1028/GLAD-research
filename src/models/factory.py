"""Instantiate model adapters named by experiment configuration."""

from importlib import import_module

import torch

from ..utils.config import Config, validate_model_config
from .base_model import BaseAnomalyModel


def create_model(input_dim: int, config: Config, device: torch.device) -> BaseAnomalyModel:
    """Load a configured adapter that implements the common model interface."""
    validate_model_config(config)
    dotted_path = config.get("adapter")
    if not isinstance(dotted_path, str) or not dotted_path.startswith("src.models."):
        raise ValueError(f"Invalid model adapter path: {dotted_path!r}")
    module_name, separator, class_name = dotted_path.rpartition(".")
    if not separator or not class_name:
        raise ValueError(f"Invalid model adapter path: {dotted_path!r}")
    module = import_module(module_name)
    adapter_class = getattr(module, class_name, None)
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, BaseAnomalyModel):
        raise TypeError(f"Model adapter must implement BaseAnomalyModel: {dotted_path}")
    return adapter_class(input_dim, config, device)
