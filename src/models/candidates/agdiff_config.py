"""Lightweight AGDiff configuration validation; safe for terminal UI planning."""
from collections.abc import Mapping
import math
from typing import Any
from ...utils.config import Config


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"AGDiff {name} must be a positive integer")


def _nonnegative(value: Any, name: str, positive: bool = False) -> None:
    if (type(value) not in {int, float} or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"AGDiff {name} must be finite and {bound}")


def validate_config(config: Config) -> None:
    """Reject unsupported or ineffective settings before constructing the model."""
    arch, settings = config.get("architecture", {}), config.get("training", {})
    if not isinstance(arch, Mapping) or not isinstance(settings, Mapping):
        raise ValueError("AGDiff architecture and training settings must be mappings")
    if arch.get("backbone", "GIN") != "GIN":
        raise ValueError("AGDiff currently implements architecture.backbone=GIN only")
    if settings.get("optimizer", "Adam") not in {"Adam", "AdamW"}:
        raise ValueError("AGDiff training.optimizer must be Adam or AdamW")
    for key in ["hidden_dim", "latent_dim", "gin_layers"]:
        _positive_integer(arch.get(key), f"architecture.{key}")
    for key in ["reconstruction_block_size", "dense_decode_max_nodes"]:
        _positive_integer(arch.get(key, 512), f"architecture.{key}")
    for key in ["pretrain_epochs", "joint_epochs", "diffusion_steps"]:
        _positive_integer(settings.get(key), f"training.{key}")
    _positive_integer(settings.get("reverse_chunk_size", 16), "training.reverse_chunk_size")
    for key in ["learning_rate", "gradient_clip_norm"]:
        _nonnegative(settings.get(key), f"training.{key}", positive=True)
    for key in ["kl_coefficient", "lambda_diff"]:
        _nonnegative(settings.get(key), f"training.{key}")
    _nonnegative(settings.get("weight_decay", 0.), "training.weight_decay")
    for value, name in [(arch.get("detector_spectral_norm", False), "detector_spectral_norm"),
                        (settings.get("reverse_checkpointing", True), "reverse_checkpointing")]:
        if type(value) is not bool:
            raise ValueError(f"AGDiff {name} must be true or false")

