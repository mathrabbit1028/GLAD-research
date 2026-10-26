"""GRAFT-AD validation without Torch/PyG imports; safe for TUI planning."""
from collections.abc import Mapping
import math


FAMILIES = {"community", "degree", "attributes", "coupling", "mixed"}
FIELDS = {
    "architecture": {"hidden_dim", "gin_layers", "dropout"},
    "features": {"mode", "infer_one_hot_tail", "one_hot_groups", "categorical_columns", "reference_nodes"},
    "prior": {"max_anchors", "matches_per_anchor", "match_attempts_per_anchor", "match_tolerance", "max_blocks",
              "max_first_level_graphs", "max_edges", "scm_hidden_dim", "scm_layers", "parameter_jitter", "seed_offset"},
    "verifier": {"neighbors", "near_quantile", "far_quantile", "scale_epsilon", "distance_chunk_size",
                 "clustering_samples", "edge_samples"},
    "generation": {"families", "strengths", "samples_per_family", "attempts_per_family", "min_samples_per_family",
                   "easy_fraction", "require_normal_base", "base_quantile", "control_samples", "control_attempts",
                   "max_descriptor_displacement"},
    "training": {"optimizer", "epochs", "learning_rate", "weight_decay", "gradient_clip_norm",
                 "negatives_per_normal", "negative_weight", "ranking_weight", "ranking_margin"},
}
# Older saved model configs retain their shape; these settings have explicit defaults.
OPTIONAL = {"generation": {"base_quantile", "control_samples", "control_attempts"}}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"GRAFT-AD {name} must be an integer >= {minimum}")


def _number(value, name, minimum=0., maximum=None, strict=False):
    if (type(value) not in {int, float} or not math.isfinite(value)
            or value < minimum or (strict and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"GRAFT-AD invalid {name}: {value!r}")


def validate_config(config):
    sections = {}
    for name in ("architecture", "features", "prior", "verifier", "generation", "training"):
        section = config.get(name)
        if not isinstance(section, Mapping):
            raise ValueError(f"GRAFT-AD {name} must be a mapping")
        unknown = set(section) - FIELDS[name]
        missing = FIELDS[name] - set(section) - OPTIONAL.get(name, set())
        if unknown or missing:
            raise ValueError(f"GRAFT-AD {name}: unknown keys={sorted(unknown)}, missing keys={sorted(missing)}")
        sections[name] = section
    arch, features, prior, verifier, generation, training = (
        sections[k] for k in ("architecture", "features", "prior", "verifier", "generation", "training"))
    for section, values in (
        ("architecture", ("hidden_dim", "gin_layers")),
        ("features", ("reference_nodes",)),
        ("prior", ("max_anchors", "matches_per_anchor", "match_attempts_per_anchor",
                   "max_blocks", "max_first_level_graphs", "max_edges", "scm_hidden_dim", "scm_layers")),
        ("verifier", ("neighbors", "distance_chunk_size", "clustering_samples", "edge_samples")),
        ("generation", ("samples_per_family", "attempts_per_family", "min_samples_per_family")),
        ("training", ("epochs", "negatives_per_normal")),
    ):
        for key in values:
            _integer(sections[section].get(key), f"{section}.{key}")
    _integer(prior.get("seed_offset"), "prior.seed_offset", 0)
    _number(arch.get("dropout"), "architecture.dropout", maximum=1.)
    if arch["dropout"] == 1:
        raise ValueError("GRAFT-AD dropout must be < 1")
    if not isinstance(features.get("mode"), str) or features["mode"] not in {"auto", "degree", "native"}:
        raise ValueError("GRAFT-AD features.mode must be auto, degree or native")
    if type(features.get("infer_one_hot_tail")) is not bool:
        raise ValueError("GRAFT-AD features.infer_one_hot_tail must be true or false")
    groups, columns = features.get("one_hot_groups"), features.get("categorical_columns")
    if not isinstance(groups, list) or not isinstance(columns, list):
        raise ValueError("GRAFT-AD feature groups and categorical columns must be lists")
    seen = set()
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError("GRAFT-AD each one-hot group needs at least two feature indices")
        for column in group:
            _integer(column, "features.one_hot_groups column", 0)
            if column in seen:
                raise ValueError("GRAFT-AD one-hot groups must not repeat or overlap columns")
            seen.add(column)
    for column in columns:
        _integer(column, "features.categorical_columns column", 0)
        if column in seen:
            raise ValueError("GRAFT-AD categorical columns must be distinct from one-hot groups")
        seen.add(column)
    if features["mode"] == "degree" and (groups or columns):
        raise ValueError("GRAFT-AD degree features cannot have categorical schema overrides")
    for section, key in (("prior", "match_tolerance"), ("verifier", "scale_epsilon"),
                         ("training", "learning_rate"), ("training", "gradient_clip_norm"),
                         ("training", "negative_weight")):
        _number(sections[section].get(key), f"{section}.{key}", strict=True)
    _number(prior.get("parameter_jitter"), "prior.parameter_jitter", maximum=1.)
    for key in ("weight_decay", "ranking_weight", "ranking_margin"):
        _number(training.get(key), f"training.{key}")
    if not isinstance(training.get("optimizer"), str) or training["optimizer"] not in {"Adam", "AdamW"}:
        raise ValueError("GRAFT-AD training.optimizer must be Adam or AdamW")
    for key in ("near_quantile", "far_quantile"):
        _number(verifier.get(key), f"verifier.{key}", maximum=1., strict=True)
    if not verifier["near_quantile"] < verifier["far_quantile"] < 1:
        raise ValueError("GRAFT-AD needs 0 < near_quantile < far_quantile < 1")
    _number(generation.get("easy_fraction"), "generation.easy_fraction", maximum=1.)
    if type(generation.get("require_normal_base")) is not bool:
        raise ValueError("GRAFT-AD generation.require_normal_base must be true or false")
    _number(generation.get("base_quantile", .98), "generation.base_quantile", maximum=1., strict=True)
    if generation.get("base_quantile", .98) == 1:
        raise ValueError("GRAFT-AD generation.base_quantile must be < 1")
    controls = generation.get("control_samples", generation["samples_per_family"])
    control_attempts = generation.get("control_attempts", generation["attempts_per_family"])
    _integer(controls, "generation.control_samples")
    _integer(control_attempts, "generation.control_attempts")
    if controls > control_attempts:
        raise ValueError("GRAFT-AD control_samples must be <= control_attempts")
    _number(generation.get("max_descriptor_displacement"), "generation.max_descriptor_displacement", strict=True)
    families, strengths = generation.get("families"), generation.get("strengths")
    if (not isinstance(families, list) or not families
            or any(not isinstance(f, str) or f not in FAMILIES for f in families)
            or len(set(families)) != len(families)):
        raise ValueError(f"GRAFT-AD families must be distinct members of {sorted(FAMILIES)}")
    if not isinstance(strengths, list) or not strengths:
        raise ValueError("GRAFT-AD generation.strengths must be a nonempty list")
    for strength in strengths:
        _number(strength, "generation.strengths entry", maximum=4., strict=True)
    if prior["matches_per_anchor"] > prior["match_attempts_per_anchor"]:
        raise ValueError("GRAFT-AD matching attempts must cover the requested matches")
    if not (generation["min_samples_per_family"] <= generation["samples_per_family"]
            <= generation["attempts_per_family"]):
        raise ValueError("GRAFT-AD needs min_samples_per_family <= samples_per_family <= attempts_per_family")
