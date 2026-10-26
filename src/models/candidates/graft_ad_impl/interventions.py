"""Explicit parameter interventions; strengths are controls, not semantic labels."""
import copy

import numpy as np


def available_families(requested, degree_mode):
    return [family for family in requested if not degree_mode or family not in {"attributes", "coupling"}]


def intervene(params, family, strength, rng, degree_mode=False):
    result = copy.deepcopy(params)
    if family == "mixed":
        choices = ["community", "degree"] if degree_mode else ["community", "degree", "attributes", "coupling"]
        selected = rng.choice(choices, size=2, replace=False).tolist()
    else:
        selected = [family]
    for kind in selected:
        sign = float(rng.choice([-1., 1.]))
        if kind == "community":
            probability = np.clip(result["mixing"], .01, .99)
            logit = np.log(probability / (1 - probability)) + sign * 2 * strength
            result["mixing"] = float(np.clip(1 / (1 + np.exp(-logit)), .001, .999))
            result["first_share"] = float(np.clip(result["first_share"] + sign * .2 * strength, .01, .99))
        elif kind == "degree":
            result["degree_sigma"] = float(np.clip(result["degree_sigma"] * np.exp(sign * strength), .01, 3.))
            result["peripheral_fraction"] = float(np.clip(result["peripheral_fraction"] + sign * .15 * strength, 0., .6))
        elif kind == "attributes" and not degree_mode:
            result["attribute_shift"] = sign * strength
        elif kind == "coupling" and not degree_mode:
            result["coupling_fraction"] = float(min(strength, 1.))
        else:
            raise ValueError(f"GRAFT-AD intervention is unavailable: {kind}")
    return result, selected
