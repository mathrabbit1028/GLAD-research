"""Bounded, family-balanced synthetic pool with an actual easy-negative quota."""
import hashlib
import json
import math

import numpy as np

from .interventions import available_families, intervene


def generate_pool(prior, verifier, normal_descriptors, describe, settings, rng):
    pool, metadata, reports = {}, [], []
    families = available_families(settings["families"], prior.schema.degree_mode)
    disabled = [family for family in settings["families"] if family not in families]
    if disabled:
        print(f"graft_ad disabled families for degree-derived features: {disabled}", flush=True)
    for family in families:
        moderate, easy = [], []
        counts = {key: 0 for key in ("invalid", "base_rejected", "unchanged", "ambiguous", "too_far", "easy_quota")}
        base_distances = []
        target, fraction = settings["samples_per_family"], settings["easy_fraction"]
        easy_cap = int(math.floor(target * fraction))
        attempts = 0
        for attempts in range(1, settings["attempts_per_family"] + 1):
            params = prior.draw(rng)
            seed = int(rng.integers(0, 2**32))
            strength = float(rng.choice(settings["strengths"]))
            changed, components = intervene(params, family, strength, rng, prior.schema.degree_mode)
            base = prior.sample(params, seed)
            candidate = prior.sample(changed, seed)
            if base is None or candidate is None:
                counts["invalid"] += 1
                continue
            base_description, description = describe(base), describe(candidate)
            base_distance = verifier.distance(base_description)
            base_distances.append(base_distance)
            if settings["require_normal_base"] and (
                    base_distance > verifier.near or
                    verifier.match_distance(base_description, normal_descriptors[params["anchor"]])
                    > prior.settings["match_tolerance"]):
                counts["base_rejected"] += 1
                continue
            displacement = verifier.match_distance(description, base_description)
            # Also reject exact no-ops (e.g. permuting a constant feature matrix).
            unchanged = (np.array_equal(base.edges, candidate.edges) and np.array_equal(base.x, candidate.x))
            if unchanged or displacement <= verifier.settings["scale_epsilon"]:
                counts["unchanged"] += 1
                continue
            if displacement > settings["max_descriptor_displacement"]:
                counts["too_far"] += 1
                continue
            distance, weight = verifier.weight(description)
            if weight <= 0.:
                counts["ambiguous"] += 1
                continue
            record = {
                "family": family, "components": components, "anchor": params["anchor"],
                "seed": seed, "strength": strength, "distance": distance,
                "base_distance": base_distance, "displacement": displacement,
                "weight": weight, "easy": distance >= verifier.far,
                "parameters": changed,
            }
            item = (candidate, record)
            if record["easy"]:
                if len(easy) < easy_cap:
                    easy.append(item)
                else:
                    counts["easy_quota"] += 1
            elif len(moderate) < target:
                moderate.append(item)
            # Stop only if a target-sized pool can satisfy its actual easy fraction.
            allowed = easy_cap if fraction == 1 else min(easy_cap, math.floor(len(moderate) * fraction / (1 - fraction)))
            if len(moderate) + min(len(easy), allowed) >= target:
                break
        allowed = easy_cap if fraction == 1 else min(easy_cap, math.floor(len(moderate) * fraction / (1 - fraction)))
        selected_easy = easy[:allowed]
        selected = moderate[:target - len(selected_easy)] + selected_easy
        usable = len(selected) >= settings["min_samples_per_family"]
        reports.append({"family": family, "attempts": attempts, "accepted": len(selected) if usable else 0,
                        "easy": len(selected_easy) if usable else 0, "rejected": counts,
                        "base_distance_mean": float(np.mean(base_distances)) if base_distances else None})
        if usable:
            pool[family] = selected
            metadata.extend(record for _, record in selected)
        print(f"graft_ad pool family={family} proposals={attempts} accepted="
              f"{len(selected) if usable else 0} easy={len(selected_easy) if usable else 0} rejected={counts}", flush=True)
    if not pool:
        raise ValueError("GRAFT-AD generated no usable screened family; inspect prior coverage and generation gate logs")
    digest = hashlib.sha256()
    for family, items in pool.items():
        for graph, record in items:
            digest.update(family.encode())
            digest.update(graph.x.astype("<f4").tobytes())
            digest.update(graph.edges.astype("<i8").tobytes())
            digest.update(json.dumps(record, sort_keys=True, allow_nan=False).encode())
    return pool, {"families": list(pool), "disabled_families": disabled,
                  "reports": reports, "samples": metadata, "sha256": digest.hexdigest()}
