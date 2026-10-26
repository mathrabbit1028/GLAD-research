"""Validated normal controls and bounded, family-balanced intervention pools."""
import copy
import hashlib
import json
import math

import numpy as np

from .interventions import available_families, intervene


def _distance_summary(values):
    if not values:
        return None
    return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}


def _normal_controls(prior, verifier, normal_descriptors, describe, settings, rng):
    """Replay matched draws before exploring fresh, jittered parameter samples.

    Prior matching retains an exact sampling seed. Replacing both that seed and
    its parameters used to discard the very normal match the library established.
    A separate LOO-normal quantile controls base acceptance; near/far still govern
    negative supervision. No gate is widened in response to rejection.
    """
    quantile = settings.get("base_quantile", .98)
    limit = verifier.normal_threshold(quantile)
    target = settings.get("control_samples", settings["samples_per_family"])
    budget = settings.get("control_attempts", settings["attempts_per_family"])
    replay_order = rng.permutation(len(prior.library)).tolist()
    controls, distances, anchor_distances = [], [], []
    rejected = {key: 0 for key in ("invalid", "normal_gate", "anchor_match")}
    attempted, replayed = 0, 0
    for attempted in range(1, budget + 1):
        if attempted <= len(replay_order):
            params = copy.deepcopy(prior.library[replay_order[attempted - 1]])
            seed, source = int(params["match_seed"]), "matched_replay"
        else:
            params = prior.draw(rng)
            seed, source = int(rng.integers(0, 2**32)), "resampled"
        base = prior.sample(params, seed)
        if base is None:
            rejected["invalid"] += 1
            continue
        description = describe(base)
        distance = verifier.distance(description)
        anchor_distance = verifier.match_distance(description, normal_descriptors[params["anchor"]])
        distances.append(distance)
        anchor_distances.append(anchor_distance)
        if settings["require_normal_base"]:
            if distance > limit:
                rejected["normal_gate"] += 1
                continue
            if anchor_distance > prior.settings["match_tolerance"]:
                rejected["anchor_match"] += 1
                continue
        controls.append({"params": params, "seed": seed, "graph": base, "description": description,
                         "distance": distance, "anchor_distance": anchor_distance, "source": source})
        replayed += int(source == "matched_replay")
        if len(controls) == target:
            break
    report = {"attempts": attempted, "accepted": len(controls), "matched_replays": replayed,
              "require_normal_base": settings["require_normal_base"], "base_quantile": quantile,
              "base_threshold": limit, "rejected": rejected,
              "distance": _distance_summary(distances), "anchor_distance": _distance_summary(anchor_distances)}
    print(f"graft_ad controls={json.dumps(report, sort_keys=True, allow_nan=False)}", flush=True)
    if not controls:
        raise ValueError(
            "GRAFT-AD found no usable normal controls before intervention screening. "
            f"controls={json.dumps(report, sort_keys=True, allow_nan=False)}; "
            "the prior lacks accepted normal coverage under the configured base gate and anchor tolerance. "
            "Inspect the feature schema and prior-match logs; negative-family quotas have not been applied.")
    return controls, report


def _allowed_easy(moderate_count, cap, fraction):
    return cap if fraction == 1 else min(cap, math.floor(moderate_count * fraction / (1 - fraction)))


def generate_pool(prior, verifier, normal_descriptors, describe, settings, rng):
    pool, metadata, reports = {}, [], []
    families = available_families(settings["families"], prior.schema.degree_mode)
    disabled = [family for family in settings["families"] if family not in families]
    if disabled:
        print(f"graft_ad disabled families for degree-derived features: {disabled}", flush=True)
    if not families:
        raise ValueError("GRAFT-AD has no intervention families enabled for the fitted feature schema")
    controls, control_report = _normal_controls(prior, verifier, normal_descriptors, describe, settings, rng)
    for family in families:
        moderate, easy = [], []
        seen = set()
        counts = {key: 0 for key in ("invalid", "unchanged", "ambiguous", "not_farther", "too_far",
                                    "duplicate", "easy_quota")}
        candidate_distances, gains = [], []
        target, fraction = settings["samples_per_family"], settings["easy_fraction"]
        easy_cap = int(math.floor(target * fraction))
        attempts = 0
        for attempts in range(1, settings["attempts_per_family"] + 1):
            control = controls[int(rng.integers(len(controls)))]
            params, seed, base = control["params"], control["seed"], control["graph"]
            strength = float(rng.choice(settings["strengths"]))
            changed, components = intervene(params, family, strength, rng, prior.schema.degree_mode)
            candidate = prior.sample(changed, seed)
            if candidate is None:
                counts["invalid"] += 1
                continue
            description = describe(candidate)
            displacement = verifier.match_distance(description, control["description"])
            # Also reject exact no-ops (e.g. permuting a constant feature matrix).
            unchanged = (np.array_equal(base.edges, candidate.edges) and np.array_equal(base.x, candidate.x))
            if unchanged or displacement <= verifier.settings["scale_epsilon"]:
                counts["unchanged"] += 1
                continue
            if displacement > settings["max_descriptor_displacement"]:
                counts["too_far"] += 1
                continue
            distance, weight = verifier.weight(description)
            gain = distance - control["distance"]
            candidate_distances.append(distance)
            gains.append(gain)
            if weight <= 0.:
                counts["ambiguous"] += 1
                continue
            if gain <= verifier.settings["scale_epsilon"]:
                counts["not_farther"] += 1
                continue
            candidate_hash = hashlib.sha256(candidate.x.astype("<f4").tobytes()
                                            + candidate.edges.astype("<i8").tobytes()).hexdigest()
            if candidate_hash in seen:
                counts["duplicate"] += 1
                continue
            seen.add(candidate_hash)
            record = {
                "family": family, "components": components, "anchor": params["anchor"],
                "seed": seed, "strength": strength, "distance": distance,
                "base_distance": control["distance"], "base_anchor_distance": control["anchor_distance"],
                "control_source": control["source"], "displacement": displacement, "distance_gain": gain,
                "weight": weight, "easy": distance >= verifier.far,
                "parameters": changed, "base_parameters": params,
                "sha256": candidate_hash,
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
            allowed = _allowed_easy(len(moderate), easy_cap, fraction)
            if len(moderate) + min(len(easy), allowed) >= target:
                break
        allowed = _allowed_easy(len(moderate), easy_cap, fraction)
        selected_easy = easy[:allowed]
        selected = moderate[:target - len(selected_easy)] + selected_easy
        usable = len(selected) >= settings["min_samples_per_family"]
        report = {"family": family, "attempts": attempts, "accepted": len(selected) if usable else 0,
                  "retained_before_minimum": len(selected), "eligible_moderate": len(moderate),
                  "eligible_easy": len(easy), "easy": len(selected_easy) if usable else 0,
                  "rejected": counts, "candidate_distance": _distance_summary(candidate_distances),
                  "distance_gain": _distance_summary(gains)}
        reports.append(report)
        if usable:
            pool[family] = selected
            metadata.extend(record for _, record in selected)
        print(f"graft_ad pool={json.dumps(report, sort_keys=True, allow_nan=False)}", flush=True)
    if not pool:
        raise ValueError(
            "GRAFT-AD generated no usable screened family after normal-control validation. "
            f"near={verifier.near:.6g} far={verifier.far:.6g}; "
            f"families={json.dumps(reports, sort_keys=True, allow_nan=False)}")
    digest = hashlib.sha256()
    for family, items in pool.items():
        for graph, record in items:
            digest.update(family.encode())
            digest.update(graph.x.astype("<f4").tobytes())
            digest.update(graph.edges.astype("<i8").tobytes())
            digest.update(json.dumps(record, sort_keys=True, allow_nan=False).encode())
    return pool, {"screening_version": 2, "controls": control_report,
                  "families": list(pool), "disabled_families": disabled,
                  "reports": reports, "samples": metadata, "sha256": digest.hexdigest()}
