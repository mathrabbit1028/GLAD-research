"""Sparse, independent adaptation of the GraphPFN v4 graph-prior construction.

Two levels of degree-corrected Poisson SBMs, a random node bijection, and
variable-degree preferential attachment. Collapsing duplicate edges yields a
simple undirected graph. Parameter fitting is descriptor matching, not PFN ICL.
"""
import copy

import numpy as np

from .graph import GraphSample, canonical_edges


def _memberships(n, blocks, rng):
    blocks = max(1, min(blocks, n))
    return rng.choice(blocks, n, p=rng.dirichlet(np.ones(blocks)))


def _sbm_edges(groups, blocks, propensities, expected_edges, mixing, rng):
    """Poisson edge events sampled in O(nodes + events), without dense adjacency."""
    parts = []
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    mass = np.bincount(inverse, weights=propensities)
    # Each group is an independent first-level graph; the second level has one group.
    budgets = rng.multinomial(expected_edges, mass / mass.sum())
    for index in range(len(unique_groups)):
        nodes = np.flatnonzero(inverse == index)
        count = budgets[index]
        if len(nodes) < 2 or count == 0:
            continue
        p = propensities[nodes] / propensities[nodes].sum()
        cross = int(rng.binomial(count, mixing))
        if cross:
            parts.append(np.stack((rng.choice(nodes, cross, p=p), rng.choice(nodes, cross, p=p))))
        labels = blocks[nodes]
        labels_unique, label_inverse = np.unique(labels, return_inverse=True)
        block_mass = np.bincount(label_inverse, weights=propensities[nodes])
        inside = rng.multinomial(count - cross, block_mass / block_mass.sum())
        for block in range(len(labels_unique)):
            members = nodes[label_inverse == block]
            if len(members) > 1 and inside[block]:
                weights = propensities[members] / propensities[members].sum()
                parts.append(np.stack((rng.choice(members, inside[block], p=weights),
                                       rng.choice(members, inside[block], p=weights))))
    return np.concatenate(parts, axis=1) if parts else np.empty((2, 0), dtype=np.int64)


class NormalGraphPrior:
    def __init__(self, settings, schema):
        self.settings, self.schema = dict(settings), schema
        self.library = []
        self.match_report = []

    def propose(self, graph, anchor, rng):
        degrees = graph.degrees()
        mean = degrees.mean()
        sigma = np.sqrt(np.log1p(degrees.var() / max(mean * mean, 1e-6)))
        return {
            "anchor": int(anchor), "n": int(graph.n), "edges": int(graph.edges.shape[1]),
            "first_graphs": int(rng.integers(1, self.settings["max_first_level_graphs"] + 1)),
            "blocks": int(rng.integers(1, self.settings["max_blocks"] + 1)),
            "mixing": float(rng.uniform(.05, .9)), "first_share": float(rng.uniform(.25, .75)),
            "degree_sigma": float(np.clip(sigma * rng.uniform(.7, 1.4), .05, 2.)),
            "peripheral_fraction": float(np.clip(np.mean(degrees <= 2) * rng.uniform(.1, .5), 0., .5)),
            "graph_mix": float(rng.uniform(.1, .9)),
            "attribute_seed": int(rng.integers(0, 2**32)),
        }

    def sample(self, params, seed):
        rng = np.random.default_rng(seed)
        n, expected = params["n"], params["edges"]
        if expected > self.settings["max_edges"]:
            raise ValueError("GRAFT-AD anchor exceeds prior.max_edges; raise the explicit generation budget")
        if n == 1 or expected == 0:
            edges = np.empty((2, 0), dtype=np.int64)
        else:
            periphery = min(n - 1, int(round(n * params["peripheral_fraction"])))
            core = n - periphery
            expected_degree = 2 * expected / n
            attachment_mean = min(expected_degree / 2, core)
            pa_budget = min(expected, int(round(periphery * attachment_mean)))
            # Poisson event counts are bounded before allocating arrays. Rejection does
            # not truncate input graphs or silently reduce their node counts.
            events = int(rng.poisson(max(expected - pa_budget, 0)))
            if events > self.settings["max_edges"]:
                return None
            first_count = int(rng.binomial(events, params["first_share"]))
            groups = _memberships(core, params["first_graphs"], rng)
            first_blocks = _memberships(core, params["blocks"], rng)
            weights = np.exp(rng.normal(0, params["degree_sigma"], core)).clip(1e-4, 1e4)
            first = _sbm_edges(groups, first_blocks, weights, first_count, params["mixing"], rng)
            # Independent second-level membership, composed with a random bijection.
            permutation = rng.permutation(core)
            second_blocks = _memberships(core, params["blocks"], rng)[permutation]
            second_weights = np.exp(rng.normal(0, params["degree_sigma"], core)).clip(1e-4, 1e4)[permutation]
            second = _sbm_edges(np.zeros(core, dtype=np.int64), second_blocks, second_weights,
                                events - first_count, params["mixing"], rng)
            edges = canonical_edges(np.concatenate((first, second), axis=1), core)
            # Degree+1 urn permits attachment even when the sampled core is edgeless.
            urn = list(range(core)) + edges.reshape(-1).tolist()
            attachment_edges = []
            remaining = pa_budget
            for node in range(core, n):
                count = min(node, remaining, int(rng.poisson(attachment_mean)))
                targets = set()
                for _ in range(4 * count):
                    if len(targets) == count:
                        break
                    targets.add(urn[int(rng.integers(len(urn)))])
                if len(targets) < count:
                    available = np.setdiff1d(np.arange(node), np.fromiter(targets, dtype=np.int64))
                    targets.update(rng.choice(available, count - len(targets), replace=False).tolist())
                for target in sorted(targets):
                    attachment_edges.append((target, node))
                    urn.extend((target, node))
                urn.append(node)
                remaining -= count
            if attachment_edges:
                edges = canonical_edges(np.concatenate((edges, np.asarray(attachment_edges).T), axis=1), n)
            if edges.shape[1] > self.settings["max_edges"]:
                return None
        x = self.schema.generate(n, edges, params, rng)
        return GraphSample(x, edges)

    def fit(self, graphs, normal_descriptors, verifier, describe, rng):
        anchors = rng.choice(len(graphs), min(len(graphs), self.settings["max_anchors"]), replace=False)
        for anchor in anchors:
            anchor = int(anchor)
            accepted, best, attempts = 0, None, 0
            for attempts in range(1, self.settings["match_attempts_per_anchor"] + 1):
                params = self.propose(graphs[anchor], anchor, rng)
                seed = int(rng.integers(0, 2**32))
                graph = self.sample(params, seed)
                if graph is None:
                    continue
                distance = verifier.match_distance(describe(graph), normal_descriptors[anchor])
                best = distance if best is None else min(best, distance)
                if distance <= self.settings["match_tolerance"]:
                    params["match_distance"], params["match_seed"] = distance, seed
                    self.library.append(params)
                    accepted += 1
                    if accepted == self.settings["matches_per_anchor"]:
                        break
            self.match_report.append({"anchor": anchor, "attempts": attempts, "accepted": accepted,
                                      "best_distance": best})
            best_text = "none" if best is None else f"{best:.6f}"
            print(f"graft_ad prior anchor={anchor} attempts={attempts} accepted={accepted} "
                  f"best_distance={best_text}", flush=True)
        if not self.library:
            raise ValueError("GRAFT-AD prior matched no training normals within prior.match_tolerance")
        print(f"graft_ad prior anchors={len(anchors)} matched_anchors="
              f"{sum(r['accepted'] > 0 for r in self.match_report)} library={len(self.library)}", flush=True)
        return self

    def draw(self, rng):
        params = copy.deepcopy(self.library[int(rng.integers(len(self.library)))])
        jitter = self.settings["parameter_jitter"]
        for key in ("mixing", "first_share", "graph_mix"):
            params[key] = float(np.clip(params[key] + rng.normal(0, jitter), .01, .99))
        params["degree_sigma"] = float(np.clip(params["degree_sigma"] * np.exp(rng.normal(0, jitter)), .01, 3.))
        return params
