"""Training-only feature schema and a small random graph-aware neural SCM."""
import numpy as np

from .graph import GraphSample, adjacency, degree_features


class FeatureSchema:
    def __init__(self, settings, prior_settings):
        self.settings = dict(settings)
        self.prior_settings = dict(prior_settings)

    def fit(self, graphs, rng):
        self.width = graphs[0].x.shape[1]
        declared = self.settings["one_hot_groups"]
        categorical = self.settings["categorical_columns"]
        if any(c >= self.width for group in declared for c in group) or any(
                c >= self.width for c in categorical):
            raise ValueError("GRAFT-AD feature schema index exceeds input_dim")
        matches_degree = self.width == 2 and all(
            np.allclose(g.x, degree_features(g.n, g.edges), rtol=1e-5, atol=1e-6) for g in graphs)
        mode = self.settings["mode"]
        self.degree_mode = mode == "degree" or (mode == "auto" and matches_degree)
        if self.degree_mode and (not matches_degree or declared or categorical):
            raise ValueError("GRAFT-AD degree mode requires the shared two-column degree encoding and no schema overrides")
        # Priority reservoir: uniform node sampling across the supplied training graphs.
        capacity = self.settings["reference_nodes"]
        rows = np.empty((0, self.width), dtype=np.float32)
        keys = np.empty(0)
        binary = np.ones(self.width, dtype=bool)
        low, high = np.full(self.width, np.inf), np.full(self.width, -np.inf)
        feature_sum, node_count = np.zeros(self.width), 0
        categorical_values = {column: set() for column in categorical}
        for graph in graphs:
            x = graph.x
            binary &= np.all((x == 0) | (x == 1), axis=0)
            low, high = np.minimum(low, x.min(axis=0)), np.maximum(high, x.max(axis=0))
            feature_sum += x.sum(axis=0, dtype=np.float64)
            node_count += graph.n
            for column in categorical:
                categorical_values[column].update(float(v) for v in np.unique(x[:, column]))
            new_keys = rng.random(graph.n)
            selected = np.argpartition(new_keys, min(capacity, graph.n) - 1)[:capacity]
            keys = np.concatenate((keys, new_keys[selected]))
            rows = np.concatenate((rows, x[selected]), axis=0)
            if len(rows) > capacity:
                keep = np.argpartition(keys, capacity - 1)[:capacity]
                rows, keys = rows[keep], keys[keep]
        self.reference = rows
        self.mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = rows.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.
        self.constant = low == high
        self.constant_values = low.astype(np.float32)
        self.binary = binary
        self.binary_probabilities = feature_sum / node_count
        self.groups = [list(g) for g in declared]
        if not self.degree_mode and not self.groups and self.settings["infer_one_hot_tail"]:
            # TU datasets concatenate continuous attributes and a one-hot node-label tail.
            for start in range(self.width - 2, -1, -1):
                group = list(range(start, self.width))
                if binary[group].all() and all(np.all(g.x[:, group].sum(axis=1) == 1) for g in graphs):
                    self.groups = [group]
                    break
        for group in self.groups:
            if not binary[group].all() or not all(np.all(g.x[:, group].sum(axis=1) == 1) for g in graphs):
                raise ValueError(f"GRAFT-AD declared group {group} is not one-hot in training normals")
        used = {column for group in self.groups for column in group}
        if used.intersection(categorical):
            raise ValueError("GRAFT-AD inferred one-hot tail overlaps categorical_columns; declare explicit groups")
        self.categorical = {}
        for column, values in categorical_values.items():
            values = np.array(sorted(values), dtype=np.float32)
            # Full-training support, smoothed reservoir frequencies. No held-out vocabulary.
            counts = np.array([(rows[:, column] == v).sum() for v in values], dtype=np.float64) + .5
            self.categorical[str(column)] = {"values": values, "probabilities": counts / counts.sum()}
        self.sorted_values = np.sort(rows, axis=0)
        self.group_probabilities = []
        for group in self.groups:
            # Never invent training-unseen categories in the unperturbed prior.
            counts = feature_sum[group]
            self.group_probabilities.append(counts / counts.sum())
        return self

    def generate(self, n, edges, params, rng):
        if self.degree_mode:
            return degree_features(n, edges)
        width = self.prior_settings["scm_hidden_dim"]
        mechanism = np.random.default_rng(params["attribute_seed"])
        graph = GraphSample(np.empty((n, 0)), edges)
        matrix, degree = adjacency(graph), graph.degrees()
        hidden = rng.standard_normal((n, width)).astype(np.float32)
        # Structural input is part of the local adaptation; PageRank is not used.
        log_degree = np.log1p(degree)
        hidden[:, 0] += ((log_degree - log_degree.mean()) / max(log_degree.std(), 1e-6)).astype(np.float32)
        for _ in range(self.prior_settings["scm_layers"]):
            local_w = mechanism.standard_normal((width, width)).astype(np.float32) / np.sqrt(width)
            graph_w = mechanism.standard_normal((width, width)).astype(np.float32) / np.sqrt(width)
            bias = mechanism.normal(0, .2, width).astype(np.float32)
            neighborhood = (matrix @ hidden) / np.maximum(degree[:, None], 1)
            local, propagated = hidden @ local_w, neighborhood @ graph_w
            graph_neurons = mechanism.random(width) < params["graph_mix"]
            hidden = np.tanh(np.where(graph_neurons, propagated, local) + bias).astype(np.float32)
        output_w = mechanism.standard_normal((width, self.width)).astype(np.float32) / np.sqrt(width)
        scores = hidden @ output_w + .05 * rng.standard_normal((n, self.width))
        order = np.argsort(np.argsort(scores, axis=0, kind="stable"), axis=0, kind="stable")
        uniform = (order + .5) / n
        shift = params.get("attribute_shift", 0.)
        if shift:
            # Mechanism shifts keep continuous values inside training-observed support.
            selected = mechanism.choice(self.width, size=max(1, self.width // 3), replace=False)
            uniform[:, selected] = np.clip(uniform[:, selected] + .2 * shift, 0., 1.)
        position = uniform * (len(self.sorted_values) - 1)
        lo, hi = np.floor(position).astype(np.int64), np.ceil(position).astype(np.int64)
        columns = np.arange(self.width)[None, :]
        x = ((1 - (position - lo)) * self.sorted_values[lo, columns]
             + (position - lo) * self.sorted_values[hi, columns])
        for column in np.flatnonzero(self.binary):
            probability = float(self.binary_probabilities[column])
            x[:, column] = uniform[:, column] > 1 - probability
        for column, specification in self.categorical.items():
            column = int(column)
            indices = np.searchsorted(np.cumsum(specification["probabilities"]), uniform[:, column], side="right")
            x[:, column] = specification["values"][np.minimum(indices, len(specification["values"]) - 1)]
        for group, probabilities in zip(self.groups, self.group_probabilities):
            log_probabilities = np.where(probabilities > 0, np.log(np.maximum(probabilities, 1e-300)), -np.inf)
            logits = scores[:, group] + log_probabilities[None, :]
            if shift:
                logits += shift * mechanism.standard_normal(len(group))[None, :]
            # Gumbel noise avoids deterministic collapse to the most common node category.
            logits -= np.log(-np.log(rng.uniform(1e-6, 1 - 1e-6, logits.shape)))
            x[:, group] = 0.
            x[np.arange(n), np.asarray(group)[logits.argmax(axis=1)]] = 1.
        # One-hot support is preserved by zero probabilities above; other constants stay fixed.
        grouped = {column for group in self.groups for column in group}
        constant = [c for c in np.flatnonzero(self.constant) if c not in grouped]
        x[:, constant] = self.constant_values[constant]
        fraction = params.get("coupling_fraction", 0.)
        if fraction:
            count = min(n, max(2, int(round(n * fraction))))
            selected = rng.choice(n, count, replace=False)
            x[selected] = x[rng.permutation(selected)].copy()
        x = x.astype(np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError("GRAFT-AD generated non-finite node features")
        return x

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("width", "degree_mode", "reference", "mean", "scale", "constant", "constant_values",
                 "binary", "binary_probabilities", "groups", "categorical", "sorted_values", "group_probabilities")}
