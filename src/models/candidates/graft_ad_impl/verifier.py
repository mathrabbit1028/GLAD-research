"""Frozen, normal-only descriptor distance gate with leave-one-out thresholds."""
import numpy as np


class DescriptorVerifier:
    def __init__(self, settings):
        self.settings = dict(settings)
        self.fitted = False

    def fit(self, descriptors):
        descriptors = np.asarray(descriptors, dtype=np.float64)
        if len(descriptors) < 2:
            raise ValueError("GRAFT-AD's verifier requires at least two training-normal graphs")
        self.center = np.median(descriptors, axis=0)
        scale = descriptors.std(axis=0)
        self.active = scale > self.settings["scale_epsilon"]
        if not self.active.any():
            raise ValueError("GRAFT-AD normal descriptors have no varying dimensions")
        self.scale = np.maximum(scale, self.settings["scale_epsilon"])
        self.references = self.transform(descriptors)
        self.k = min(self.settings["neighbors"], len(descriptors) - 1)
        chunk = self.settings["distance_chunk_size"]
        loo = []
        for start in range(0, len(self.references), chunk):
            query = self.references[start:start + chunk]
            distances = self._distances(query)
            distances[np.arange(len(query)), np.arange(start, start + len(query))] = np.inf
            loo.extend(np.partition(distances, self.k - 1, axis=1)[:, :self.k].mean(axis=1))
        self.near, self.far = (float(np.quantile(loo, self.settings[key]))
                               for key in ("near_quantile", "far_quantile"))
        if self.far - self.near <= self.settings["scale_epsilon"]:
            raise ValueError("GRAFT-AD verifier quantiles coincide; normal descriptors are too degenerate")
        self.fitted = True
        return self

    def transform(self, values):
        return ((np.asarray(values) - self.center) / self.scale)[..., self.active]

    def _distances(self, query):
        # A chunk x reference matrix, not chunk x reference x descriptor activations.
        squared = ((query * query).sum(axis=1)[:, None]
                   + (self.references * self.references).sum(axis=1)[None, :]
                   - 2 * query @ self.references.T)
        return np.sqrt(np.maximum(squared, 0.) / self.references.shape[1])

    def distance(self, descriptor):
        if not self.fitted:
            raise RuntimeError("GRAFT-AD verifier is not fitted")
        values = self._distances(self.transform(np.asarray(descriptor)[None]))[0]
        return float(np.partition(values, self.k - 1)[:self.k].mean())

    def match_distance(self, first, second):
        delta = self.transform(first) - self.transform(second)
        return float(np.sqrt(np.mean(delta * delta)))

    def weight(self, descriptor):
        distance = self.distance(descriptor)
        weight = float(np.clip((distance - self.near) / (self.far - self.near), 0., 1.))
        return distance, weight

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("center", "scale", "active", "references", "k", "near", "far")}
