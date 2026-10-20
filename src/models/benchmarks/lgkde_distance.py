"""Vectorized evaluation of the official LGKDE graph-pair MMD formula."""
import torch
from torch.nn import functional as F
from ..vendor.lgkde.model import DGMMD


class VectorizedDGMMD(DGMMD):
    def _compute_distance_matrix_node_level(self, graphs, graphs_b=None):
        a, ids_a, n_a = self._compute_embeddings_node_level(graphs)
        if graphs_b is None:
            b, ids_b, n_b = a, ids_a, n_a
        else:
            b, ids_b, n_b = self._compute_embeddings_node_level(graphs_b)
        # Columns are normalized node membership indicators. Multiplication
        # computes each graph-pair's mean node kernel, including diagonal terms.
        weights_a = F.one_hot(ids_a, n_a).to(a.dtype)
        weights_b = F.one_hot(ids_b, n_b).to(b.dtype)
        weights_a = weights_a / weights_a.sum(0)
        weights_b = weights_b / weights_b.sum(0)
        aa = self._pairwise_distances(a)
        bb = aa if graphs_b is None else self._pairwise_distances(b)
        ab = aa if graphs_b is None else self._pairwise_distances_cross(a, b)
        distance = a.new_zeros((n_a, n_b))
        for bandwidth in self.bandwidths:
            kernel_aa = torch.exp(-aa / bandwidth)
            kernel_bb = kernel_aa if graphs_b is None else torch.exp(-bb / bandwidth)
            kernel_ab = kernel_aa if graphs_b is None else torch.exp(-ab / bandwidth)
            within_a = (weights_a * (kernel_aa @ weights_a)).sum(0)
            within_b = (weights_b * (kernel_bb @ weights_b)).sum(0)
            cross = weights_a.T @ kernel_ab @ weights_b
            mmd = (within_a[:, None] + within_b[None, :] - 2 * cross).clamp_min(1e-8).sqrt()
            distance = torch.maximum(distance, mmd)
        if graphs_b is None:
            # Official code computes the upper triangle and mirrors it.
            upper = torch.triu(distance, diagonal=1)
            distance = upper + upper.T
        return distance
