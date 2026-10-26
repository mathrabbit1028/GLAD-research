"""CPU-side sparse graph preprocessing for prior generation, never query fitting."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GraphSample:
    """Finite node features and unique undirected edges with source < target."""
    x: np.ndarray
    edges: np.ndarray

    @property
    def n(self):
        return self.x.shape[0]

    def degrees(self):
        return np.bincount(self.edges.reshape(-1), minlength=self.n).astype(np.float64)


def canonical_edges(edges, n):
    edges = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("GRAFT-AD edge endpoint is outside the graph")
    pairs = np.sort(edges, axis=0).T
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0).T.copy()


def adjacency(graph):
    # scipy is already a pinned GLAD Lab dependency.
    from scipy.sparse import csr_matrix
    ends = graph.edges
    row = np.concatenate((ends[0], ends[1]))
    col = np.concatenate((ends[1], ends[0]))
    return csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)),
                      shape=(graph.n, graph.n))


def degree_features(n, edges):
    degrees = np.bincount(edges.reshape(-1), minlength=n)
    return np.column_stack((np.ones(n), np.log1p(degrees))).astype(np.float32)


def descriptors(graph, clustering_samples, edge_samples):
    """Bounded triangle sampling plus exact sparse component/degree statistics."""
    from scipy.sparse.csgraph import connected_components
    n, edges = graph.n, graph.edges
    deg = graph.degrees()
    matrix = adjacency(graph)
    _, components = connected_components(matrix, directed=False)
    largest = np.bincount(components).max() / n
    # Fixed local randomness makes descriptor evaluation independent of training RNG.
    rng = np.random.default_rng(0)
    centers = np.flatnonzero(deg >= 2)
    closed = 0
    if centers.size:
        centers = rng.choice(centers, size=clustering_samples, replace=True)
        for center in centers:
            neighbors = matrix.indices[matrix.indptr[center]:matrix.indptr[center + 1]]
            a, b = rng.choice(neighbors, size=2, replace=False)
            start, stop = matrix.indptr[a:a + 2]
            linked = matrix.indices[start:stop]
            pos = np.searchsorted(linked, b)
            closed += int(pos < len(linked) and linked[pos] == b)
    clustering = closed / clustering_samples
    if edges.shape[1]:
        selected = rng.choice(edges.shape[1], size=min(edge_samples, edges.shape[1]), replace=False)
        u, v = edges[:, selected]
        agreement = np.abs(graph.x[u].astype(np.float64) - graph.x[v]).mean(axis=0)
    else:
        agreement = np.zeros(graph.x.shape[1])
    structural = np.array([
        np.log1p(n), 2 * edges.shape[1] / max(n * (n - 1), 1),
        *np.quantile(np.log1p(deg), [0., .25, .5, .75, 1.]),
        float(np.mean(deg == 0)), largest, clustering,
    ])
    result = np.concatenate((structural, graph.x.mean(axis=0, dtype=np.float64),
                             graph.x.std(axis=0, dtype=np.float64), agreement))
    if not np.isfinite(result).all():
        raise FloatingPointError("GRAFT-AD graph descriptors are not finite")
    return result
