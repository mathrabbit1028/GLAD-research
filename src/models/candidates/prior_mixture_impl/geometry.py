"""Training-only normal geometry, represented by checkpoint-safe tensors."""
import numpy as np
import torch
from sklearn.cluster import KMeans


def fit_geometry(embeddings, settings, seed):
    x = embeddings.detach().double().cpu()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False)
    keep = scale > 1e-8
    if not keep.any() or len(x) < 2:
        raise ValueError('Insufficient distinct normal embeddings')
    scale = scale.clamp_min(1e-8)
    standardized = (x - mean) / scale
    standardized[:, ~keep] = 0
    _, singular, vh = torch.linalg.svd(standardized, full_matrices=False)
    rank = int((singular > max(float(singular[0]) * 1e-8, 1e-10)).sum())
    dimension = min(settings.dimension, rank, max(1, len(x) // 2))
    if dimension == 0:
        raise ValueError('Insufficient geometric rank')
    projection = vh[:dimension].T.contiguous()
    z = standardized @ projection
    distinct = len(torch.unique(z, dim=0))
    if distinct < 2:
        raise ValueError('Fewer than two distinct normal embeddings')
    minimum = max(settings.min_occupancy, 2 * dimension)
    regions = min(settings.regions, distinct, max(1, len(x) // minimum))
    labels = KMeans(n_clusters=regions, random_state=seed, n_init=10).fit_predict(z.numpy())
    # Merge the smallest undersupported group into its nearest surviving group.
    while len(np.unique(labels)) > 1:
        ids, counts = np.unique(labels, return_counts=True)
        small = int(ids[np.argmin(counts)])
        if counts.min() >= minimum:
            break
        center = z[labels == small].mean(0)
        others = [int(j) for j in ids if j != small]
        nearest = min(others, key=lambda j: float((center - z[labels == j].mean(0)).square().sum()))
        labels[labels == small] = nearest
    means, factors, radii, counts = [], [], [], []
    for j in np.unique(labels):
        members = z[labels == j]
        center = members.mean(0)
        residual = members - center
        covariance = residual.T @ residual / max(1, len(members) - 1)
        identity = torch.eye(dimension, dtype=torch.float64)
        covariance = ((1 - settings.shrinkage) * covariance +
                      settings.shrinkage * covariance.trace() / dimension * identity +
                      settings.floor * identity)
        factor = torch.linalg.cholesky(covariance)
        distances = torch.linalg.solve_triangular(factor, residual.T, upper=False).square().sum(0)
        radius = torch.quantile(distances, settings.initial_quantile).clamp_min(settings.floor)
        means.append(center)
        factors.append(factor)
        radii.append(radius)
        counts.append(len(members))
    return dict(mean=mean, scale=scale, keep=keep, projection=projection,
                centers=torch.stack(means), factors=torch.stack(factors),
                initial_radii=torch.stack(radii), log_scales=torch.zeros(len(means), dtype=torch.float64)), counts


def distances(embeddings, state):
    x = (embeddings.double() - state['mean']) / state['scale']
    x = x * state['keep']
    z = x @ state['projection']
    residual = z[:, None, :] - state['centers'][None, :, :]
    solved = torch.linalg.solve_triangular(state['factors'], residual.permute(1, 2, 0), upper=False)
    return solved.square().sum(1).T


def scores(squared_distances, state, preliminary=False):
    radii = state['initial_radii']
    if not preliminary:
        radii = radii * state['log_scales'].exp()
    return (squared_distances / radii).min(1).values
