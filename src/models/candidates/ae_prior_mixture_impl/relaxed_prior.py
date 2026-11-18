"""Differentiable, Section-4-inspired prior for the Level C surrogate.

This is a new relaxed generator, not the pathwise derivative of the discrete
``PriorBank``. It preserves its six binary regimes, two block layers, random
correspondence, peripheral attachment, and random local/message feature maps.
Probabilities use smooth exponential saturation; peripheral counts become
independent relaxed Bernoulli edges with an expected-count target. Features
use two weighted-mean message layers and a conditional categorical head with
empirical log-frequency bias. That bias does NOT guarantee fitted marginals.

All noise, sizes, memberships, permutations, and random network weights are
independent of the fitted settings. Holding ``seed`` fixed holds them fixed.
Only graph node counts/schema are read from the reference graphs by the bank;
every fitted floating-point quantity is supplied explicitly in ``eta``.
"""
from collections import OrderedDict
import math

import torch
from torch.nn import functional as F


def fit_prior(graphs, smoothing=1e-4, variance_floor=1e-6):
    """Fit differentiable empirical settings to masked soft graphs.

    Degree is averaged per graph, matching the original bank. Attribute
    moments/frequencies are node weighted. ``smoothing`` is a pseudocount per
    category; ``variance_floor`` is added to variance before the square root.
    """
    x, adjacency, mask = graphs
    valid = mask.to(x.dtype)
    count = valid.sum()
    mean = (x * valid.unsqueeze(-1)).sum((0, 1)) / count
    variance = ((x - mean).square() * valid.unsqueeze(-1)).sum((0, 1)) / count
    frequencies = (x * valid.unsqueeze(-1)).sum((0, 1)) + smoothing
    return dict(
        mean_degree=(adjacency.sum((1, 2)) / valid.sum(1)).mean(),
        category_probs=frequencies / frequencies.sum(),
        feature_mean=mean,
        feature_scale=(variance + variance_floor).sqrt(),
    )


class RelaxedPriorBank:
    """Conditional-size bank with component-major, reproducible soft draws."""

    def __init__(self, reference_dense, components=64, seed=0,
                 edge_temperature=.67, node_temperature=.67, hidden=16):
        x, _, mask = reference_dense
        if not 1 <= components <= 64:
            raise ValueError('Relaxed prior supports 1 to 64 component regimes')
        if edge_temperature <= 0 or node_temperature <= 0:
            raise ValueError('Relaxation temperatures must be positive')
        self.sizes = mask.sum(1).detach().to('cpu', torch.long)
        if len(self.sizes) == 0 or int(self.sizes.min()) < 1:
            raise ValueError('Relaxed prior requires nonempty graphs')
        self.width, self.max_nodes = x.shape[-1], int(self.sizes.max())
        self.components, self.seed, self.hidden = int(components), int(seed), int(hidden)
        self.edge_temperature, self.node_temperature = float(edge_temperature), float(node_temperature)
        indices = torch.linspace(0, 63, self.components).round().long().tolist()
        self.specs = [dict(bits=[(i >> b) & 1 for b in range(6)]) for i in indices]
        self._cache = OrderedDict()
        self.specification = dict(
            version=1, generator='relaxed_two_layer_block_attachment',
            feature_mode='categorical', components=self.components,
            regimes=self.specs, sizes=self.sizes.tolist(), seed=self.seed,
            edge_temperature=self.edge_temperature, node_temperature=self.node_temperature,
            hidden=self.hidden, size_law='fixed_empirical_reference',
            peripheral_rule='sequential_soft_degree_independent_bernoulli',
            features='two_random_local_weighted_mean_layers_log_frequency_bias',
            marginal_calibration=False,
        )

    def _noise(self, per_component, seed, device, dtype):
        key = (int(per_component), int(seed), str(device), dtype)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if per_component < 1:
            raise ValueError('At least one sample per component is required')
        generator = torch.Generator(device='cpu')
        generator.manual_seed((self.seed * 1000003 + int(seed)) % (2**63 - 1))
        def rand(*shape):
            return torch.rand(shape, generator=generator, dtype=torch.float64)
        def normal(*shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64)
        def randint(low, high):
            return int(torch.randint(low, high, (), generator=generator))
        def uniform(low, high):
            return low + (high - low) * float(rand())
        nmax, hidden = self.max_nodes, self.hidden
        records = []
        for component in range(self.components):
            bits = self.specs[component]['bits']
            for _ in range(per_component):
                n = int(self.sizes[randint(0, len(self.sizes))])
                peripheral = uniform(*((.05, .2) if bits[3] == 0 else (.25, .5)))
                core = max(1, n - int(n * peripheral))
                groups = min(core, randint(*((1, 4) if bits[0] == 0 else (4, 9))))
                contrast = uniform(*((1.1, 3.) if bits[1] == 0 else (4., 12.)))
                spread = uniform(*((.05, .3) if bits[2] == 0 else (.6, 1.2)))
                compartments = torch.randint(randint(1, 4), (core,), generator=generator)
                affinities = []
                for layer in range(2):
                    category = -rand(groups).clamp_min(1e-12).log()
                    labels = torch.multinomial(category / category.sum(), core, replacement=True, generator=generator)
                    propensity = (-spread**2 / 2 + spread * normal(core)).exp()
                    affinity = torch.where(labels[:, None] == labels[None, :], contrast, 1.)
                    affinity = affinity * propensity[:, None] * propensity[None, :]
                    if layer == 0:
                        affinity = affinity * (compartments[:, None] == compartments[None, :])
                    affinity.fill_diagonal_(0.)
                    if layer == 1:
                        permutation = torch.randperm(core, generator=generator)
                        affinity = affinity[permutation][:, permutation]
                    mean = affinity.sum() / max(1, core * (core - 1))
                    padded = torch.zeros(nmax, nmax, dtype=torch.float64)
                    padded[:core, :core] = affinity / mean.clamp_min(1e-12)
                    affinities.append(padded)
                mix = uniform(*((0., .3) if bits[4] == 0 else (.7, 1.)))
                amplitude = uniform(*((.05, .2) if bits[5] == 0 else (.4, .9)))
                u = rand(3, nmax, nmax).clamp(1e-8, 1 - 1e-8)
                logistic = u.log() - (-u).log1p()
                logistic = logistic.triu(1) + logistic.triu(1).transpose(-1, -2)
                node_uniform = rand(nmax, self.width).clamp(1e-8, 1 - 1e-8)
                records.append(dict(
                    mask=torch.arange(nmax) < n, core=torch.tensor(core),
                    affinity=torch.stack(affinities), degree_scale=torch.tensor(uniform(.6, 1.5)),
                    logistic=logistic, attachment_fraction=rand(nmax),
                    initial=normal(nmax, hidden), degree_gate=torch.tensor(float(rand() < .5)),
                    local_weights=normal(2, hidden, hidden) / math.sqrt(hidden),
                    graph_weights=normal(2, hidden, hidden) / math.sqrt(hidden),
                    feature_gate=(rand(2, hidden) < mix).double(),
                    feature_noise=normal(2, nmax, hidden) * amplitude,
                    output_weights=normal(hidden, self.width) / math.sqrt(hidden),
                    output_noise=normal(nmax, self.width) * amplitude,
                    gumbel=-(-node_uniform.log()).log(),
                ))
        result = {}
        for name in records[0]:
            value = torch.stack([record[name] for record in records])
            result[name] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        self._cache[key] = result
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return result

    def sample(self, eta, per_component, seed, hard=False):
        """Draw with all fitted settings attached to their computation graph.

        ``hard`` thresholds/argmaxes exactly the same draws and is for evaluation
        only; it is deliberately not a straight-through gradient estimator.
        """
        degree = eta['mean_degree']
        noise = self._noise(per_component, seed, degree.device, degree.dtype)
        mask, core = noise['mask'], noise['core']
        nmax = self.max_nodes
        valid = mask.unsqueeze(1) & mask.unsqueeze(2)
        valid = valid & ~torch.eye(nmax, device=degree.device, dtype=torch.bool).unsqueeze(0)
        core_mask = torch.arange(nmax, device=degree.device)[None, :] < core[:, None]
        core_pairs = core_mask.unsqueeze(1) & core_mask.unsqueeze(2) & valid
        # Saturation avoids clipping boundaries while keeping probabilities < 1.
        density = .8 * (-torch.expm1(-degree * noise['degree_scale'] / (.8 * (core - 1).clamp_min(1))))
        probability = .95 * (-torch.expm1(-noise['affinity'] * density[:, None, None, None] / (2 * .95)))
        def relaxed(p, logistic):
            # Smoothing supplies finite logits even at structural zero affinities;
            # structural zero edges are then exactly masked below.
            p = 1e-7 + (1 - 2e-7) * p
            return torch.sigmoid((torch.logit(p) + logistic) / self.edge_temperature)
        layers = relaxed(probability, noise['logistic'][:, :2])
        # Compartment-forbidden edges are structural zeros, not tiny extra edges.
        layers = layers * (noise['affinity'] > 0)
        adjacency = (1 - (1 - layers[:, 0]) * (1 - layers[:, 1])) * core_pairs
        positions = torch.arange(nmax, device=degree.device)
        # Each step uses existing soft degree; no in-place writes break the graph.
        for node in range(int(core.min()), nmax):
            active = (core <= node) & mask[:, node]
            eligible = (positions < node)[None, :] & mask
            existing = adjacency * eligible.unsqueeze(1) * eligible.unsqueeze(2)
            attractiveness = (existing.sum(-1) + 1.) * eligible
            expected = node * (-torch.expm1(-degree / max(1, node))) * noise['attachment_fraction'][:, node]
            probability = -torch.expm1(-expected[:, None] * attractiveness / attractiveness.sum(-1, keepdim=True).clamp_min(1.))
            attached = relaxed(probability, noise['logistic'][:, 2, node]) * eligible * active[:, None]
            selector = F.one_hot(torch.tensor(node, device=degree.device), nmax).to(degree.dtype)
            addition = selector[None, :, None] * attached[:, None, :]
            adjacency = adjacency + addition + addition.transpose(-1, -2)
        # Copy one triangle so floating-point affinity multiplication cannot
        # introduce an ulp-sized asymmetry between nominally identical edges.
        upper = (adjacency * valid).triu(1)
        adjacency = upper + upper.transpose(-1, -2)
        soft_degree = adjacency.sum(-1)
        initial = noise['initial']
        first = (1 - noise['degree_gate'][:, None]) * initial[:, :, 0] + noise['degree_gate'][:, None] * soft_degree.log1p()
        features = torch.cat([first.unsqueeze(-1), initial[:, :, 1:]], -1) * mask.unsqueeze(-1)
        normalized = adjacency / (soft_degree.unsqueeze(-1) + 1.)
        for layer in range(2):
            local = torch.bmm(features, noise['local_weights'][:, layer])
            message = torch.bmm(torch.bmm(normalized, features), noise['graph_weights'][:, layer])
            gate = noise['feature_gate'][:, layer, None, :]
            features = torch.tanh((1 - gate) * local + gate * message + noise['feature_noise'][:, layer]) * mask.unsqueeze(-1)
        conditional = torch.bmm(features, noise['output_weights']) + noise['output_noise']
        logits = eta['category_probs'].log()[None, None, :] + conditional
        x = F.softmax((logits + noise['gumbel']) / self.node_temperature, -1) * mask.unsqueeze(-1)
        if hard:
            adjacency = (adjacency >= .5).to(degree.dtype) * valid
            x = F.one_hot(x.argmax(-1), self.width).to(degree.dtype) * mask.unsqueeze(-1)
        return x, adjacency, mask
