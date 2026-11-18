"""Lightweight validation for the finite-update Level B candidate."""
import math
from .prior_mixture_config import validate_config as validate_proposal


def validate_config(config):
    validate_proposal(config.proposal)
    if config.mode not in ('level_b', 'frozen_gin', 'no_ae'):
        raise ValueError('mode must be level_b, frozen_gin, or no_ae')
    if config.proposal.mixture.min_shell_rate != 0:
        raise ValueError('Level B uses no geometry')
    for section, keys in {
        'architecture': ['hidden_dim', 'layers'], 'ae': ['hidden_dim'],
        'mixture': ['normal_steps', 'abnormal_steps', 'restarts'],
        'inner': ['warmup_steps', 'steps', 'per_component'],
        'outer': ['steps', 'select_every'],
        'sampling': ['train_per_component', 'meta_per_component', 'select_per_component'],
        'limits': ['max_nodes'],
    }.items():
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    for section, keys in {
        'ae': ['identity_logit'], 'mixture': ['step_size', 'restart_temperature'],
        'inner': ['warmup_lr', 'learning_rate'],
        'outer': ['learning_rate', 'gradient_clip'],
        'loss': ['edge_budget', 'node_budget'],
    }.items():
        for key in keys:
            if not math.isfinite(config[section][key]) or config[section][key] <= 0:
                raise ValueError(f'{section}.{key} must be positive and finite')
    for key in ['raw', 'gap', 'reconstruction', 'edit', 'anchor', 'separation']:
        if not math.isfinite(config.loss[key]) or config.loss[key] < 0:
            raise ValueError(f'loss.{key} must be nonnegative and finite')
    if not 0 <= config.inner.momentum < 1 or config.inner.weight_decay < 0:
        raise ValueError('Invalid inner momentum or weight decay')
    if not 0 < config.mixture.cap <= 1 or config.mixture.cap * config.proposal.prior.components < 1:
        raise ValueError('Infeasible mixture cap')
    if not 0 < config.split.fit_fraction < 1 or not 0 < config.split.meta_fraction < 1 - config.split.fit_fraction:
        raise ValueError('Invalid normal-only data fractions')
    for value in [config.split.seed, config.sampling.seed]:
        if type(value) is not int or value < 0:
            raise ValueError('Seeds must be nonnegative integers')
