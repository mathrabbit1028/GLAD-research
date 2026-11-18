"""Validation for the categorical, finite-unroll Level C candidate."""
import math


def validate_config(c):
    if c.mode not in ('no_ae', 'initial_ae', 'fixed_prior', 'level_c'):
        raise ValueError('Unknown Level C comparison mode')
    integers = dict(architecture=['hidden_dim', 'layers'], prior=['components'], ae=['hidden_dim'],
                    ratio=['warmup_steps', 'steps', 'per_component', 'calibration_per_component'],
                    temperature=['steps'], mixture=['normal_steps', 'abnormal_steps', 'restarts'],
                    inner=['warmup_steps', 'steps', 'per_component'], outer=['steps', 'select_every'],
                    sampling=['meta_per_component', 'select_per_component'], limits=['max_nodes'])
    for section, names in integers.items():
        for name in names:
            if type(c[section][name]) is not int or c[section][name] < 1:
                raise ValueError(f'{section}.{name} must be a positive integer')
    positives = dict(prior=['edge_temperature', 'node_temperature'], ae=['identity_logit'],
                     ratio=['warmup_lr', 'learning_rate'], temperature=['learning_rate'],
                     mixture=['step_size', 'restart_temperature'], inner=['warmup_lr', 'learning_rate'],
                     outer=['learning_rate', 'gradient_clip'], loss=['edge_budget', 'node_budget'])
    for section, names in positives.items():
        for name in names:
            if not math.isfinite(c[section][name]) or c[section][name] <= 0:
                raise ValueError(f'{section}.{name} must be positive and finite')
    for name in ('raw', 'gap', 'reconstruction', 'edit', 'anchor', 'separation'):
        if not math.isfinite(c.loss[name]) or c.loss[name] < 0:
            raise ValueError('Loss weights must be finite and nonnegative')
    for section in ('ratio', 'inner'):
        if not 0 <= c[section].momentum < 1 or not math.isfinite(c[section].weight_decay) or c[section].weight_decay < 0:
            raise ValueError('Invalid momentum or weight decay')
    if not 0 < c.mixture.cap <= 1 or c.mixture.cap*c.prior.components < 1:
        raise ValueError('Infeasible mixture cap')
    if not 0 < c.split.fit_fraction < 1 or not 0 < c.split.meta_fraction < 1-c.split.fit_fraction:
        raise ValueError('Invalid normal-only partition')
    for seed in (c.split.seed, c.sampling.seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Seeds must be nonnegative integers')
