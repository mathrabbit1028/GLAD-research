"""Lightweight configuration validation for the GIN alternative."""
import math
from .prior_mixture_config import validate_config as validate_proposal


def validate_config(config):
    validate_proposal(config.proposal)
    if config.proposal.mixture.min_shell_rate != 0:
        raise ValueError('The GIN alternative requires min_shell_rate=0; it uses no geometry')
    for section, keys in [('architecture', ['hidden_dim', 'layers']),
                          ('training', ['epochs', 'batch_size', 'log_every']),
                          ('sampling', ['train_per_class', 'validation_per_class'])]:
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if type(config.sampling.seed) is not int or config.sampling.seed < 0:
        raise ValueError('sampling.seed must be a nonnegative integer')
    for key in ['learning_rate', 'gradient_clip']:
        if not math.isfinite(config.training[key]) or config.training[key] <= 0:
            raise ValueError(f'training.{key} must be finite and positive')
