"""Lightweight configuration checks, without importing the training stack."""
import math


def validate_config(config):
    for section in ('architecture', 'prior', 'training', 'mixture', 'geometry', 'generation'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'prior_mixture requires {section}')
        for key, value in config[section].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{section}.{key} must be finite')
    positive_ints = {
        'architecture': ['hidden_dim', 'layers'],
        'prior': ['components', 'train_per_component', 'calibration_per_component',
                  'pilot_per_component', 'max_nodes'],
        'training': ['ratio_epochs', 'radius_epochs', 'batch_size', 'log_every'],
        'mixture': ['normal_steps', 'abnormal_steps', 'restarts'],
        'geometry': ['dimension', 'regions', 'min_occupancy'],
        'generation': ['proposal_multiplier'],
    }
    for section, keys in positive_ints.items():
        for key in keys:
            if type(config[section].get(key)) is not int or config[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if not 4 <= config.prior.components <= 64:
        raise ValueError('prior.components must be in [4, 64]')
    if config.prior.feature_mode not in ('auto', 'continuous', 'categorical', 'degree'):
        raise ValueError('Unsupported prior.feature_mode')
    if config.mixture.abnormal_mode not in ('constrained', 'literal'):
        raise ValueError('mixture.abnormal_mode must be constrained or literal')
    for section, key in [('training', 'learning_rate'), ('training', 'gradient_clip'),
                         ('mixture', 'tolerance'), ('geometry', 'floor'),
                         ('geometry', 'radius_min'), ('geometry', 'radius_max'),
                         ('geometry', 'beta'), ('generation', 'target_ratio')]:
        if config[section][key] <= 0:
            raise ValueError(f'{section}.{key} must be positive')
    for section, key in [('geometry', 'shrinkage'), ('mixture', 'min_shell_rate')]:
        if not 0 <= config[section][key] <= 1:
            raise ValueError(f'{section}.{key} must be in [0, 1]')
    if not 0 <= config.geometry.synthetic_positive_fraction < 1:
        raise ValueError('geometry.synthetic_positive_fraction must be in [0, 1)')
    if not 1 / config.prior.components <= config.mixture.cap <= 1:
        raise ValueError('mixture.cap is infeasible for the configured bank')
    if not 0 < config.generation.shell_inner < config.generation.shell_outer:
        raise ValueError('Shell offsets must satisfy 0 < inner < outer')
    if config.generation.contrast < 0 or config.geometry.radius_penalty < 0:
        raise ValueError('Contrast and radius penalty must be nonnegative')
    if not config.geometry.radius_min <= 1 <= config.geometry.radius_max:
        raise ValueError('Radius bounds must contain the initial radius scale 1')
    if not 0 < config.geometry.initial_quantile < 1:
        raise ValueError('geometry.initial_quantile must be in (0, 1)')
