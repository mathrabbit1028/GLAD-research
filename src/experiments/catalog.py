"""Discover benchmark, candidate and dataset YAML files without importing models."""
from pathlib import Path
import re
import yaml
from ..utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def discover():
    methods, datasets = {}, {}
    for folder, role in [('benchmarks', 'benchmark'), ('models', 'candidate')]:
        for path in sorted((ROOT / 'configs' / folder).glob('*.yaml')):
            key = path.stem
            config = yaml.safe_load(path.read_text())
            if not re.fullmatch(r'[a-z][a-z0-9_]*', key):
                raise ValueError(f'Invalid model config filename: {path.name}')
            if key in methods or config.get('name') != key:
                raise ValueError(f'Model names must be unique and match filenames: {path}')
            methods[key] = dict(key=key, role=role, label=config.get('display_name', key),
                                config=config, path=str(path))
    for folder in ['dataset', 'extra/dataset']:
        for path in sorted((ROOT / 'configs' / folder).glob('*.yaml')):
            key = path.stem
            config = yaml.safe_load(path.read_text())
            if key in datasets:
                raise ValueError(f'Duplicate dataset config: {key}')
            processed = ROOT / config['data_root'] / config['name'] / 'processed/data.pt'
            datasets[key] = dict(key=key, label=config['name'], config=config, path=str(path),
                                 ready=processed.is_file(), extra=folder.startswith('extra'))
    return dict(methods=methods, datasets=datasets, defaults=load_config())


def flatten(mapping, prefix=''):
    result = {}
    for key, value in mapping.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def editable_fields(catalog, kind, key=None):
    if kind == 'model':
        config = catalog['methods'][key]['config']
        allowed = {k: v for k, v in config.items()
                   if k not in {'name', 'adapter', 'validator', 'role', 'display_name', 'description'}}
        return flatten(allowed, 'model')
    if kind == 'dataset':
        config = catalog['datasets'][key]['config']
        allowed = {k: config[k] for k in ['batch_size', 'max_nodes', 'latent_dim', 'categorical_tail_dim']
                   if k in config}
        return flatten(allowed, 'dataset')
    protocol = catalog['defaults'].protocol
    return {**flatten(protocol.dataset, 'dataset'), **flatten(protocol.evaluation, 'evaluation')}


def create_candidate(name):
    """Create an editable starter model and config; never execute either file."""
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError('Use a lowercase name beginning with a letter: letters, digits and underscores only')
    if name in discover()['methods']:
        raise ValueError(f'A method named {name} already exists')
    module = ROOT / 'src/models/candidates' / f'{name}.py'
    config = ROOT / 'configs/models' / f'{name}.yaml'
    if module.exists() or config.exists():
        raise ValueError('The target model or config file already exists')
    code = (ROOT / 'templates/model_adapter.py').read_text()
    settings = yaml.safe_load((ROOT / 'templates/model_config.yaml').read_text())
    settings.update(name=name, display_name=name.replace('_', ' ').title(),
                    adapter=f'src.models.candidates.{name}.CandidateAdapter')
    with module.open('x') as handle:
        handle.write(code)
    try:
        with config.open('x') as handle:
            yaml.safe_dump(settings, handle, sort_keys=False)
    except BaseException:
        module.unlink()
        raise
    return module, config
