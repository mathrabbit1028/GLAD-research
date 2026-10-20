"""Serializable experiment profiles and validation; no model or torch imports."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from .catalog import ROOT, discover, editable_fields
from ..utils.config import load_config, to_plain, validate_model_config
from ..utils.device import cuda_request, gpu_requests


@dataclass
class ExperimentSpec:
    version: int = 1
    name: str = 'Experiment'
    models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: list(range(5)))
    device: str = 'cuda'
    gpu_devices: list[str] = field(default_factory=list)
    max_parallel_jobs: int = 0
    min_free_gpu_memory_mb: int = 1024
    threads: int = 1
    train_only: bool = False
    python_executable: str = sys.executable
    output_dir: str = ''
    shared_overrides: dict = field(default_factory=dict)
    model_overrides: dict = field(default_factory=dict)
    dataset_overrides: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError('An experiment profile must be a JSON object')
        spec = cls(**value)
        if type(spec.version) is not int or spec.version != 1:
            raise ValueError('Unsupported experiment profile version')
        return spec


def new_spec(catalog=None):
    catalog = catalog or discover()
    defaults = catalog['defaults']
    return ExperimentSpec(models=list(defaults.suite.models), datasets=list(defaults.suite.datasets),
                          seeds=list(defaults.suite.seeds), device=defaults.device)


def _validate_overrides(values, fields, label):
    if not isinstance(values, dict):
        raise ValueError(f'{label} overrides must be a mapping')
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f'Unknown {label} parameters: {sorted(unknown)}')
    for key, value in values.items():
        # JSON-compatible data only: NaN/Infinity cannot silently enter manifests.
        json.dumps(value, allow_nan=False)
        original = fields[key]
        if key.endswith('.max_nodes'):
            if value != 'auto' and (type(value) is not int or value < 1):
                raise ValueError(f'{key} expects auto or a positive integer')
            continue
        if key.endswith('.normal_label'):
            if value != 'majority' and type(value) is not int:
                raise ValueError(f'{key} expects majority or an integer class index')
            continue
        if isinstance(original, str) and original.startswith('${'):
            continue
        if isinstance(original, float):
            valid = type(value) in {int, float}
        elif original is None:
            valid = True
        else:
            valid = type(value) is type(original)
        if not valid:
            raise ValueError(f'{key} expects {type(original).__name__}, got {type(value).__name__}')


def build_plan(spec, catalog=None, require_data=True, check_output=True):
    catalog = catalog or discover()
    for kind, choices in [('models', catalog['methods']), ('datasets', catalog['datasets'])]:
        selected = getattr(spec, kind)
        if not isinstance(selected, list) or not selected or any(not isinstance(k, str) for k in selected):
            raise ValueError(f'Select at least one {kind[:-1]}')
        if len(selected) != len(set(selected)) or set(selected) - set(choices):
            raise ValueError(f'Unknown or duplicate {kind}: {selected}')
    if (not isinstance(spec.seeds, list) or not spec.seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in spec.seeds)
            or len(spec.seeds) != len(set(spec.seeds))):
        raise ValueError('Seeds must be distinct integers from 0 to 4294967295')
    if type(spec.threads) is not int or spec.threads < 1:
        raise ValueError('Thread count must be a positive integer')
    if type(spec.train_only) is not bool:
        raise ValueError('train_only must be true or false')
    device = cuda_request(spec.device)
    devices = gpu_requests(device, spec.gpu_devices)
    if devices:
        device = devices[0]
    if type(spec.max_parallel_jobs) is not int or spec.max_parallel_jobs < 0:
        raise ValueError('Maximum parallel jobs must be a nonnegative integer (0 = one per available GPU).')
    if type(spec.min_free_gpu_memory_mb) is not int or spec.min_free_gpu_memory_mb < 1:
        raise ValueError('Minimum free GPU memory must be a positive integer in MiB.')
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise ValueError('Provide an experiment name')
    python = Path(spec.python_executable).expanduser()
    if not python.is_file():
        raise ValueError(f'Python interpreter not found: {python}')
    _validate_overrides(spec.shared_overrides, editable_fields(catalog, 'shared'), 'shared')
    for kind, attribute, available in [('model', 'model_overrides', catalog['methods']),
                                      ('dataset', 'dataset_overrides', catalog['datasets'])]:
        mapping = getattr(spec, attribute)
        if not isinstance(mapping, dict) or set(mapping) - set(available):
            raise ValueError(f'Overrides refer to an unknown {kind}')
        for key, values in mapping.items():
            _validate_overrides(values, editable_fields(catalog, kind, key), key)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', spec.name.strip()).strip('-') or 'experiment'
    output = Path(spec.output_dir).expanduser() if spec.output_dir else ROOT / 'outputs/experiments' / f'{slug}-{stamp}'
    output = (ROOT / output).resolve()
    if check_output and output.exists():
        raise ValueError(f'Choose a new output directory: {output}')
    jobs = []
    for model in spec.models:
        for dataset in spec.datasets:
            for seed in spec.seeds:
                work = output / f'{model}_{dataset}_seed{seed}'
                values = {**spec.shared_overrides, **spec.dataset_overrides.get(dataset, {}),
                          **spec.model_overrides.get(model, {})}
                overrides = [f'model={model}', f'dataset={dataset}', f'seed={seed}',
                             f'device={device}', f'output_dir={work}']
                overrides += [f'{key}={json.dumps(value, allow_nan=False)}' for key, value in values.items()]
                cfg = load_config(overrides)
                validate_model_config(cfg.model)
                cfg.dataset.data_root = str((ROOT / cfg.dataset.data_root).resolve())
                if not 0 < cfg.dataset.normal_train_fraction < 1:
                    raise ValueError('Training fraction must be between zero and one')
                if not 0 <= cfg.dataset.normal_validation_fraction < 1 - cfg.dataset.normal_train_fraction:
                    raise ValueError('Train and calibration fractions must leave normal test graphs')
                if cfg.evaluation.threshold_source == 'validation_normals' and cfg.dataset.normal_validation_fraction <= 0:
                    raise ValueError('Held-out-normal calibration needs a positive calibration fraction')
                if cfg.evaluation.threshold_source not in {'validation_normals', 'training_normals'}:
                    raise ValueError('Threshold source must be validation_normals or training_normals')
                if not 0 < cfg.evaluation.threshold_quantile < 1:
                    raise ValueError('Threshold quantile must be between zero and one')
                if cfg.evaluation.model_selection != 'final_epoch':
                    raise ValueError('Only final_epoch selection is supported')
                if type(cfg.dataset.batch_size) is not int or cfg.dataset.batch_size < 1:
                    raise ValueError('Dataset batch sizes must be positive integers')
                if cfg.dataset.max_nodes != 'auto' and (type(cfg.dataset.max_nodes) is not int or cfg.dataset.max_nodes < 1):
                    raise ValueError('max_nodes must be auto or a positive integer')
                for key, value in cfg.model.get('training', {}).items():
                    if key.endswith('epochs') and (type(value) is not int or value < 1):
                        raise ValueError(f'{model}: {key} must be a positive integer')
                    if key.endswith('learning_rate') and (type(value) not in {int, float} or value <= 0):
                        raise ValueError(f'{model}: {key} must be positive')
                processed = Path(cfg.dataset.data_root) / cfg.dataset.name / 'processed/data.pt'
                if require_data and not processed.is_file():
                    raise ValueError(f'Processed dataset missing: {processed}')
                jobs.append(dict(model=model, role=catalog['methods'][model]['role'], dataset=dataset,
                                 seed=seed, output_dir=str(work), overrides=overrides,
                                 config=to_plain(cfg), status='pending'))
    return output, jobs
