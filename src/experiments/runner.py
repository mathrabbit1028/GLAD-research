"""Run a frozen experiment plan in isolated subprocesses; shared by TUI and CLI."""
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import yaml
from .catalog import ROOT
from .plan import ExperimentSpec, build_plan
from ..utils.aggregate import write_summary
from ..utils.device import compute_processes, cuda_request


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _stop(process):
    _stop_many([process])


def _stop_many(processes):
    """Signal all active groups together, then enforce one shared grace period."""
    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while any(process.poll() is None for process in live) and time.monotonic() < deadline:
        time.sleep(0.05)
    for process in live:
        try:
            if process.poll() is None:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.wait()
        except ProcessLookupError:
            pass


def _check_gpu(spec, cancel):
    """Check the worker interpreter before creating output or launching training."""
    command = [spec.python_executable, '-m', 'src.utils.device', '--list-gpus',
               '--device', spec.device, '--min-free-memory-mb', str(spec.min_free_gpu_memory_mb)]
    if spec.gpu_devices:
        command += ['--devices', *spec.gpu_devices]
    if 'lgkde' in spec.models:
        command.append('--dgl')
    with tempfile.TemporaryDirectory(prefix='glad-gpu-') as cache:
        env = dict(os.environ, DGLBACKEND='pytorch', DGLDEFAULTDIR=cache)
        with subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=os.name == 'posix') as process:
            started = time.monotonic()
            try:
                while True:
                    if cancel.is_set():
                        raise RuntimeError('GPU check cancelled; no experiment jobs were started.')
                    if time.monotonic() - started > 180:
                        raise RuntimeError('GPU check timed out in the experiment interpreter; no jobs were started.')
                    try:
                        output, _ = process.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    _stop(process)
            if process.returncode:
                raise RuntimeError('GPU check failed; no experiment jobs were started.\n' + output.strip())
    try:
        details = json.loads(output.strip().splitlines()[-1])
        if not isinstance(details, dict) or not isinstance(details.get('gpus'), list) or not details['gpus']:
            raise ValueError('Missing GPU pool')
        for gpu in details['gpus']:
            if not isinstance(gpu, dict) or not {'device', 'gpu', 'uuid', 'visibility_token'} <= gpu.keys():
                raise ValueError('Missing GPU details')
    except (ValueError, IndexError) as error:
        raise RuntimeError('GPU check returned invalid details; no experiment jobs were started.\n' + output) from error
    return details


def _spawn(state, source):
    return subprocess.Popen(state['job']['commands'][state['command']], cwd=source,
                            env=state['env'], stdin=subprocess.DEVNULL, stdout=state['log'],
                            stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')


def _launch_job(index, job, gpu, spec, source, env):
    """Each worker sees only its assigned GPU as local cuda:0."""
    work = Path(job['output_dir'])
    work.mkdir()
    job['config']['device'] = 'cuda:0'
    job['overrides'] = ['device=cuda:0' if value.startswith('device=') else value
                        for value in job['overrides']]
    job.update(gpu_device=gpu['device'], gpu_uuid=gpu['uuid'],
               cuda_visible_devices=gpu['visibility_token'])
    config_file = work / 'planned_config.yaml'
    config_file.write_text(yaml.safe_dump(job['config'], sort_keys=False))
    checkpoint = work / 'checkpoints' / f"{job['model']}_{job['config']['dataset']['name']}_seed{job['seed']}.pt"
    commands = [[spec.python_executable, '-u', '-m', 'src.train', '--config-file', str(config_file)]]
    if not spec.train_only:
        commands.append([spec.python_executable, '-u', '-m', 'src.evaluate', '--checkpoint', str(checkpoint)])
    job.update(status='running', stage='training', commands=commands)
    log = (work / 'run.log').open('w')
    state = dict(index=index, job=job, gpu=gpu, log=log, command=0,
                 started=time.perf_counter(), env=dict(env, CUDA_VISIBLE_DEVICES=gpu['visibility_token'],
                                                       DGLDEFAULTDIR=str(work / 'dgl-cache')))
    try:
        log.write(f"Assigned {gpu['device']} ({gpu['gpu']}, {gpu['uuid']}); worker device cuda:0\n")
        log.flush()
        state['process'] = _spawn(state, source)
    except BaseException:
        if 'process' in state:
            _stop(state['process'])
        log.close()
        raise
    return state


def _collect_result(job, train_only, paired_splits):
    work = Path(job['output_dir'])
    cfg = yaml.safe_load((work / 'resolved_config.yaml').read_text())
    split = json.loads((work / 'split.json').read_text())
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    pair = (job['dataset'], job['seed'])
    if pair in paired_splits and paired_splits[pair] != split_hash:
        raise ValueError(f'Unpaired graph splits for {pair}')
    if not train_only:
        stem = f"{job['model']}_{cfg['dataset']['name']}_seed{job['seed']}"
        metrics = json.loads((work / 'results' / f'{stem}.json').read_text())
        if (metrics['algorithm'], metrics['dataset'], metrics['seed']) != (job['model'], cfg['dataset']['name'], job['seed']):
            raise ValueError('Result identity does not match the scheduled job')
        job['metrics'] = metrics
    paired_splits[pair] = split_hash
    job.update(config=cfg, split_sha256=split_hash,
               status='trained' if train_only else 'completed', stage='done')


def run_experiment(spec, cancel=None, on_event=None):
    spec = ExperimentSpec.from_dict(copy.deepcopy(spec.to_dict()))
    spec.device = cuda_request(spec.device)
    spec.python_executable = str(Path(spec.python_executable).expanduser().resolve())
    cancel = cancel or threading.Event()
    emit = on_event or (lambda event: None)
    output, jobs = build_plan(spec)
    emit(dict(type='gpu_check'))
    inventory = _check_gpu(spec, cancel)
    gpus = inventory['gpus']
    parallel = min(len(gpus), spec.max_parallel_jobs or len(gpus), len(jobs))
    if cancel.is_set():
        raise RuntimeError('Experiment cancelled before starting; no jobs were started.')
    emit(dict(type='gpu_ready', gpus=gpus, max_parallel_jobs=parallel,
              skipped=inventory.get('skipped', [])))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, name=spec.name, created_utc=datetime.now(timezone.utc).isoformat(),
                    status='preparing', python=spec.python_executable, experiment=spec.to_dict(),
                    gpu=gpus[0], gpu_pool=inventory, max_parallel_jobs=parallel,
                    source_sha256={}, data_sha256={}, runs=jobs)
    def save():
        temporary = output / 'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(output / 'manifest.json')
    save()
    (output / 'experiment.json').write_text(json.dumps(spec.to_dict(), indent=2) + '\n')
    emit(dict(type='prepared', output=str(output), total=len(jobs), jobs=jobs,
              gpus=gpus, max_parallel_jobs=parallel))
    active = {}
    try:
        # Child jobs import the source snapshot, so editing the workspace cannot change a running experiment.
        source = output / 'source'
        source.mkdir()
        for directory in ['src', 'configs', 'scripts', 'docs', 'reports', 'templates']:
            for path in sorted((ROOT / directory).rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                if cancel.is_set():
                    break
                relative = path.relative_to(ROOT)
                manifest['source_sha256'][str(relative)] = digest(path)
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        for path in [*ROOT.glob('requirements*.txt'), ROOT / 'README.md', ROOT / 'launch_tui.py']:
            if path.is_file():
                manifest['source_sha256'][path.name] = digest(path)
                shutil.copy2(path, source / path.name)
        for job in jobs:
            data = job['config']['dataset']
            root = Path(data['data_root']) / data['name']
            for path in [*(root / 'raw').glob('*.txt'), root / 'processed/data.pt']:
                if cancel.is_set():
                    break
                key = str(path.resolve())
                if key not in manifest['data_sha256']:
                    manifest['data_sha256'][key] = digest(path)
        env = dict(os.environ, OMP_NUM_THREADS=str(spec.threads), MKL_NUM_THREADS=str(spec.threads),
                   OPENBLAS_NUM_THREADS=str(spec.threads), PYTHONHASHSEED='0',
                   DGLBACKEND='pytorch', DGLDEFAULTDIR=str(output / 'dgl-cache'))
        if not cancel.is_set():
            freeze = subprocess.run([spec.python_executable, '-m', 'pip', 'freeze'],
                                    capture_output=True, text=True, timeout=60)
            (output / 'environment.txt').write_text(freeze.stdout + freeze.stderr)
        manifest['status'] = 'running'
        save()
        paired_splits, next_job = {}, 0
        busy, checked_at, waiting = None, -float('inf'), False

        def finished(state):
            state['log'].close()
            state['job']['seconds'] = time.perf_counter() - state['started']
            save()
            emit(dict(type='job_finished', index=state['index'], job=state['job']))

        while next_job < len(jobs) or active:
            if cancel.is_set():
                _stop_many([state['process'] for state in active.values()])
                for state in active.values():
                    state['job']['status'] = 'cancelled'
                    finished(state)
                active.clear()
                for job in jobs[next_job:]:
                    job['status'] = 'cancelled'
                break

            # Only this coordinator mutates jobs, split checks and the manifest.
            for slot, state in list(active.items()):
                code = state['process'].poll()
                if code is None:
                    continue
                job = state['job']
                try:
                    if code:
                        raise RuntimeError(f'{job["stage"]} exited with code {code}; see run.log')
                    if state['command'] + 1 < len(job['commands']):
                        if cancel.is_set():
                            continue
                        state['command'] += 1
                        state['process'] = _spawn(state, source)
                        job['stage'] = 'evaluation'
                        save()
                        emit(dict(type='job_updated', index=state['index'], job=job))
                        continue
                    _collect_result(job, spec.train_only, paired_splits)
                except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
                    _stop(state['process'])
                    job.update(status='failed', error=str(error))
                finished(state)
                del active[slot]

            if cancel.is_set():
                continue
            if next_job < len(jobs) and len(active) < parallel:
                if time.monotonic() - checked_at >= 2:
                    busy, checked_at = compute_processes(), time.monotonic()
                for gpu in gpus:
                    if next_job >= len(jobs) or len(active) >= parallel or cancel.is_set():
                        break
                    slot = gpu['device']
                    if slot in active or (busy is not None and busy.get(gpu['uuid'])):
                        continue
                    index, job = next_job, jobs[next_job]
                    next_job += 1
                    started = time.perf_counter()
                    try:
                        state = _launch_job(index, job, gpu, spec, source, env)
                        active[slot] = state
                        save()
                        emit(dict(type='job_started', index=index, job=job))
                    except (RuntimeError, OSError, ValueError) as error:
                        if slot in active:
                            raise
                        job.update(status='failed', error=str(error), seconds=time.perf_counter() - started)
                        save()
                        emit(dict(type='job_finished', index=index, job=job))
                now_waiting = not active and next_job < len(jobs)
                if now_waiting and not waiting:
                    emit(dict(type='waiting_gpus'))
                waiting = now_waiting
            if next_job < len(jobs) or active:
                cancel.wait(0.2)
        statuses = {job['status'] for job in jobs}
        manifest['status'] = ('cancelled' if cancel.is_set() else 'failed' if 'failed' in statuses else 'completed')
        save()
        if any(job['status'] == 'completed' for job in jobs):
            try:
                write_summary(output)
            except (ValueError, OSError) as error:
                manifest['summary_error'] = str(error)
        save()
        emit(dict(type='finished', output=str(output), status=manifest['status']))
        return output
    except BaseException as error:
        _stop_many([state['process'] for state in active.values()])
        for state in active.values():
            state['log'].close()
            state['job']['seconds'] = time.perf_counter() - state['started']
        manifest.update(status='cancelled' if cancel.is_set() else 'failed', error=str(error))
        for job in jobs:
            if job['status'] in {'running', 'pending'}:
                job['status'] = 'cancelled' if cancel.is_set() else 'interrupted'
        save()
        raise
