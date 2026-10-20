"""Curses profile editor, experiment monitor and results browser for SSH terminals."""
import copy
import curses
from datetime import datetime
import json
from pathlib import Path
import queue
import re
import signal
import sys
import threading
import textwrap

import yaml

from ..experiments.catalog import ROOT, create_candidate, discover, editable_fields
from ..experiments.plan import ExperimentSpec, build_plan, new_spec
from ..experiments.runner import run_experiment
from ..utils.aggregate import aggregate, write_summary
from ..utils.latex_exporter import export_mean_std
from ..utils.device import gpu_requests

ERRORS = (ValueError, TypeError, KeyError, OSError, yaml.YAMLError)
ENTER = (10, 13, curses.KEY_ENTER)
BACK = (27, ord('q'), curses.KEY_LEFT, curses.KEY_F10)


class Workbench:
    def __init__(self, screen):
        self.screen = screen
        self.catalog = discover()
        self.spec = new_spec(self.catalog)
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.run_dir = None
        self.jobs = []
        self.gpus, self.parallel = [], 0
        self.status = 'Ready. No experiment is running.'
        self.run_error = None
        self.screen.timeout(200)
        self.screen.keypad(True)
        self.highlight = curses.A_REVERSE
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
            self.highlight |= curses.color_pair(1)
        self._cursor(False)

    @property
    def running(self):
        return self.worker is not None and self.worker.is_alive()

    def _cursor(self, visible):
        try:
            curses.curs_set(int(visible))
        except curses.error:
            pass

    def _put(self, y, x, value, attr=0):
        height, width = self.screen.getmaxyx()
        if 0 <= y < height and 0 <= x < width - 1:
            # Logs and file content cannot inject terminal control characters.
            value = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', str(value))
            try:
                self.screen.addnstr(y, x, value, width - x - 1, attr)
            except curses.error:
                pass

    def _frame(self, title, hint):
        self._poll()
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 16 or width < 60:
            self._put(0, 0, 'Enlarge terminal to at least 60 columns x 16 rows.')
            self._put(2, 0, 'Resize to continue; Esc goes back / quits.')
            self.screen.refresh()
            return False
        self._put(0, 1, 'GLAD Lab | ' + title, curses.A_BOLD)
        total = len(self.spec.models) * len(self.spec.datasets) * len(self.spec.seeds)
        self._put(1, 1, f'{self.spec.name} | {len(self.spec.models)} methods x '
                  f'{len(self.spec.datasets)} datasets x {len(self.spec.seeds)} seeds = {total:,} jobs')
        self._put(2, 1, f'Parallel jobs: {self.spec.max_parallel_jobs or "auto (one per GPU)"}')
        self._put(height - 3, 1, self.status)
        for i, line in enumerate(textwrap.wrap(hint, width - 3)[:2]):
            self._put(height - 2 + i, 1, line)
        return True

    def _key(self):
        try:
            return self.screen.get_wch()
        except curses.error:
            return -1

    @staticmethod
    def _code(key):
        return ord(key) if isinstance(key, str) else key

    def _menu(self, title, rows, hint='', on_select=None, keys=None):
        """Rows may be callable so background status and edited values stay live."""
        index = 0
        while True:
            entries = rows() if callable(rows) else rows
            index = max(0, min(index, len(entries) - 1))
            visible = self._frame(title, hint or 'Up/Down: move | Enter: select | q/Esc: back')
            if visible:
                page = max(1, self.screen.getmaxyx()[0] - 7)
                start = max(0, index - page + 1)
                for offset, (label, _) in enumerate(entries[start:start + page]):
                    self._put(3 + offset, 2, label,
                              self.highlight if start + offset == index else 0)
                if entries:
                    self._put(self.screen.getmaxyx()[0] - 4, 2, f'{index + 1}/{len(entries)}')
                self.screen.refresh()
            code = self._code(self._key())
            if code in BACK:
                return None
            if not visible:
                continue
            if code in (curses.KEY_UP, ord('k')):
                index = max(0, index - 1)
            elif code in (curses.KEY_DOWN, ord('j')):
                index = min(len(entries) - 1, index + 1)
            elif code == curses.KEY_NPAGE:
                index += max(1, self.screen.getmaxyx()[0] - 7)
            elif code == curses.KEY_PPAGE:
                index -= max(1, self.screen.getmaxyx()[0] - 7)
            elif code == curses.KEY_HOME:
                index = 0
            elif code == curses.KEY_END:
                index = len(entries) - 1
            elif keys and code in keys:
                self._safe(keys[code])
            elif entries and code in (*ENTER, ord(' ')):
                value = entries[index][1]
                if on_select:
                    self._safe(lambda: on_select(value))
                else:
                    return value

    def _prompt(self, title, initial='', description=''):
        value, cursor = str(initial), len(str(initial))
        self._cursor(True)
        try:
            while True:
                visible = self._frame(title, 'Enter: accept | Esc: cancel | Ctrl-U: clear')
                if visible:
                    self._put(3, 2, description)
                    width = self.screen.getmaxyx()[1] - 5
                    offset = max(0, cursor - width + 1)
                    self._put(5, 2, ' ' * width, self.highlight)
                    self._put(5, 2, value[offset:offset + width], self.highlight)
                    self.screen.move(5, 2 + cursor - offset)
                    self.screen.refresh()
                key = self._key()
                code = self._code(key)
                if code == 27:
                    return None
                if not visible:
                    continue
                if code in ENTER:
                    return value
                if code in (curses.KEY_BACKSPACE, 127, 8) and cursor:
                    value, cursor = value[:cursor - 1] + value[cursor:], cursor - 1
                elif code == curses.KEY_DC:
                    value = value[:cursor] + value[cursor + 1:]
                elif code == curses.KEY_LEFT:
                    cursor = max(0, cursor - 1)
                elif code == curses.KEY_RIGHT:
                    cursor = min(len(value), cursor + 1)
                elif code in (curses.KEY_HOME, 1):
                    cursor = 0
                elif code in (curses.KEY_END, 5):
                    cursor = len(value)
                elif code == 21:
                    value, cursor = '', 0
                elif isinstance(key, str) and key.isprintable():
                    value, cursor = value[:cursor] + key + value[cursor:], cursor + 1
        finally:
            self._cursor(False)

    def _view(self, title, content):
        offset = 0
        while True:
            if self._frame(title, 'Up/Down/PgUp/PgDn: scroll | Home/End | q/Esc: back'):
                height, width = self.screen.getmaxyx()
                lines = [part for line in content.splitlines()
                         for part in (textwrap.wrap(line, width - 5, replace_whitespace=False) or [''])]
                page = height - 7
                offset = max(0, min(offset, len(lines) - page))
                for i, line in enumerate(lines[offset:offset + page]):
                    self._put(3 + i, 2, line)
                self.screen.refresh()
            code = self._code(self._key())
            if code in BACK or code in ENTER:
                return
            if code in (curses.KEY_DOWN, ord('j')):
                offset += 1
            elif code in (curses.KEY_UP, ord('k')):
                offset -= 1
            elif code == curses.KEY_NPAGE:
                offset += max(1, self.screen.getmaxyx()[0] - 7)
            elif code == curses.KEY_PPAGE:
                offset -= max(1, self.screen.getmaxyx()[0] - 7)
            elif code == curses.KEY_HOME:
                offset = 0
            elif code == curses.KEY_END:
                offset = 10**9

    def _safe(self, action):
        try:
            action()
        except ERRORS as error:
            self._view('Cannot complete action', str(error))

    @staticmethod
    def _path(value):
        path = Path(value).expanduser()
        return (ROOT / path).resolve()

    def _select(self, kind):
        entries = self.catalog['methods' if kind == 'models' else 'datasets']
        selected = getattr(self.spec, kind)

        def rows():
            result = []
            for key, entry in entries.items():
                detail = entry['role'] if kind == 'models' else ('ready' if entry['ready'] else 'DATA MISSING')
                if key == 'uniform':
                    detail += ', experimental'
                result.append((f"[{'x' if key in selected else ' '}] {key} | {entry['label']} | {detail}", key))
            return result

        def toggle(key):
            selected.remove(key) if key in selected else selected.append(key)

        keys = {ord('a'): lambda: selected.__setitem__(slice(None), list(entries)),
                ord('n'): selected.clear}
        if kind == 'models':
            for shortcut, role in [('b', 'benchmark'), ('c', 'candidate')]:
                keys[ord(shortcut)] = lambda role=role: selected.__setitem__(
                    slice(None), [key for key, value in entries.items() if value['role'] == role])
        self._menu(kind.title(), rows, 'Space/Enter: toggle | a: all | n: none | '
                   + ('b: benchmarks | c: candidates | ' if kind == 'models' else '') + 'q: back',
                   on_select=toggle, keys=keys)

    def _seeds(self):
        mode = self._menu('Seeds', [('Generate from count and first seed', 'count'),
                                    ('Enter exact seed IDs', 'exact')])
        if mode is None:
            return
        if mode == 'exact':
            text = self._prompt('Exact seeds', ','.join(map(str, self.spec.seeds)), 'Comma or space separated integers')
            if text is None:
                return
            seeds = [int(item) for item in re.split(r'[,\s]+', text.strip()) if item]
        else:
            count = self._prompt('Number of seeds', len(self.spec.seeds), 'Between 1 and 10000')
            if count is None:
                return
            count = int(count)
            if not 1 <= count <= 10000:
                raise ValueError('Seed count must be between 1 and 10000')
            first = self._prompt('First seed', self.spec.seeds[0] if self.spec.seeds else 0)
            if first is None:
                return
            seeds = list(range(int(first), int(first) + count))
        if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError('Seeds must be distinct integers from 0 to 4294967295')
        self.spec.seeds = seeds

    def _parameters(self):
        kind = self._menu('Hyperparameter scope', [('Model', 'model'), ('Dataset', 'dataset'),
                                                 ('Shared protocol', 'shared')])
        if kind is None:
            return
        target = None
        if kind != 'shared':
            choices = self.spec.models if kind == 'model' else self.spec.datasets
            if not choices:
                raise ValueError(f'Select a {kind} first')
            target = self._menu('Choose ' + kind, [(key, key) for key in choices])
            if target is None:
                return
        fields = editable_fields(self.catalog, kind, target)
        overrides = (self.spec.shared_overrides if kind == 'shared' else
                     getattr(self.spec, kind + '_overrides').setdefault(target, {}))

        def edit(key):
            text = self._prompt(key, json.dumps(overrides.get(key, fields[key])),
                                'YAML value: numbers, true/false, lists; strings may be quoted')
            if text is not None:
                value = yaml.safe_load(text)
                if isinstance(fields[key], float) and isinstance(value, str):
                    value = float(value)
                json.dumps(value, allow_nan=False)
                overrides[key] = value

        self._menu(f'{kind}: {target or "normal_only"}',
                   lambda: [(f'{"*" if key in overrides else " "} {key} = '
                             + json.dumps(overrides.get(key, value)), key) for key, value in fields.items()],
                   'Enter: edit | * = override | r: reset scope | q: back', on_select=edit,
                   keys={ord('r'): overrides.clear})

    def _settings(self):
        labels = {'name': 'Experiment name', 'gpu_devices': 'GPUs (auto or IDs: 0,1)',
                  'max_parallel_jobs': 'Maximum parallel jobs (0 = automatic)',
                  'min_free_gpu_memory_mb': 'Minimum free GPU memory (MiB)',
                  'threads': 'Threads per job', 'python_executable': 'Experiment Python (absolute path)',
                  'output_dir': 'New output directory (blank = automatic)',
                  'train_only': 'Train + calibrate only (skip final evaluation)'}

        def edit(key):
            if key == 'train_only':
                self.spec.train_only = not self.spec.train_only
                return
            initial = getattr(self.spec, key)
            if key == 'gpu_devices':
                initial = ','.join(gpu_requests(self.spec.device, self.spec.gpu_devices)) or 'auto'
            value = self._prompt(labels[key], initial)
            if value is not None:
                value = value.strip()
                if key in {'threads', 'max_parallel_jobs', 'min_free_gpu_memory_mb'}:
                    value = int(value)
                    if value < (0 if key == 'max_parallel_jobs' else 1):
                        raise ValueError('Parallel jobs must be nonnegative; threads and free memory must be positive.')
                elif key == 'gpu_devices':
                    choices = [] if value.lower() in {'', 'auto', '[]'} else [
                        f'cuda:{item}' if item.isdigit() else item
                        for item in re.split(r'[,\s]+', value) if item]
                    value = gpu_requests('cuda', choices)
                    self.spec.device = 'cuda'
                setattr(self.spec, key, value)

        def rows():
            values = []
            for key, label in labels.items():
                value = getattr(self.spec, key)
                if key == 'gpu_devices':
                    value = ', '.join(gpu_requests(self.spec.device, value)) or 'auto'
                values.append((f'{label}: {value}', key))
            return values

        self._menu('Execution settings',
                   rows,
                   on_select=edit)

    def _profiles(self):
        action = self._menu('Profiles', [('Open existing JSON profile', 'open'),
                                        ('Save JSON profile', 'save'), ('New default profile', 'new')])
        if action == 'new':
            self.spec = new_spec(self.catalog)
        elif action == 'open':
            paths = sorted((ROOT / 'experiments').glob('*.json'))
            chosen = self._menu('Open profile', [(path.name, str(path)) for path in paths]
                                + [('Enter another path...', 'other')])
            if chosen == 'other':
                chosen = self._prompt('Profile path', 'experiments/')
            if not chosen:
                return
            spec = ExperimentSpec.from_dict(json.loads(self._path(chosen).read_text()))
            # A profile copied from another server may name an unavailable Python.
            # Validate its configuration while allowing that path to be edited.
            if not isinstance(spec.python_executable, str):
                raise ValueError('The experiment Python path must be a string')
            validation = ExperimentSpec.from_dict(copy.deepcopy(spec.to_dict()))
            validation.python_executable = sys.executable
            build_plan(validation, self.catalog, require_data=False, check_output=False)
            self.spec = spec
            self.status = f'Loaded {chosen}. Check Experiment Python in Execution settings.'
        elif action == 'save':
            build_plan(self.spec, self.catalog, require_data=False, check_output=False)
            chosen = self._prompt('Save profile', 'experiments/my_comparison.json')
            if not chosen:
                return
            path = self._path(chosen)
            if path.exists() and self._menu('Replace existing profile?', [('Keep existing file', False),
                                                                        ('Replace it', True)]) is not True:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.spec.to_dict(), indent=2, allow_nan=False) + '\n')
            self.status = f'Saved {path}'

    def _preview(self):
        output, jobs = build_plan(self.spec, self.catalog, require_data=False)
        missing = [key for key in self.spec.datasets if not self.catalog['datasets'][key]['ready']]
        content = dict(output=str(output), jobs=len(jobs), missing_processed_data=missing,
                       profile=self.spec.to_dict(), first_job=jobs[0]['config'])
        self._view('Plan preview (no workers started)', yaml.safe_dump(content, sort_keys=False))

    def _start(self):
        if self.running:
            self._monitor()
            return
        self._poll()
        build_plan(self.spec, self.catalog)
        frozen = ExperimentSpec.from_dict(copy.deepcopy(self.spec.to_dict()))
        self.cancel = threading.Event()
        self.jobs, self.run_dir = [], None
        self.gpus, self.parallel = [], 0
        self.run_error = None
        self.status = 'Checking GPU in the experiment interpreter...'

        def work():
            try:
                run_experiment(frozen, self.cancel, lambda event: self.events.put(copy.deepcopy(event)))
            except BaseException as error:
                self.events.put(dict(type='error', message=str(error)))
            finally:
                self.events.put(dict(type='worker_done'))

        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()
        self._monitor()

    def _poll(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            kind = event['type']
            if kind == 'prepared':
                self.run_dir, self.jobs = Path(event['output']), event['jobs']
                self.gpus, self.parallel = event['gpus'], event['max_parallel_jobs']
            elif kind == 'gpu_check':
                self.status = 'Checking GPU in the experiment interpreter...'
            elif kind == 'gpu_ready':
                self.gpus, self.parallel = event['gpus'], event['max_parallel_jobs']
                self.status = f'{len(self.gpus)} GPU(s) ready; up to {self.parallel} parallel jobs. Preparing snapshot...'
            elif kind == 'waiting_gpus':
                self.status = 'Waiting for a selected GPU to become idle; s stops the run.'
            elif kind in {'job_started', 'job_finished', 'job_updated'}:
                self.jobs[event['index']] = event['job']
                job = event['job']
                self.status = f'{job.get("gpu_device", "-")}: {job["status"]} '
                self.status += f'{job["model"]}/{job["dataset"]}/seed{job["seed"]}'
            elif kind == 'finished':
                self.status = f'{event["status"].capitalize()}: {event["output"]}'
            elif kind == 'error':
                self.run_error = event['message']
                self.status = 'Experiment error: ' + event['message']
            elif kind == 'worker_done' and self.run_dir:
                try:
                    self.jobs = json.loads((self.run_dir / 'manifest.json').read_text())['runs']
                except (OSError, ValueError, KeyError):
                    pass

    @staticmethod
    def _tail(path, lines):
        try:
            with path.open('rb') as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - 65536))
                return handle.read().decode('utf-8', errors='replace').splitlines()[-lines:]
        except OSError:
            return ['Waiting for log output...']

    def _stop(self):
        if self.running:
            self.cancel.set()
            self.status = 'Stopping all active process groups; pending jobs will be cancelled...'

    def _monitor(self):
        index, follow = 0, True
        while True:
            if self._frame('Run monitor', 'Up/Down: job | Tab: next active | f: follow active | s: stop all | q: back'):
                running = [(i, job) for i, job in enumerate(self.jobs) if job['status'] == 'running']
                if follow:
                    if not any(i == index for i, _ in running):
                        index = running[0][0] if running else index
                index = max(0, min(index, len(self.jobs) - 1))
                height = self.screen.getmaxyx()[0]
                self._put(3, 2, str(self.run_dir or ('GPU check / startup failed.' if self.run_error else
                                                  'No run directory yet.')))
                done = sum(job['status'] not in {'pending', 'running'} for job in self.jobs)
                self._put(4, 2, f'Progress: {done}/{len(self.jobs)} | Running: {len(running)}/{self.parallel} | Follow: {follow}')
                shown = running[:max(1, min(6, (height - 12) // 3))]
                for row, (i, job) in enumerate(shown):
                    self._put(5 + row, 2, f'{job.get("gpu_device", "-")} | {job["model"]}/{job["dataset"]}'
                              f'/seed{job["seed"]} | {job.get("stage", "running")}',
                              self.highlight if i == index else 0)
                selected_row = 6 + len(shown)
                log_start = selected_row + 2
                if self.jobs:
                    job = self.jobs[index]
                    self._put(selected_row, 2, f'Job {index + 1}: {job.get("gpu_device", "-")} '
                              f'{job["model"]}/{job["dataset"]}/seed{job["seed"]} [{job["status"]}]', curses.A_BOLD)
                    lines = self._tail(Path(job['output_dir']) / 'run.log', max(1, height - 3 - log_start))
                    for i, line in enumerate(lines):
                        self._put(log_start + i, 2, line)
                    if job.get('error'):
                        self._put(selected_row + 1, 2, job['error'])
                elif self.run_error:
                    lines = textwrap.wrap(self.run_error, self.screen.getmaxyx()[1] - 5)
                    for i, line in enumerate(lines[:height - 10]):
                        self._put(7 + i, 2, line)
                self.screen.refresh()
            code = self._code(self._key())
            if code in BACK:
                return
            if code == ord('s'):
                self._stop()
            elif code == ord('f'):
                follow = True
            elif code == 9:
                indices = [i for i, job in enumerate(self.jobs) if job['status'] == 'running']
                if indices:
                    index = indices[(indices.index(index) + 1) % len(indices)] if index in indices else indices[0]
                    follow = False
            elif code in (curses.KEY_UP, ord('k')):
                follow, index = False, max(0, index - 1)
            elif code in (curses.KEY_DOWN, ord('j')):
                follow, index = False, min(len(self.jobs) - 1, index + 1)

    def _results(self):
        directory = self.run_dir

        def show():
            if directory is None:
                raise ValueError('Open a run directory containing manifest.json first')
            _, summary = aggregate(directory)

            def metric(row, key):
                value, std = row[key + '_mean'], row[key + '_std']
                return 'N/A' if value is None else f'{100 * value:.2f}' + (f' +/- {100 * std:.2f}' if std is not None else '')

            lines = [str(directory), 'Metrics in percent; mean +/- sample standard deviation.', '']
            for row in summary:
                lines += [f'{row["algorithm"]} ({row["role"]}) | {row["dataset"]} | seeds '
                          f'{row["n_seeds"]}/{row["n_expected"]} | {"complete" if row["complete"] else "PARTIAL"}',
                          f'  AUROC {metric(row, "auc")} | AP {metric(row, "average_precision")} | F1 {metric(row, "f1")}', '']
            self._view('Results', '\n'.join(lines))

        while True:
            action = self._menu('Results', [('Show / refresh results', 'show'),
                                          ('Open run directory', 'open'), ('Export CSV / JSON / LaTeX', 'export')])
            if action is None:
                return
            if action == 'show':
                self._safe(show)
            elif action == 'open':
                path = self._prompt('Run directory', directory or ROOT / 'outputs/experiments')
                if path:
                    directory = self._path(path)
                    self._safe(show)
            elif action == 'export':
                def export():
                    if directory is None:
                        raise ValueError('Open a run directory first')
                    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                    path = self._prompt('New export directory', str(directory / f'export-{stamp}'))
                    if path:
                        output = write_summary(directory, self._path(path))
                        export_mean_std(output / 'runs.csv', output / 'table.tex')
                        self._view('Export complete', str(output))
                self._safe(export)

    def _candidate(self):
        name = self._prompt('New candidate ID', '', 'Lowercase letters, digits and underscores; begin with a letter')
        if name:
            module, config = create_candidate(name.strip())
            self.catalog = discover()
            self.spec.models.append(name.strip())
            self._view('Candidate created (no jobs started)', f'Edit these files:\n{module}\n{config}\n\n'
                       'Replace the starter graph-feature autoencoder with your model.')

    def _refresh(self):
        self.catalog = discover()
        self.status = 'Refreshed method and dataset configs.'

    def _shutdown(self):
        self._stop()
        while self.running:
            try:
                self._frame('Stopping experiment', 'Waiting for the coordinator to stop all active child processes...')
                self.screen.refresh()
            except curses.error:
                # The terminal can disappear on SSH disconnect; still stop workers.
                pass
            self.worker.join(timeout=0.2)

    def run(self):
        actions = [('Methods: include / exclude', lambda: self._select('models')),
                   ('Datasets: include / exclude', lambda: self._select('datasets')),
                   ('Seeds: count / exact IDs', self._seeds), ('Hyperparameters', self._parameters),
                   ('Execution settings', self._settings), ('Profiles: open / save / new', self._profiles),
                   ('Preview plan', self._preview), ('Start experiment / monitor active run', self._start),
                   ('Run monitor / stop', self._monitor), ('Results / export', self._results),
                   ('New candidate model', self._candidate), ('Refresh configs', self._refresh), ('Quit', None)]
        try:
            while True:
                chosen = self._menu('Experiment workbench', [(label, i) for i, (label, _) in enumerate(actions)])
                if chosen is None or actions[chosen][1] is None:
                    if not self.running or self._menu('Stop experiment and quit?', [('Keep running', False),
                                                                                    ('Stop and quit', True)]) is True:
                        return
                else:
                    self._safe(actions[chosen][1])
        finally:
            self._shutdown()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = {}
    for name in ('SIGTERM', 'SIGHUP'):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous[sig] = signal.signal(sig, interrupted)
    try:
        curses.wrapper(lambda screen: Workbench(screen).run())
    except KeyboardInterrupt:
        pass
    except curses.error as error:
        raise SystemExit(f'Cannot initialize terminal UI: {error}. Use an interactive SSH/tmux terminal with a valid TERM.') from error
    except ERRORS as error:
        raise SystemExit(f'Cannot open terminal workbench: {error}') from error
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
