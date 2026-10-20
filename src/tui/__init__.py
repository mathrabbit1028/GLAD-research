"""SSH-friendly terminal workbench; importing this package starts no jobs."""


def main():
    import sys
    import os

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit('The terminal UI needs an interactive terminal. Use ssh -t or open it inside tmux.')
    if os.environ.get('TERM', '') in {'', 'dumb'}:
        raise SystemExit('Use a terminal with a valid TERM setting, such as xterm-256color.')
    try:
        from .app import main as open_workbench
    except ModuleNotFoundError as error:
        if error.name in {'curses', '_curses'}:
            raise SystemExit('This Python build needs curses support. Use a Linux Python with ncurses enabled.') from error
        if error.name == 'yaml':
            raise SystemExit('Install the environment: python -m pip install -r requirements.txt') from error
        raise
    open_workbench()
