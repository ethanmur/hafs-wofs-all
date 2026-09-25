"""Mirror a run's console output into a log file.

A job on a compute node loses its stdout when the allocation ends unless the
scheduler was told to keep it, and an interactive `srun` keeps nothing at all.
Every command therefore writes its own timestamped log beside the case output
as well as printing to the terminal.

Note that only this process's output is captured. MET is invoked with its
output captured in-process, so its messages reach the log through the normal
prints; a subprocess that wrote straight to the inherited terminal would not.
"""

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _Tee:
    """Write-through to the real stream and the log file."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, text):
        self._stream.write(text)
        self._fh.write(text)
        self._fh.flush()      # a killed job still leaves a usable log
        return len(text)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def log_path(case, command, when=None):
    """<log_dir or out_dir/logs>/<case>_<command>_<stamp>.log"""
    when = when or datetime.now()
    base = getattr(case, "log_dir", None) or Path(case.out_dir) / "logs"
    slug = getattr(case, "case_slug", None) or "case"
    return Path(base) / f"{slug}_{command}_{when:%Y%m%dT%H%M%S}.log"


@contextmanager
def tee(case, command):
    """Duplicate stdout and stderr into a per-run log for the duration.

    A log that cannot be opened is a warning, never a failure: losing the
    record of a run is not a reason to lose the run.
    """
    path = log_path(case, command)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "w")
    except OSError as err:
        print(f"WARNING: no log file ({path}): {err}", flush=True)
        yield None
        return
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(saved_out, fh), _Tee(saved_err, fh)
    started = datetime.now()
    try:
        print(f"# {command}  started {started:%Y-%m-%d %H:%M:%S}")
        print(f"# log {path}")
        yield path
    finally:
        elapsed = (datetime.now() - started).total_seconds()
        try:
            print(f"# {command}  finished in {elapsed / 60:.1f} min")
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
            fh.close()
