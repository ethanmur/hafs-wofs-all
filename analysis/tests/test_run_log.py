"""Unit tests for the per-run log tee.

Run directly:   python3 analysis/tests/test_run_log.py
Or via pytest:  pytest analysis/tests/test_run_log.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_log


class _Case:
    def __init__(self, out_dir, slug="beryl", log_dir=None):
        self.out_dir = Path(out_dir)
        self.case_slug = slug
        self.log_dir = log_dir


def test_log_path_defaults_under_out_dir():
    case = _Case("/out")
    path = run_log.log_path(case, "regrid-obs", datetime(2024, 7, 8, 13, 5, 9))
    assert path == Path("/out/logs/beryl_regrid-obs_20240708T130509.log")


def test_log_path_honours_log_dir_override():
    case = _Case("/out", log_dir="/scratch/logs")
    path = run_log.log_path(case, "build-grid", datetime(2024, 7, 8))
    assert path.parent == Path("/scratch/logs")


def test_tee_writes_stdout_and_stderr_to_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        case = _Case(tmp)
        with run_log.tee(case, "regrid-obs") as path:
            print("regridded 2024070800")
            print("a warning", file=sys.stderr)
        text = Path(path).read_text()
        assert "regridded 2024070800" in text
        assert "a warning" in text
        assert "# regrid-obs  started" in text
        assert "finished in" in text
        # the real streams are restored afterwards
        assert not isinstance(sys.stdout, run_log._Tee)


def test_tee_restores_streams_after_an_exception():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with run_log.tee(_Case(tmp), "regrid-obs") as path:
                print("before the failure")
                raise RuntimeError("MET failed")
        except RuntimeError:
            pass
        assert not isinstance(sys.stdout, run_log._Tee)
        # whatever ran before the failure is still on disk
        assert "before the failure" in Path(path).read_text()


def test_tee_survives_an_unwritable_log_dir():
    case = _Case("/proc/nonexistent-and-unwritable")
    with run_log.tee(case, "regrid-obs") as path:
        print("the run must not fail because the log could not open")
    assert path is None
    assert not isinstance(sys.stdout, run_log._Tee)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
