import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run
from run import parse_args


def test_parse_args_defaults_to_all():
    yaml_path, command = run.parse_args(["case.yaml"])
    assert yaml_path == "case.yaml" and command == "all"


def test_parse_args_explicit_command():
    yaml_path, command = run.parse_args(["case.yaml", "ets"])
    assert command == "ets"


def test_parse_args_rejects_unknown_command():
    try:
        run.parse_args(["case.yaml", "bogus"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_parse_args_accepts_compare():
    yaml_path, command = run.parse_args(["storms/helene_compare.yaml", "compare"])
    assert command == "compare"


def test_parse_args_accepts_rmse():
    yaml_path, command = run.parse_args(["case.yaml", "rmse"])
    assert command == "rmse"


def test_parse_args_accepts_cycles():
    yaml_path, command = parse_args(["case.yaml", "cycles"])
    assert yaml_path == "case.yaml"
    assert command == "cycles"


def test_parse_args_accepts_cycles_compare():
    yaml_path, command = parse_args(["compare.yaml", "cycles-compare"])
    assert yaml_path == "compare.yaml"
    assert command == "cycles-compare"


def _run_all():
    """Discover and run all test_* functions."""
    import inspect
    for name, func in inspect.getmembers(sys.modules[__name__]):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"✓ {name}")
            except Exception as e:
                print(f"✗ {name}: {e}")
                raise


if __name__ == "__main__":
    _run_all()
