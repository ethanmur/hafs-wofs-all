import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run
from run import parse_args


def test_parse_args_requires_command():
    try:
        run.parse_args(["case.yaml"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_parse_args_accepts_every_current_command():
    for command in run.COMMANDS:
        assert run.parse_args(["case.yaml", command]) == (
            "case.yaml", command, None)


def test_parse_args_res_km_option_anywhere():
    assert run.parse_args(["c.yaml", "build-grid", "--res-km", "3"]) == (
        "c.yaml", "build-grid", 3.0)
    assert run.parse_args(["--res-km", "3", "c.yaml", "regrid-obs"]) == (
        "c.yaml", "regrid-obs", 3.0)
    for bad in (["c.yaml", "build-grid", "--res-km"],
                ["c.yaml", "build-grid", "--res-km", "fine"]):
        try:
            run.parse_args(bad)
            assert False, f"{bad} should be rejected"
        except SystemExit:
            pass


def test_build_mask_is_a_command():
    assert "build-mask" in run.COMMANDS
    assert run.parse_args(["c.yaml", "build-mask"]) == ("c.yaml", "build-mask",
                                                        None)


def test_parse_args_rejects_unknown_command():
    try:
        run.parse_args(["case.yaml", "bogus"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_parse_args_rejects_removed_commands():
    # HAFS-only products, and commands waiting for their plan steps (C, D)
    for command in ("parent", "ets", "rmse", "cycles", "cycles-compare", "all",
                    "compare", "replot", "build-masks", "plot-masked",
                    "list-windows", "regrid-hafs", "plot-hafs"):
        try:
            run.parse_args(["case.yaml", command])
            assert False, f"{command} should be rejected"
        except SystemExit:
            pass


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
