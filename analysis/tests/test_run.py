import pytest
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
            "case.yaml", command, None, None, None)


def test_parse_args_res_km_option_anywhere():
    assert run.parse_args(["c.yaml", "build-grid", "--res-km", "3"]) == (
        "c.yaml", "build-grid", 3.0, None, None)
    assert run.parse_args(["--res-km", "3", "c.yaml", "regrid-obs"]) == (
        "c.yaml", "regrid-obs", 3.0, None, None)
    for bad in (["c.yaml", "build-grid", "--res-km"],
                ["c.yaml", "build-grid", "--res-km", "fine"]):
        try:
            run.parse_args(bad)
            assert False, f"{bad} should be rejected"
        except SystemExit:
            pass


def test_build_mask_is_a_command():
    assert "build-mask" in run.COMMANDS
    assert run.parse_args(["c.yaml", "build-mask"]) == (
        "c.yaml", "build-mask", None, None, None)


def test_parse_args_sources_option():
    assert run.parse_args(["c.yaml", "regrid-obs", "--sources", "mrms"]) == (
        "c.yaml", "regrid-obs", None, ("mrms",), None)
    assert run.parse_args(["c.yaml", "regrid-obs", "--sources", "stage4",
                           "mrms", "--res-km", "3"]) == (
        "c.yaml", "regrid-obs", 3.0, ("stage4", "mrms"), None)
    for bad in (["c.yaml", "regrid-obs", "--sources"],
                ["c.yaml", "regrid-obs", "--sources", "aorc"]):
        try:
            run.parse_args(bad)
            assert False, f"{bad} should be rejected"
        except SystemExit:
            pass


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
                    "list-windows", "plot-hafs"):
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


def test_parse_args_coast_km_option():
    assert run.parse_args(["c.yaml", "build-mask", "--coast-km", "50"]) == (
        "c.yaml", "build-mask", None, None, 50.0)
    assert run.parse_args(["--coast-km", "50", "c.yaml", "build-mask",
                           "--res-km", "3"]) == (
        "c.yaml", "build-mask", 3.0, None, 50.0)
    for bad in (["c.yaml", "build-mask", "--coast-km"],
                ["c.yaml", "build-mask", "--coast-km", "near"],
                ["c.yaml", "build-mask", "--coast-km", "-5"]):
        try:
            run.parse_args(bad)
            assert False, f"{bad} should be rejected"
        except SystemExit:
            pass


def test_plot_flags_are_split_off_before_the_usual_parse():
    argv, plot = run.split_plot_options(
        ["c.yaml", "plot-eyeball", "--members", "0", "3", "6", "9",
         "--windows", "5", "--cycle-every-h", "12", "--domain", "wofs",
         "--res-km", "3"])
    assert argv == ["c.yaml", "plot-eyeball", "--res-km", "3"]
    assert plot == {"members": (0, 3, 6, 9), "windows": 5,
                    "cycle_every_h": 12, "domain": "wofs"}
    assert run.split_plot_options(["c.yaml", "grid-stat"]) == (
        ["c.yaml", "grid-stat"], {})


def test_plot_flags_reject_bad_values():
    for bad in (["--domain", "tiny"], ["--domain"], ["--windows", "0"],
                ["--windows", "x"], ["--members"], ["--members", "-1"]):
        with pytest.raises(SystemExit):
            run.split_plot_options(["c.yaml", "plot-eyeball", *bad])


def test_plot_flags_only_apply_to_the_plot_commands(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["c.yaml", "grid-stat", "--domain", "wofs"])
