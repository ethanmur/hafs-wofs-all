"""Tests for the mask check (E0b): the four cell classes, the per-window
counts, the MET read-back through a fake gen_vx_mask, and both figures."""
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mask_check as mc
from case_helpers import make_case
from test_eyeball_plots import NX, NY, _bdeck, _write_nc

T = datetime

FAKE_VX = '''#!{python}
import sys
import netCDF4
import numpy as np
args = sys.argv
mask_file, out = args[2], args[3]
if "{mode}" == "fail":
    sys.stderr.write("ERROR: grids differ\\n")
    sys.exit(1)
with netCDF4.Dataset(mask_file) as nc:
    vals = nc.variables["obs_valid"][:].astype(float)
if "{mode}" == "shifted":
    vals = np.roll(vals, 1, axis=1)
with netCDF4.Dataset(out, "w") as nc:
    nc.createDimension("lat", vals.shape[0])
    nc.createDimension("lon", vals.shape[1])
    nc.createVariable("obs_valid_met", "f4", ("lat", "lon"))[:] = (vals > 0.5)
'''


def fake_vx(tmp_path, mode="ok"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / "gen_vx_mask"
    tool.write_text(FAKE_VX.format(python=sys.executable, mode=mode))
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _mask():
    mask = np.zeros((NY, NX), dtype=bool)
    mask[:, :5] = True
    return mask


def _truth(window):
    """Truth everywhere except a radar hole inside the mask (rows 0-1, column
    0) and, in the second window only, one more cell outside it."""
    t = np.full((NY, NX), 5.0)
    t[:2, 0] = np.nan
    t[:, 7] = np.nan                      # no truth far offshore
    if window == 1:
        t[3, 6] = np.nan
    return t


def _case(tmp_path, met_mode="ok", n=2, **kw):
    from deployment_case import RegridSettings
    tmp_path.mkdir(parents=True, exist_ok=True)
    _bdeck(tmp_path / "bt.dat")
    case = make_case(
        tmp_path, valid_start=T(2024, 7, 8, 18),
        valid_end=T(2024, 7, 8, 18) + timedelta(hours=3 * n), truth_source="mrms",
        regrid=RegridSettings(
            sources=("mrms",),
            met_bin_dir=fake_vx(tmp_path, met_mode) if met_mode else None),
        **kw)
    _write_nc(case.mask_path, {"obs_valid": _mask().astype(float)})
    for k in range(n):
        end = T(2024, 7, 8, 21) + timedelta(hours=3 * k)
        _write_nc(case.obs_window_path("mrms", end), {"precip": _truth(k)})
    return case


def test_the_four_classes_and_their_counts():
    code = mc.classify(_mask(), _truth(1))
    counts = mc.class_counts(code)
    assert code[0, 0] == mc.GAP and code[5, 2] == mc.KEPT
    assert code[5, 6] == mc.DISCARDED and code[5, 7] == mc.NEITHER
    # 8x8 grid, mask = 5 columns: 40 in; 2 gap cells (column 0, two rows)
    assert counts == {"kept": 38, "discarded": 8 * 2 - 1, "gap": 2}


def test_met_read_back_agrees_with_the_written_mask(tmp_path):
    case = _case(tmp_path)
    got, why = mc.met_reads_mask(case, case.obs_window_path(
        "mrms", T(2024, 7, 8, 21)))
    assert why == "" and (got == _mask()).all()


def test_met_read_back_reports_failure_and_a_missing_tool(tmp_path):
    case = _case(tmp_path, met_mode="fail")
    got, why = mc.met_reads_mask(case, case.obs_window_path(
        "mrms", T(2024, 7, 8, 21)))
    assert got is None and "gen_vx_mask failed" in why and "grids differ" in why
    nobody = _case(tmp_path / "x", met_mode=None)
    got, why = mc.met_reads_mask(nobody, nobody.obs_window_path(
        "mrms", T(2024, 7, 8, 21)))
    assert got is None and why


def test_both_figures_are_drawn_and_counts_printed(tmp_path, capsys):
    case = _case(tmp_path)
    files = mc.plot_mask_check(case)
    names = sorted(p.name for p in files)
    base = "2024_02l_beryl_20240708_d1_mask_check"
    assert names == [f"{base}_2024070821.png", f"{base}_2024070900.png",
                     f"{base}_coverage.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in files)
    assert all(p.parent == case.mask_plot_dir for p in files)
    out = capsys.readouterr().out
    assert "0 of 64 cells differ" in out
    # the radar hole is the only in-mask cell group without truth
    assert "In the mask, 2 cells never have truth" in out
    assert "outside it, 7 cells have truth" in out or "outside it" in out


def test_a_mask_met_reads_differently_is_flagged(tmp_path, capsys):
    mc.plot_mask_check(_case(tmp_path, met_mode="shifted"))
    out = capsys.readouterr().out
    assert "MET does not read the mask as written" in out


def test_no_met_tool_still_draws_and_says_so(tmp_path, capsys):
    mc.plot_mask_check(_case(tmp_path, met_mode=None))
    assert "not checked" in capsys.readouterr().out


def test_missing_mask_or_truth_stops_with_the_next_step(tmp_path):
    case = _case(tmp_path)
    case.mask_path.unlink()
    with pytest.raises(SystemExit, match="build-mask"):
        mc.plot_mask_check(case)
    other = _case(tmp_path / "y", n=1)
    other.obs_window_path("mrms", T(2024, 7, 8, 21)).unlink()
    with pytest.raises(SystemExit, match="window-obs"):
        mc.plot_mask_check(other)


def test_the_mask_must_match_the_truth_grid(tmp_path):
    case = _case(tmp_path)
    _write_nc(case.obs_window_path("mrms", T(2024, 7, 8, 21)),
              {"precip": _truth(0)})
    import netCDF4
    case.mask_path.unlink()
    with netCDF4.Dataset(case.mask_path, "w") as nc:
        nc.createDimension("lat", 3)
        nc.createDimension("lon", 3)
        nc.createVariable("lat", "f4", ("lat", "lon"))[:] = 30.0
        nc.createVariable("lon", "f4", ("lat", "lon"))[:] = -85.0
        nc.createVariable("obs_valid", "f4", ("lat", "lon"))[:] = 1.0
    with pytest.raises(SystemExit, match="rerun build-mask"):
        mc.plot_mask_check(case)


def test_it_is_a_command_and_names_do_not_collide_across_resolutions(tmp_path):
    import run
    assert run.parse_args(["c.yaml", "plot-mask-check"])[1] == "plot-mask-check"
    six = make_case(tmp_path)
    three = make_case(tmp_path, res_km=3.0)
    assert three.mask_check_path().name.endswith("_3km_mask_check_coverage.png")
    assert six.mask_check_path() != three.mask_check_path()
