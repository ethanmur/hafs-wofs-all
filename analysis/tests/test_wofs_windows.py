"""Tests for the WoFS 1 h and 3 h accumulations: step selection, the native
sums, the staged MET input, the MET call itself (a stand-in regrid_data_plane),
and a full run on a synthetic archive."""
import csv
import stat
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import deployment_case as dc
import met_regrid
import verification_grid as vg
import wofs_windows as ww
from case_helpers import make_case
from test_window_manifest import GRID, _write_wofs

T = datetime


# --- steps and windows ------------------------------------------------------

def test_window_steps_for_3h_1h_and_a_half_hour_init():
    init = T(2024, 9, 27, 18)
    w1, w2 = aw.clock_windows(init, T(2024, 9, 28, 0))
    assert ww.window_steps(init, w1) == list(range(1, 37))
    assert ww.window_steps(init, w2) == list(range(37, 73))
    half = T(2024, 9, 27, 17, 30)
    assert ww.window_steps(half, w1) == list(range(7, 43))
    hours = ww.hour_windows(w1)
    assert [ww.window_steps(init, h) for h in hours] == [
        list(range(1, 13)), list(range(13, 25)), list(range(25, 37))]


def test_hour_windows_tile_the_window():
    w = aw.Window(T(2024, 9, 27, 18), T(2024, 9, 27, 21))
    hs = ww.hour_windows(w)
    assert [(h.start.hour, h.end.hour) for h in hs] == [(18, 19), (19, 20),
                                                        (20, 21)]
    assert hs[0].start == w.start and hs[-1].end == w.end


def test_member_names_and_fields_line_up_one_to_one():
    assert ww.member_names(3) == ["rain_mm_m00", "rain_mm_m01", "rain_mm_m02"]
    assert ww.member_fields(2) == ['name="rain_mm"; level="(0,*,*)";',
                                   'name="rain_mm"; level="(1,*,*)";']


# --- the native sums --------------------------------------------------------

def _files(steps):
    return {s: s for s in steps}


def test_accumulate_gives_three_hours_and_their_total_in_mm():
    init = T(2024, 9, 27, 18)
    win = aw.Window(init, init + timedelta(hours=3))
    loads = []

    def loader(step):
        loads.append(step)
        return np.full((2, 3, 3), 0.001 * step)

    hours, total, nan = ww.accumulate(_files(range(1, 37)), init, win, loader)
    assert len(hours) == 3 and nan == 0
    assert np.allclose(hours[0], 0.001 * sum(range(1, 13)) * 25.4)
    assert np.allclose(hours[2], 0.001 * sum(range(25, 37)) * 25.4)
    assert np.allclose(total, sum(hours))
    assert np.allclose(total, 0.001 * sum(range(1, 37)) * 25.4)
    assert sorted(loads) == list(range(1, 37))          # each file read once


def test_each_member_keeps_its_own_total():
    init = T(2024, 9, 27, 18)
    win = aw.Window(init, init + timedelta(hours=1))
    field = np.stack([np.full((2, 2), 0.1), np.full((2, 2), 0.3)])
    hours, total, _ = ww.accumulate(_files(range(1, 13)), init, win,
                                    lambda s: field)
    assert np.allclose(total[0], 12 * 0.1 * 25.4)
    assert np.allclose(total[1], 12 * 0.3 * 25.4)


def test_a_missing_step_means_nothing_is_summed():
    init = T(2024, 9, 27, 18)
    win = aw.Window(init, init + timedelta(hours=3))
    with pytest.raises(FileNotFoundError) as err:
        ww.accumulate(_files(set(range(1, 37)) - {20}), init, win,
                      lambda s: np.zeros((1, 2, 2)))
    assert "not complete" in str(err.value)


def test_nan_in_a_step_propagates_to_the_hour_and_the_total():
    init = T(2024, 9, 27, 18)
    win = aw.Window(init, init + timedelta(hours=3))

    def loader(step):
        f = np.ones((1, 2, 2))
        if step == 5:
            f[0, 0, 0] = np.nan
        return f

    hours, total, nan = ww.accumulate(_files(range(1, 37)), init, win, loader)
    assert np.isnan(hours[0][0, 0, 0]) and np.isfinite(hours[1][0, 0, 0])
    assert np.isnan(total[0, 0, 0]) and nan == 1


def test_block_mean_averages_2x2_blocks_and_keeps_the_mean():
    f = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    c = ww.block_mean(f, 2)
    assert c.shape == (2, 2, 2) and c[0, 0, 0] == f[0, :2, :2].mean()
    assert np.isclose(c.mean(), f.mean())
    assert ww.block_mean(f, 1) is f
    f[0, 0, 0] = np.nan
    assert np.isnan(ww.block_mean(f, 2)[0, 0, 0])
    with pytest.raises(ValueError):
        ww.block_mean(np.ones((1, 5, 4)), 2)


# --- the staged input -------------------------------------------------------

def test_staged_input_is_laid_out_like_a_wofs_file_with_its_grid_attributes(
        tmp_path):
    import netCDF4
    attrs = ww.native_grid_attrs(dict(GRID, Projection="Lambert Conformal",
                                      hemisphere="N", cape="ignore me"))
    assert "Projection" in attrs and "cape" not in attrs
    data = np.random.default_rng(3).random((3, 4, 4))
    path = ww.write_native_input(tmp_path / "s" / "in.nc", data, attrs)
    with netCDF4.Dataset(path) as nc:
        assert nc.variables[ww.VAR].dimensions == ("ne", "lat", "lon")
        assert set(nc.variables) == {ww.VAR}            # no coordinate arrays
        assert nc.Projection == "Lambert Conformal" and nc.nx == "4"
        assert nc.variables[ww.VAR].units == "mm"
        assert np.allclose(nc.variables[ww.VAR][:], data, atol=1e-5)


# --- a stand-in for regrid_data_plane --------------------------------------

FAKE_TOOL = '''#!{python}
import re, sys
import netCDF4
import numpy as np
MODE = "{mode}"
args = sys.argv[1:]
inp, grid, out = args[:3]
fields, names, i = [], None, 3
while i < len(args):
    if args[i] == "-field":
        fields.append(args[i + 1]); i += 2
    elif args[i] == "-name":
        names = args[i + 1].split(","); i += 2
    elif args[i] in ("-method", "-width", "-vld_thresh", "-v"):
        i += 2
    else:
        i += 1
if MODE == "fail":
    print("ERROR: could not read the input", file=sys.stderr)
    sys.exit(1)
assert len(fields) == len(names)
nx, ny = int(grid.split()[1]), int(grid.split()[2])
with netCDF4.Dataset(inp) as nc:
    data = np.ma.filled(nc.variables["rain_mm"][:], np.nan)
f = data.shape[2] // nx
with netCDF4.Dataset(out, "w") as nc:
    nc.createDimension("lat", ny); nc.createDimension("lon", nx)
    nc.createVariable("lat", "f4", ("lat", "lon"))[:] = 30.0
    nc.createVariable("lon", "f4", ("lat", "lon"))[:] = -85.0
    for spec, name in zip(fields, names):
        m = int(re.search(r"\\((\\d+),", spec).group(1))
        coarse = data[m].reshape(ny, f, nx, f).mean(axis=(1, 3))
        if MODE == "scaled":
            coarse = coarse * 1.1
        v = nc.createVariable(name, "f4", ("lat", "lon"), fill_value=-9999.0)
        v[:] = -9999.0 if MODE == "missing" else coarse
'''


def fake_met(tmp_path, mode="exact"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / "regrid_data_plane"
    tool.write_text(FAKE_TOOL.format(python=sys.executable, mode=mode))
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _cfg(tmp_path, bin_dir, ne=2):
    spec = vg.wofs_grid_spec(GRID, 6.0, name="t")
    return met_regrid.RegridConfig(
        cache_dir=tmp_path / "out", grid_spec=spec.met_spec, grid_name="t",
        res_label="6km", met_bin_dir=bin_dir)


def test_one_call_regrids_every_member_and_the_file_reads_back(tmp_path):
    cfg = _cfg(tmp_path, fake_met(tmp_path))
    data = np.random.default_rng(4).random((3, 4, 4)) * 30
    out = cfg.cache_dir / "2024092721_i202409271800_wofs_6km.nc"
    tool = met_regrid.met_tool("regrid_data_plane", cfg.met_bin_dir)
    ww.regrid_one(tool, cfg, data, {"nx": "4", "ny": "4"}, out,
                  T(2024, 9, 27, 18), T(2024, 9, 27, 21), 3)
    got = ww.read_window(out, ne=3)
    assert got.shape == (3, 2, 2)
    assert np.allclose(got, ww.block_mean(data, 2), atol=1e-4)
    import netCDF4
    with netCDF4.Dataset(out) as nc:
        assert nc.accumulation_hours == 3 and nc.members == 3
        assert nc.init_time == "20240927_180000"
        assert nc.valid_time == "20240927_210000"
    assert not list(cfg.staging_dir.glob("*_native.nc"))      # staged input gone


def test_a_failing_met_call_keeps_the_staged_input_for_inspection(tmp_path):
    cfg = _cfg(tmp_path, fake_met(tmp_path, "fail"))
    tool = met_regrid.met_tool("regrid_data_plane", cfg.met_bin_dir)
    out = cfg.cache_dir / "x.nc"
    with pytest.raises(RuntimeError) as err:
        ww.regrid_one(tool, cfg, np.ones((2, 4, 4)), {"nx": "4"}, out,
                      T(2024, 9, 27, 18), T(2024, 9, 27, 21), 3)
    assert "could not read the input" in str(err.value)
    assert not out.exists()
    assert list(cfg.staging_dir.glob("*_native.nc"))


def test_a_member_that_comes_back_all_missing_is_a_failure(tmp_path):
    cfg = _cfg(tmp_path, fake_met(tmp_path, "missing"))
    tool = met_regrid.met_tool("regrid_data_plane", cfg.met_bin_dir)
    with pytest.raises(RuntimeError) as err:
        ww.regrid_one(tool, cfg, np.ones((2, 4, 4)), {"nx": "4"},
                      cfg.cache_dir / "y.nc", T(2024, 9, 27, 18),
                      T(2024, 9, 27, 21), 3)
    assert "all-missing" in str(err.value)


def test_regrid_command_repeats_field_and_names_every_member():
    cfg = met_regrid.RegridConfig(cache_dir=Path("/c"), grid_spec="g",
                                  grid_name="t")
    cmd = met_regrid.regrid_command("rdp", "in.nc", "g", "out.nc",
                                    ww.member_fields(3), cfg,
                                    names=ww.member_names(3))
    assert cmd.count("-field") == 3
    assert cmd[cmd.index("-name") + 1] == "rain_mm_m00,rain_mm_m01,rain_mm_m02"
    with pytest.raises(ValueError):
        met_regrid.regrid_command("rdp", "i", "g", "o", ww.member_fields(3),
                                  cfg, names=["only_one"])


# --- end to end on a synthetic archive -------------------------------------

def _archive(tmp_path, mode="exact", nan_step=None):
    wofs_root = tmp_path / "wofs" / "2024_BERYL" / "20240708_d1"
    init = T(2024, 7, 8, 18)
    _write_wofs(wofs_root, init, range(0, 73))
    for s in range(0, 73):
        valid = init + timedelta(minutes=5 * s)
        path = (wofs_root / "1800" / f"wofs_ALL_{s:02d}_20240708_1800_"
                                     f"{valid:%H%M}.nc")
        rain = np.stack([np.full((4, 4), 0.001 * s * (m + 1), dtype="f4")
                         for m in range(2)])
        if s == nan_step:
            rain[0, 0, 0] = np.nan
        ds = xr.Dataset({"rain": (("ne", "lat", "lon"), rain)})
        ds.attrs = dict(GRID, init_time="20240708_180000", valid_time="x")
        ds.to_netcdf(path)
    spec = vg.wofs_grid_spec(GRID, 6.0, name="t")
    case = make_case(tmp_path, wofs_root=wofs_root,
                     valid_start=T(2024, 7, 8, 18), valid_end=T(2024, 7, 9, 0),
                     hafs_models={},
                     regrid=dc.RegridSettings(
                         met_bin_dir=fake_met(tmp_path, mode)))
    vg.write_grid_json(case.grid_json_path, vg.grid_to_dict(spec))
    return case


def test_regrid_wofs_writes_1h_and_3h_files_with_the_right_sums(tmp_path):
    case = _archive(tmp_path)
    rows = ww.regrid_wofs_case(case)
    assert len(rows[3]) == 2 and len(rows[1]) == 6          # 2 windows x 3 h
    init = T(2024, 7, 8, 18)
    three = case.wofs_window_path(T(2024, 7, 8, 21), init, 3)
    one = case.wofs_window_path(T(2024, 7, 8, 19), init, 1)
    assert three.parent == case.wofs_regrid_3h_dir
    assert one.parent == case.wofs_regrid_1h_dir
    assert three.name == "2024070821_i202407081800_wofs_6km.nc"
    assert one.name == "2024070819_i202407081800_wofs_6km.nc"
    a, b = ww.read_window(three, ne=2), ww.read_window(one, ne=2)
    assert a.shape == b.shape == (2, 2, 2)
    assert np.allclose(a[0], 0.001 * sum(range(1, 37)) * 25.4, atol=1e-3)
    assert np.allclose(a[1], 2 * 0.001 * sum(range(1, 37)) * 25.4, atol=1e-3)
    assert np.allclose(b[0], 0.001 * sum(range(1, 13)) * 25.4, atol=1e-3)
    # the three 1 h files of a window add up to its 3 h file
    ones = [ww.read_window(case.wofs_window_path(T(2024, 7, 8, h), init, 1))
            for h in (19, 20, 21)]
    assert np.allclose(sum(ones), a, atol=1e-3)
    for hours in (1, 3):
        with open(case.wofs_budget_csv_path(hours)) as fh:
            budget = list(csv.DictReader(fh))
        assert {r["status"] for r in budget} == {"regridded"}
        assert all(r["flag"] == "" for r in budget)
        assert all(float(r["met_vs_block_max_mm"]) < 1e-3 for r in budget)


def test_a_rerun_reuses_existing_files_and_does_not_call_met_again(tmp_path):
    case = _archive(tmp_path)
    ww.regrid_wofs_case(case)
    out = case.wofs_window_path(T(2024, 7, 8, 21), T(2024, 7, 8, 18), 3)
    before = out.stat().st_mtime_ns
    # a MET that would fail if it were called
    fake_met(tmp_path, "fail")
    rows = ww.regrid_wofs_case(case)
    assert {r["status"] for r in rows[3] + rows[1]} == {"cached"}
    assert out.stat().st_mtime_ns == before


def test_met_disagreeing_with_the_block_mean_is_flagged(tmp_path):
    case = _archive(tmp_path, mode="scaled")        # MET returns 10% too much
    rows = ww.regrid_wofs_case(case)
    assert all(r["flag"] == "CHECK" for r in rows[3] + rows[1])
    assert float(rows[3][0]["met_vs_block_max_mm"]) > ww.MET_VS_BLOCK_TOL_MM


def test_nan_in_a_step_is_flagged_and_stays_missing(tmp_path):
    case = _archive(tmp_path, nan_step=5)           # inside the first hour
    rows = ww.regrid_wofs_case(case)
    first_3h = next(r for r in rows[3] if r["end"].endswith("21:00"))
    assert first_3h["flag"] == "CHECK" and int(first_3h["nan_cells"]) == 1
    h1 = ww.read_window(case.wofs_window_path(T(2024, 7, 8, 19),
                                              T(2024, 7, 8, 18), 1))
    assert np.isnan(h1[0, 0, 0]) and np.isfinite(h1[1, 0, 0])


def test_regrid_wofs_needs_a_grid_of_the_runs_resolution(tmp_path):
    case = _archive(tmp_path)
    three = make_case(tmp_path, wofs_root=case.wofs_root, res_km=3.0,
                      valid_start=case.valid_start, valid_end=case.valid_end)
    with pytest.raises(SystemExit) as err:           # only the 6 km grid exists
        ww.regrid_wofs_case(three)
    assert "build-grid" in str(err.value)


def test_regrid_wofs_skips_an_unavailable_deployment(tmp_path):
    case = make_case(tmp_path, wofs_available=False, valid_start=None,
                     valid_end=None)
    with pytest.raises(SystemExit) as err:
        ww.regrid_wofs_case(case)
    assert "no complete WoFS window" in str(err.value)


def test_the_case_keeps_separate_1h_and_3h_folders(tmp_path):
    case = make_case(tmp_path)
    init, end = T(2024, 7, 8, 18), T(2024, 7, 8, 21)
    assert case.wofs_window_path(end, init, 1).parent != \
        case.wofs_window_path(end, init, 3).parent
    assert case.wofs_budget_csv_path(1).parent == case.wofs_regrid_1h_dir
    with pytest.raises(ValueError):
        case.wofs_regrid_dir(6)
