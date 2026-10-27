"""Tests for the WoFS 3 h windows: step selection, the sum and unit
conversion, block-mean coarsening, the MET-readable NetCDF, and a full run on
a synthetic archive."""
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import verification_grid as vg
import wofs_windows as ww
from case_helpers import make_case
from test_window_manifest import GRID, _write_wofs

T = datetime


# --- step selection ---------------------------------------------------------

def test_window_steps_at_the_init_second_window_and_a_half_hour_init():
    init = T(2024, 9, 27, 18)
    w1, w2 = aw.clock_windows(init, T(2024, 9, 28, 0))
    assert ww.window_steps(init, w1) == list(range(1, 37))
    assert ww.window_steps(init, w2) == list(range(37, 73))
    half = T(2024, 9, 27, 17, 30)
    assert ww.window_steps(half, w1) == list(range(7, 43))
    assert all(len(ww.window_steps(i, w)) == 36 for i, w in
               ((init, w1), (half, w1)))


# --- the sum ----------------------------------------------------------------

def _loader(per_step):
    return lambda path: per_step[path]


def test_sum_steps_adds_inches_then_converts_to_mm():
    files = {s: s for s in range(1, 4)}
    fields = {s: np.full((2, 3, 3), 0.01 * s) for s in files}
    total, nan = ww.sum_steps(files, [1, 2, 3], loader=_loader(fields))
    assert np.allclose(total, 0.06 * 25.4) and nan == 0
    assert total.shape == (2, 3, 3)


def test_each_member_keeps_its_own_total():
    files = {1: 1, 2: 2}
    fields = {1: np.stack([np.full((2, 2), 0.1), np.full((2, 2), 0.2)]),
              2: np.stack([np.full((2, 2), 0.3), np.full((2, 2), 0.0)])}
    total, _ = ww.sum_steps(files, [1, 2], loader=_loader(fields))
    assert np.allclose(total[0], 0.4 * 25.4) and np.allclose(total[1],
                                                            0.2 * 25.4)


def test_a_missing_step_means_the_window_is_not_summed_at_all():
    with pytest.raises(FileNotFoundError) as err:
        ww.sum_steps({1: 1, 3: 3}, [1, 2, 3], loader=lambda p: np.zeros((1, 2, 2)))
    assert "not complete" in str(err.value)


def test_nan_in_any_step_makes_the_cell_nan_and_is_counted():
    files = {1: 1, 2: 2}
    a = np.ones((1, 2, 2)); b = np.ones((1, 2, 2)); b[0, 0, 0] = np.nan
    total, nan = ww.sum_steps(files, [1, 2], loader=_loader({1: a, 2: b}))
    assert np.isnan(total[0, 0, 0]) and nan == 1 and np.isfinite(total[0, 0, 1])


# --- coarsening -------------------------------------------------------------

def test_block_mean_averages_each_2x2_block_and_keeps_the_mean():
    f = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    c = ww.block_mean(f, 2)
    assert c.shape == (2, 2, 2)
    assert c[0, 0, 0] == f[0, :2, :2].mean() and c[1, 1, 1] == f[1, 2:, 2:].mean()
    assert np.isclose(c.mean(), f.mean())           # nothing gained or lost


def test_block_mean_at_native_resolution_is_the_identity():
    f = np.random.default_rng(1).random((3, 4, 4))
    assert ww.block_mean(f, 1) is f


def test_block_mean_nan_propagates_and_bad_shapes_are_refused():
    f = np.ones((1, 4, 4)); f[0, 0, 0] = np.nan
    c = ww.block_mean(f, 2)
    assert np.isnan(c[0, 0, 0]) and np.isfinite(c[0, 0, 1])
    with pytest.raises(ValueError):
        ww.block_mean(np.ones((1, 5, 4)), 2)


# --- the NetCDF -------------------------------------------------------------

def _spec(res=6.0):
    return vg.wofs_grid_spec(GRID, res, name="t")


def test_window_netcdf_round_trips_with_missing_cells_and_met_attributes(
        tmp_path):
    import netCDF4
    spec = _spec()
    lat, lon = vg.grid_latlon(spec)
    data = np.random.default_rng(2).random((3, spec.ny, spec.nx)) * 50
    data[1, 0, 0] = np.nan
    init = T(2024, 7, 8, 18)
    win = aw.Window(T(2024, 7, 8, 18), T(2024, 7, 8, 21))
    out = ww.write_window_netcdf(tmp_path / "a" / "w.nc", spec, data, lat, lon,
                                 init, win, 36, [Path("a.nc"), Path("b.nc")])
    got = ww.read_window(out)
    assert got.shape == data.shape
    assert np.isnan(got[1, 0, 0]) and np.allclose(got[0], data[0], atol=1e-4)
    with netCDF4.Dataset(out) as nc:
        assert nc.dimensions["ne"].size == 3
        assert nc.Projection == "Lambert Conformal"
        assert nc.nx == str(spec.nx) and nc.window_end == "20240708_210000"
        assert nc.variables[ww.VAR].dimensions == ("ne", "lat", "lon")
        assert nc.variables[ww.VAR].units == "mm"
    assert not list(tmp_path.rglob("*.tmp.nc"))      # written atomically


# --- end to end on a synthetic archive -------------------------------------

def _archive(tmp_path, nan_step=None):
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
    spec = _spec()
    case = make_case(tmp_path, wofs_root=wofs_root,
                     valid_start=T(2024, 7, 8, 18), valid_end=T(2024, 7, 9, 0),
                     hafs_models={})
    vg.write_grid_json(case.grid_json_path, vg.grid_to_dict(spec))
    return case


def test_regrid_wofs_writes_one_file_per_complete_pair_with_the_right_sums(
        tmp_path):
    case = _archive(tmp_path)
    rows = ww.regrid_wofs_case(case)
    assert len(rows) == 2
    init = T(2024, 7, 8, 18)
    first = case.wofs_window_path(T(2024, 7, 8, 21), init)
    second = case.wofs_window_path(T(2024, 7, 9, 0), init)
    assert first.name == "2024070821_i202407081800_wofs_6km.nc"
    a, b = ww.read_window(first), ww.read_window(second)
    assert a.shape == (2, 2, 2)                     # 4 x 4 coarsened to 2 x 2
    # member m: 0.001 * (m+1) * sum(steps) inches -> mm
    assert np.allclose(a[0], 0.001 * sum(range(1, 37)) * 25.4, atol=1e-3)
    assert np.allclose(a[1], 2 * 0.001 * sum(range(1, 37)) * 25.4, atol=1e-3)
    assert np.allclose(b[0], 0.001 * sum(range(37, 73)) * 25.4, atol=1e-3)
    assert case.wofs_budget_csv_path.exists()
    with open(case.wofs_budget_csv_path) as fh:
        budget = list(csv.DictReader(fh))
    assert {r["status"] for r in budget} == {"regridded"}
    assert all(r["flag"] == "" and r["nan_cells"] == "0" for r in budget)


def test_a_rerun_reuses_existing_windows(tmp_path):
    case = _archive(tmp_path)
    ww.regrid_wofs_case(case)
    out = case.wofs_window_path(T(2024, 7, 8, 21), T(2024, 7, 8, 18))
    before = out.stat().st_mtime_ns
    rows = ww.regrid_wofs_case(case)
    assert {r["status"] for r in rows} == {"cached"}
    assert out.stat().st_mtime_ns == before


def test_nan_in_a_step_is_flagged_and_stays_missing_in_the_window(tmp_path):
    case = _archive(tmp_path, nan_step=5)           # inside the first window
    rows = ww.regrid_wofs_case(case)
    first = next(r for r in rows if r["window_end"].endswith("21:00"))
    assert first["flag"] == "CHECK" and int(first["nan_cells"]) == 1
    data = ww.read_window(case.wofs_window_path(T(2024, 7, 8, 21),
                                                T(2024, 7, 8, 18)))
    assert np.isnan(data[0, 0, 0]) and np.isfinite(data[1, 0, 0])


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
