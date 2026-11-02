"""Tests for the HAFS 3 h regrid: the work list from the manifest, file and
folder naming, the MET call (a stand-in regrid_data_plane), the budget check,
reruns, and a full run on a synthetic archive."""
import csv
import stat
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deployment_case as dc
import hafs_windows as hw
import met_regrid
import verification_grid as vg
import window_manifest as wm
from case_helpers import make_case
from test_window_manifest import GRID, _write_hafs, _write_wofs

T = datetime

# A stand-in regrid_data_plane: the "input" is an empty .grb2, and the native
# field sits beside it as <file>.npz (lat, lon, vals). The output is the 2 x 2
# block mean on the coarse grid, like MET's BUDGET on a nested grid.
FAKE_TOOL = '''#!{python}
import sys
import netCDF4
import numpy as np
MODE = "{mode}"
args = sys.argv[1:]
inp, grid, out = args[:3]
if MODE == "fail":
    print("ERROR: could not read the input", file=sys.stderr)
    sys.exit(1)
d = np.load(inp + ".npz")
nx, ny = int(grid.split()[1]), int(grid.split()[2])
f = d["vals"].shape[1] // nx
def coarse(a):
    return a.reshape(ny, f, nx, f).mean(axis=(1, 3))
vals = coarse(d["vals"])
if MODE == "scaled":
    vals = vals * 1.1
with netCDF4.Dataset(out, "w") as nc:
    nc.createDimension("lat", ny); nc.createDimension("lon", nx)
    nc.createVariable("lat", "f4", ("lat", "lon"))[:] = coarse(d["lat"])
    nc.createVariable("lon", "f4", ("lat", "lon"))[:] = coarse(d["lon"])
    v = nc.createVariable("precip", "f4", ("lat", "lon"), fill_value=-9999.0)
    v[:] = -9999.0 if MODE == "missing" else vals
'''


def fake_met(tmp_path, mode="exact"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / "regrid_data_plane"
    tool.write_text(FAKE_TOOL.format(python=sys.executable, mode=mode))
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _reader(shift=0.0, missing=()):
    """Read the native bucket from the sidecar; `missing` paths have none."""
    def read(path, valid_end, hours):
        if Path(path).name in missing:
            return None
        d = np.load(str(path) + ".npz")
        return d["lat"] + shift, d["lon"] + shift, d["vals"]
    return read


def _archive(tmp_path, mode="exact", budget_check=True):
    """A Beryl 20240708_d1 case: one WoFS init (18Z, 73 steps), HFSA cycles
    for the 48 h before, HFSB for two of them, with native fields written
    beside each window's file."""
    wofs_root = tmp_path / "wofs" / "2024_BERYL" / "20240708_d1"
    _write_wofs(wofs_root, T(2024, 7, 8, 18), range(0, 73))
    cycles = [T(2024, 7, 6, 18) + timedelta(hours=6 * k) for k in range(8)]
    _write_hafs(tmp_path / "A", "02l", "HFSA", cycles)
    _write_hafs(tmp_path / "B", "02l", "HFSB", cycles[:2])
    spec3 = vg.wofs_grid_spec(GRID, 3.0, name="t3")
    lat, lon = vg.grid_latlon(spec3)
    rng = np.random.default_rng(5)
    for root in (tmp_path / "A", tmp_path / "B"):
        for f in root.rglob("*.grb2"):
            np.savez(str(f) + ".npz", lat=lat, lon=lon,
                     vals=rng.random(lat.shape) * 20)
    spec = vg.wofs_grid_spec(GRID, 6.0, name="t")
    case = make_case(
        tmp_path, wofs_root=wofs_root, valid_start=T(2024, 7, 8, 18),
        valid_end=T(2024, 7, 9, 0),
        hafs_models={"HFSA": tmp_path / "A", "HFSB": tmp_path / "B"},
        hafs_regrid_dir=tmp_path / "hafs-regrid" / "20240708_d1_beryl",
        regrid=dc.RegridSettings(met_bin_dir=fake_met(tmp_path, mode),
                                 budget_check=budget_check))
    case.hafs_min_members = 8
    vg.write_grid_json(case.grid_json_path, vg.grid_to_dict(spec))
    return case


# --- naming and paths -------------------------------------------------------

def test_files_are_named_like_the_wofs_files_in_a_folder_per_configuration(
        tmp_path):
    case = make_case(tmp_path,
                     hafs_regrid_dir=tmp_path / "d" / "20241009_d1_milton")
    p = case.hafs_window_path("HFSA", T(2024, 10, 9, 21), T(2024, 10, 9, 12))
    assert p.parent.name == "20241009_d1_milton_HFSA"
    assert p.name == "2024100921_i202410091200_hfsa_6km.nc"
    assert case.hafs_window_path("HFSB", T(2024, 10, 9, 21),
                                 T(2024, 10, 9, 12)).parent.name == (
        "20241009_d1_milton_HFSB")
    w = case.wofs_window_path(T(2024, 10, 9, 21), T(2024, 10, 9, 12), 3)
    assert w.name.split("_")[:2] == p.name.split("_")[:2]   # valid end, cycle
    three = make_case(tmp_path, res_km=3.0,
                      hafs_regrid_dir=tmp_path / "d" / "20241009_d1_milton")
    assert three.hafs_window_path("HFSA", T(2024, 10, 9, 21),
                                  T(2024, 10, 9, 12)).name.endswith("_3km.nc")
    assert case.hafs_budget_csv_path("HFSA").name == (
        "regrid_budget_2024092600_2024092602.csv")


# --- the work list ----------------------------------------------------------

def test_work_list_is_every_complete_cycle_of_windows_with_wofs(tmp_path):
    case = _archive(tmp_path)
    m = wm.build_manifest(case)
    jobs = hw.work_list(case, m)
    assert jobs and all(j.window.end.hour in (21, 0) for j in jobs)
    assert {j.config for j in jobs} == {"HFSA", "HFSB"}
    first = [j for j in jobs if j.window.end == T(2024, 7, 8, 21)]
    # 48 h lookback, six-hourly, strictly before the 18Z start: 8 HFSA + 2 HFSB
    assert sum(j.config == "HFSA" for j in first) == 8
    assert sum(j.config == "HFSB" for j in first) == 2
    assert all(j.init < j.window.start for j in jobs)
    assert all(j.lead_h >= 6 for j in jobs)          # f003 is never a window
    assert all(j.source.name.endswith(f"f{j.lead_h:03d}.grb2") for j in jobs)


def test_a_window_without_wofs_is_left_out(tmp_path):
    case = _archive(tmp_path)
    m = wm.build_manifest(case)
    for r in m.summary:
        if r["window_start"].startswith("2024-07-08 21"):
            r["category"] = "no_wofs"
    jobs = hw.work_list(case, m)
    assert jobs and all(j.window.end != T(2024, 7, 9, 0) for j in jobs)


# --- end to end -------------------------------------------------------------

def test_regrid_hafs_writes_a_file_per_cycle_and_window_with_a_clean_budget(
        tmp_path):
    case = _archive(tmp_path)
    rows = hw.regrid_hafs_case(case, reader=_reader())
    # the 21Z window has 8 HFSA cycles back to 48 h, the 00Z window 7 (the
    # 06-18Z cycle is now over 48 h back); HFSB has two cycles, one per window
    assert len(rows["HFSA"]) == 15 and len(rows["HFSB"]) == 3
    assert {r["status"] for m in rows.values() for r in m} == {"regridded"}
    init = T(2024, 7, 8, 12)
    out = case.hafs_window_path("HFSA", T(2024, 7, 8, 21), init)
    assert out.name == "2024070821_i202407081200_hfsa_6km.nc"
    lat, lon, vals = met_regrid.read_regridded(out)
    assert vals.shape == (2, 2)
    src = next(j.source for j in hw.work_list(case, wm.build_manifest(case))
               if j.init == init and j.window.end == T(2024, 7, 8, 21)
               and j.config == "HFSA")
    native = np.load(str(src) + ".npz")["vals"]
    assert np.allclose(vals, native.reshape(2, 2, 2, 2).mean(axis=(1, 3)),
                       atol=1e-4)
    import netCDF4
    with netCDF4.Dataset(out) as nc:
        assert nc.model == "HFSA" and nc.accumulation_hours == 3
        assert nc.init_time == "20240708_120000"
        assert nc.valid_time == "20240708_210000"
    for c in ("HFSA", "HFSB"):
        with open(case.hafs_budget_csv_path(c)) as fh:
            budget = list(csv.DictReader(fh))
        assert all(r["flag"] == "" for r in budget)
        assert all(abs(float(r["mean_pct_diff"])) < 0.5 for r in budget)
        assert case.hafs_regrid_dir_for(c).name.endswith(f"_{c}")


def test_met_that_returns_too_much_is_flagged(tmp_path):
    case = _archive(tmp_path, mode="scaled")
    rows = hw.regrid_hafs_case(case, reader=_reader())
    assert all(r["flag"] == "CHECK" for m in rows.values() for r in m)
    assert all(float(r["mean_pct_diff"]) > 5 for m in rows.values()
               for r in m)


def test_a_rerun_reuses_files_and_keeps_their_budget_rows(tmp_path):
    case = _archive(tmp_path)
    first = hw.regrid_hafs_case(case, reader=_reader())
    out = case.hafs_window_path("HFSA", T(2024, 7, 8, 21), T(2024, 7, 8, 12))
    before = out.stat().st_mtime_ns
    fake_met(tmp_path, "fail")             # would fail if MET were called
    again = hw.regrid_hafs_case(case, reader=_reader())
    assert {r["status"] for m in again.values() for r in m} == {"cached"}
    assert out.stat().st_mtime_ns == before
    a = [float(r["mean_pct_diff"]) for r in first["HFSA"]]
    b = [float(r["mean_pct_diff"]) for r in again["HFSA"]]
    assert a == b


def test_a_cycle_without_the_bucket_or_off_the_grid_is_a_row_not_a_failure(
        tmp_path):
    case = _archive(tmp_path)
    one = next(j for j in hw.work_list(case, wm.build_manifest(case))
               if j.config == "HFSB")
    rows = hw.regrid_hafs_case(case, reader=_reader(missing={one.source.name}))
    bad = [r for r in rows["HFSB"] if r["status"] == "no_bucket"]
    assert len(bad) == 1
    assert not case.hafs_window_path("HFSB", one.window.end, one.init).exists()

    case2 = _archive(tmp_path / "x")
    rows2 = hw.regrid_hafs_case(case2, reader=_reader(shift=40.0))
    assert {r["status"] for m in rows2.values() for r in m} == {"no_overlap"}


def test_budget_check_off_skips_the_native_read(tmp_path):
    case = _archive(tmp_path, budget_check=False)

    def boom(*a):
        raise AssertionError("the native file should not be read")
    rows = hw.regrid_hafs_case(case, reader=boom)
    assert {r["status"] for m in rows.values() for r in m} == {"regridded"}
    assert all(r["mean_pct_diff"] == "" for m in rows.values() for r in m)


def test_a_failing_met_call_stops_the_run_with_its_message(tmp_path):
    case = _archive(tmp_path, mode="fail")
    with pytest.raises(RuntimeError) as err:
        hw.regrid_hafs_case(case, reader=_reader())
    assert "could not read the input" in str(err.value)


def test_regrid_hafs_needs_a_grid_of_the_runs_resolution(tmp_path):
    case = _archive(tmp_path)
    three = make_case(tmp_path, wofs_root=case.wofs_root, res_km=3.0,
                      valid_start=case.valid_start, valid_end=case.valid_end,
                      hafs_models=case.hafs_models)
    with pytest.raises(SystemExit) as err:
        hw.regrid_hafs_case(three, reader=_reader())
    assert "build-grid" in str(err.value)


def test_regrid_hafs_skips_an_unavailable_deployment(tmp_path):
    case = make_case(tmp_path, wofs_available=False, valid_start=None,
                     valid_end=None)
    with pytest.raises(SystemExit) as err:
        hw.regrid_hafs_case(case)
    assert "no complete WoFS window" in str(err.value)


def test_grid_key_separates_two_domains_of_the_same_shape():
    lat, lon = np.meshgrid(np.linspace(20, 25, 4), np.linspace(-90, -85, 4))
    assert hw.grid_key(lat, lon) == hw.grid_key(lat.copy(), lon.copy())
    assert hw.grid_key(lat, lon) != hw.grid_key(lat + 1, lon)
