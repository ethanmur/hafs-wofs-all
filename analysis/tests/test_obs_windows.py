"""Tests for the obs 3 h windows: which hourly files make a window, the sum,
missing hours, NaN, file layout and reruns."""
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import deployment_case as dc
import met_regrid
import obs_windows as ow
from case_helpers import make_case

T = datetime


def _hourly(case, source, t, value):
    """A file shaped like MET's regrid_data_plane output, every cell `value`
    (an array, or a scalar)."""
    import netCDF4
    path = ow.hourly_path(case, source, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.broadcast_to(np.asarray(value, dtype="f4"), (3, 3))
    with netCDF4.Dataset(path, "w") as nc:
        nc.MET_version = "V12.2.0"
        nc.Projection = "Lambert Conformal"
        nc.createDimension("lat", 3)
        nc.createDimension("lon", 3)
        nc.createVariable("lat", "f4", ("lat", "lon"))[:] = 30.0
        nc.createVariable("lon", "f4", ("lat", "lon"))[:] = -85.0
        v = nc.createVariable("precip", "f4", ("lat", "lon"),
                              fill_value=-9999.0)
        v[:] = np.where(np.isnan(value), -9999.0, value)
    return path


def _case(tmp_path, **kw):
    return make_case(
        tmp_path, valid_start=T(2024, 9, 26, 0), valid_end=T(2024, 9, 26, 6),
        regrid=dc.RegridSettings(sources=("mrms",)), **kw)


def _fill(case, source="mrms", skip=()):
    """Hour ending at t holds the value t.hour, so a sum identifies its hours."""
    for h in range(1, 7):
        t = T(2024, 9, 26, h)
        if t not in skip:
            _hourly(case, source, t, float(h))


def test_a_window_is_the_three_hours_ending_inside_it_not_starting_in_it():
    w = aw.Window(T(2024, 9, 26, 15), T(2024, 9, 26, 18))
    assert ow.hour_ends(w) == [T(2024, 9, 26, 16), T(2024, 9, 26, 17),
                               T(2024, 9, 26, 18)]
    over_midnight = aw.Window(T(2024, 9, 26, 21), T(2024, 9, 27, 0))
    assert ow.hour_ends(over_midnight)[-1] == T(2024, 9, 27, 0)


def test_sum_propagates_nan_and_rejects_shape_mismatch():
    a = np.array([[1.0, 2.0], [3.0, np.nan]])
    b = np.ones((2, 2))
    s = ow.sum_hours([a, b, b])
    assert s[0, 0] == 3 and np.isnan(s[1, 1])
    with pytest.raises(ValueError):
        ow.sum_hours([a, np.ones((3, 3))])


def test_windows_are_summed_from_the_right_hours_and_named_for_the_end(tmp_path):
    case = _case(tmp_path)
    _fill(case)
    res = ow.window_obs_case(case)["mrms"]
    assert [r["status"] for r in res] == ["written", "written"]
    first = case.obs_window_path("mrms", T(2024, 9, 26, 3))
    assert first.parent == case.mrms_regrid_3h_dir
    assert first.name == "2024092603_mrms_6km.nc"
    assert case.regrid_3h_dir_for("stage4") == case.stage4_regrid_3h_dir
    # (00Z, 03Z] = hours ending 01, 02, 03 -> 1 + 2 + 3; (03Z, 06Z] -> 4 + 5 + 6
    assert np.allclose(met_regrid.read_regridded(first)[2], 6.0)
    second = case.obs_window_path("mrms", T(2024, 9, 26, 6))
    assert np.allclose(met_regrid.read_regridded(second)[2], 15.0)


def test_the_window_file_keeps_the_met_grid_and_says_what_it_is(tmp_path):
    import netCDF4
    case = _case(tmp_path)
    _fill(case)
    ow.window_obs_case(case)
    with netCDF4.Dataset(case.obs_window_path("mrms", T(2024, 9, 26, 3))) as nc:
        assert nc.MET_version == "V12.2.0" and nc.Projection
        assert {"lat", "lon", "precip"} <= set(nc.variables)
        assert nc.accumulation_hours == 3
        assert nc.window_start == "20240926_000000"
        assert nc.valid_time == "20240926_030000"
        assert nc.hours_summed == "2024092601,2024092602,2024092603"
        assert nc.source_product == "mrms"


def test_a_window_with_a_missing_hour_is_missing_not_rescaled(tmp_path):
    case = _case(tmp_path)
    _fill(case, skip={T(2024, 9, 26, 2)})
    res = ow.window_obs_case(case)["mrms"]
    assert [r["status"] for r in res] == ["missing_hours", "written"]
    assert res[0]["missing_hours"] == "2024092602"
    assert res[0]["hours_found"] == "2/3"
    assert not case.obs_window_path("mrms", T(2024, 9, 26, 3)).exists()
    assert case.obs_window_path("mrms", T(2024, 9, 26, 6)).exists()


def test_an_all_missing_hourly_file_counts_as_missing(tmp_path):
    case = _case(tmp_path)
    _fill(case)
    _hourly(case, "mrms", T(2024, 9, 26, 5), np.nan)
    res = ow.window_obs_case(case)["mrms"]
    assert res[1]["status"] == "missing_hours"
    assert res[1]["missing_hours"] == "2024092605"


def test_a_cell_missing_in_one_hour_is_missing_in_the_window(tmp_path):
    case = _case(tmp_path)
    _fill(case)
    patchy = np.full((3, 3), 2.0)
    patchy[1, 1] = np.nan
    _hourly(case, "mrms", T(2024, 9, 26, 2), patchy)
    res = ow.window_obs_case(case)["mrms"]
    assert res[0]["nan_cells"] == 1
    out = met_regrid.read_regridded(case.obs_window_path("mrms",
                                                         T(2024, 9, 26, 3)))[2]
    assert np.isnan(out[1, 1]) and out[0, 0] == 6.0


def test_a_rerun_reuses_files_and_reports_them_cached(tmp_path):
    case = _case(tmp_path)
    _fill(case)
    ow.window_obs_case(case)
    out = case.obs_window_path("mrms", T(2024, 9, 26, 3))
    before = out.stat().st_mtime_ns
    res = ow.window_obs_case(case)["mrms"]
    assert {r["status"] for r in res} == {"cached"}
    assert out.stat().st_mtime_ns == before
    assert res[0]["mean_mm"] == 6.0


def test_each_source_has_its_own_folder_and_csv(tmp_path):
    case = make_case(tmp_path, valid_start=T(2024, 9, 26, 0),
                     valid_end=T(2024, 9, 26, 6),
                     regrid=dc.RegridSettings(sources=("stage4", "mrms")))
    _fill(case, "mrms")
    _fill(case, "stage4", skip={T(2024, 9, 26, 5)})
    res = ow.window_obs_case(case)
    assert [r["status"] for r in res["mrms"]] == ["written", "written"]
    assert [r["status"] for r in res["stage4"]] == ["written", "missing_hours"]
    for source in ("mrms", "stage4"):
        with open(case.obs_window_csv_path(source)) as fh:
            rows = list(csv.DictReader(fh))
        assert {r["source"] for r in rows} == {source}
        assert case.obs_window_csv_path(source).parent == (
            case.regrid_3h_dir_for(source))
    assert case.mrms_regrid_3h_dir != case.stage4_regrid_3h_dir


def test_other_resolutions_get_their_own_names(tmp_path):
    three = make_case(tmp_path, res_km=3.0,
                      valid_start=T(2024, 9, 26, 0),
                      valid_end=T(2024, 9, 26, 6))
    assert three.obs_window_path("mrms", T(2024, 9, 26, 3)).name == (
        "2024092603_mrms_3km.nc")
    assert three.obs_window_csv_path("mrms").name.endswith("_3km.csv")


def test_an_unavailable_deployment_is_skipped(tmp_path):
    case = make_case(tmp_path, wofs_available=False, valid_start=None,
                     valid_end=None)
    with pytest.raises(SystemExit) as err:
        ow.window_obs_case(case)
    assert "no complete WoFS window" in str(err.value)
