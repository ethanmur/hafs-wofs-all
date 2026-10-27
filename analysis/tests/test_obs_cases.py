"""Unit tests for obs_cases' pure logic (no network, no cluster data)."""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deployment_case as dc
import obs_cases
from case_helpers import make_case
import stage4_hourly
from obs_cases import (hourly_timestamps, check_cache_complete, mrms_s3_key,
                       regrid_config, grid_shape_from_json, read_grid_meta, download_obs,
                       regrid_obs)


def _case(tmp_path, **overrides):
    # this module's directories keep the short names its fixtures expect
    names = dict(regrid_dir=tmp_path / "rg", mrms_cache_dir=tmp_path / "mrms",
                 stage4_cache_dir=tmp_path / "st4",
                 grid_json_dir=tmp_path / "json")
    names.update(overrides)
    return make_case(tmp_path, **names)


def _touch_mrms(cache_dir, valid_dt):
    _, fname = mrms_s3_key(valid_dt)
    path = Path(cache_dir) / fname.replace(".gz", "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_grid_json(case, nx=150, ny=150, res_km=6.0):
    case.grid_json_dir.mkdir(parents=True, exist_ok=True)
    case.grid_json_path.write_text(json.dumps({
        "name": case.case_slug, "nx": nx, "ny": ny, "res_km": res_km,
        "met_spec": f"lambert {nx} {ny} 29.0 -86.8 -82.25 6.0 6371.2 30.0 "
                    "60.0 N"}))


def test_hourly_timestamps_end_of_hour_convention():
    ts = hourly_timestamps(datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 3))
    assert ts == [datetime(2024, 9, 26, 1), datetime(2024, 9, 26, 2),
                  datetime(2024, 9, 26, 3)]


def test_mrms_key_names_the_hour_end():
    key, fname = mrms_s3_key(datetime(2024, 9, 26, 1))
    assert fname == ("MRMS_MultiSensor_QPE_01H_Pass2_00.00_"
                     "20240926-010000.grib2.gz")
    assert key == f"CONUS/MultiSensor_QPE_01H_Pass2_00.00/20240926/{fname}"


def test_check_cache_complete_mrms(tmp_path):
    case = _case(tmp_path, truth_source="mrms")
    for t in hourly_timestamps(case.valid_start, case.valid_end):
        _touch_mrms(case.mrms_cache_dir, t)
    assert check_cache_complete(case) == []


def test_check_cache_complete_flags_missing_mrms_hours(tmp_path):
    case = _case(tmp_path, truth_source="mrms")
    _touch_mrms(case.mrms_cache_dir, datetime(2024, 9, 26, 1))
    missing = check_cache_complete(case)
    assert len(missing) == 1 and "02Z" in missing[0]


def test_check_cache_complete_stage4(tmp_path, monkeypatch):
    case = _case(tmp_path, valid_end=datetime(2024, 9, 26, 3))
    monkeypatch.setattr(stage4_hourly, "index_stage4_hourly",
                        lambda d: {datetime(2024, 9, 26, 2): [Path(d) / "x"]})
    missing = check_cache_complete(case)
    assert len(missing) == 2
    assert any("01Z" in m for m in missing) and any("03Z" in m for m in missing)


def test_the_truth_product_decides_what_must_be_cached(tmp_path, monkeypatch):
    # stage4 is the truth: MRMS files are not needed unless asked for
    case = _case(tmp_path)
    monkeypatch.setattr(stage4_hourly, "index_stage4_hourly",
                        lambda d: {t: [Path(d) / "x"] for t in
                                   hourly_timestamps(case.valid_start,
                                                     case.valid_end)})
    assert check_cache_complete(case) == []
    both = _case(tmp_path, regrid=dc.RegridSettings(sources=("stage4", "mrms")))
    assert len(check_cache_complete(both)) == 2     # no MRMS cached


def test_regrid_config_reads_the_grid_spec_and_names_files_by_hour_and_product(
        tmp_path):
    case = _case(tmp_path)
    _write_grid_json(case)
    cfg = regrid_config(case)
    assert cfg.grid_spec.startswith("lambert 150 150")
    assert cfg.cache_dir == tmp_path / "rg"     # this deployment's own folder
    assert cfg.method == "BUDGET" and cfg.width == 2
    # the folder names the deployment, so the file needs only hour + product
    assert cfg.output_path("stage4", datetime(2024, 9, 26, 1)) == (
        tmp_path / "rg" / "2024092601_stage4_6km.nc")
    assert cfg.settings_path.name == "regrid_settings_6km.txt"
    assert cfg.staging_dir == tmp_path / "rg" / "_staging"


def test_regrid_config_without_a_grid_says_to_run_build_grid(tmp_path):
    with pytest.raises(SystemExit) as err:
        regrid_config(_case(tmp_path))
    assert "build-grid" in str(err.value)


def test_grid_shape_from_json_is_rows_then_columns(tmp_path):
    case = _case(tmp_path)
    _write_grid_json(case, nx=150, ny=140)
    assert grid_shape_from_json(case.grid_json_path) == (140, 150)


def test_unavailable_deployment_is_skipped_by_every_obs_command(tmp_path):
    case = _case(tmp_path, wofs_available=False, valid_start=None,
                 valid_end=None)
    for command in (download_obs, regrid_obs):
        with pytest.raises(SystemExit) as err:
            command(case)
        assert "no complete WoFS window" in str(err.value)


def test_download_obs_for_a_stage4_deployment_fetches_nothing(tmp_path, capsys):
    download_obs(_case(tmp_path))
    out = capsys.readouterr().out
    assert "not fetched by this command" in out
    assert "nothing to fetch" in out


def test_read_grid_meta_includes_the_resolution(tmp_path):
    case = _case(tmp_path)
    _write_grid_json(case, nx=150, ny=140, res_km=6.0)
    assert read_grid_meta(case.grid_json_path) == {"ny": 140, "nx": 150,
                                                   "res_km": 6.0}


def test_regrid_config_refuses_a_grid_json_of_another_resolution(tmp_path):
    case = _case(tmp_path, res_km=3.0)
    # a 3 km run reads <case>_3km_grid.json; if that file somehow holds a 6 km
    # grid, regridding would write files labelled 3km on the wrong grid
    _write_grid_json(case, res_km=6.0)
    with pytest.raises(SystemExit) as err:
        regrid_config(case)
    assert "6 km grid" in str(err.value) and "3 km" in str(err.value)


def test_a_3km_run_finds_its_own_grid_not_the_6km_one(tmp_path):
    six, three = _case(tmp_path), _case(tmp_path, res_km=3.0)
    _write_grid_json(six, res_km=6.0)
    with pytest.raises(SystemExit) as err:         # only the 6 km grid exists
        regrid_config(three)
    assert "build-grid" in str(err.value)
    _write_grid_json(three, nx=300, ny=300, res_km=3.0)
    cfg = regrid_config(three)
    assert cfg.res_label == "3km"
    assert cfg.output_path("stage4", datetime(2024, 9, 26, 1)).name == \
        "2024092601_stage4_3km.nc"
    assert regrid_config(six).res_label == "6km"
