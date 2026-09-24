"""Unit tests for obs_cases' pure logic (no network, no Hercules data).

Run directly:   python3 analysis/tests/test_obs_cases.py
Or via pytest:  pytest analysis/tests/test_obs_cases.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from obs_cases import (
    ObsCase, from_yaml, hourly_timestamps, check_cache_complete,
)
from hafs_common import mrms_s3_key
import aorc_common
import stage4_hourly


def _minimal_case(**overrides):
    kwargs = dict(
        storm_name="Test Storm", best_track=Path("/tmp/bt.dat"),
        valid_start=datetime(2024, 9, 26, 0), valid_end=datetime(2024, 9, 26, 6),
        domain=(15.0, 42.0, -100.0, -60.0), out_dir=Path("/tmp/out"),
        mrms_cache_dir=Path("/tmp/mrms"), stage4_cache_dir=Path("/tmp/st4"),
        aorc_cache_dir=Path("/tmp/aorc"),
    )
    kwargs.update(overrides)
    return ObsCase(**kwargs)


def test_hourly_timestamps_end_of_hour_convention():
    ts = hourly_timestamps(datetime(2024, 9, 26, 0), datetime(2024, 9, 26, 3))
    assert ts == [datetime(2024, 9, 26, 1), datetime(2024, 9, 26, 2),
                 datetime(2024, 9, 26, 3)]


def test_from_yaml_round_trip_and_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "case.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "storm_name": "Test Storm",
            "best_track": "/tmp/bt.dat",
            "valid_start": 2024092400,
            "valid_end": 2024092906,
            "domain": [15.0, 42.0, -100.0, -60.0],
            "skip_mrms": True,
            "skip_stage4": True,
        }))
        case = from_yaml(yaml_path)
        assert isinstance(case, ObsCase)
        assert case.storm_name == "Test Storm"
        assert case.valid_start == datetime(2024, 9, 24, 0)
        assert case.valid_end == datetime(2024, 9, 29, 6)
        assert case.skip_mrms is True
        assert case.skip_stage4 is True
        assert case.skip_aorc is False          # default
        assert case.output_slug == "case_2024092400_2024092906"


def test_from_yaml_regrid_block_is_optional():
    with tempfile.TemporaryDirectory() as tmp:
        base = {"best_track": "/tmp/bt.dat", "valid_start": 2024092400,
                "valid_end": 2024092406, "domain": [15.0, 42.0, -100.0, -60.0]}
        plain = Path(tmp) / "plain.yaml"
        plain.write_text(yaml.safe_dump(base))
        assert from_yaml(plain).regrid is None

        with_regrid = Path(tmp) / "regrid.yaml"
        with_regrid.write_text(yaml.safe_dump({**base, "regrid": {
            "grid_template": "/hafs/**/*parent.atm.f*.grb2",
            "cache_dir": "/cache", "tolerance_pct": 1.5}}))
        cfg = from_yaml(with_regrid).regrid
        assert cfg.method == "BUDGET" and cfg.tolerance_pct == 1.5
        assert cfg.cache_dir == Path("/cache")


def test_from_yaml_requires_core_fields():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "case.yaml"
        yaml_path.write_text(yaml.safe_dump({"storm_name": "Test Storm"}))
        try:
            from_yaml(yaml_path)
            assert False, "expected a KeyError for the missing fields"
        except KeyError:
            pass


def _touch_mrms(cache_dir, valid_dt):
    _, fname = mrms_s3_key(valid_dt)
    (Path(cache_dir) / fname.replace(".gz", "")).parent.mkdir(
        parents=True, exist_ok=True)
    (Path(cache_dir) / fname.replace(".gz", "")).touch()


def _touch_aorc(cache_dir, valid_dt):
    path = aorc_common.aorc_cache_path(cache_dir, valid_dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_check_cache_complete_empty_when_everything_cached():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        case = _minimal_case(
            skip_stage4=True, mrms_cache_dir=base / "mrms",
            aorc_cache_dir=base / "aorc",
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 2),
        )
        for t in hourly_timestamps(case.valid_start, case.valid_end):
            _touch_mrms(case.mrms_cache_dir, t)
            _touch_aorc(case.aorc_cache_dir, t)
        assert check_cache_complete(case) == []


def test_check_cache_complete_flags_missing_hours():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        case = _minimal_case(
            skip_stage4=True, mrms_cache_dir=base / "mrms",
            aorc_cache_dir=base / "aorc",
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 2),
        )
        # Only cache one of the two needed hours, and only for MRMS.
        _touch_mrms(case.mrms_cache_dir, datetime(2024, 9, 26, 1))
        missing = check_cache_complete(case)
        assert len(missing) == 3   # MRMS h2, AORC h1, AORC h2
        assert any("MRMS" in m and "02Z" in m for m in missing)
        assert any("AORC" in m and "01Z" in m for m in missing)


def test_check_cache_complete_flags_missing_stage4_hours(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        case = _minimal_case(
            skip_stage4=False, skip_mrms=True, skip_aorc=True,
            stage4_cache_dir=Path(tmp),
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 3),
        )
        # Only hour 2 is "in the cache" -- fake the parsed ST4.<day> index
        # so this test doesn't need a real GRIB fixture.
        monkeypatch.setattr(
            stage4_hourly, "index_stage4_hourly",
            lambda cache_dir_str: {
                datetime(2024, 9, 26, 2): [Path(tmp) / "ST4.20240926"],
            },
        )
        missing = check_cache_complete(case)
        assert len(missing) == 2   # hours 1 and 3 missing
        assert any("01Z" in m for m in missing)
        assert any("03Z" in m for m in missing)
        assert not any("02Z" in m for m in missing)


def test_check_cache_complete_empty_when_stage4_cached(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        case = _minimal_case(
            skip_stage4=False, skip_mrms=True, skip_aorc=True,
            stage4_cache_dir=Path(tmp),
            valid_start=datetime(2024, 9, 26, 0),
            valid_end=datetime(2024, 9, 26, 2),
        )
        fake_path = Path(tmp) / "ST4.20240926"
        monkeypatch.setattr(
            stage4_hourly, "index_stage4_hourly",
            lambda cache_dir_str: {
                datetime(2024, 9, 26, 1): [fake_path],
                datetime(2024, 9, 26, 2): [fake_path],
            },
        )
        assert check_cache_complete(case) == []


class _FakeMonkeypatch:
    """Minimal stand-in so this file also runs standalone (no pytest)."""

    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def test_from_yaml_init_range_is_optional_and_independent_of_window():
    with tempfile.TemporaryDirectory() as tmp:
        base = {"best_track": "/tmp/bt.dat", "valid_start": 2024070800,
                "valid_end": 2024071000,
                "domain": [15.0, 42.0, -100.0, -60.0]}
        plain = Path(tmp) / "plain.yaml"
        plain.write_text(yaml.safe_dump(base))
        case = from_yaml(plain)
        assert case.init_start is None and case.init_end is None

        # init_start may precede valid_start to reach long lead times
        ranged = Path(tmp) / "ranged.yaml"
        ranged.write_text(yaml.safe_dump({**base, "init_start": 2024070500}))
        case = from_yaml(ranged)
        assert case.init_start == datetime(2024, 7, 5, 0)
        assert case.init_end == datetime(2024, 7, 10, 0)   # defaults to window


def test_from_yaml_rejects_impossible_init_range():
    with tempfile.TemporaryDirectory() as tmp:
        base = {"best_track": "/tmp/bt.dat", "valid_start": 2024070800,
                "valid_end": 2024071000,
                "domain": [15.0, 42.0, -100.0, -60.0]}
        for bad in ({"init_start": 2024070500, "init_end": 2024070400},
                    {"init_start": 2024071200},          # after valid_end
                    {"init_start": "not-a-time"}):
            path = Path(tmp) / "bad.yaml"
            path.write_text(yaml.safe_dump({**base, **bad}))
            try:
                from_yaml(path)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {bad}")


def test_from_yaml_grid_and_wofs_domain():
    with tempfile.TemporaryDirectory() as tmp:
        base = {"best_track": "/tmp/bt.dat", "valid_start": 2024070800,
                "valid_end": 2024071000,
                "domain": [15.0, 42.0, -100.0, -60.0]}
        plain = Path(tmp) / "plain.yaml"
        plain.write_text(yaml.safe_dump(base))
        case = from_yaml(plain)
        assert case.grid.res_km == 6.0 and case.grid.pad_km == 750.0
        assert case.wofs_domain is None

        full = Path(tmp) / "full.yaml"
        full.write_text(yaml.safe_dump({
            **base, "wofs_domain": [31.5, 39.5, -87.0, -78.0],
            "verification_grid": {"res_km": 5.0, "pad_km": 900.0}}))
        case = from_yaml(full)
        assert case.wofs_domain == (31.5, 39.5, -87.0, -78.0)
        assert case.grid.res_km == 5.0 and case.grid.pad_km == 900.0
        assert case.grid.grid_name == "lambert5km"


def test_from_yaml_reports_bad_grid_block_with_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.yaml"
        path.write_text(yaml.safe_dump({
            "best_track": "/tmp/bt.dat", "valid_start": 2024070800,
            "valid_end": 2024071000, "domain": [15.0, 42.0, -100.0, -60.0],
            "verification_grid": {"pad_km": 100.0}}))
        try:
            from_yaml(path)
        except ValueError as err:
            assert "pad_km" in str(err) and "bad.yaml" in str(err)
            return
        raise AssertionError("expected ValueError")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        needs_mp = "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]
        mp = _FakeMonkeypatch() if needs_mp else None
        try:
            fn(mp) if needs_mp else fn()
        finally:
            if mp is not None:
                mp.undo()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
