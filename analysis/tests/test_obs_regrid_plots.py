"""Unit tests for obs_regrid_plots' pure logic (cropping, config, budget CSV).

Run directly:   python3 analysis/tests/test_obs_regrid_plots.py
Or via pytest:  pytest analysis/tests/test_obs_regrid_plots.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from obs_cases import from_yaml
from obs_regrid_plots import (crop_slices, load_budget, track_segment,
                              available_sources, _budget_note)


_TRACK = [
    (datetime(2024, 9, 26, 0), 28.0, -84.0),
    (datetime(2024, 9, 26, 6), 29.0, -84.0),
    (datetime(2024, 9, 26, 12), 30.0, -84.0),
]


def test_track_segment_interpolates_hourly():
    seg = track_segment(_TRACK, datetime(2024, 9, 26, 0),
                        datetime(2024, 9, 26, 6))
    assert len(seg) == 7
    assert np.isclose(seg[0][0], 28.0)
    assert np.isclose(seg[3][0], 28.5)   # halfway between 28 and 29
    assert np.isclose(seg[-1][0], 29.0)


def _grid():
    lon, lat = np.meshgrid(np.arange(-90.0, -80.0, 0.5),
                           np.arange(25.0, 35.0, 0.5))
    return lat, lon


def test_crop_slices_bounds_domain_with_margin():
    lat, lon = _grid()
    rows, cols = crop_slices(lat, lon, (30.0, 31.0, -85.0, -84.0))
    assert lat[rows][0, 0] == 29.5 and lat[rows][-1, 0] == 31.5
    assert lon[:, cols][0, 0] == -85.5 and lon[:, cols][0, -1] == -83.5


def test_crop_slices_strides_to_max_cells():
    lat, lon = _grid()
    rows, cols = crop_slices(lat, lon, (25.0, 35.0, -90.0, -80.0), max_cells=5)
    assert lat[rows, cols].shape[0] <= 5 and lat[rows, cols].shape[1] <= 5


def test_crop_slices_none_outside_grid():
    lat, lon = _grid()
    assert crop_slices(lat, lon, (40.0, 41.0, -70.0, -69.0)) is None


def _write_yaml(tmp, **extra):
    cfg = {"best_track": "bt.dat", "valid_start": 2024092401,
           "valid_end": 2024092404, "domain": [15, 42, -100, -60],
           "out_dir": str(tmp / "out"), **extra}
    path = tmp / "case.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_from_yaml_regrid_plots_block():
    tmp = Path(tempfile.mkdtemp())
    case = from_yaml(_write_yaml(tmp, regrid_plots={
        "out_dir": str(tmp / "maps"), "zoom_domain": [34, 35, -83, -81]}))
    assert case.regrid_plot_dir == tmp / "maps"
    assert case.zoom_domain == (34.0, 35.0, -83.0, -81.0)
    default = from_yaml(_write_yaml(tmp))
    assert default.regrid_plot_dir == tmp / "out" / "regrid"
    assert default.zoom_domain is None


def test_from_yaml_rejects_bad_zoom_domain():
    tmp = Path(tempfile.mkdtemp())
    try:
        from_yaml(_write_yaml(tmp, regrid_plots={"zoom_domain": [34, 35]}))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_load_budget_reads_conservation_csv():
    tmp = Path(tempfile.mkdtemp())
    case = from_yaml(_write_yaml(tmp))
    assert load_budget(case) == {}
    case.out_dir.mkdir(parents=True)
    (case.out_dir / f"regrid_budget_{case.output_slug}.csv").write_text(
        "valid,source,mean_pct_diff\n2024-09-24 02:00,aorc,-0.198\n")
    budget = load_budget(case)
    t = datetime(2024, 9, 24, 2)
    assert "-0.20%" in _budget_note(budget, t, "aorc")
    assert _budget_note(budget, t, "mrms") == ""


class _Cfg:
    def __init__(self, present):
        self._present = present

    def output_path(self, source, t):
        class _P:
            def __init__(self, ok):
                self._ok = ok

            def exists(self):
                return self._ok

            def __str__(self):
                return f"{source}_{t:%Y%m%d%H}.nc"
        return _P((source, t) in self._present)


class _Case:
    def __init__(self, sources, present):
        self.regrid = _Cfg(present)
        self._sources = sources
        self.truth_source = "stage4"
        self.skip_mrms = self.skip_stage4 = self.skip_aorc = False


def _patch_sources(monkeypatch, sources):
    import obs_cases
    monkeypatch.setattr(obs_cases, "regrid_sources", lambda case: sources)


def test_available_sources_keeps_only_fully_regridded(monkeypatch):
    times = [datetime(2024, 9, 26, h) for h in (0, 1, 2)]
    _patch_sources(monkeypatch, ["stage4", "mrms", "aorc"])
    present = {("stage4", t) for t in times}
    case = _Case(["stage4", "mrms", "aorc"], present)
    # only the truth product has been regridded; the others drop out quietly
    assert available_sources(case, times, "plot-regrid") == ["stage4"]


def test_available_sources_errors_on_a_partial_product(monkeypatch):
    times = [datetime(2024, 9, 26, h) for h in (0, 1, 2)]
    _patch_sources(monkeypatch, ["stage4"])
    present = {("stage4", times[0])}          # 1 of 3 hours
    try:
        available_sources(_Case(["stage4"], present), times, "plot-regrid")
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for a half-regridded product")


def test_available_sources_errors_when_nothing_is_regridded(monkeypatch):
    times = [datetime(2024, 9, 26, 0)]
    _patch_sources(monkeypatch, ["stage4"])
    try:
        available_sources(_Case(["stage4"], set()), times, "plot-regrid")
    except SystemExit:
        return
    raise AssertionError("expected SystemExit when nothing is regridded")


def test_from_yaml_panels_subset():
    tmp = Path(tempfile.mkdtemp())
    case = from_yaml(_write_yaml(tmp, regrid_plots={
        "panels": ["compare-regrid"]}))
    assert case.regrid_panels == ("compare-regrid",)
    assert from_yaml(_write_yaml(tmp)).regrid_panels == ()


def test_from_yaml_rejects_unknown_panel():
    tmp = Path(tempfile.mkdtemp())
    try:
        from_yaml(_write_yaml(tmp, regrid_plots={"panels": ["compare-zoom"]}))
    except ValueError as err:
        assert "compare-zoom" in str(err)
        return
    raise AssertionError("expected ValueError")


class _FakeMonkeypatch:
    def __init__(self):
        self._saved = []

    def setattr(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        n_args = fn.__code__.co_argcount
        mp = _FakeMonkeypatch() if n_args else None
        try:
            fn(mp) if mp else fn()
        finally:
            if mp:
                mp.undo()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
