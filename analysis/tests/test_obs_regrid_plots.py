"""Unit tests for obs_regrid_plots' pure logic (cropping, source selection, budget CSV).

Run directly:   python3 analysis/tests/test_obs_regrid_plots.py
Or via pytest:  pytest analysis/tests/test_obs_regrid_plots.py -v
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from case_helpers import make_case
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


def _case(tmp):
    return make_case(tmp, case_slug="c", valid_start=datetime(2024, 9, 24, 0),
                     valid_end=datetime(2024, 9, 24, 3))


def test_load_budget_reads_conservation_csv():
    tmp = Path(tempfile.mkdtemp())
    case = _case(tmp)
    assert load_budget(case) == {}
    case.regrid_dir.mkdir(parents=True)
    case.budget_csv_path.write_text(
        "valid,source,mean_pct_diff\n2024-09-24 02:00,stage4,-0.198\n")
    budget = load_budget(case)
    t = datetime(2024, 9, 24, 2)
    assert "-0.20%" in _budget_note(budget, t, "stage4")
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
        return _P((source, t) in self._present)


class _Case:
    def __init__(self, sources):
        self.regrid_sources = tuple(sources)


def test_available_sources_keeps_only_fully_regridded():
    times = [datetime(2024, 9, 26, h) for h in (0, 1, 2)]
    cfg = _Cfg({("stage4", t) for t in times})
    # only the truth product has been regridded; the other drops out quietly
    assert available_sources(_Case(["stage4", "mrms"]), cfg, times) == ["stage4"]


def test_available_sources_errors_on_a_partial_product():
    times = [datetime(2024, 9, 26, h) for h in (0, 1, 2)]
    cfg = _Cfg({("stage4", times[0])})          # 1 of 3 hours
    try:
        available_sources(_Case(["stage4"]), cfg, times)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for a half-regridded product")


def test_available_sources_errors_when_nothing_is_regridded():
    times = [datetime(2024, 9, 26, 0)]
    try:
        available_sources(_Case(["stage4"]), _Cfg(set()), times)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit when nothing is regridded")
