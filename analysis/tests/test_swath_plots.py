"""Tests for the member-exceedance swath maps: the percentage logic, the
member selection (the 0-3 h cycle only when included), and a real render on
synthetic regridded files."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import swath_plots as sp
from test_eyeball_plots import _case, _field, _write_nc

T = datetime


def test_exceedance_percent_counts_members_at_or_above_the_threshold():
    f = np.array([[[0., 30.]], [[25., 10.]], [[24.9, np.nan]], [[50., 40.]]])
    got = sp.exceedance_percent(f, 25.0)
    assert got[0, 0] == 50.0                    # 25 and 50 of 4
    assert got[0, 1] == 200 / 3                 # NaN member leaves the count
    allnan = sp.exceedance_percent(np.full((2, 1, 1), np.nan), 25.0)
    assert np.isnan(allnan[0, 0])


def test_member_text_counts_each_configuration():
    hafs = [("HFSA", None, None)] * 3 + [("HFSB", None, None)] * 2
    assert sp.member_text(18, hafs) == (
        "WoFS 18 members   |   HAFS 5 members (3 HFSA + 2 HFSB)")


def test_hafs_members_skip_the_spinup_cycle_unless_included(tmp_path):
    case = _case(tmp_path)
    w = [x for x in __import__("accum_windows").clock_windows(
        case.valid_start, case.valid_end)][0]
    _write_nc(case.hafs_window_path("HFSA", w.end, w.start), {"precip": _field(5)})
    plain = sp.hafs_members(case, w)
    assert all(i < w.start for _, i, _ in plain) and len(plain) == 8
    case.hafs_include_spinup = True
    assert len(sp.hafs_members(case, w)) == 9


def test_plot_swaths_writes_one_figure_per_window(tmp_path):
    case = _case(tmp_path)
    logs = []
    written = sp.plot_swaths(case, 25.0, log=logs.append)
    assert len(written) == 2 and all(p.exists() for p in written)
    assert written[0].name.endswith("_ge25mm.png")
    assert written[0].parent == sp.swath_dir(case)
    assert any("WoFS 12" in l and "HAFS 8" in l for l in logs)
