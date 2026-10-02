"""Unit tests for the absolute-clock accumulation windows (D2).

Run directly:   python3 analysis/tests/test_accum_windows.py
Or via pytest:  pytest analysis/tests/test_accum_windows.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import accum_windows as aw


def test_windows_snap_to_the_00z_clock():
    ws = aw.clock_windows(datetime(2023, 8, 19, 17), datetime(2023, 8, 20, 3))
    assert [w.start.hour for w in ws] == [18, 21, 0]
    assert ws[0].start == datetime(2023, 8, 19, 18)
    assert ws[-1].end == datetime(2023, 8, 20, 3)


def test_partial_windows_at_both_ends_are_dropped():
    ws = aw.clock_windows(datetime(2023, 8, 19, 17), datetime(2023, 8, 19, 23))
    assert [w.label for w in ws] == ["2023-08-19 18Z-21Z"]


def test_wofs_1730z_run_covers_18_21z_at_f00_30():
    run = aw.ModelRun("wofs", datetime(2023, 8, 19, 17, 30), 360, 5)
    window = aw.Window(datetime(2023, 8, 19, 18), datetime(2023, 8, 19, 21))
    cov = aw.covers(run, window)
    assert cov is not None
    assert (cov.lead_start_min, cov.lead_end_min) == (30, 210)
    assert cov.lead_label == "f00:30-f03:30"
    assert cov.lead_h == 3.5


def test_wofs_1430z_run_ends_before_the_window_does():
    run = aw.ModelRun("wofs", datetime(2023, 8, 19, 14, 30), 360, 5)
    window = aw.Window(datetime(2023, 8, 19, 18), datetime(2023, 8, 19, 21))
    assert aw.covers(run, window) is None


def test_run_starting_inside_the_window_is_dropped():
    run = aw.ModelRun("wofs", datetime(2023, 8, 19, 18, 30), 360, 5)
    window = aw.Window(datetime(2023, 8, 19, 18), datetime(2023, 8, 19, 21))
    assert aw.covers(run, window) is None


def test_hafs_bucket_granularity_rejects_off_bucket_leads():
    # A 3 h window two hours after a HAFS init is not a whole APCP bucket.
    window = aw.Window(datetime(2023, 8, 19, 20), datetime(2023, 8, 19, 23))
    assert aw.covers(aw.ModelRun("hafs", datetime(2023, 8, 19, 18),
                                 126 * 60, 180), window) is None
    window = aw.Window(datetime(2023, 8, 19, 21), datetime(2023, 8, 20, 0))
    cov = aw.covers(aw.ModelRun("hafs", datetime(2023, 8, 19, 18),
                                126 * 60, 180), window)
    assert cov is not None and cov.lead_label == "f003-f006"


def test_hafs_cycles_land_on_the_clock_so_every_window_is_reachable():
    runs = aw.hafs_runs(datetime(2023, 8, 17, 0), datetime(2023, 8, 19, 18))
    windows = aw.clock_windows(datetime(2023, 8, 19, 0),
                               datetime(2023, 8, 20, 0))
    cov = aw.coverage(runs, windows)
    covered = {c.window for c in cov}
    assert covered == set(windows)
    assert all(c.lead_end_min % 180 == 0 for c in cov)


def test_lead_is_measured_at_the_window_end():
    run = aw.ModelRun("hafs", datetime(2023, 8, 19, 0), 126 * 60, 180)
    window = aw.Window(datetime(2023, 8, 19, 18), datetime(2023, 8, 19, 21))
    assert aw.covers(run, window).lead_h == 21.0


def test_window_beyond_max_lead_is_dropped():
    run = aw.ModelRun("hafs", datetime(2023, 8, 14, 0), 126 * 60, 180)
    window = aw.Window(datetime(2023, 8, 19, 9), datetime(2023, 8, 19, 12))
    assert aw.covers(run, window) is None            # 132 h > 126 h


def test_wofs_runs_span_the_deployment_and_reach_past_its_end():
    class Dom:
        name = "wofs_1"
        valid_start = datetime(2023, 8, 19, 17)
        valid_end = datetime(2023, 8, 20, 3)
    runs = aw.wofs_runs(Dom())
    assert len(runs) == 21                            # 10 h at 30 min, inclusive
    last = runs[-1]
    assert last.init == datetime(2023, 8, 20, 3)
    window = aw.Window(datetime(2023, 8, 20, 6), datetime(2023, 8, 20, 9))
    assert aw.covers(last, window) is not None        # f03-f06, still valid


def test_format_lead_reads_as_hours_and_minutes():
    assert aw.format_lead(180) == "f003"
    assert aw.format_lead(210) == "f03:30"
    assert aw.format_lead(0) == "f000"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
