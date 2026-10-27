"""Unit tests for the absolute-clock accumulation windows.

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


def test_window_end_stamp_and_label():
    w = aw.Window(datetime(2024, 9, 27, 18), datetime(2024, 9, 27, 21))
    assert w.stamp == "2024092721" and w.hours == 3.0
    assert w.label == "2024-09-27 18Z-21Z"


def test_the_clock_is_anchored_at_00z_for_any_valid_start():
    ws = aw.clock_windows(datetime(2024, 9, 27, 6, 30), datetime(2024, 9, 28, 0))
    assert [w.start.hour for w in ws] == [9, 12, 15, 18, 21]
