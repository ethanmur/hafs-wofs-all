import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

from best_track import (parse_bdeck, parse_bdeck_fixes,
                        parse_bdeck_status, bdeck_summary)


def test_parse_bdeck_times_from_column_and_dedup():
    track = parse_bdeck(FIX / "bal092024_sample.dat")
    # 5 lines but the first two share 2024092400 -> 4 unique fixes.
    assert len(track) == 4
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    times = [t for t, _, _ in track]
    assert times == sorted(times)


def test_parse_bdeck_no_fixes_raises(tmp_path=None):
    import tempfile
    p = Path(tempfile.mkdtemp()) / "empty.dat"
    p.write_text("AL, 09, 2024092400,   , CARQ,   0, 168N,  832W\n")  # not BEST
    try:
        parse_bdeck(p)
        assert False, "expected ValueError"
    except ValueError as e:
        assert str(p) in str(e)


def test_parse_bdeck_fixes_reads_rmw_nautical_miles():
    import tempfile
    p = Path(tempfile.mkdtemp()) / "rmw.dat"
    cols = ["AL", "01", "2025010100", "", "BEST", "0", "100N", "500W",
            "50", "990", "TS", "34", "NEQ", "0", "0", "0", "0",
            "1010", "200", "25"]
    p.write_text(", ".join(cols) + "\n")
    fixes = parse_bdeck_fixes(p)
    assert len(fixes) == 1
    assert fixes[0][3] == 25 * 1.852


def test_parse_bdeck_status_keeps_every_stage():
    track = parse_bdeck_status(FIX / "bal022024_tail.dat")
    # 6 lines, the first two share 2024070900 -> 5 unique fixes.
    assert len(track) == 5
    assert [s for _, _, _, s in track] == ["HU", "TS", "TD", "LO", "EX"]
    assert track[-1][0] == datetime(2024, 7, 10, 0)


def test_bdeck_summary_reports_extent_and_statuses():
    info = bdeck_summary(FIX / "bal022024_tail.dat")
    assert info["n"] == 5
    assert info["first"] == datetime(2024, 7, 9, 0)
    assert info["last"] == datetime(2024, 7, 10, 0)
    assert info["lat_min"] == 32.0 and info["lat_max"] == 43.1
    assert info["lon_min"] == -95.0 and info["lon_max"] == -78.6
    # remnant-low and extratropical stages must survive into the extent
    assert info["statuses"] == ["HU", "TS", "TD", "LO", "EX"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
