"""Tests for the storm-location printout (plan G3, step 1): the distance
arithmetic, nearest-fix rule with its 3 h limit, landfall detection on a
synthetic coast and on real Natural Earth land, and the printed table."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import storm_location as sl
import verification_grid as vg
from case_helpers import make_case
from test_verification_grid import HELENE_ATTRS

T = datetime
# a straight coast at 90W: everything east of it is mainland
COAST = lambda lat, lon: lon > -90.0           # noqa: E731


def _track(points):
    """[(time, lat, lon, status)] from (hours after 2024-07-08 00Z, lat, lon)."""
    return [(T(2024, 7, 8) + timedelta(hours=h), la, lo, "TS")
            for h, la, lo in points]


# --- arithmetic ---------------------------------------------------------------

def test_distance_is_a_geodesic_in_km():
    assert sl.distance_km(30.0, -90.0, 31.0, -90.0) == pytest.approx(111.0,
                                                                     abs=0.8)
    assert sl.distance_km(30.0, -90.0, 30.0, -90.0) == 0.0


def test_the_domain_centre_is_the_middle_of_the_grid(tmp_path):
    spec = vg.wofs_grid_spec(HELENE_ATTRS, 6.0, name="h", source="f.nc")
    path = vg.write_grid_json(tmp_path / "g.json", vg.grid_to_dict(spec))
    case = make_case(tmp_path, grid_json_dir=tmp_path)
    case_path = case.grid_json_path
    case_path.write_text(path.read_text())
    lat, lon = sl.domain_center(case)
    glat, glon = vg.grid_latlon(spec)
    assert glat.min() < lat < glat.max() and glon.min() < lon < glon.max()
    # an even grid: the mean of the four central cells
    ny, nx = glat.shape
    assert lat == pytest.approx(glat[ny // 2 - 1:ny // 2 + 1,
                                     nx // 2 - 1:nx // 2 + 1].mean())


# --- the nearest fix and its limit ----------------------------------------------

def test_nearest_fix_within_three_hours_has_a_signed_offset():
    track = _track([(0, 28, -95), (6, 29, -94), (12, 30, -93)])
    fix, off = sl.nearest_fix(track, T(2024, 7, 8, 7, 30))
    assert fix[0] == T(2024, 7, 8, 6) and off == pytest.approx(-1.5)
    fix, off = sl.nearest_fix(track, T(2024, 7, 8, 10, 30))
    assert fix[0] == T(2024, 7, 8, 12) and off == pytest.approx(1.5)


def test_a_fix_more_than_three_hours_away_is_out_of_bounds():
    track = _track([(0, 28, -95), (6, 29, -94)])
    fix, off = sl.nearest_fix(track, T(2024, 7, 8, 13, 30))   # track ended 7.5 h ago
    assert fix is None and off == pytest.approx(-7.5)
    fix, off = sl.nearest_fix(track, T(2024, 7, 7, 20))        # not yet formed
    assert fix is None and off == pytest.approx(4.0)
    # exactly three hours is still in bounds
    assert sl.nearest_fix(track, T(2024, 7, 8, 9))[0] is not None


# --- landfall -------------------------------------------------------------------

def test_landfall_is_where_the_track_crosses_the_coast():
    track = _track([(0, 28, -92), (6, 28, -88)])      # ocean -> land, halfway
    (t, la, lo), = sl.landfalls(track, COAST)
    assert lo == pytest.approx(-90.0, abs=1e-6) and la == pytest.approx(28.0)
    assert abs(t - T(2024, 7, 8, 3)) < timedelta(seconds=1)


def test_a_track_that_starts_on_land_or_stays_at_sea_has_no_landfall():
    assert sl.landfalls(_track([(0, 28, -85), (6, 28, -80)]), COAST) == []
    assert sl.landfalls(_track([(0, 28, -95), (6, 28, -92)]), COAST) == []


def test_every_ocean_to_land_step_is_found_and_the_last_is_the_reference():
    track = _track([(0, 28, -92), (6, 28, -88), (12, 28, -92),
                    (18, 28, -88)])
    falls = sl.landfalls(track, COAST)
    assert len(falls) == 2
    assert abs(falls[0][0] - T(2024, 7, 8, 3)) < timedelta(seconds=1)
    assert abs(falls[1][0] - T(2024, 7, 8, 15)) < timedelta(seconds=1)


def test_real_land_polygons_know_the_texas_coast_and_ignore_islands():
    try:
        land = sl.mainland_land_test()
    except Exception as exc:                    # no Natural Earth data offline
        pytest.skip(f"Natural Earth land unavailable: {exc}")
    assert land(30.0, -96.0) and land(20.5, -88.0)      # Texas, Yucatan
    assert not land(27.0, -92.0)                          # Gulf of Mexico
    assert not land(12.48, -61.45)                        # Carriacou (island)
    assert not land(18.1, -77.3)                          # Jamaica (island)
    track = _track([(0, 27.0, -95.0), (6, 30.0, -95.0)])   # Gulf -> Galveston
    (_, la, lo), = sl.landfalls(track, land)
    assert 28.5 < la < 29.8 and lo == pytest.approx(-95.0)


# --- the table --------------------------------------------------------------------

def _case(tmp_path, end):
    return make_case(tmp_path, valid_start=T(2024, 7, 8, 0), valid_end=end,
                     best_track=tmp_path / "bt.dat")


def test_rows_report_distance_hours_from_landfall_and_the_out_of_bounds(tmp_path):
    case = _case(tmp_path, T(2024, 7, 8, 18))
    # fixes at 00, 06, 12Z only: the 15-18Z window is out of bounds
    track = _track([(0, 28, -92), (6, 28, -88), (12, 29, -87)])
    res = sl.locate(case, track, (29.0, -86.0), COAST)
    rows = {r["window_end"].hour: r for r in res["rows"]}
    assert sorted(rows) == [3, 6, 9, 12, 15, 18]
    assert rows[3]["fix"][0] == T(2024, 7, 8, 0)
    assert rows[3]["tc_km"] == pytest.approx(
        sl.distance_km(29.0, -86.0, 28.0, -92.0))
    assert rows[18]["fix"] is None and rows[18]["offset_h"] < -3
    # landfall at 03Z: a window midpoint 1.5 h later is +... h after it
    assert rows[6]["from_landfall_h"] == pytest.approx(
        (T(2024, 7, 8, 4, 30) - T(2024, 7, 8, 3)).total_seconds() / 3600)
    assert rows[3]["from_landfall_h"] < 0
    assert rows[3]["landfall_km"] == rows[18]["landfall_km"]


def test_the_printout_names_the_out_of_bounds_hours(tmp_path, capsys):
    case = _case(tmp_path, T(2024, 7, 8, 18))
    track = _track([(0, 28, -92), (6, 28, -88), (12, 29, -87)])
    res = sl.locate(case, track, (29.0, -86.0), COAST)
    sl.print_location(case, track, (29.0, -86.0), res)
    out = capsys.readouterr().out
    assert "Landfall      : 2024-07-08 03:00Z" in out
    assert "time out of bounds: 4.5 h off" in out
    assert "dist to TC" in out and "29.00N 86.00W" in out


def test_a_storm_that_never_made_landfall_says_so(tmp_path, capsys):
    case = _case(tmp_path, T(2024, 7, 8, 6))
    track = _track([(0, 28, -95), (6, 28, -94)])
    res = sl.locate(case, track, (29.0, -86.0), COAST)
    sl.print_location(case, track, (29.0, -86.0), res)
    out = capsys.readouterr().out
    assert "none found" in out
    assert all(r["landfall_km"] is None for r in res["rows"])


def test_a_domain_without_complete_windows_is_refused(tmp_path):
    case = make_case(tmp_path, wofs_available=False)
    with pytest.raises(SystemExit, match="no complete WoFS window"):
        sl.storm_location_case(case, is_land=COAST)


def test_it_is_a_command():
    import run
    assert run.parse_args(["c.yaml", "storm-location"])[1] == "storm-location"
