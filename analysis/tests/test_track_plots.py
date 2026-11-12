"""Tests for the a-deck reader, the a-deck fetch name, and the two track maps
(really rendered on a synthetic domain, best track and HAFS cycles)."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import adeck
import fetch_bdecks as fb
import storm_location as sl
import track_plots as tp
import verification_grid as vg
from case_helpers import make_case
from test_verification_grid import HELENE_ATTRS

T = datetime


def _line(init, tech, tau, lat, lon, extra=""):
    return (f"AL, 09, {init:%Y%m%d%H}, 03, {tech}, {tau:03d}, "
            f"{int(round(lat * 10))}N, {int(round(-lon * 10))}W, 60, 990, "
            f"XX{extra}")


def _adeck(tmp_path, cycles=((T(2024, 9, 26, 0), 0.0), (T(2024, 9, 26, 6), 0.3))):
    """HFSA and HFSB tracks moving north along 84W, one per cycle, with an
    offset per cycle so they differ; plus a repeated wind-radii line."""
    lines = []
    for init, off in cycles:
        for tech, sign in (("HFSA", 1), ("HFSB", -1)):
            for tau in range(0, 49, 6):
                t = init + timedelta(hours=tau)
                lat = 26.0 + 0.1 * (t - T(2024, 9, 25, 18)).total_seconds() / 3600
                lines.append(_line(init, tech, tau, lat, -84.0 + sign * off))
    lines.append(_line(T(2024, 9, 26, 0), "HFSA", 6, 99.0, -1.0))     # dup tau
    lines.append(_line(T(2024, 9, 26, 0), "GFSO", 6, 30.0, -80.0))    # other tech
    lines.append("garbage line")
    path = tmp_path / "aal092024.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


def _bdeck(tmp_path):
    lines = []
    for h in range(0, 60, 6):
        t = T(2024, 9, 25, 18) + timedelta(hours=h)
        lat = 26.0 + 0.1 * h
        lines.append(f"AL, 09, {t:%Y%m%d%H},   , BEST,   0, {int(lat * 10)}N, "
                     "840W, 70, 980, HU, 34, NEQ, 0100, 0100, 0100, 0100")
    path = tmp_path / "bal092024.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def case(tmp_path):
    spec = vg.wofs_grid_spec(HELENE_ATTRS, 6.0, name="h", source="f.nc")
    c = make_case(
        tmp_path, res_km=3.0, storm_name="Helene", deployment="20240926_d1",
        case_slug="2024_09l_helene_20240926_d1", storm_id="2024-09l",
        valid_start=T(2024, 9, 26, 6), valid_end=T(2024, 9, 26, 12),
        best_track=_bdeck(tmp_path),
        eyeball_plot_dir=tmp_path / "figures" / "eyeball-panels" / "x",
        hafs_models={"HFSA": tmp_path / "A", "HFSB": tmp_path / "B"})
    c.grid_json_path.parent.mkdir(parents=True, exist_ok=True)
    vg.write_grid_json(c.grid_json_path, vg.grid_to_dict(spec))
    return c


# --- the a-deck -----------------------------------------------------------------

def test_the_adeck_reader_keeps_hafs_once_per_cycle_and_hour(tmp_path):
    tracks = adeck.parse_adeck(_adeck(tmp_path))
    assert sorted(tracks) == [("HFSA", T(2024, 9, 26, 0)),
                              ("HFSA", T(2024, 9, 26, 6)),
                              ("HFSB", T(2024, 9, 26, 0)),
                              ("HFSB", T(2024, 9, 26, 6))]
    tr = tracks[("HFSA", T(2024, 9, 26, 0))]
    assert [t for t, *_ in tr] == [T(2024, 9, 26, 0) + timedelta(hours=h)
                                   for h in range(0, 49, 6)]
    assert tr[1][1] != 99.0                      # the duplicate line is ignored


def test_the_summary_counts_cycles_per_model(tmp_path):
    info = adeck.adeck_summary(_adeck(tmp_path))
    assert info["HFSA"] == (2, T(2024, 9, 26, 0), T(2024, 9, 26, 6))
    assert info["HFSB"][0] == 2


def test_a_track_is_cut_at_the_window_end_with_an_interpolated_point():
    tr = [(T(2024, 9, 26, h), 26.0 + h / 10, -84.0, 60.0) for h in (0, 6, 12)]
    seg, reached = adeck.track_to(tr, T(2024, 9, 26, 9))
    assert reached and [p[0].hour for p in seg] == [0, 6, 9]
    assert seg[-1][1] == pytest.approx(26.9)
    seg, reached = adeck.track_to(tr, T(2024, 9, 26, 6))   # on a fix
    assert reached and [p[0].hour for p in seg] == [0, 6]
    seg, reached = adeck.track_to(tr, T(2024, 9, 27, 0))   # track too short
    assert not reached and len(seg) == 3


def test_adeck_file_names_follow_the_bdeck_ones():
    assert fb.adeck_name("2024-14l") == "aal142024.dat"
    assert fb.adeck_name("2023-09e") == "aep092023.dat"
    assert fb.adeck_name("bal022024.dat") == "aal022024.dat"
    assert fb.bdeck_year(fb.adeck_name("2024-14l")) == "2024"


def test_paths_sit_beside_the_other_data_and_figures(case, tmp_path):
    assert tp.adeck_path(case) == tmp_path.parent / "adeck" / "aal092024.dat"
    assert tp.track_dir(case) == tmp_path / "figures" / "storm-tracks"


# --- the maps ---------------------------------------------------------------------

def _patch(monkeypatch, case, tmp_path, cycles=None):
    adeck_file = _adeck(tmp_path)
    monkeypatch.setattr(tp, "adeck_path", lambda c: adeck_file)
    monkeypatch.setattr(tp, "_landfall_test",
                        lambda log=print: (lambda la, lo: lo > 1000))
    cyc = cycles or {"HFSA": [T(2024, 9, 26, 0), T(2024, 9, 26, 6)],
                     "HFSB": [T(2024, 9, 26, 0)]}
    monkeypatch.setattr(tp, "hafs_cycles", lambda c, w: cyc)


def test_hafs_tracks_one_figure_per_window_named_by_case_and_end(
        case, tmp_path, monkeypatch):
    _patch(monkeypatch, case, tmp_path)
    written = tp.plot_hafs_tracks(case, dpi=60, log=lambda *a: 0)
    assert [p.name for p in written] == [
        "hafs-tracks_2024_09l_helene_20240926_d1_2024092609.png",
        "hafs-tracks_2024_09l_helene_20240926_d1_2024092612.png"]
    assert all(p.exists() and p.stat().st_size > 5000 for p in written)
    assert {p.parent for p in written} == {tmp_path / "figures"
                                           / "storm-tracks"}


def test_cycles_missing_from_the_adeck_are_counted_not_fatal(
        case, tmp_path, monkeypatch):
    _patch(monkeypatch, case, tmp_path,
           cycles={"HFSA": [T(2024, 9, 26, 0), T(2024, 9, 25, 0)],
                   "HFSB": []})
    track = tp.bt.parse_bdeck_status(case.best_track)
    tracks = adeck.parse_adeck(tp.adeck_path(case))
    olat, olon, center = tp.domain_outline(case)
    w = aw.Window(T(2024, 9, 26, 6), T(2024, 9, 26, 9))
    path, drawn, missing = tp.plot_window_hafs_tracks(
        case, w, track, tracks, tp.hafs_cycles(case, w), olat, olon, center,
        None, dpi=60)
    assert (drawn, missing) == (1, 1) and path.exists()


def test_a_window_without_cycles_is_skipped(case, tmp_path, monkeypatch,
                                             capsys):
    _patch(monkeypatch, case, tmp_path, cycles={"HFSA": [], "HFSB": []})
    assert tp.plot_hafs_tracks(case, dpi=60) == []
    assert "no HAFS cycles" in capsys.readouterr().out


def test_tc_location_figures_include_an_out_of_bounds_window(
        case, tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, case, tmp_path)
    import dataclasses
    late = dataclasses.replace(case, valid_end=T(2024, 9, 28, 6))
    written = tp.plot_tc_location(late, dpi=60)
    out = capsys.readouterr().out
    assert len(written) == 16
    assert "time out of bounds" in out
    assert all(p.name.startswith("tc-location_2024_09l_helene") for p in written)


def test_a_missing_adeck_says_how_to_fetch_it(case, tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "adeck_path", lambda c: tmp_path / "none.dat")
    with pytest.raises(SystemExit, match="fetch_bdecks.py --adeck"):
        tp.plot_hafs_tracks(case)


def test_unavailable_domains_are_refused_and_commands_exist(tmp_path):
    with pytest.raises(SystemExit, match="no complete WoFS window"):
        tp.plot_tc_location(make_case(tmp_path, wofs_available=False))
    import run
    for c in ("plot-hafs-tracks", "plot-tc-location"):
        assert run.parse_args(["c.yaml", c])[1] == c
