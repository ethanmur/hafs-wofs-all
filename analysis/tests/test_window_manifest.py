"""Tests for the window manifest: expected cycles, per-run statuses, the
window summary, and a build from a synthetic WoFS/HAFS archive on disk."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import window_manifest as wm
from case_helpers import make_case

T = datetime


def _win(start, hours=3):
    return aw.Window(start, start + timedelta(hours=hours))


# --- expected HAFS cycles ---------------------------------------------------

def test_expected_cycles_include_48h_back_and_exclude_the_window_start():
    got = wm.expected_cycles(T(2024, 7, 10, 18))
    assert got[0] == T(2024, 7, 8, 18) and got[-1] == T(2024, 7, 10, 12)
    assert len(got) == 8 and T(2024, 7, 10, 18) not in got


def test_expected_cycles_are_eight_for_any_window_on_the_3h_clock():
    for hour in range(0, 24, 3):
        assert len(wm.expected_cycles(T(2024, 7, 10, hour))) == 8, hour


def test_expected_cycles_follow_lookback_and_cycle_spacing():
    assert len(wm.expected_cycles(T(2024, 7, 10, 18), lookback_h=24)) == 4
    assert len(wm.expected_cycles(T(2024, 7, 10, 18), 48, cycle_h=12)) == 4


# --- WoFS rows --------------------------------------------------------------

def _dep(inits, empty=None, ne=18):
    import wofs_inventory as wi
    dep = wi.Deployment(storm="2024_BERYL", name="20240708_d1", path=Path("/w"))
    for t, steps in inits:
        dep.inits.append(wi.InitDir(Path(t.strftime("%H%M")), t, list(steps),
                                    Path("f.nc")))
    dep.empty_inits = empty or {}
    dep.sizes = {"ne": ne}
    return dep


def test_wofs_rows_status_per_init_and_window(tmp_path):
    case = make_case(tmp_path, hafs_models={})
    windows = aw.clock_windows(T(2024, 7, 8, 18), T(2024, 7, 9, 0))
    dep = _dep([(T(2024, 7, 8, 18), range(0, 73)),          # both windows
                (T(2024, 7, 8, 19), range(0, 37)),          # 3 h run
                (T(2024, 7, 8, 20), [1])],                  # one file
               empty={"1830": T(2024, 7, 8, 18, 30)})
    rows = wm.wofs_rows(case, dep, windows)
    by = {(r["run"], r["window_start"][-5:]): r for r in rows}
    assert by[("202407081800", "1800Z")]["status"] == "complete"
    assert by[("202407081800", "2100Z")]["status"] == "complete"
    # init 19:00 starts after the 18Z window, so only the 21Z one is listed
    assert ("202407081900", "1800Z") not in by
    assert by[("202407081900", "2100Z")]["status"] == "short_run"
    assert by[("202407082000", "2100Z")]["status"] == "short_run"
    assert by[("202407081830", "2100Z")]["status"] == "no_files"
    assert all(r["n_members"] == 18 and r["model"] == "wofs" for r in rows)


# --- HAFS rows and the summary ---------------------------------------------

LEADS = list(range(0, 127, 3))


def _hafs(tmp_path, inits_a, inits_b=()):
    case = make_case(tmp_path, hafs_models={"HFSA": tmp_path / "A",
                                            "HFSB": tmp_path / "B"})
    cycles = {"HFSA": {t: LEADS for t in inits_a},
              "HFSB": {t: LEADS for t in inits_b}}
    files = {c: {(t, l): Path(f"/h/{c}/{t:%Y%m%d%H}/f{l:03d}.grb2")
                 for t in cycles[c] for l in LEADS} for c in cycles}
    return case, cycles, files


def test_hafs_rows_cover_every_expected_cycle_including_missing_ones(tmp_path):
    ws = T(2024, 7, 10, 18)
    present = [T(2024, 7, 8, 18), T(2024, 7, 9, 0)]
    case, cycles, files = _hafs(tmp_path, present)
    rows = wm.hafs_rows(case, cycles, files, [_win(ws)])
    a = [r for r in rows if r["model"] == "hfsa"]
    assert len(a) == 8 and len([r for r in rows if r["model"] == "hfsb"]) == 8
    assert sum(r["status"] == "complete" for r in a) == 2
    assert sum(r["status"] == "no_cycle" for r in a) == 6
    done = next(r for r in a if r["status"] == "complete")
    # the 15-18Z... bucket for this window lives in the file named for its end
    assert done["source"].endswith(f"f{int(done['lead_h']):03d}.grb2")
    assert all(r["source"] == "" for r in a if r["status"] != "complete")


def test_hafs_rows_distinguish_short_runs_and_missing_files(tmp_path):
    ws = T(2024, 7, 10, 18)
    case, cycles, files = _hafs(tmp_path, [T(2024, 7, 9, 0), T(2024, 7, 9, 6)])
    cycles["HFSA"][T(2024, 7, 9, 0)] = [0, 3, 6]                # ends early
    # the window ends 39 h after the 06Z init; that is the file that is missing
    cycles["HFSA"][T(2024, 7, 9, 6)] = [l for l in LEADS if l != 39]
    rows = wm.hafs_rows(case, cycles, files, [_win(ws)])
    st = {r["run"]: r["status"] for r in rows if r["model"] == "hfsa"
          and r["status"] != "no_cycle"}
    assert st == {"2024070900": "short_run", "2024070906": "missing_file"}


def _summary(tmp_path, n_a, n_b, wofs_complete=True):
    ws = T(2024, 7, 10, 18)
    base = T(2024, 7, 8, 18)
    case, cycles, files = _hafs(
        tmp_path, [base + timedelta(hours=6 * k) for k in range(n_a)],
        [base + timedelta(hours=6 * k) for k in range(n_b)])
    case.hafs_min_members = 8
    window = _win(ws)
    dep = _dep([(ws - timedelta(hours=0),
                 range(0, 73) if wofs_complete else [1])])
    rows = wm.wofs_rows(case, dep, [window]) + wm.hafs_rows(
        case, cycles, files, [window])
    return wm.summarize(case, rows, [window])[0]


def test_window_is_testable_with_a_wofs_run_and_enough_hafs_members(tmp_path):
    s = _summary(tmp_path, 4, 4)
    assert s["hfsa_members"] == 4 and s["hfsb_members"] == 4
    assert (s["hafs_total"], s["hafs_status"], s["category"]) == (
        8, "ok", "testable")


def test_a_window_with_too_few_hafs_members_is_wofs_only(tmp_path):
    s = _summary(tmp_path, 1, 0)
    assert (s["hafs_total"], s["hafs_status"], s["category"]) == (
        1, "thin", "wofs_only")
    s = _summary(tmp_path, 0, 0)
    assert (s["hafs_status"], s["category"]) == ("none", "wofs_only")


def test_a_window_without_a_complete_wofs_run_is_no_wofs(tmp_path):
    assert _summary(tmp_path, 4, 4, wofs_complete=False)["category"] == \
        "no_wofs"


def test_pooled_threshold_lets_one_configuration_carry_a_window(tmp_path):
    s = _summary(tmp_path, 8, 0)
    assert s["hfsa_members"] == 8 and s["hfsb_members"] == 0
    assert s["category"] == "testable"


def test_complete_wofs_pairs_are_what_regrid_wofs_will_process(tmp_path):
    case = make_case(tmp_path, hafs_models={})
    windows = aw.clock_windows(T(2024, 7, 8, 18), T(2024, 7, 9, 0))
    dep = _dep([(T(2024, 7, 8, 18), range(0, 73)),
                (T(2024, 7, 8, 19), range(0, 37))])
    pairs = wm.wofs_complete_pairs(wm.wofs_rows(case, dep, windows))
    assert sorted((i, w.start.hour) for i, w in pairs) == [
        (T(2024, 7, 8, 18), 18), (T(2024, 7, 8, 18), 21)]


# --- built from disk --------------------------------------------------------

GRID = dict(start_lat=29.0, start_lon=-86.8, lon_orient=-82.25,
            scale_lat_1=30.0, scale_lat_2=60.0, d_km=3.0, r_km=6371.2,
            nx="4", ny="4", CEN_LAT=33.0, CEN_LON=-82.2)


def _write_wofs(root, init, steps, ne=2):
    folder = root / f"{init:%H%M}"
    folder.mkdir(parents=True, exist_ok=True)
    for s in steps:
        valid = init + timedelta(minutes=5 * s)
        ds = xr.Dataset({"rain": (("ne", "lat", "lon"),
                                  np.zeros((ne, 4, 4), dtype="f4"))})
        ds.attrs = dict(GRID, init_time=f"{init:%Y%m%d_%H%M%S}",
                        valid_time="x")
        ds.to_netcdf(folder / (f"wofs_ALL_{s:02d}_{init:%Y%m%d}_{init:%H%M}_"
                               f"{valid:%H%M}.nc"))


def _write_hafs(config_dir, sid, config, inits):
    for t in inits:
        d = config_dir / f"{t:%Y%m%d%H}"
        d.mkdir(parents=True, exist_ok=True)
        for lead in LEADS:
            (d / f"{sid}.{t:%Y%m%d%H}.{config.lower()}.parent.atm."
                 f"f{lead:03d}.grb2").write_text("")


def test_build_manifest_from_a_synthetic_archive(tmp_path):
    init = T(2024, 7, 8, 18)
    wofs_root = tmp_path / "wofs" / "2024_BERYL" / "20240708_d1"
    _write_wofs(wofs_root, init, range(0, 73))
    _write_wofs(wofs_root, T(2024, 7, 8, 19), range(0, 5))    # truncated run
    (wofs_root / "2000").mkdir()                              # empty folder
    cycles = [T(2024, 7, 6, 18) + timedelta(hours=6 * k) for k in range(6)]
    _write_hafs(tmp_path / "A", "02l", "HFSA", cycles)
    _write_hafs(tmp_path / "B", "02l", "HFSB", cycles[:2])
    case = make_case(tmp_path, wofs_root=wofs_root,
                     valid_start=T(2024, 7, 8, 18), valid_end=T(2024, 7, 9, 0),
                     hafs_models={"HFSA": tmp_path / "A",
                                  "HFSB": tmp_path / "B"})
    m = wm.build_manifest(case)
    assert [w.start.hour for w in m.windows] == [18, 21]
    summary = {r["window_start"][-5:]: r for r in m.summary}
    first, second = summary["1800Z"], summary["2100Z"]
    # init 18:00 is complete for both windows; the truncated 19:00 run is not
    assert first["wofs_complete_runs"] == 1 and second["wofs_complete_runs"] == 1
    assert first["hfsa_members"] == 6 and first["hfsb_members"] == 2
    assert first["hafs_total"] == 8 and first["category"] == "testable"
    assert any("empty init" in n for n in m.notes)
    models = {r["model"] for r in m.rows}
    assert models == {"wofs", "hfsa", "hfsb"}


def test_build_manifest_case_writes_both_tables(tmp_path, capsys):
    init = T(2024, 7, 8, 18)
    wofs_root = tmp_path / "wofs" / "2024_BERYL" / "20240708_d1"
    _write_wofs(wofs_root, init, range(0, 73))
    case = make_case(tmp_path, wofs_root=wofs_root,
                     valid_start=T(2024, 7, 8, 18), valid_end=T(2024, 7, 9, 0),
                     hafs_models={})
    wm.build_manifest_case(case)
    assert case.manifest_path.exists() and case.window_summary_path.exists()
    out = capsys.readouterr().out
    assert "wofs_only" in out and "2 windows" in out


def test_unavailable_deployment_gets_no_manifest(tmp_path, capsys):
    case = make_case(tmp_path, wofs_available=False, valid_start=None,
                     valid_end=None)
    assert wm.build_manifest_case(case) is None
    assert not case.manifest_path.exists()
