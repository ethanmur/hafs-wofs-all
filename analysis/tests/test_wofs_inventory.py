import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wofs_inventory as wi


def test_parse_wofs_name():
    step, init, valid = wi.parse_wofs_name(
        "wofs_ALL_36_20240927_0900_1200.nc")
    assert step == 36
    assert init == datetime(2024, 9, 27, 9, 0)
    assert valid == "1200"


def test_parse_wofs_name_rejects_other_files():
    assert wi.parse_wofs_name("wofs_ALL_36_20240927_0900.nc") is None
    assert wi.parse_wofs_name("notes.txt") is None


def test_expected_valid_matches_names_and_wraps_midnight():
    init = datetime(2024, 9, 27, 9, 0)
    assert wi.expected_valid_hhmm(init, 36) == "1200"
    assert wi.expected_valid_hhmm(datetime(2024, 9, 27, 23, 30), 24) == "0130"


def test_missing_steps():
    assert wi.missing_steps([0, 1, 2, 3]) == []
    assert wi.missing_steps([0, 1, 4, 5]) == [2, 3]
    assert wi.missing_steps([]) == []


def test_cycle_gaps_ignores_regular_cycling():
    t = lambda h, m: datetime(2024, 9, 26, h, m)
    inits = [t(0, 0), t(0, 30), t(1, 0), t(17, 0), t(17, 30)]
    assert wi.cycle_gaps(inits) == [(t(1, 0), t(17, 0))]


def test_grid_id_distinguishes_domains_and_ignores_nx_type():
    base = dict(start_lat=29.02168, start_lon=-86.84998, lon_orient=-82.25424,
                scale_lat_1=30.0, scale_lat_2=60.0, d_km=3.0, r_km=6371.2,
                nx="300", ny="300")
    same = dict(base)
    moved = dict(base, start_lat=24.06089)
    assert wi.grid_id(base) == wi.grid_id(same)
    assert wi.grid_id(base) != wi.grid_id(moved)


def test_scan_init_dir_prefers_lowest_step_and_flags_name_mismatch(tmp_path):
    d = tmp_path / "0900"
    d.mkdir()
    for step, valid in [(1, "0905"), (2, "0911")]:
        (d / f"wofs_ALL_{step:02d}_20240927_0900_{valid}.nc").write_text("")
    info = wi.scan_init_dir(d)
    assert info.steps == [1, 2]
    assert info.first_file.name.startswith("wofs_ALL_01")
    assert len(info.mismatches) == 1 and "step 2" in info.mismatches[0]


def test_scan_init_dir_empty_returns_none(tmp_path):
    assert wi.scan_init_dir(tmp_path) is None


# --- window availability ----------------------------------------------------

def _w(start, end):
    return wi.clock_windows(start, end)[0]


def test_compact_ranges():
    assert wi.compact_ranges([2, 3, 4, 7]) == "2-4,7"
    assert wi.compact_ranges([]) == ""
    assert wi.compact_ranges([5]) == "5"


def test_init_windows_for_six_hour_run_from_1800():
    init = datetime(2024, 9, 27, 18, 0)
    wins = wi.init_windows(init, range(0, 73))
    assert [(w.start.hour, w.end.hour) for w in wins] == [(18, 21), (21, 0)]


def test_init_windows_for_half_hour_init_gets_one_window():
    init = datetime(2024, 9, 27, 17, 30)
    wins = wi.init_windows(init, range(0, 73))
    assert [(w.start.hour, w.end.hour) for w in wins] == [(18, 21)]


def test_window_status_complete_needs_steps_s_plus_1_to_s_plus_36():
    init = datetime(2024, 9, 27, 18, 0)
    win = wi.init_windows(init, range(0, 73))[0]            # 18-21Z, s = 0
    status, present, required, missing = wi.window_status(
        init, set(range(1, 37)), win)                        # no step 00
    assert (status, present, required, missing) == ("complete", 36, 36, [])


def test_window_status_second_window_uses_later_steps():
    init = datetime(2024, 9, 27, 18, 0)
    win = wi.init_windows(init, range(0, 73))[1]            # 21-00Z, s = 36
    assert wi.window_status(init, set(range(0, 37)), win)[0] == "short_run"
    assert wi.window_status(init, set(range(0, 73)), win)[0] == "complete"


def test_window_status_hole_vs_short_run():
    init = datetime(2024, 9, 27, 18, 0)
    win = wi.init_windows(init, range(0, 73))[0]
    steps = set(range(1, 37)) - {10, 11}
    status, _, _, missing = wi.window_status(init, steps, win)
    assert status == "missing_steps" and missing == [10, 11]
    status, present, _, missing = wi.window_status(init, {1}, win)   # Idalia
    assert status == "short_run" and present == 1 and missing[0] == 2


def test_window_status_no_files():
    init = datetime(2024, 9, 27, 18, 0)
    win = wi.init_windows(init, [])[0]
    assert wi.window_status(init, set(), win)[0] == "no_files"


def test_infer_init_dates_overnight_run_rolls_to_next_day():
    hhmms = ["0000", "0030", "0300", "1700", "1730", "2330"]
    got = wi.infer_init_dates("20230819", hhmms)
    assert got["1700"] == datetime(2023, 8, 19, 17, 0)
    assert got["2330"] == datetime(2023, 8, 19, 23, 30)
    assert got["0000"] == datetime(2023, 8, 20, 0, 0)
    assert got["0300"] == datetime(2023, 8, 20, 3, 0)


def test_infer_init_dates_same_day_morning_start():
    got = wi.infer_init_dates("20240927", ["0500", "0530", "1500"])
    assert got["0500"] == datetime(2024, 9, 27, 5, 0)
    assert got["1500"] == datetime(2024, 9, 27, 15, 0)


def _dep(inits, empty=None):
    dep = wi.Deployment(storm="2023_IDALIA", name="20230830_d1",
                        path=Path("."))
    dep.grids = {("g",): ["x"]}
    for t, steps in inits:
        dep.inits.append(wi.InitDir(Path(t.strftime("%H%M")), t,
                                    list(steps), Path("f.nc")))
    dep.empty_inits = empty or {}
    return dep


def test_availability_rows_flag_short_empty_and_complete_runs():
    t1800 = datetime(2023, 8, 30, 18, 0)
    dep = _dep([(t1800, range(0, 73)), (datetime(2023, 8, 30, 19, 0), [1])],
               empty={"1830": datetime(2023, 8, 30, 18, 30)})
    rows = wi.availability_rows(dep, {("g",): "G1"})
    by = {(r["init_folder"], r["window_start"][-5:]): r for r in rows}
    assert by[("1800", "1800Z")]["status"] == "complete"
    assert by[("1900", "2100Z")]["status"] == "short_run"
    assert by[("1830", "2100Z")]["status"] == "no_files"
    assert by[("1830", "2100Z")]["init_inferred"] is True
    summary = wi.summarize_availability(rows)[("2023_IDALIA", "20230830_d1")]
    assert summary["windows_served"] == 2 and summary["no_files"] >= 1


def test_deployment_with_nothing_complete_is_flagged_unserved():
    dep = _dep([(datetime(2023, 8, 30, 18, 0), [1])])
    rows = wi.availability_rows(dep, {("g",): "G1"})
    summary = wi.summarize_availability(rows)[("2023_IDALIA", "20230830_d1")]
    assert summary["windows_served"] == 0


def test_deployment_overlaps_same_storm_only():
    a = _dep([(datetime(2024, 9, 26, 17, 0), range(0, 73))])
    b = _dep([(datetime(2024, 9, 26, 20, 0), range(0, 73))])
    a.name, b.name = "d1", "d2"
    assert len(wi.deployment_overlaps([a, b])) == 1
    b.storm = "other"
    assert wi.deployment_overlaps([a, b]) == []
