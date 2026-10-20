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
