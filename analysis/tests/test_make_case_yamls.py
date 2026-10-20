import csv
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_case_yamls as mc


def _win(start, end, status="complete"):
    return {"window_start": start, "window_end": end, "status": status}


def test_floor_to_cycle():
    assert mc.floor_to_cycle(datetime(2024, 7, 6, 21, 0)) == \
        datetime(2024, 7, 6, 18, 0)
    assert mc.floor_to_cycle(datetime(2024, 7, 6, 0, 0)) == \
        datetime(2024, 7, 6, 0, 0)


def test_served_span_uses_only_complete_windows():
    rows = [_win("2024-07-08 1800Z", "2024-07-08 2100Z"),
            _win("2024-07-08 2100Z", "2024-07-09 0000Z", "short_run"),
            _win("2024-07-09 0300Z", "2024-07-09 0600Z"),
            _win("2024-07-09 0300Z", "2024-07-09 0600Z")]   # second init
    start, end, n = mc.served_span(rows)
    assert start == datetime(2024, 7, 8, 18) and end == datetime(2024, 7, 9, 6)
    assert n == 2


def test_served_span_none_when_nothing_complete():
    assert mc.served_span([_win("a", "b", "short_run")]) is None
    assert mc.served_span([]) is None


def test_build_case_hafs_range_is_48h_back_floored_to_a_cycle():
    inv = {"storm": "2024_BERYL", "deployment": "20240708_d1"}
    case = mc.build_case(inv, [_win("2024-07-08 2100Z", "2024-07-09 0000Z")])
    assert case["init_start"] == datetime(2024, 7, 6, 18)
    assert case["init_end"] == datetime(2024, 7, 9, 0)


def test_rendered_available_case_parses_with_expected_keys():
    inv = {"storm": "2024_BERYL", "deployment": "20240708_d1"}
    case = mc.build_case(inv, [_win("2024-07-08 1800Z", "2024-07-08 2100Z")])
    cfg = yaml.safe_load(mc.render_case(case))
    assert cfg["storm_id"] == "2024-02l" and cfg["truth_source"] == "stage4"
    assert cfg["valid_start"] == 2024070818 and cfg["valid_end"] == 2024070821
    assert cfg["wofs"]["available"] is True
    assert cfg["hafs"]["lookback_h"] == 48
    assert set(cfg["hafs"]["models"]) == {"HFSA", "HFSB"}
    assert cfg["hafs"]["models"]["HFSB"].endswith("/hafs/2024-02l/HFSB")
    assert cfg["wofs"]["root"].endswith("/2024_BERYL/20240708_d1")


def test_rendered_unavailable_case_has_no_window_or_hafs_block():
    inv = {"storm": "2023_IDALIA", "deployment": "20230830_d1"}
    cfg = yaml.safe_load(mc.render_case(mc.build_case(inv, [])))
    assert cfg["wofs"]["available"] is False
    assert "valid_start" not in cfg and "hafs" not in cfg
    assert cfg["storm_id"] == "2023-10l"


def test_case_filename():
    inv = {"storm": "2023_HILARY", "deployment": "20230820_d2"}
    assert mc.case_filename(mc.build_case(inv, [])) == \
        "2023_09e_hilary_20230820_d2.yaml"


def test_main_writes_one_file_per_deployment(tmp_path):
    inv = tmp_path / "inv.csv"
    win = tmp_path / "win.csv"
    with open(inv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["storm", "deployment"])
        w.writeheader()
        w.writerows([{"storm": "2024_BERYL", "deployment": "20240708_d1"},
                     {"storm": "2023_IDALIA", "deployment": "20230830_d1"}])
    with open(win, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["storm", "deployment",
                                           "window_start", "window_end",
                                           "status"])
        w.writeheader()
        w.writerow({"storm": "2024_BERYL", "deployment": "20240708_d1",
                    "window_start": "2024-07-08 1800Z",
                    "window_end": "2024-07-08 2100Z", "status": "complete"})
    out = tmp_path / "storms"
    mc.main([str(inv), str(win), "--out-dir", str(out)])
    assert sorted(p.name for p in out.glob("*.yaml")) == [
        "2023_10l_idalia_20230830_d1.yaml",
        "2024_02l_beryl_20240708_d1.yaml"]
