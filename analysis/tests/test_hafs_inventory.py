import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hafs_inventory as hi
from accum_windows import Window


def _win(h0, day=10):
    start = datetime(2024, 7, day, h0)
    end = datetime(2024, 7, day + (h0 + 3) // 24, (h0 + 3) % 24)
    return Window(start, end)


def test_window_status_bucket_file_is_named_for_the_window_end():
    init = datetime(2024, 7, 9, 18)
    win = _win(15, 10)                                  # 15-18Z on the 10th
    assert (win.end - init).total_seconds() / 3600 == 24
    assert hi.hafs_window_status(init, [0, 3, 24], win) == "complete"
    assert hi.hafs_window_status(init, [0, 3, 21], win) == "short_run"
    assert hi.hafs_window_status(init, [0, 3, 6, 12, 18, 27], win) == \
        "missing_file"
    assert hi.hafs_window_status(init, [0, 3, 6], win) == "short_run"
    assert hi.hafs_window_status(init, [], win) == "no_files"


def test_window_status_window_before_init_is_not_served():
    init = datetime(2024, 7, 10, 18)
    assert hi.hafs_window_status(init, [3, 6], _win(12)) == \
        "not_on_bucket_grid"


def test_members_include_exactly_48h_back_and_exclude_window_start_cycle():
    ws = datetime(2024, 7, 10, 18)
    inits = [datetime(2024, 7, 8, 12), datetime(2024, 7, 8, 18),
             datetime(2024, 7, 9, 0), datetime(2024, 7, 10, 12),
             datetime(2024, 7, 10, 18), datetime(2024, 7, 10, 21)]
    got = hi.members_for_window(ws, inits)
    assert got == [datetime(2024, 7, 8, 18), datetime(2024, 7, 9, 0),
                   datetime(2024, 7, 10, 12)]


def test_six_hourly_cycles_give_eight_members_for_any_window_start():
    for hour in (18, 21):
        ws = datetime(2024, 7, 10, hour)
        inits = [datetime(2024, 7, 6, 0) + timedelta(hours=6 * k)
                 for k in range(40)]
        assert len(hi.members_for_window(ws, inits)) == 8


def test_cycle_gaps():
    t = lambda d, h: datetime(2024, 7, d, h)
    assert hi.cycle_gaps([t(9, 0), t(9, 6), t(9, 18)]) == [(t(9, 6), t(9, 18))]


def test_window_members_counts_only_cycles_with_the_bucket_file():
    win = _win(15, 10)
    cycles = {datetime(2024, 7, 9, 18): [0, 3, 24],     # complete
              datetime(2024, 7, 9, 12): [0, 3, 6],      # short_run
              datetime(2024, 7, 9, 6): [0, 3, 6, 12, 30, 33]}   # missing_file
    n, usable = hi.window_members(win, cycles)
    assert n == 1 and usable == [datetime(2024, 7, 9, 18)]


def test_scan_config_reads_leads_and_flags_empty_cycle(tmp_path):
    cfg = tmp_path / "HFSA"
    good = cfg / "2024070812"
    good.mkdir(parents=True)
    for lead in (0, 3, 6):
        (good / f"02l.2024070812.hfsa.parent.atm.f{lead:03d}.grb2").write_text("")
    (good / "02l.2024070812.hfsa.storm.atm.f003.grb2").write_text("")
    (cfg / "2024070818").mkdir()
    cycles, notes = hi.scan_config(cfg)
    assert cycles == {datetime(2024, 7, 8, 12): [0, 3, 6]}
    assert any("2024070818" in n and "no parent" in n for n in notes)


def test_scan_config_missing_dir(tmp_path):
    cycles, notes = hi.scan_config(tmp_path / "nope")
    assert cycles == {} and "not found" in notes[0]


def test_hafs_status_thresholds():
    assert hi.hafs_status(8) == "ok" and hi.hafs_status(16) == "ok"
    assert hi.hafs_status(7) == "thin" and hi.hafs_status(1) == "thin"
    assert hi.hafs_status(0) == "none"
    assert hi.hafs_status(4, min_members=4) == "ok"


def _wofs_rows():
    return [{"storm": "2024_BERYL", "deployment": "20240710_d1",
             "window_start": "2024-07-10 1800Z",
             "window_end": "2024-07-10 2100Z", "status": "complete"},
            {"storm": "2024_BERYL", "deployment": "20240710_d1",
             "window_start": "2024-07-10 1800Z",
             "window_end": "2024-07-10 2100Z", "status": "complete"},
            {"storm": "2024_BERYL", "deployment": "20240710_d1",
             "window_start": "2024-07-10 2100Z",
             "window_end": "2024-07-11 0000Z", "status": "short_run"}]


def test_member_rows_carry_total_and_status_for_thin_windows():
    archive = {("2024-02l", "HFSA"): ({datetime(2024, 7, 8, 18): [3, 51]}, []),
               ("2024-02l", "HFSB"): ({}, [])}
    rows = hi.member_rows(archive, _wofs_rows())
    assert len(rows) == 2                      # one window x two configs
    a = next(r for r in rows if r["config"] == "HFSA")
    assert a["n_members"] == 1 and a["members"] == "2024070818"
    assert a["total_members"] == 1 and a["hafs_status"] == "thin"
    summary = hi.summarize_members(rows)[("2024_BERYL", "20240710_d1")]
    assert summary["windows"] == 1 and summary["thin"] == 1 and not summary["ok"]


def test_pooled_total_of_eight_passes_even_with_one_configuration():
    leads = list(range(0, 127, 3))
    inits = {datetime(2024, 7, 8, 18) + timedelta(hours=6 * k): leads
             for k in range(10)}
    archive = {("2024-02l", "HFSA"): (inits, []), ("2024-02l", "HFSB"): ({}, [])}
    wofs = [{"storm": "2024_BERYL", "deployment": "d",
             "window_start": "2024-07-10 2100Z",
             "window_end": "2024-07-11 0000Z", "status": "complete"}]
    rows = hi.member_rows(archive, wofs)
    assert {r["hafs_status"] for r in rows} == {"ok"}
    assert [r["n_members"] for r in rows] == [8, 0]
