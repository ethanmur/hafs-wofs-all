"""Tests for E4 pooling: scores come from summed counts and partial sums, a
single pooled run reproduces MET's own scores, and the files and flags work.
Real grid_stat output (tests/fixtures/stat) is the reference."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool_stats as ps
import stat_tables as st
from test_stat_tables import (END, HFSA, T_HFSA, T_WOFS, WOFS, _case,
                              _install)


def _frames(tmp_path):
    """{line type: DataFrame} of the two fixture runs, as parse-stat made
    them."""
    case = _case(tmp_path)
    _install(case)
    tables = {lt: [] for lt in st.LINE_TYPES}
    for tag, name in ((T_WOFS, WOFS), (T_HFSA, HFSA)):
        out, problems = st.parse_run(
            case, END, tag, case.grid_stat_window_dir(END) / name)
        assert problems == []
        for lt, rows in out.items():
            tables[lt] += rows
    frames = {}
    for lt, rows in tables.items():
        df = pd.DataFrame(rows).replace("", np.nan)
        df["member"] = df["member"].fillna("")
        for col in df.columns:               # numbers as read_csv gives them
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        frames[lt.lower()] = df
    return frames


def _ctc(source, hits, fa, miss, cn, window="2024-10-10 00:00", lead=6,
         case="2024_14l_milton_20241009_d1"):
    return {"case": case, "storm": "Milton", "deployment": "d1",
            "truth": "mrms", "source": source, "member": "", "init_time": "x",
            "valid_end": window, "lead_h": lead, "thresh_mm": 10.0,
            "total": hits + fa + miss + cn, "fy_oy": hits, "fy_on": fa,
            "fn_oy": miss, "fn_on": cn}


def _ets(h, fa, m, cn):
    n = h + fa + m + cn
    hr = (h + m) * (h + fa) / n
    return (h - hr) / (h + m + fa - hr)


# --- the point of pooling -----------------------------------------------------

def test_pooled_ets_is_the_ets_of_the_summed_counts_not_the_mean():
    a, b = (20, 10, 10, 60), (5, 30, 40, 25)
    df = pd.DataFrame([_ctc("hfsa", *a), _ctc("hfsa", *b,
                                              window="2024-10-10 03:00")])
    out = ps.pool(df, "ctc", "all")
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["n_runs"], row["n_windows"]) == (2, 2)
    assert row["fy_oy"] == 25 and row["obs_events"] == 25 + 50
    summed = tuple(x + y for x, y in zip(a, b))
    assert row["ets"] == pytest.approx(_ets(*summed))
    assert row["ets"] != pytest.approx((_ets(*a) + _ets(*b)) / 2)
    assert row["pod"] == pytest.approx(25 / 75)
    assert row["fbias"] == pytest.approx((25 + 40) / 75)


def test_undefined_scores_stay_empty_not_zero():
    df = pd.DataFrame([_ctc("hfsa", 0, 0, 0, 100)])
    row = ps.pool(df, "ctc", "all").iloc[0]
    assert math.isnan(row["pod"]) and math.isnan(row["far"])
    assert math.isnan(row["ets"]) and row["acc"] == 1.0


def test_levels_group_by_case_storm_or_everything():
    rows = [_ctc("wofs", 10, 5, 5, 80, case="2024_14l_milton_20241009_d1"),
            _ctc("wofs", 20, 5, 5, 70, case="2024_09l_helene_20240926_d1")]
    rows[1]["storm"] = "Helene"
    df = pd.DataFrame(rows)
    assert len(ps.pool(df, "ctc", "case")) == 2
    assert len(ps.pool(df, "ctc", "storm")) == 2
    allr = ps.pool(df, "ctc", "all")
    assert len(allr) == 1 and allr.iloc[0]["fy_oy"] == 30


# --- one run in, MET's own numbers out -----------------------------------------

def test_one_pooled_run_reproduces_mets_cts(tmp_path):
    f = _frames(tmp_path)
    pooled = ps.pool(f["ctc"], "ctc", "all")
    cts = f["cts"]
    for src, mtag in (("wofs", T_WOFS), ("hfsa", T_HFSA)):
        mine = pooled[pooled.source == src].set_index("thresh_mm")
        mets = cts[cts.source == src].set_index("thresh_mm")
        for col in ("pod", "far", "csi", "ets", "fbias", "hk", "hss", "acc",
                    "baser", "fmean", "podn", "pofd"):
            for t in mets.index:
                a, b = mine.loc[t, col], mets.loc[t, col]
                if pd.isna(b):
                    assert pd.isna(a), (src, t, col)
                else:
                    assert a == pytest.approx(b, abs=3e-5), (src, t, col)


def test_one_pooled_sl1l2_run_reproduces_mets_cnt(tmp_path):
    f = _frames(tmp_path)
    pooled = ps.pool(f["sl1l2"], "sl1l2", "all")
    cnt = f["cnt"]
    for src in ("wofs", "hfsa"):
        for cell in ("all", "wet"):
            mine = pooled[(pooled.source == src)
                          & (pooled.cell_set == cell)].iloc[0]
            mets = cnt[(cnt.source == src) & (cnt.cell_set == cell)].iloc[0]
            for col in ("total", "fbar", "obar", "me", "mae", "mse", "rmse",
                        "mbias"):
                assert mine[col] == pytest.approx(mets[col], rel=1e-4,
                                                  abs=1e-4), (src, cell, col)
            # MET's standard deviations and correlation use N-1
            for col in ("fstdev", "ostdev", "pr_corr"):
                assert mine[col] == pytest.approx(mets[col], rel=2e-3), col


def test_one_pooled_nbrcnt_run_reproduces_mets_fss(tmp_path):
    f = _frames(tmp_path)
    pooled = ps.pool(f["nbrcnt"], "nbrcnt", "all")
    nb = f["nbrcnt"]
    keys = ["source", "thresh_mm", "nbr_cells"]
    mine = pooled.set_index(keys)
    mets = nb.set_index(keys)
    assert len(mine) == len(mets)
    for k in mets.index:
        for col in ("fbs", "fss", "ufss", "f_rate", "o_rate"):
            a, b = mine.loc[k, col], mets.loc[k, col]
            if pd.isna(b):
                assert pd.isna(a), (k, col)       # nothing to misplace
            else:
                assert a == pytest.approx(b, abs=2e-5), (k, col)


def test_pooling_a_run_twice_changes_counts_not_scores(tmp_path):
    f = _frames(tmp_path)
    one = ps.pool(f["ctc"], "ctc", "all")
    twice = ps.pool(pd.concat([f["ctc"], f["ctc"]]), "ctc", "all")
    assert list(twice["n_runs"]) == [2 * n for n in one["n_runs"]]
    assert list(twice["total"]) == [2 * n for n in one["total"]]
    np.testing.assert_allclose(twice["ets"].fillna(-1), one["ets"].fillna(-1))


def test_nbrcnt_fss_pools_by_fbs_and_worst_case_not_by_averaging():
    base = {"case": "c", "storm": "S", "deployment": "d", "truth": "mrms",
            "source": "wofs", "lead_h": 3, "thresh_mm": 1.0, "nbr_cells": 3,
            "nbr_width_km": 9.0, "member": "", "init_time": "x"}
    # run A: fbs 0.1, worst 0.5 (fss 0.8); run B: fbs 0.3, worst 0.4 (fss 0.25)
    rows = [{**base, "valid_end": "w1", "total": 100, "fbs": 0.1, "fss": 0.8,
             "f_rate": 0.2, "o_rate": 0.3},
            {**base, "valid_end": "w2", "total": 300, "fbs": 0.3,
             "fss": 0.25, "f_rate": 0.1, "o_rate": 0.1}]
    out = ps.pool(pd.DataFrame(rows), "nbrcnt", "all").iloc[0]
    fbs = (100 * 0.1 + 300 * 0.3) / 400
    worst = (100 * 0.5 + 300 * 0.4) / 400
    assert out["fss"] == pytest.approx(1 - fbs / worst)
    assert out["fss"] != pytest.approx((0.8 + 0.25) / 2)
    assert out["o_rate"] == pytest.approx((100 * 0.3 + 300 * 0.1) / 400)
    assert out["ufss"] == pytest.approx(0.5 + out["o_rate"] / 2)


# --- matching, files, command ---------------------------------------------------

def test_matched_keeps_the_windows_or_leads_every_source_has():
    df = pd.DataFrame([_ctc("wofs", 1, 1, 1, 1, window="w1"),
                       _ctc("wofs", 1, 1, 1, 1, window="w2", lead=3),
                       _ctc("wofs", 1, 1, 1, 1, window="w2", lead=6),
                       _ctc("hfsa", 1, 1, 1, 1, window="w2", lead=6),
                       _ctc("hfsa", 1, 1, 1, 1, window="w2", lead=9)])
    kept = ps.matched(df, "windows")
    assert set(kept["valid_end"]) == {"w2"} and len(kept) == 4
    kept = ps.matched(df, "leads")
    assert set(kept["lead_h"]) == {6} and len(kept) == 2


def test_file_matching_respects_the_tag(tmp_path):
    for name in ("stat_ctc_2024_14l_milton_20241009_d1_3km.csv",
                 "stat_ctc_2024_14l_milton_20241009_d1.csv",
                 "stat_ctc_2024_14l_milton_20241009_d1_3km_coast50km.csv",
                 "pooled_all_ctc_3km.csv"):
        (tmp_path / name).write_text("x\n")
    names = lambda tag: [p.name for p in ps.csv_files(tmp_path, "ctc", tag)]
    assert names("_3km") == ["stat_ctc_2024_14l_milton_20241009_d1_3km.csv"]
    assert names("") == ["stat_ctc_2024_14l_milton_20241009_d1.csv"]
    assert names("_3km_coast50km") == [
        "stat_ctc_2024_14l_milton_20241009_d1_3km_coast50km.csv"]


def test_the_command_writes_a_file_per_level_and_line_type(tmp_path, capsys):
    case = _case(tmp_path)
    _install(case)
    st.parse_stat_case(case)
    out = tmp_path / "pooled"
    written = ps.pool_all(case.stat_csv_dir, out, "_3km")
    assert sorted(written) == sorted((lv, lt) for lv in ps.LEVELS
                                     for lt in ps.LINE_TYPES)
    path = out / "pooled_all_ctc_3km.csv"
    table = pd.read_csv(path)
    assert set(table["source"]) == {"wofs", "hfsa"}
    assert len(table) == 2 * 11
    assert "n_windows" in table and "obs_events" in table
    both = ps.pool_all(case.stat_csv_dir, out, "_3km", ("all",),
                       mode="leads")
    assert both[("all", "ctc")].name == "pooled_all_ctc_3km_matched_leads.csv"
    assert "windows / runs per source" in capsys.readouterr().out


def test_an_empty_folder_or_wrong_tag_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="parse-stat"):
        ps.pool_all(tmp_path, tag="_3km")
