"""Tests for E4b: the window bootstrap, paired differences and leave-one-storm-
out, with known-answer cases (identical windows, a mean with a known standard
error, a score a window can switch on or off)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap_stats as bs
import pool_stats as ps
import stat_tables as st
from test_pool_stats import _ctc, _ets
from test_stat_tables import _case, _install


def _windows(n, counts, source="wofs", storm="Milton", case="c1", lead=6):
    """n windows with the same (hits, fa, miss, cn) in each."""
    rows = []
    for k in range(n):
        r = _ctc(source, *counts, window=f"w{k:03d}", lead=lead, case=case)
        r["storm"] = storm
        rows.append(r)
    return pd.DataFrame(rows)


def _sl1l2(values, source="wofs", total=1000):
    rows = []
    for k, mae in enumerate(values):
        rows.append({"case": "c1", "storm": "S", "deployment": "d",
                     "truth": "mrms", "source": source, "member": "",
                     "valid_end": f"w{k:04d}", "lead_h": 6,
                     "cell_set": "all", "wet_thresh_mm": np.nan,
                     "total": total, "fbar": 1.0, "obar": 1.0, "fobar": 1.0,
                     "ffbar": 1.0, "oobar": 1.0, "mae": mae})
    return pd.DataFrame(rows)


def _boot(df, lt="ctc", level="all", n=300, seed=0, diffs=False):
    return bs.bootstrap(df, lt, level, n_boot=n, seed=seed, diffs=diffs)


# --- known answers --------------------------------------------------------------

def test_identical_windows_give_a_zero_width_interval_at_the_estimate():
    est, _ = _boot(_windows(12, (20, 10, 10, 60)))
    row = est.iloc[0]
    assert row["ets"] == pytest.approx(_ets(20, 10, 10, 60))
    assert row["ets_lo"] == pytest.approx(row["ets"])
    assert row["ets_hi"] == pytest.approx(row["ets"])
    assert row["n_windows"] == 12 and row["obs_events"] == 12 * 30


def test_the_estimate_is_the_pooled_value_and_lies_inside_its_interval():
    rng = np.random.default_rng(1)
    rows = [_ctc("hfsa", *rng.integers(5, 60, 4), window=f"w{k:02d}")
            for k in range(12)]
    df = pd.DataFrame(rows)
    est, _ = _boot(df)
    pooled = ps.pool(df, "ctc", "all")
    assert est.iloc[0]["ets"] == pytest.approx(pooled.iloc[0]["ets"])
    assert est.iloc[0]["pod"] == pytest.approx(pooled.iloc[0]["pod"])
    assert est.iloc[0]["ets_lo"] < est.iloc[0]["ets"] < est.iloc[0]["ets_hi"]


def test_a_mean_has_the_textbook_standard_error():
    """The bootstrap of a mean over n windows has spread sd / sqrt(n)."""
    n = 200
    vals = np.random.default_rng(7).normal(10.0, 2.0, n)
    est, _ = _boot(_sl1l2(vals), lt="sl1l2", n=1000)
    row = est.iloc[0]
    assert row["mae"] == pytest.approx(vals.mean())
    se = vals.std() / math.sqrt(n)
    assert (row["mae_hi"] - row["mae_lo"]) / (2 * 1.96) == pytest.approx(
        se, rel=0.15)


def test_the_same_seed_repeats_and_another_seed_moves_the_interval():
    rng = np.random.default_rng(3)
    df = pd.DataFrame([_ctc("hfsa", *rng.integers(5, 60, 4),
                            window=f"w{k:02d}") for k in range(12)])
    a, _ = _boot(df, seed=5)
    b, _ = _boot(df, seed=5)
    c, _ = _boot(df, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert a.iloc[0]["ets_lo"] != c.iloc[0]["ets_lo"]
    assert a.iloc[0]["ets"] == c.iloc[0]["ets"]


def test_a_score_few_windows_can_define_gets_no_interval():
    """Hits exist in one window of 12: POD is undefined in ~35 % of the
    resamples, so its interval is withheld rather than shown misleadingly."""
    rows = [_ctc("hfsa", 5, 5, 5, 85 if k == 0 else 0, window=f"w{k:02d}")
            for k in range(12)]
    for k in range(1, 12):
        rows[k].update(fy_oy=0, fy_on=0, fn_oy=0, fn_on=100, total=100)
    est, _ = _boot(pd.DataFrame(rows), n=400)
    row = est.iloc[0]
    assert not math.isnan(row["pod"])
    assert math.isnan(row["pod_lo"]) and math.isnan(row["pod_hi"])


# --- paired differences --------------------------------------------------------

def test_differences_share_the_resampled_windows():
    a = _windows(10, (30, 10, 10, 50), source="wofs")
    b = _windows(10, (20, 10, 20, 50), source="hfsa")
    est, diff = _boot(pd.concat([a, b]), diffs=True)
    d = diff.iloc[0]
    assert (d["source_a"], d["source_b"]) == ("wofs", "hfsa")
    pa, pb = (est[est.source == s].iloc[0] for s in ("wofs", "hfsa"))
    assert d["pod_diff"] == pytest.approx(pa["pod"] - pb["pod"])
    assert d["pod_diff"] > 0 and d["pod_lo"] == pytest.approx(d["pod_diff"])


def test_two_sources_with_the_same_counts_differ_by_exactly_zero():
    rng = np.random.default_rng(4)
    base = [rng.integers(5, 60, 4) for _ in range(12)]
    rows = [_ctc(s, *c, window=f"w{k:02d}") for s in ("wofs", "hfsa")
            for k, c in enumerate(base)]
    _, diff = _boot(pd.DataFrame(rows), diffs=True)
    assert diff["ets_diff"].iloc[0] == 0
    assert diff["ets_lo"].iloc[0] == 0 and diff["ets_hi"].iloc[0] == 0


def test_no_differences_unless_asked():
    a = _windows(4, (30, 10, 10, 50), source="wofs")
    b = _windows(4, (20, 10, 20, 50), source="hfsa")
    _, diff = _boot(pd.concat([a, b]), diffs=False)
    assert diff is None


# --- levels and leave one storm out ---------------------------------------------

def test_storm_level_resamples_inside_each_storm():
    df = pd.concat([_windows(5, (30, 10, 10, 50), storm="Milton", case="m"),
                    _windows(3, (10, 10, 30, 50), storm="Helene", case="h")])
    est, _ = _boot(df, level="storm")
    assert sorted(est["storm"]) == ["Helene", "Milton"]
    assert dict(zip(est["storm"], est["n_windows"])) == {"Helene": 3,
                                                         "Milton": 5}


def test_leave_one_storm_out_is_the_pooled_table_without_that_storm():
    df = pd.concat([_windows(2, (30, 10, 10, 50), storm="Milton", case="m"),
                    _windows(2, (10, 10, 30, 50), storm="Helene", case="h"),
                    _windows(2, (5, 5, 5, 85), storm="Beryl", case="b")])
    loo = bs.leave_one_storm_out(df, "ctc")
    assert list(loo["dropped_storm"]) == ["(none)", "Beryl", "Helene",
                                          "Milton"]
    without = ps.pool(df[df.storm != "Helene"], "ctc", "all")
    got = loo[loo.dropped_storm == "Helene"].iloc[0]
    assert got["ets"] == pytest.approx(without.iloc[0]["ets"])
    assert got["fy_oy"] == without.iloc[0]["fy_oy"]
    full = ps.pool(df, "ctc", "all")
    assert loo[loo.dropped_storm == "(none)"].iloc[0]["ets"] == \
        pytest.approx(full.iloc[0]["ets"])


# --- the command ------------------------------------------------------------------

def test_the_command_writes_boot_loo_and_only_matched_diffs(tmp_path, capsys):
    case = _case(tmp_path)
    _install(case)
    st.parse_stat_case(case)
    out = tmp_path / "boot"
    written = bs.bootstrap_all(case.stat_csv_dir, out, "_3km", n_boot=100)
    assert ("boot", "all", "ctc") in written
    assert ("loo", "storm", "nbrcnt") in written
    assert not any(k[0] == "bootdiff" for k in written)
    table = pd.read_csv(out / "boot_all_ctc_3km.csv")
    assert {"ets", "ets_lo", "ets_hi", "n_windows"} <= set(table)
    loo = pd.read_csv(out / "loo_storm_ctc_3km.csv")
    # one storm only: dropping it leaves nothing, so just the full sample
    assert set(loo["dropped_storm"]) == {"(none)"}


def test_matching_that_leaves_nothing_is_skipped_not_a_crash(tmp_path, capsys):
    case = _case(tmp_path)
    _install(case)          # WoFS lead 3 and HFSA lead 6: no shared lead
    st.parse_stat_case(case)
    with pytest.raises(SystemExit, match="after --matched leads"):
        bs.bootstrap_all(case.stat_csv_dir, tmp_path / "b", "_3km",
                         mode="leads", n_boot=100)
    assert "no windows left" in capsys.readouterr().out


def test_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="parse-stat"):
        bs.bootstrap_all(tmp_path, tag="_3km", n_boot=100)


def test_differences_work_for_a_blank_key_under_pandas_3_strings():
    """The all-cells rows of sl1l2 have an empty wet_thresh_mm; pandas 3's
    string dtype keeps that missing, which broke joining the key as text."""
    old = pd.options.future.infer_string
    try:
        pd.options.future.infer_string = True
        df = pd.concat([_sl1l2([1.0, 2.0, 3.0, 4.0], source="wofs"),
                        _sl1l2([2.0, 2.0, 5.0, 4.0], source="hfsa")])
        _, diff = _boot(df, lt="sl1l2", diffs=True)
    finally:
        pd.options.future.infer_string = old
    assert len(diff) == 1
    assert diff.iloc[0]["mae_diff"] == pytest.approx(2.5 - 3.25)
