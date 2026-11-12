"""Tests for the score figures: the fixed-window and pooled-lead samples, the
success-ratio conversion, and every figure kind on synthetic bootstrap tables
(really rendered)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap_stats as bs
import plot_stats as pl
import pool_stats as ps
import stat_tables as st
from test_pool_stats import _ctc
from test_stat_tables import _case, _install

THRESH = (1.0, 10.0, 50.0)


def _ctc_frame(leads, windows=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for src, ls in leads.items():
        for w in range(windows):
            for lead in ls:
                for t in THRESH:
                    r = _ctc(src, *rng.integers(200, 2000, 4),
                             window=f"w{w:02d}", lead=lead)
                    r["thresh_mm"] = t
                    rows.append(r)
    return pd.DataFrame(rows)


def _sl1l2_frame(leads, windows=4, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for src, ls in leads.items():
        for w in range(windows):
            for lead in ls:
                for cell in ("all", "wet"):
                    rows.append({
                        "case": "c1", "storm": "Milton", "deployment": "d",
                        "truth": "mrms", "source": src, "member": "",
                        "valid_end": f"w{w:02d}", "lead_h": lead,
                        "cell_set": cell,
                        "wet_thresh_mm": np.nan if cell == "all" else 0.5,
                        "total": 1000, "fbar": 1.0, "obar": 1.1,
                        "fobar": 2.0 + rng.random(), "ffbar": 3.0,
                        "oobar": 3.2, "mae": 1.0 + rng.random()})
    return pd.DataFrame(rows)


LEADS = {"hfsa": (6, 12, 18), "hfsb": (6, 12, 18), "wofs": (3, 4, 5, 6)}


@pytest.fixture
def tables(tmp_path):
    """boot_all_* tables for the three samples, from synthetic runs."""
    out = tmp_path / "pooled"
    out.mkdir()
    for sample, (ending, _) in pl.SAMPLES.items():
        leads = LEADS if sample == "fixed" else {k: (6,) for k in LEADS}
        ctc, sl = _ctc_frame(leads), _sl1l2_frame(leads)
        if sample == "allleads":
            ctc, sl = ctc.assign(lead_h="all"), sl.assign(lead_h="all")
        for lt, df in (("ctc", ctc), ("sl1l2", sl)):
            est, _ = bs.bootstrap(df, lt, "all", n_boot=100, seed=0)
            est.to_csv(out / f"boot_all_{lt}_3km{ending}.csv", index=False)
    return out


# --- the samples -------------------------------------------------------------

def test_success_ratio_is_one_minus_far_with_a_flipped_interval():
    df = pd.DataFrame({"far": [0.2], "far_lo": [0.1], "far_hi": [0.3]})
    out = pl.add_sr(df).iloc[0]
    assert (out["sr"], out["sr_lo"], out["sr_hi"]) == pytest.approx(
        (0.8, 0.7, 0.9))


def test_fixed_windows_need_every_lead_of_every_source():
    rows = []
    for w, hafs_leads in (("w1", range(6, 49, 6)), ("w2", (6, 12, 18)),
                          ("w3", range(6, 49, 6))):
        for lead in hafs_leads:
            rows.append(_ctc("hfsa", 1, 1, 1, 1, window=w, lead=lead))
        for lead in (3, 4, 5, 6) if w != "w3" else (3, 4):
            rows.append(_ctc("wofs", 1, 1, 1, 1, window=w, lead=lead))
    kept = ps.fixed_windows(pd.DataFrame(rows))
    assert set(kept["valid_end"]) == {"w1"}
    assert sorted(kept[kept.source == "hfsa"]["lead_h"].unique()) == list(
        range(6, 49, 6))


def test_a_source_without_fixed_leads_is_an_error():
    df = pd.DataFrame([_ctc("other", 1, 1, 1, 1)])
    with pytest.raises(ValueError, match="other"):
        ps.fixed_windows(df)


def test_pooling_all_leads_labels_them_and_names_the_files(tmp_path):
    case = _case(tmp_path)
    _install(case)
    st.parse_stat_case(case)
    written = ps.pool_all(case.stat_csv_dir, tmp_path / "p", "_3km",
                          levels=("all",), pool_leads=True, log=lambda *a: 0)
    path = written[("all", "ctc")]
    assert path.name == "pooled_all_ctc_3km_allleads.csv"
    assert set(pd.read_csv(path)["lead_h"].astype(str)) == {"all"}
    boot = bs.bootstrap_all(case.stat_csv_dir, tmp_path / "b", "_3km",
                            n_boot=100, pool_leads=True, log=lambda *a: 0)
    assert boot[("boot", "all", "ctc")].name == "boot_all_ctc_3km_allleads.csv"
    assert ps.name_suffix("windows", True) == "_matched_windows_allleads"


# --- the figures -------------------------------------------------------------

def test_every_figure_kind_is_written(tables, tmp_path):
    figs = tmp_path / "figs"
    written = pl.make_all(tables, figs, "_3km", dpi=60, log=lambda *a: 0)
    names = {p.name for p in written}
    assert all(p.exists() and p.stat().st_size > 3000 for p in written)
    # 5 scores x (3 models + 3 thresholds) + RMSE (3 models + 2 cell sets)
    # + 2 samples x (5 scores + RMSE bars) + 2 samples x (3 + 3 + 1) diagrams
    assert len(written) == 5 * 6 + 5 + 2 * 6 + 2 * 7
    for expected in ("lead_ets_model-hfsa_3km.png",
                     "lead_fbias_thresh-10mm_3km.png",
                     "lead_rmse_model-wofs_3km.png",
                     "lead_rmse_cells-wet_3km.png",
                     "thresh_sr_lead6_3km.png", "thresh_pod_allleads_3km.png",
                     "rmse_lead6_3km.png", "rmse_allleads_3km.png",
                     "perfdiag_model-hfsb_lead6_3km.png",
                     "perfdiag_thresh-50mm_allleads_3km.png",
                     "perfdiag_all_lead6_3km.png"):
        assert expected in names, expected
    assert {p.parent for p in written} == {figs}


def test_only_selects_figure_families(tables, tmp_path):
    written = pl.make_all(tables, tmp_path / "f", "_3km", only=("perf",),
                          dpi=60, log=lambda *a: 0)
    assert written and all(p.name.startswith("perfdiag_") for p in written)


def test_a_missing_sample_is_skipped_not_a_crash(tables, tmp_path):
    (tables / "boot_all_ctc_3km_matched_windows_allleads.csv").unlink()
    written = pl.make_all(tables, tmp_path / "f", "_3km", only=("thresh",),
                          dpi=60, log=lambda *a: 0)
    assert not any("allleads" in p.name for p in written
                   if p.name.startswith("thresh_"))
    assert any("lead6" in p.name for p in written)


def test_no_tables_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="bootstrap_stats"):
        pl.make_all(tmp_path, tmp_path / "f")


def test_bands_can_be_turned_off_and_thresholds_capped(tables, tmp_path):
    full = pl.make_all(tables, tmp_path / "a", "_3km", only=("lead",), dpi=60,
                       log=lambda *a: 0)
    capped = pl.make_all(tables, tmp_path / "b", "_3km", only=("lead",),
                         dpi=60, log=lambda *a: 0, bands=False, max_thresh=10)
    # 3 thresholds (1, 10, 50) -> 2 under the cap: 5 x (3 + 2) + 5 RMSE
    assert len(full) == 5 * 6 + 5 and len(capped) == 5 * 5 + 5
    assert not any("thresh-50mm" in p.name for p in capped)
    assert pl.SETTINGS == {"bands": False, "max_thresh": 10}
    pl.make_all(tables, tmp_path / "c", "_3km", only=("perf",), dpi=60,
                log=lambda *a: 0)             # defaults restore themselves
    assert pl.SETTINGS == {"bands": True, "max_thresh": None}


def test_per_model_figures_share_one_y_range_from_the_largest_values(tables):
    import pandas as pd
    ctc = pd.read_csv(tables / "boot_all_ctc_3km_matched_fixed.csv")
    ctc = pl.add_sr(ctc)
    pl.SETTINGS.update(bands=True, max_thresh=None)
    lims = pl.lock_ylims(ctc, ["ets", "fbias"])
    assert lims["ets"][1] <= 1.0 and lims["ets"][0] <= 0.0
    top = max(ctc["ets_hi"].max(), ctc["ets"].max())
    assert lims["ets"][1] == pytest.approx(min(1.0, 1.05 * top))
    assert lims["fbias"][0] == 0.0 and 1.5 <= lims["fbias"][1] <= 4.0


def test_the_all_windows_set_reads_the_unmatched_tables_and_names_its_figures(
        tables, tmp_path):
    import shutil
    # the unmatched tables: all leads in one file, and the pooled-lead one
    shutil.copy(tables / "boot_all_ctc_3km_matched_fixed.csv",
                tables / "boot_all_ctc_3km.csv")
    shutil.copy(tables / "boot_all_sl1l2_3km_matched_fixed.csv",
                tables / "boot_all_sl1l2_3km.csv")
    shutil.copy(tables / "boot_all_ctc_3km_matched_windows_allleads.csv",
                tables / "boot_all_ctc_3km_allleads.csv")
    shutil.copy(tables / "boot_all_sl1l2_3km_matched_windows_allleads.csv",
                tables / "boot_all_sl1l2_3km_allleads.csv")
    written = pl.make_all(tables, tmp_path / "f", "_3km", dpi=60,
                          log=lambda *a: 0, sample_set="all")
    names = {p.name for p in written}
    assert "lead_ets_model-hfsa_allwindows_3km.png" in names
    assert "thresh_ets_lead6_allwindows_3km.png" in names
    assert "perfdiag_all_allleads_allwindows_3km.png" in names
    assert all("_allwindows_" in n for n in names)
    assert pl.SAMPLES["lead6"][1] == "lead 6 h, all available windows"
    pl.make_all(tables, tmp_path / "g", "_3km", dpi=60, log=lambda *a: 0,
                only=("perf",))                       # back to the matched set
    assert pl.SAMPLES["lead6"][1] == "lead 6 h, same windows"


def test_the_footnote_gives_a_range_when_models_have_different_window_counts():
    import pandas as pd
    df = pd.DataFrame({"n_windows": [12, 39, 24], "obs_events": [5000] * 3})
    pl.SETTINGS.update(bands=True, max_thresh=None)
    assert "12-39 windows" in pl._windows_note(df, "x")
    assert "12 windows" in pl._windows_note(df.iloc[[0]], "x")
