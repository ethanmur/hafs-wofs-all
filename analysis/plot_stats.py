"""Categorical and continuous score figures from the pooled tables (plan F).

    python analysis/plot_stats.py --pooled-dir data/grid-stat/csvs_pooled
        --fig-dir figures/stats-categorical-continuous [--tag _3km]
        [--only lead thresh perf] [--dpi 150] [--no-bands] [--max-thresh 75]
        [--sample-set matched|all|both]

Reads the bootstrap tables bootstrap_stats.py wrote and draws one figure per
score and partition, never everything at once. Plain matplotlib: runs on a
laptop or on Orion, no MET and no map libraries.

Inputs (tag _3km), from bootstrap_stats.py --level all. Sample set "matched"
(the models on the same windows, the fair comparison):
    boot_all_{ctc,sl1l2}_3km_matched_fixed.csv            lead curves
    boot_all_{ctc,sl1l2}_3km_matched_leads.csv            lead 6 h, head to head
    boot_all_{ctc,sl1l2}_3km_matched_windows_allleads.csv all leads summed
Sample set "all" (every available window of each model; the models see different
windows, so it is not a fair comparison but shows the whole sample; figure names
gain _allwindows):
    boot_all_{ctc,sl1l2}_3km.csv                          lead curves, lead 6 h
    boot_all_{ctc,sl1l2}_3km_allleads.csv                 all leads summed

With --first-bucket include the same figures read the *_withspinup tables
(HAFS 0-3 h bucket pooled; fixed lead curves from *_matched_fixed-spinup_*)
and their names gain _withspinup.

Scores: ETS, CSI, frequency bias, POD, success ratio (1 - FAR) and RMSE, each
with its 95 % window-bootstrap band.

  lead_<score>_model-<m>_...      score vs lead, one line per threshold
  lead_<score>_thresh-<t>mm_...   score vs lead, one line per model
  thresh_<score>_<sample>_...     score vs threshold, one line per model, at
                                  lead 6 h (lead6) or all leads summed (allleads)
  lead_rmse_*, rmse_<sample>_...  RMSE has no threshold: all cells and wet cells
  perfdiag_model-<m>_<sample>_...  performance diagrams: POD vs success ratio
  perfdiag_thresh-<t>mm_...        with CSI contours and bias lines; shapes are
  perfdiag_all_...                 models, colours thresholds

`--no-bands` leaves out the bootstrap bands, crosses and error bars;
`--max-thresh` (default 75 mm) drops the thinner, noisier thresholds above it.
The per-model figures of a score share one y range, set from the largest values
across the models, so they compare at a glance.

Markers are hollow where the threshold has fewer than MIN_EVENTS observed-event
cells (a score from a handful of cells is noise). Colours are Okabe-Ito for the
models and viridis for the thresholds: neither pairs red with green.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

MIN_EVENTS = 1000
# Set by make_all for one run: draw the bootstrap bands, and the highest
# threshold (mm) plotted (None: all).
SETTINGS = {"bands": True, "max_thresh": None}
MODELS = ("hfsa", "hfsb", "wofs")
LABEL = {"hfsa": "HFSA", "hfsb": "HFSB", "wofs": "WoFS"}
COLOR = {"hfsa": "#0072B2", "hfsb": "#E69F00", "wofs": "#222222"}
MARKER = {"hfsa": "D", "hfsb": "s", "wofs": "o"}
CELLS = {"all": "all cells", "wet": "wet cells"}

SCORES = {                                  # key: (axis label, table)
    "ets": ("ETS", "ctc"), "csi": ("CSI", "ctc"),
    "fbias": ("Frequency bias", "ctc"), "pod": ("POD", "ctc"),
    "sr": ("Success ratio (1 - FAR)", "ctc"),
}
# key: (table file ending, description). "matched" compares the models on the
# same windows (fair); "all" uses every available window of each model, so the
# models see different windows (not a fair comparison, but the whole sample).
SAMPLE_SETS = {
    "matched": {
        "fixed": ("_matched_fixed", "fixed windows"),
        "lead6": ("_matched_leads", "lead 6 h, same windows"),
        "allleads": ("_matched_windows_allleads", "all leads summed, matched "
                     "windows"),
    },
    "all": {
        "fixed": ("", "all available windows"),
        "lead6": ("", "lead 6 h, all available windows"),
        "allleads": ("_allleads", "all leads summed, all available windows"),
    },
}
SAMPLES = dict(SAMPLE_SETS["matched"])       # the set in use; make_all sets it


def spinup_samples(samples):
    """The samples for tables pooled with --first-bucket include: every file
    ending gains _withspinup, and the fixed-lead curves use the other window
    family (HAFS leads 3, 9 ... 51; --matched fixed-spinup), which is the one
    with a 0-3 h point."""
    out = {k: (e + "_withspinup", d) for k, (e, d) in samples.items()}
    if "_matched_fixed" in samples["fixed"][0]:
        out["fixed"] = ("_matched_fixed-spinup_withspinup",
                        "fixed windows, with the HAFS 0-3 h bucket")
    else:
        out["fixed"] = (out["fixed"][0],
                        "all available windows, with the HAFS 0-3 h bucket")
    return out


# =============================================================================
# Data
# =============================================================================

def add_sr(df):
    """Success ratio = 1 - FAR; its interval is the FAR interval flipped."""
    df = df.copy()
    df["sr"], df["sr_lo"], df["sr_hi"] = 1 - df["far"], 1 - df["far_hi"], \
        1 - df["far_lo"]
    return df


def load(pooled_dir, tag):
    """{(sample, line type): DataFrame} for the tables that exist."""
    out = {}
    for sample, (ending, _) in SAMPLES.items():
        for lt in ("ctc", "sl1l2"):
            path = Path(pooled_dir) / f"boot_all_{lt}{tag}{ending}.csv"
            if path.exists():
                df = pd.read_csv(path)
                out[(sample, lt)] = add_sr(df) if lt == "ctc" else df
    return out


def thresholds(df):
    """The thresholds to draw, up to SETTINGS["max_thresh"]."""
    cap = SETTINGS["max_thresh"]
    return sorted(t for t in df["thresh_mm"].unique()
                  if cap is None or t <= cap)


def threshold_colors(values):
    """One colour per threshold, stable across figures (viridis)."""
    cmap = plt.get_cmap("viridis")
    pos = np.linspace(0.05, 0.92, len(values))
    return {t: cmap(p) for t, p in zip(values, pos)}


def _fmt_t(t):
    return f"{t:g}"


def _windows_note(df, what):
    """Footnote: the sample and how many windows (a range when models or leads
    differ), whether bands are drawn, and that WoFS pools its members."""
    d = df[df["obs_events"] >= MIN_EVENTS] if "obs_events" in df else df
    d = d if len(d) else df
    lo, hi = int(d["n_windows"].min()), int(d["n_windows"].max())
    n = f"{lo}" if lo == hi else f"{lo}-{hi}"
    band = "; 95 % window-bootstrap band" if SETTINGS["bands"] else ""
    return f"{what}; {n} windows{band}; WoFS pools its members"


# =============================================================================
# Drawing helpers
# =============================================================================

def _style(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _line(ax, x, y, lo, hi, color, label=None, ls="-", marker="o",
          thin=None, band=True):
    """A line with markers and a band. `thin` is a boolean array: those points
    get hollow markers and no band."""
    x, y, lo, hi = (np.asarray(v, float) for v in (x, y, lo, hi))
    thin = np.zeros(len(x), bool) if thin is None else np.asarray(thin, bool)
    ax.plot(x, y, color=color, ls=ls, lw=1.6, label=label, zorder=3)
    solid = ~thin & np.isfinite(y)
    ax.plot(x[solid], y[solid], ls="none", marker=marker, ms=5, color=color,
            zorder=4)
    ax.plot(x[thin & np.isfinite(y)], y[thin & np.isfinite(y)], ls="none",
            marker=marker, ms=5, mfc="white", mec=color, zorder=4)
    if band and SETTINGS["bands"]:
        ok = ~thin & np.isfinite(lo) & np.isfinite(hi)
        ax.fill_between(x, np.where(ok, lo, np.nan), np.where(ok, hi, np.nan),
                        color=color, alpha=0.11, lw=0, zorder=2)


def _finish(fig, ax, path, title, note, score=None, legend=True,
            legend_title=None, handles=None, dpi=150, ylim=None):
    ax.set_title(title, fontsize=11, loc="left")
    fig.text(0.01, 0.01, note, fontsize=7, color="#555555", ha="left",
             va="bottom")
    if score == "fbias":
        ax.axhline(1.0, color="#888888", lw=0.9, ls="--", zorder=1)
    elif score in ("ets", "csi", "pod", "sr"):
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0.0) if lo < 0.02 else lo, min(max(hi, 0.05), 1.0))
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        if handles is None:
            ax.legend(title=legend_title, fontsize=8, title_fontsize=8,
                      loc="center left", bbox_to_anchor=(1.01, 0.5),
                      frameon=False)
        else:
            ax.legend(handles=handles, title=legend_title, fontsize=8,
                      title_fontsize=8, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.subplots_adjust(left=0.1, right=0.76, bottom=0.17, top=0.9)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def lock_ylims(df, scores, ts=None):
    """{score: (low, high)}: one y range per score across all models (and
    thresholds drawn), so the per-model figures can be compared side by side."""
    out = {}
    sub = df if ts is None else df[df["thresh_mm"].isin(ts)]
    for score in scores:
        est = sub[score].to_numpy(float)
        hi = sub[f"{score}_hi"].to_numpy(float) if SETTINGS["bands"] else est
        lo = sub[f"{score}_lo"].to_numpy(float) if SETTINGS["bands"] else est
        top = np.nanmax(np.where(np.isfinite(hi), hi, est))
        bottom = np.nanmin(np.where(np.isfinite(lo), lo, est))
        if score == "fbias":
            out[score] = (0.0, min(max(1.5, 1.05 * top), 4.0))
        elif score == "rmse":
            out[score] = (0.0, 1.05 * top)
        else:
            out[score] = (min(0.0, bottom), min(1.0, 1.05 * top))
    return out


def _fbias_ylim(ax, df, col="fbias"):
    top = float(np.nanmax(df[col])) if len(df) else 2.0
    ax.set_ylim(0, min(max(1.5, 1.1 * top), 4.0))


def _axes():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    _style(ax)
    return fig, ax


def _name(fig_dir, stem, tag):
    return Path(fig_dir) / f"{stem}{tag}.png"


# =============================================================================
# Lead-time curves
# =============================================================================

def plot_lead_by_model(df, score, model, fig_dir, tag, dpi, ylim=None):
    """Score vs lead for one model, one line per threshold."""
    label, _ = SCORES[score]
    sub = df[df["source"] == model]
    ts = thresholds(df)
    colors = threshold_colors(ts)
    fig, ax = _axes()
    for t in ts:
        d = sub[sub["thresh_mm"] == t].sort_values("lead_h")
        if d.empty:
            continue
        _line(ax, d["lead_h"], d[score], d[f"{score}_lo"], d[f"{score}_hi"],
              colors[t], label=_fmt_t(t), thin=d["obs_events"] < MIN_EVENTS)
    ax.set_xlabel("Lead time at window end (h)")
    ax.set_ylabel(label)
    if score == "fbias":
        _fbias_ylim(ax, sub)
    _finish(fig, ax, _name(fig_dir, f"lead_{score}_model-{model}", tag),
            f"{label}: {LABEL[model]}, by rainfall threshold",
            _windows_note(sub, SAMPLES["fixed"][1]) + "; hollow = fewer than "
            f"{MIN_EVENTS} event cells", score,
            legend_title="3 h threshold (mm)", dpi=dpi, ylim=ylim)


def plot_lead_by_threshold(df, score, t, fig_dir, tag, dpi):
    """Score vs lead at one threshold, one line per model."""
    label, _ = SCORES[score]
    sub = df[df["thresh_mm"] == t]
    fig, ax = _axes()
    for m in MODELS:
        d = sub[sub["source"] == m].sort_values("lead_h")
        if d.empty:
            continue
        _line(ax, d["lead_h"], d[score], d[f"{score}_lo"], d[f"{score}_hi"],
              COLOR[m], label=LABEL[m], marker=MARKER[m],
              thin=d["obs_events"] < MIN_EVENTS)
    ax.set_xlabel("Lead time at window end (h)")
    ax.set_ylabel(label)
    if score == "fbias":
        _fbias_ylim(ax, sub)
    _finish(fig, ax, _name(fig_dir, f"lead_{score}_thresh-{_fmt_t(t)}mm", tag),
            f"{label}: models at {_fmt_t(t)} mm per 3 h",
            _windows_note(sub, SAMPLES["fixed"][1]) + "; hollow = fewer than "
            f"{MIN_EVENTS} event cells", score, dpi=dpi)


def plot_rmse_lead(sl, fig_dir, tag, dpi):
    ylim = lock_ylims(sl, ["rmse"])["rmse"]
    """RMSE vs lead: per model (all and wet cells) and per cell set (models)."""
    paths = []
    for m in MODELS:
        d = sl[sl["source"] == m]
        if d.empty:
            continue
        fig, ax = _axes()
        for cell, ls in (("all", "-"), ("wet", "--")):
            e = d[d["cell_set"] == cell].sort_values("lead_h")
            _line(ax, e["lead_h"], e["rmse"], e["rmse_lo"], e["rmse_hi"],
                  COLOR[m], label=CELLS[cell], ls=ls, marker=MARKER[m])
        ax.set_xlabel("Lead time at window end (h)")
        ax.set_ylabel("RMSE (mm per 3 h)")
        p = _name(fig_dir, f"lead_rmse_model-{m}", tag)
        _finish(fig, ax, p, f"RMSE: {LABEL[m]}",
                _windows_note(d, SAMPLES["fixed"][1]), dpi=dpi,
                legend_title="cells scored", ylim=ylim)
        paths.append(p)
    for cell in ("all", "wet"):
        fig, ax = _axes()
        for m in MODELS:
            e = sl[(sl["source"] == m) & (sl["cell_set"] == cell)
                   ].sort_values("lead_h")
            if e.empty:
                continue
            _line(ax, e["lead_h"], e["rmse"], e["rmse_lo"], e["rmse_hi"],
                  COLOR[m], label=LABEL[m], marker=MARKER[m])
        ax.set_xlabel("Lead time at window end (h)")
        ax.set_ylabel("RMSE (mm per 3 h)")
        p = _name(fig_dir, f"lead_rmse_cells-{cell}", tag)
        _finish(fig, ax, p, f"RMSE over {CELLS[cell]}: models",
                _windows_note(sl, SAMPLES["fixed"][1]), dpi=dpi)
        paths.append(p)
    return paths


# =============================================================================
# Threshold curves
# =============================================================================

def plot_threshold(df, score, sample, fig_dir, tag, dpi):
    """Score vs threshold, one line per model, for one sample."""
    label, _ = SCORES[score]
    sub = df[df["lead_h"].astype(str).isin(["6", "all"])]
    ts = thresholds(sub)
    sub = sub[sub["thresh_mm"].isin(ts)]
    pos = {t: i for i, t in enumerate(ts)}
    fig, ax = _axes()
    for m in MODELS:
        d = sub[sub["source"] == m].sort_values("thresh_mm")
        if d.empty:
            continue
        _line(ax, d["thresh_mm"].map(pos), d[score], d[f"{score}_lo"],
              d[f"{score}_hi"], COLOR[m], label=LABEL[m], marker=MARKER[m],
              thin=d["obs_events"] < MIN_EVENTS)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([_fmt_t(t) for t in ts])
    ax.set_xlabel("3 h rainfall threshold (mm)")
    ax.set_ylabel(label)
    if score == "fbias":
        _fbias_ylim(ax, sub)
    _finish(fig, ax, _name(fig_dir, f"thresh_{score}_{sample}", tag),
            f"{label} vs threshold: {SAMPLES[sample][1]}",
            _windows_note(sub, SAMPLES[sample][1]) + "; hollow = fewer than "
            f"{MIN_EVENTS} event cells", score, dpi=dpi)


def plot_rmse_bars(sl, sample, fig_dir, tag, dpi):
    """RMSE for each model over all and wet cells, grouped bars with CIs."""
    sub = sl[sl["lead_h"].astype(str).isin(["6", "all"])]
    fig, ax = _axes()
    width = 0.26
    for k, m in enumerate(MODELS):
        for i, cell in enumerate(("all", "wet")):
            e = sub[(sub["source"] == m) & (sub["cell_set"] == cell)]
            if e.empty:
                continue
            r = e.iloc[0]
            err = ([[r["rmse"] - r["rmse_lo"]], [r["rmse_hi"] - r["rmse"]]]
                   if SETTINGS["bands"] else None)
            ax.bar(i + (k - 1) * width, r["rmse"], width * 0.92,
                   color=COLOR[m], label=LABEL[m] if i == 0 else None,
                   yerr=err, ecolor="#666666", capsize=3, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all cells", "wet cells"])
    ax.set_ylabel("RMSE (mm per 3 h)")
    _finish(fig, ax, _name(fig_dir, f"rmse_{sample}", tag),
            f"RMSE: {SAMPLES[sample][1]}", _windows_note(sub,
                                                          SAMPLES[sample][1]),
            dpi=dpi)


# =============================================================================
# Performance diagrams
# =============================================================================

def _perf_background(ax):
    x = np.linspace(0.005, 1, 300)
    xx, yy = np.meshgrid(x, x)
    csi = 1.0 / (1.0 / xx + 1.0 / yy - 1.0)
    cs = ax.contour(xx, yy, csi, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                         0.8, 0.9], colors="#bbbbbb",
                    linewidths=0.8, zorder=1)
    ax.clabel(cs, fmt="%.1f", fontsize=7, inline=True)
    for b in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        ax.plot([0, min(1.0, 1.0 / b)], [0, min(1.0, b)], color="#999999",
                lw=1.4 if b == 1 else 0.7, ls="-" if b == 1 else ":",
                zorder=1)
        if b <= 1:
            ax.text(1.005, b, f"{b:g}", fontsize=7, color="#666666",
                    va="center", clip_on=False)
        else:
            ax.text(1.0 / b, 1.012, f"{b:g}", fontsize=7, color="#666666",
                    ha="center", clip_on=False)
    ax.text(0.5, 1.045, "bias lines", fontsize=7, color="#666666", ha="center",
            clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Success ratio (1 - FAR)")
    ax.set_ylabel("Probability of detection")


def plot_performance(df, sample, fig_dir, tag, dpi, models=MODELS,
                     only_t=None, stem="perfdiag_all", ci=False,
                     title="Performance diagram"):
    """One performance diagram: shapes are models, colours are thresholds."""
    sub = df[df["lead_h"].astype(str).isin(["6", "all"])]
    sub = sub[sub["source"].isin(models)]
    all_ts = thresholds(df)
    colors = threshold_colors(all_ts)
    ts = [only_t] if only_t is not None else all_ts
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    _perf_background(ax)
    for m in models:
        d = sub[sub["source"] == m].sort_values("thresh_mm")
        d = d[d["thresh_mm"].isin(ts)]
        d = d[np.isfinite(d["pod"]) & np.isfinite(d["sr"])]
        if len(d) > 1:
            ax.plot(d["sr"], d["pod"], color="#888888", lw=0.7, alpha=0.6,
                    zorder=2)
        for _, r in d.iterrows():
            c = colors[r["thresh_mm"]]
            thin = r["obs_events"] < MIN_EVENTS
            if (ci and SETTINGS["bands"]
                    and np.isfinite([r["pod_lo"], r["sr_lo"]]).all()):
                ax.plot([r["sr_lo"], r["sr_hi"]], [r["pod"], r["pod"]],
                        color=c, lw=1.0, alpha=0.7, zorder=3)
                ax.plot([r["sr"], r["sr"]], [r["pod_lo"], r["pod_hi"]],
                        color=c, lw=1.0, alpha=0.7, zorder=3)
            ax.plot(r["sr"], r["pod"], marker=MARKER[m], ms=9, ls="none",
                    mfc="white" if thin else c, mec=c, mew=1.6, zorder=4)
    handles = []
    if only_t is None:
        handles += [Line2D([], [], marker="o", ls="none", color=colors[t],
                           label=f"{_fmt_t(t)} mm") for t in ts]
    if len(models) > 1:
        handles += [Line2D([], [], marker=MARKER[m], ls="none", mfc="#999999",
                           mec="#444444", label=LABEL[m]) for m in models]
    if handles:
        ax.legend(handles=handles, fontsize=8, frameon=False,
                  loc="center left", bbox_to_anchor=(1.12, 0.5))
    ax.set_title(title, fontsize=10.5, loc="left", pad=22)
    fig.text(0.01, 0.01, _windows_note(sub, SAMPLES[sample][1]).replace(
        "; 95 %", "\n95 %") + "; hollow = fewer than "
        f"{MIN_EVENTS} event cells" + ("; crosses are the bands" if ci and SETTINGS["bands"] else ""),
        fontsize=7, color="#555555", va="bottom")
    fig.subplots_adjust(left=0.1, right=0.72, bottom=0.12, top=0.86)
    p = Path(fig_dir) / f"{stem}_{sample}{tag}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_all_performance(ctc, sample, fig_dir, tag, dpi):
    """Per model, per threshold (all models) and everything on one plot."""
    paths = []
    what = SAMPLES[sample][1]
    for m in MODELS:
        if (ctc["source"] == m).any():
            paths.append(plot_performance(
                ctc, sample, fig_dir, tag, dpi, models=(m,), ci=True,
                stem=f"perfdiag_model-{m}",
                title=f"Performance diagram: {LABEL[m]}, all thresholds"))
    for t in thresholds(ctc):
        paths.append(plot_performance(
            ctc, sample, fig_dir, tag, dpi, only_t=t, ci=True,
            stem=f"perfdiag_thresh-{_fmt_t(t)}mm",
            title=f"Performance diagram: {_fmt_t(t)} mm per 3 h, all models"))
    paths.append(plot_performance(
        ctc, sample, fig_dir, tag, dpi, stem="perfdiag_all",
        title="Performance diagram: all models and thresholds"))
    return paths


# =============================================================================
# Command line
# =============================================================================

def make_all(pooled_dir, fig_dir, tag="_3km", only=("lead", "thresh", "perf"),
             dpi=150, log=print, bands=True, max_thresh=None,
             sample_set="matched", first_bucket="exclude"):
    """[Path] of every figure written. `bands` draws the bootstrap bands and
    crosses; `max_thresh` drops thresholds above it (mm); `sample_set` is
    "matched" (same windows for every model) or "all" (every available window;
    figure names gain _allwindows). `first_bucket` "include" reads the
    _withspinup tables (HAFS 0-3 h bucket pooled) and adds _withspinup to the
    figure names."""
    SETTINGS["bands"], SETTINGS["max_thresh"] = bands, max_thresh
    SAMPLES.clear()
    SAMPLES.update(spinup_samples(SAMPLE_SETS[sample_set])
                   if first_bucket == "include" else SAMPLE_SETS[sample_set])
    tables = load(pooled_dir, tag)
    if not tables:
        raise SystemExit(f"plot_stats: no boot_all_*{tag}*.csv tables in "
                         f"{pooled_dir}; run bootstrap_stats.py first.")
    # from here `tag` only names the figures
    tag = (("_allwindows" if sample_set == "all" else "") + tag
           + ("_withspinup" if first_bucket == "include" else ""))
    written = []

    def note(what, paths):
        written.extend(paths)
        log(f"{what:<28} {len(paths):>3} figures")

    if "lead" in only:
        ctc = tables.get(("fixed", "ctc"))
        if ctc is not None:
            paths = []
            ylims = lock_ylims(ctc, list(SCORES), thresholds(ctc))
            for score in SCORES:
                for m in MODELS:
                    if (ctc["source"] == m).any():
                        plot_lead_by_model(ctc, score, m, fig_dir, tag, dpi,
                                           ylim=ylims[score])
                        paths.append(_name(fig_dir, f"lead_{score}_model-{m}",
                                           tag))
                for t in thresholds(ctc):
                    plot_lead_by_threshold(ctc, score, t, fig_dir, tag, dpi)
                    paths.append(_name(
                        fig_dir, f"lead_{score}_thresh-{_fmt_t(t)}mm", tag))
            note("lead curves (categorical)", paths)
        else:
            log(f"lead curves: boot_all_ctc{SAMPLES['fixed'][0]}.csv missing; "
                "skipped")
        sl = tables.get(("fixed", "sl1l2"))
        if sl is not None:
            note("lead curves (RMSE)", plot_rmse_lead(sl, fig_dir, tag, dpi))
    if "thresh" in only:
        paths = []
        for sample in ("lead6", "allleads"):
            ctc = tables.get((sample, "ctc"))
            if ctc is None:
                log(f"threshold curves: {SAMPLES[sample][0]} table missing; "
                    f"{sample} skipped")
                continue
            for score in SCORES:
                plot_threshold(ctc, score, sample, fig_dir, tag, dpi)
                paths.append(_name(fig_dir, f"thresh_{score}_{sample}", tag))
            sl = tables.get((sample, "sl1l2"))
            if sl is not None:
                plot_rmse_bars(sl, sample, fig_dir, tag, dpi)
                paths.append(_name(fig_dir, f"rmse_{sample}", tag))
        note("threshold curves + RMSE bars", paths)
    if "perf" in only:
        for sample in ("lead6", "allleads"):
            ctc = tables.get((sample, "ctc"))
            if ctc is not None:
                note(f"performance diagrams {sample}",
                     plot_all_performance(ctc, sample, fig_dir, tag, dpi))
    log(f"\n{len(written)} figures in {fig_dir}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pooled-dir", required=True,
                    help="folder with the boot_all_*.csv tables")
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--tag", default="_3km")
    ap.add_argument("--only", nargs="+", choices=("lead", "thresh", "perf"),
                    default=["lead", "thresh", "perf"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-bands", action="store_true",
                    help="no bootstrap bands, crosses or error bars")
    ap.add_argument("--max-thresh", type=float, default=75.0,
                    help="highest 3 h threshold (mm) drawn; default 75, use "
                         "150 for all")
    ap.add_argument("--sample-set", choices=("matched", "all", "both"),
                    default="both",
                    help="matched: models on the same windows (fair); all: "
                         "every available window of each model (names gain "
                         "_allwindows); default both")
    ap.add_argument("--first-bucket", choices=("exclude", "include"),
                    default="exclude",
                    help="HAFS 0-3 h bucket: exclude (default) or include (reads "
                         "the _withspinup tables; names gain _withspinup)")
    a = ap.parse_args(argv)
    sets = ("matched", "all") if a.sample_set == "both" else (a.sample_set,)
    for s in sets:
        print(f"--- sample set: {s}")
        make_all(a.pooled_dir, a.fig_dir, a.tag, tuple(a.only), a.dpi,
                 bands=not a.no_bands, max_thresh=a.max_thresh, sample_set=s,
                 first_bucket=a.first_bucket)


if __name__ == "__main__":
    main(sys.argv[1:])
