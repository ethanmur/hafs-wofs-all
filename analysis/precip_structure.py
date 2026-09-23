"""Precipitation-distribution and pattern verification helpers."""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skill_metrics import cell_area_km2
from plot_units import cubic_miles, inches, miles, square_miles


DIST_FIELDS = ("p50", "p90", "p95", "p99", "max_mm", "volume_km3",
               "wet_frac")
_PLOT_TYPOGRAPHY = {
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
}


def _apply_plot_typography():
    """Keep cycle-structure text consistently bold and legible."""
    plt.rcParams.update(_PLOT_TYPOGRAPHY)


def _nan_dict(keys):
    return {key: np.nan for key in keys}


def distribution_stats(field, swath, grid_lat, grid_res):
    """Distribution and volume statistics over finite points in ``swath``."""
    field = np.asarray(field, dtype=float)
    valid = np.asarray(swath, dtype=bool) & np.isfinite(field)
    if not valid.any():
        return _nan_dict(DIST_FIELDS)
    values = field[valid]
    percentiles = np.percentile(values, [50, 90, 95, 99])
    area = cell_area_km2(grid_lat, grid_res)
    return {
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "max_mm": float(np.max(values)),
        "volume_km3": float(np.sum(field[valid] * area[valid]) * 1e-6),
        "wet_frac": float(np.count_nonzero(values >= 1.0) / values.size),
    }


def qq_percentiles(fcst, obs, swath, q=np.arange(1, 100)):
    """Forecast and observation percentiles over their common valid mask."""
    fcst = np.asarray(fcst, dtype=float)
    obs = np.asarray(obs, dtype=float)
    valid = (np.asarray(swath, dtype=bool) & np.isfinite(fcst)
             & np.isfinite(obs))
    q = np.asarray(q, dtype=float)
    if not valid.any():
        empty = np.full(q.shape, np.nan, dtype=float)
        return empty.copy(), empty
    return np.percentile(fcst[valid], q), np.percentile(obs[valid], q)


def _pooled_values(cycles, swath, key, fallback=None):
    arrays = []
    for cycle in cycles:
        field = cycle.get(key, fallback)
        if field is None:
            continue
        field = np.asarray(field, dtype=float)
        valid = np.asarray(swath, dtype=bool) & np.isfinite(field)
        if valid.any():
            arrays.append(field[valid])
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def _cdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, values.size + 1, dtype=float) / values.size


def _cycle_x(ccase, rows):
    rows = sorted(rows, key=lambda row: row["_init_dt"])
    inits = [row["_init_dt"] for row in rows]
    if ccase.landfall_time is not None:
        x = [(ccase.landfall_time - init_dt).total_seconds() / 3600.0
             for init_dt in inits]
    else:
        x = inits
    return rows, inits, x


def _format_cycle_axis(ax, ccase, inits, x):
    ax.set_xticks(x)
    if ccase.landfall_time is not None:
        ax.set_xlabel("Hours Before Landfall (Forecast Initialization)")
        ax.set_xticklabels([f"{value:.0f}" for value in x])
        ax.invert_xaxis()
    else:
        ax.set_xlabel("Initialization")
        ax.set_xticklabels([value.strftime("%m-%d %HZ") for value in inits],
                           rotation=45, ha="right")


def plot_distributions(ccase, fields, out_path):
    """Plot pooled PDF/CDF and per-cycle plus pooled forecast-MRMS Q-Q."""
    _apply_plot_typography()
    cycles = fields["cycles"]
    swath = fields["swath"]
    sources = [
        ("Forecast", "parent_win", "#2563a6"),
        ("MRMS", "mrms_win", "#222222"),
        ("Stage IV", "stage4_win", "#d97941"),
    ]
    pooled = [(name, _pooled_values(cycles, swath, key, fields.get(key)), color)
              for name, key, color in sources]
    if not any(values.size for _, values, _ in pooled):
        return False

    finite = np.concatenate([inches(values) for _, values, _ in pooled
                             if values.size])
    upper = max(1.0, float(np.percentile(finite, 99.9)), float(np.max(finite)))
    bins = np.linspace(0.0, upper, 51)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for name, values, color in pooled:
        if not values.size:
            continue
        values_in = inches(values)
        axes[0].hist(values_in, bins=bins, density=True, histtype="step", lw=2,
                     color=color, label=name)
        vx, vy = _cdf(values_in)
        axes[1].plot(vx, vy, lw=2, color=color, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Window-total precipitation (inches)")
    axes[0].set_ylabel("Probability density")
    axes[1].set_xlabel("Window-total precipitation (inches)")
    axes[1].set_ylabel("Cumulative probability")
    for cycle in cycles:
        fcst_q, obs_q = qq_percentiles(
            cycle["parent_win"], cycle.get("mrms_win", fields.get("mrms_win")),
            swath)
        if np.isfinite(fcst_q).any():
            axes[2].plot(inches(obs_q), inches(fcst_q), color="#9aa0a6",
                         lw=0.8, alpha=0.7)
    common_fcst = []
    common_obs = []
    for cycle in cycles:
        fgrid = np.asarray(cycle["parent_win"], dtype=float)
        ogrid = np.asarray(cycle.get("mrms_win", fields.get("mrms_win")),
                           dtype=float)
        valid = swath & np.isfinite(fgrid) & np.isfinite(ogrid)
        if valid.any():
            common_fcst.append(fgrid[valid])
            common_obs.append(ogrid[valid])
    if common_fcst:
        fq = inches(np.percentile(np.concatenate(common_fcst), np.arange(1, 100)))
        oq = inches(np.percentile(np.concatenate(common_obs), np.arange(1, 100)))
        axes[2].plot(oq, fq, color="#2563a6", lw=2.5, label="pooled")
        limit = max(float(np.nanmax(fq)), float(np.nanmax(oq)), 1.0)
        axes[2].plot([0, limit], [0, limit], color="#555555", ls=":",
                     lw=1.2, label="1:1")
    axes[2].set_xlabel("MRMS percentile (inches)")
    axes[2].set_ylabel("Forecast percentile (inches)")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    for ax, title in zip(axes, ("Pooled PDF", "Pooled CDF", "Forecast–MRMS Q–Q")):
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(title)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} precipitation distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_percentiles_by_cycle(ccase, summary_rows, out_path):
    """Plot upper percentiles and precipitation volume by cycle."""
    _apply_plot_typography()
    rows = [row for row in summary_rows
            if any(np.isfinite(row.get(key, np.nan))
                   for key in ("fcst_p90", "obs_p90", "fcst_volume_km3",
                               "obs_volume_km3"))]
    if not rows:
        return False
    rows, inits, x = _cycle_x(ccase, rows)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
    colors = {90: "#2a9d78", 95: "#e9a23b", 99: "#c43d4d"}
    for percentile in (90, 95, 99):
        axes[0].plot(x, [inches(row.get(f"fcst_p{percentile}", np.nan))
                         for row in rows],
                     color=colors[percentile], marker="o", lw=2,
                     label=f"{ccase.model_label} {percentile}%")
        axes[0].plot(x, [inches(row.get(f"obs_p{percentile}", np.nan))
                         for row in rows],
                     color=colors[percentile], marker="s", lw=1.4, ls="--",
                     label=f"MRMS {percentile}%")
    axes[1].plot(x, [cubic_miles(row.get("fcst_volume_km3", np.nan))
                     for row in rows],
                 color="#2563a6", marker="o", lw=2,
                 label=ccase.model_label)
    axes[1].plot(x, [cubic_miles(row.get("obs_volume_km3", np.nan))
                     for row in rows],
                 color="#222222", marker="s", lw=2, ls="--", label="MRMS")
    axes[0].set_ylabel("Precipitation (in)")
    axes[1].set_ylabel("Volume (Cubic Miles)")
    axes[0].legend(ncols=2, fontsize=8, frameon=False)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    _format_cycle_axis(axes[-1], ccase, inits, x)
    fig.suptitle(
        f"{ccase.storm_name} — {ccase.model_label} Precipitation Structure "
        "by Cycle")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_pattern_r(ccase, summary_rows, out_path):
    """Plot unshifted and optional track-shifted pattern correlation."""
    _apply_plot_typography()
    rows = [row for row in summary_rows
            if (np.isfinite(row.get("pattern_r", np.nan))
                or np.isfinite(row.get("pattern_r_shifted", np.nan)))]
    if not rows:
        return False
    rows, inits, x = _cycle_x(ccase, rows)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(x, [row.get("pattern_r", np.nan) for row in rows], marker="o",
            lw=2.2, color="#2563a6", label="Forecast")
    shifted = np.asarray([row.get("pattern_r_shifted", np.nan) for row in rows])
    if np.isfinite(shifted).any():
        ax.plot(x, shifted, marker="s", lw=2, ls="--", color="#d97941",
                label="Best-Track Shifted")
    ax.set_ylabel("Pearson Pattern Correlation")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False)
    _format_cycle_axis(ax, ccase, inits, x)
    ax.set_title(
        f"{ccase.storm_name} — {ccase.model_label} Pattern Correlation vs MRMS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True
