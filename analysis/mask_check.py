"""Spatial check of the observation-validity mask against the truth product
(plan step E0b), before any statistics are computed.

The mask says where the truth can be trusted, from geography alone. This
reads it back the way the statistics will and sets it against the real
regridded MRMS (or `--sources`) 3 h windows, so a mask that is too tight, too
loose or misplaced shows up on a map instead of as a strange score.

    python analysis/run.py storms/<case>.yaml plot-mask-check [--res-km N]

Every cell of a window falls in one of four classes:

  kept        inside the mask, truth present        what grid_stat will score
  discarded   outside the mask, truth present       real data the mask throws away
  gap         inside the mask, truth missing        scored cells with no truth
  outside     outside the mask, truth missing       nothing to say

`gap` should only be real radar holes, and `discarded` should be a thin rim of
offshore or foreign cells. Two products:

  <case>_mask_check_<end>.png      per window (a few, spread across the period):
                                   truth | the four classes | truth as scored
  <case>_mask_check_coverage.png   over every window: where in-mask cells lack
                                   truth, and where out-of-mask cells have it

It also hands the mask file to MET's gen_vx_mask on the truth window's grid and
compares the cells MET reads with the ones written, which is the one place a
bad grid attribute in the mask NetCDF would show.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import masks
import met_regrid
from eyeball_plots import pick_evenly
from obs_regrid_plots import (LABELS, PANEL_WIDTH_IN, _figure_title,
                              _panel_frame, centred_domain, grid_overlays,
                              track_segment)
from best_track import parse_bdeck
from plot_common import (DEFAULT_COLORBAR_STYLE, draw_precip, precip_scale)

NEITHER, KEPT, DISCARDED, GAP = 0, 1, 2, 3
CLASS_NAMES = {KEPT: "kept", DISCARDED: "discarded", GAP: "gap"}
# Okabe-Ito: blue kept, orange discarded, vermillion gap; no red/green pair
CLASS_COLORS = ["#f2f2f2", "#56b4e9", "#e69f00", "#d55e00"]
MET_MASK_NAME = "obs_valid_met"


def classify(mask, truth):
    """Class code (NEITHER/KEPT/DISCARDED/GAP) of every cell."""
    present = np.isfinite(truth)
    code = np.zeros(mask.shape, dtype=int)
    code[mask & present] = KEPT
    code[~mask & present] = DISCARDED
    code[mask & ~present] = GAP
    return code


def class_counts(code):
    return {name: int((code == c).sum()) for c, name in CLASS_NAMES.items()}


def met_reads_mask(case, truth_path):
    """Run the mask file through gen_vx_mask on the truth window's grid.

    Returns (boolean array MET produced, "") or (None, reason). The truth
    window is MET's own output, so its grid is one MET reads; if MET reads
    the mask file with a different grid it refuses or returns other cells."""
    try:
        tool = met_regrid.met_tool("gen_vx_mask", case.regrid.met_bin_dir)
    except FileNotFoundError as err:
        return None, str(err)
    import netCDF4
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mask_by_met.nc"
        cmd = [str(tool), str(truth_path), str(case.mask_path), str(out),
               "-type", "data", "-mask_field",
               f'name="{masks.MASK_NAME}"; level="(*,*)";',
               "-thresh", "gt0.5", "-name", MET_MASK_NAME]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            return None, "gen_vx_mask failed: " + " | ".join(tail)
        with netCDF4.Dataset(out) as nc:
            values = np.ma.filled(
                nc.variables[MET_MASK_NAME][:].astype(float), np.nan)
    return values > 0.5, ""


# =============================================================================
# Figures
# =============================================================================

def _frame(n, domain, title_text):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    lat_min, lat_max, lon_min, lon_max = domain
    map_h = PANEL_WIDTH_IN * (lat_max - lat_min) / (lon_max - lon_min)
    left, right, top, bottom, gap = 0.9, 0.3, 1.0, 1.5, 1.0
    fig_w = left + right + n * PANEL_WIDTH_IN + (n - 1) * gap
    fig_h = map_h + top + bottom
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=left / fig_w, right=1 - right / fig_w,
                        bottom=bottom / fig_h, top=1 - top / fig_h,
                        wspace=gap / PANEL_WIDTH_IN)
    return fig, list(axes[0]), fig_w, fig_h


def _save(fig, extra, out_path):
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=extra, pad_inches=0.4)
    plt.close(fig)
    return out_path


def plot_window(case, window, mask, truth, lat, lon, domain, overlays,
                track_line, out_path, style=DEFAULT_COLORBAR_STYLE):
    """truth | the four classes | the truth grid_stat will score."""
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    code = classify(mask, truth)
    counts = class_counts(code)
    fig, axes, fig_w, fig_h = _frame(3, domain, "")
    label = LABELS[case.plot_truth]
    titles = [f"{label} 3 h", "mask vs truth",
              f"{label} inside the mask (as scored)"]
    extra = []
    for ax, title in zip(axes, titles):
        extra += _panel_frame(ax, domain, track_line, title, overlays)
    plain = ccrs.PlateCarree()

    mesh, scale = draw_precip(axes[0], lon, lat, truth, style)
    # outside the mask nothing is scored: send it below the floor so the map
    # shows through, leaving only inside-mask gaps grey
    scored = np.where(mask, truth, 0.0)
    draw_precip(axes[2], lon, lat, scored, style)

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    axes[1].pcolormesh(lon, lat, code, cmap=cmap, norm=norm,
                       shading="nearest", transform=plain, zorder=1)
    axes[1].legend(handles=[
        Patch(facecolor=CLASS_COLORS[KEPT],
              label=f"kept: in mask, truth present ({counts['kept']:,})"),
        Patch(facecolor=CLASS_COLORS[DISCARDED],
              label=f"discarded: outside mask, truth present "
                    f"({counts['discarded']:,})"),
        Patch(facecolor=CLASS_COLORS[GAP],
              label=f"gap: in mask, truth missing ({counts['gap']:,})"),
        Patch(facecolor=CLASS_COLORS[NEITHER], edgecolor="#999999",
              label="outside mask, no truth")],
        loc="upper left", fontsize=7, framealpha=0.9)

    for ax in axes:
        ax.apply_aspect()
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    cax = fig.add_axes([x0, axes[0].get_position().y0 - 0.8 / fig_h,
                        x1 - x0, 0.14 / fig_h])
    fig.colorbar(mesh, cax=cax, orientation="horizontal",
                 ticks=scale["ticks"], extend=scale["extend"],
                 format=scale["format"], label="Precipitation (in)")
    extra += [cax, _figure_title(
        fig, axes, f"{case.storm_name} {case.deployment}: mask check, 3 h "
        f"ending {window.end:%Y-%m-%d %HZ}, {case.res_km:g} km grid")]
    return _save(fig, extra, out_path), counts


def plot_coverage(case, mask, gap_frac, data_frac, n_windows, lat, lon,
                  domain, overlays, track_line, out_path):
    """Per cell over every window: how often in-mask cells have no truth, and
    how often out-of-mask cells have some."""
    import cartopy.crs as ccrs
    from matplotlib.colors import Normalize
    plain = ccrs.PlateCarree()
    fig, axes, fig_w, fig_h = _frame(2, domain, "")
    label = LABELS[case.plot_truth]
    panels = [
        (f"inside the mask: windows with no {label} truth",
         np.where(mask, gap_frac, np.nan), "Oranges"),
        (f"outside the mask: windows with {label} truth",
         np.where(~mask, data_frac, np.nan), "Blues")]
    extra = []
    for ax, (title, field, cmap) in zip(axes, panels):
        extra += _panel_frame(ax, domain, track_line, title, overlays)
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=cmap,
                             norm=Normalize(0.0, 1.0), shading="nearest",
                             transform=plain, zorder=1)
        ax.apply_aspect()
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.8 / fig_h, pos.width,
                            0.14 / fig_h])
        fig.colorbar(mesh, cax=cax, orientation="horizontal",
                     label="fraction of windows")
        extra.append(cax)
    extra.append(_figure_title(
        fig, axes, f"{case.storm_name} {case.deployment}: mask coverage over "
        f"{n_windows} windows, {case.res_km:g} km grid"))
    return _save(fig, extra, out_path)


# =============================================================================
# Driver
# =============================================================================

def plot_mask_check(case):
    """Driver for `plot-mask-check`."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-mask-check: {case.case_slug} has no complete "
                         "WoFS window; nothing to plot.")
    if not case.mask_path.exists():
        raise SystemExit(f"plot-mask-check: {case.mask_path} is missing; run "
                         "build-mask first.")
    source = case.plot_truth
    windows = [w for w in aw.clock_windows(case.valid_start, case.valid_end)
               if case.obs_window_path(source, w.end).exists()]
    if not windows:
        raise SystemExit(f"plot-mask-check: no {LABELS[source]} 3 h windows "
                         "found; run window-obs first.")
    mask = masks.read_mask(case.mask_path)
    lat, lon, first = met_regrid.read_regridded(
        case.obs_window_path(source, windows[0].end))
    if mask.shape != first.shape:
        raise SystemExit(f"plot-mask-check: the mask is {mask.shape} but the "
                         f"truth grid is {first.shape}; rerun build-mask "
                         "at the same resolution.")
    print(f"Truth  : {LABELS[source]}, {len(windows)} windows  |  mask "
          f"{case.mask_path.name}: {int(mask.sum()):,} of {mask.size:,} cells")

    met_mask, why = met_reads_mask(case, case.obs_window_path(
        source, windows[0].end))
    if met_mask is None:
        print(f"MET    : not checked ({why})")
    else:
        differ = int((met_mask != mask).sum())
        print(f"MET    : gen_vx_mask reads the mask on the truth grid; "
              f"{differ} of {mask.size:,} cells differ from the file"
              + ("" if differ == 0 else "  <-- MET does not read the mask "
                 "as written; see the plan's C1 caveat"))

    truths = [met_regrid.read_regridded(case.obs_window_path(source, w.end))[2]
              for w in windows]
    for w, t in zip(windows, truths):
        if t.shape != mask.shape:
            raise SystemExit(f"plot-mask-check: the {w.end:%Y-%m-%d %HZ} "
                             f"window is {t.shape}, not {mask.shape}")
    present = np.array([np.isfinite(t) for t in truths])
    gap_frac = (~present).mean(axis=0)
    data_frac = present.mean(axis=0)

    domain = centred_domain(lat, lon)
    overlays = grid_overlays(lat, lon)
    overlays["contours"] = [(lat, lon, mask.astype(float), "#222222")]
    track = parse_bdeck(case.best_track)
    print(f"Output : {case.mask_plot_dir}\n")
    print(f"{'window end':>16} {'kept':>9} {'discarded':>10} {'gap':>9}  "
          "gap % of mask")
    written = []
    chosen = pick_evenly(list(zip(windows, truths)), case.eyeball.windows)
    for w, t in zip(windows, truths):
        counts = class_counts(classify(mask, t))
        print(f"{w.end:%Y-%m-%d %HZ} {counts['kept']:>9,} "
              f"{counts['discarded']:>10,} {counts['gap']:>9,}  "
              f"{100.0 * counts['gap'] / max(int(mask.sum()), 1):.1f}"
              + ("   (plotted)" if (w, t) in chosen else ""))
    for w, t in chosen:
        out, _ = plot_window(case, w, mask, t, lat, lon, domain, overlays,
                             track_segment(track, w.start, w.end),
                             case.mask_check_path(w.end),
                             case.eyeball.colorbar_style)
        written.append(out)
    written.append(plot_coverage(
        case, mask, gap_frac, data_frac, len(windows), lat, lon, domain,
        overlays, track_segment(track, windows[0].start, windows[-1].end),
        case.mask_check_path()))
    never = int((mask & (gap_frac == 1.0)).sum())
    lost = int((~mask & (data_frac > 0.0)).sum())
    print(f"\nIn the mask, {never:,} cells never have truth; outside it, "
          f"{lost:,} cells have truth in some window.")
    print(f"Saved {len(written)} figures under {case.mask_plot_dir}")
    return written
