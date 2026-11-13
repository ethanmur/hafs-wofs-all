"""Member-exceedance swath maps, WoFS vs HAFS vs MRMS, one figure per window.

    python analysis/run.py storms/<case>.yaml plot-swaths [--thresh-mm 25]
                                                          [--domain wide|wofs]

Three panels per window:

  WoFS (18 Members)   the percentage of the WoFS members of the window's
                      shortest-lead init whose 3 h rain is at or above the
                      threshold (default 25 mm, about 1 inch); white to blue,
                      colorbar on the left.
  HAFS (16 Members)   the same for every regridded HFSA and HFSB cycle of the
                      window pooled as one ensemble (the 0-3 h cycle only with
                      hafs.include_spinup / --include-spinup); white to
                      vermillion, colorbar to its right.
  MRMS (Truth)        the truth 3 h rain on the eyeball-panel scale, with the full
                      NHC best track and the storm's position at the window
                      midpoint; colorbar on the right.

A black contour marks where MRMS reaches the threshold on all three panels, so
each ensemble's swath can be judged against the truth. Cells where no member
reaches the threshold are left blank. One figure per window, flat in
figures/ensemble-swaths/, named swaths_<case>_<window end>_ge<thr>mm.png. No MET;
reads the regridded files.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import adeck
import best_track as bt
import met_regrid
import wofs_windows as ww
from eyeball_plots import (_check_same_grid, _overlays, lead_text,
                           plotting_domain, runs_for)
from obs_regrid_plots import LABELS, _draw_overlays
from matplotlib.colors import LogNorm
from plot_common import NODATA_COLOR, add_us_geography, qpf_continuous_cmap

try:
    import cartopy.crs as ccrs
except ImportError:                                  # pragma: no cover
    ccrs = None

DEFAULT_THRESH_MM = 25.0
# Okabe-Ito blue and vermillion; the 100 % end is a darker shade of each
WOFS_COLORS = ("#0072B2", "#003A63")
HAFS_COLORS = ("#D55E00", "#7A2D00")
LEVELS = np.arange(0, 101, 10)
PANEL_WIDTH_IN = 4.7
DPI = 150
FS_TICK, FS_PANEL, FS_TITLE, FS_SUB, FS_BAR = 13, 17, 21, 14, 14
# The MRMS panel in mm: the eyeball-panel colormap (0.01-5 in) stretched over
# 0.25-125 mm (0.01-4.9 in), with round colorbar ticks.
MM_FLOOR, MM_MAX, MM_LEVELS = 0.25, 125.0, 20
MM_TICKS = [0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100]


# =============================================================================
# Pure logic
# =============================================================================

def exceedance_percent(fields, thresh_mm):
    """(ny, nx) percentage of members at or above `thresh_mm`. `fields` is a
    (n, ny, nx) array in mm; NaN members at a cell are left out of that cell's
    denominator, and a cell with no valid member is NaN."""
    fields = np.asarray(fields, dtype=float)
    valid = np.isfinite(fields)
    n = valid.sum(axis=0)
    hit = ((fields >= thresh_mm) & valid).sum(axis=0)
    out = np.full(n.shape, np.nan)
    np.divide(100.0 * hit, n, out=out, where=n > 0)
    return out


def white_to(colors, name):
    """A 10-step colormap from near-white to the dark end of `colors` (mid, dark):
    the first step is a pale tint, not white, so a 0-10 % cell is still visible
    against blank; 100 % is the dark shade."""
    full = LinearSegmentedColormap.from_list(
        name, ["#ffffff", colors[0], colors[1]], N=256)
    return ListedColormap([full(x) for x in np.linspace(0.12, 1.0, 10)], name)


def hafs_members(case, window):
    """[(config, init, path)] of the HAFS cycles of a window, HFSA then HFSB,
    oldest first. The cycle initialized at the window start (0-3 h) only when
    the case includes the spin-up."""
    out = []
    for config in case.hafs_models:
        for init, path in runs_for(case.hafs_regrid_dir_for(config), window.end,
                                   config.lower(), case.res_label):
            if init < window.start or case.hafs_include_spinup:
                out.append((config, init, path))
    return out


def swath_dir(case):
    """figures/ensemble-swaths/, beside the other figure folders."""
    return Path(case.eyeball_plot_dir).parent.parent / "ensemble-swaths"


def member_text(n_wofs, hafs):
    """'WoFS 18 members   |   HAFS 14 members (7 HFSA + 7 HFSB)' (the log)."""
    counts = {}
    for config, _, _ in hafs:
        counts[config] = counts.get(config, 0) + 1
    split = " + ".join(f"{n} {c}" for c, n in counts.items())
    return (f"WoFS {n_wofs} members   |   HAFS {len(hafs)} members "
            f"({split})")


def storm_position(track, when):
    """(lat, lon) of the best track at `when` (linear between fixes), or None
    when the track does not reach it."""
    seg, reached = adeck.track_to(track, when)
    return (seg[-1][1], seg[-1][2]) if reached and seg else None


# =============================================================================
# Drawing
# =============================================================================

def _frame(ax, domain, title, overlays, show_lat):
    """Extent, geography, gridlines and a panel title; returns the title text
    (it must go to savefig's bbox_extra_artists)."""
    lat_min, lat_max, lon_min, lon_max = domain
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    add_us_geography(ax)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--",
                      alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": FS_TICK}
    _draw_overlays(ax, overlays)
    return ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center",
                   va="bottom", fontsize=FS_PANEL)


def draw_precip_mm(ax, lon, lat, data_mm):
    """Filled log-scale contours of 3 h rain in mm on the eyeball colormap;
    cells under MM_FLOOR are left unpainted, missing cells grey. Returns the
    mappable for the colorbar."""
    data = np.ma.masked_invalid(np.asarray(data_mm, dtype=float))
    bad = np.ma.getmaskarray(data)
    if bad.any():
        ax.pcolormesh(lon, lat, np.where(bad, 1.0, np.nan),
                      cmap=ListedColormap([NODATA_COLOR]), shading="nearest",
                      transform=ccrs.PlateCarree(), zorder=1)
    levels = np.logspace(np.log10(MM_FLOOR), np.log10(MM_MAX), MM_LEVELS)
    painted = np.ma.masked_where(bad | (np.nan_to_num(data_mm, nan=0.0)
                                        < MM_FLOOR), data)
    kwargs = dict(levels=levels, extend="max", cmap=qpf_continuous_cmap(),
                  norm=LogNorm(vmin=MM_FLOOR, vmax=MM_MAX),
                  transform=ccrs.PlateCarree(), zorder=2)
    if painted.mask.all():          # a dry field: an invisible set for the bar
        cf = ax.contourf(lon, lat, np.full(np.shape(lon), levels[1]), **kwargs)
        for artist in getattr(cf, "collections", [cf]):
            artist.set_visible(False)
        return cf
    return ax.contourf(lon, lat, painted, **kwargs)


def _truth_contour(ax, lat, lon, truth, thresh_mm):
    if np.isfinite(truth).any() and np.nanmax(truth) >= thresh_mm:
        ax.contour(lon, lat, np.nan_to_num(truth, nan=0.0),
                   levels=[thresh_mm], colors="black", linewidths=2.2,
                   transform=ccrs.PlateCarree(), zorder=8)


def _draw_best_track(ax, track, position):
    """The full best track (thin, with fixes) and a star at `position`."""
    ax.plot([f[2] for f in track], [f[1] for f in track], color="#111111",
            lw=1.6, transform=ccrs.PlateCarree(), zorder=9)
    ax.plot([f[2] for f in track], [f[1] for f in track], ls="none",
            marker="o", ms=3.5, color="#111111", transform=ccrs.PlateCarree(),
            zorder=9)
    if position:
        ax.plot(position[1], position[0], marker="*", ms=24, mfc="#E69F00",
                mec="black", mew=1.4, ls="none", transform=ccrs.PlateCarree(),
                zorder=10)


def swath_figure(lat, lon, wofs_pct, hafs_pct, truth, domain, track, position,
                 titles, title, subtitle, thresh_mm, out_path, overlays=None):
    """Three map panels (WoFS | HAFS | MRMS) with the truth threshold contour
    on each; returns `out_path`."""
    lat_min, lat_max, lon_min, lon_max = domain
    map_h = PANEL_WIDTH_IN * (lat_max - lat_min) / (lon_max - lon_min)
    left, mid1, mid2, right = 2.3, 1.3, 2.6, 1.9      # margins and gaps, inches
    top, bottom = 1.5, 0.8
    fig_w = left + 3 * PANEL_WIDTH_IN + mid1 + mid2 + right
    fig_h = map_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    xs = [left, left + PANEL_WIDTH_IN + mid1,
          left + 2 * PANEL_WIDTH_IN + mid1 + mid2]
    axes = [fig.add_axes([x / fig_w, bottom / fig_h, PANEL_WIDTH_IN / fig_w,
                          map_h / fig_h], projection=ccrs.PlateCarree())
            for x in xs]
    extra = [_frame(ax, domain, t, overlays, k == 0)
             for k, (ax, t) in enumerate(zip(axes, titles))]
    meshes = []
    for ax, pct, colors, name in ((axes[0], wofs_pct, WOFS_COLORS, "WoFS"),
                                  (axes[1], hafs_pct, HAFS_COLORS, "HAFS")):
        cmap = white_to(colors, f"white_{name}")
        shown = np.ma.masked_where(~(np.nan_to_num(pct, nan=0.0) > 0), pct)
        meshes.append(ax.pcolormesh(lon, lat, shown, cmap=cmap,
                                    norm=BoundaryNorm(LEVELS, cmap.N),
                                    shading="nearest", zorder=3,
                                    transform=ccrs.PlateCarree()))
    mesh = draw_precip_mm(axes[2], lon, lat, truth)
    for ax in axes:
        _truth_contour(ax, lat, lon, truth, thresh_mm)
        ax.apply_aspect()
    _draw_best_track(axes[2], track, position)

    label = f"% Members \u2265 {thresh_mm:g} mm / 3h"
    pos = [ax.get_position() for ax in axes]
    bar_w = 0.24 / fig_w
    cax = fig.add_axes([0.5 / fig_w, pos[0].y0, bar_w, pos[0].height])
    cb = fig.colorbar(meshes[0], cax=cax, ticks=[0, 20, 40, 60, 80, 100])
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cb.set_label(label, fontsize=FS_BAR)
    cax2 = fig.add_axes([pos[1].x1 + 0.3 / fig_w, pos[1].y0, bar_w,
                         pos[1].height])
    cb2 = fig.colorbar(meshes[1], cax=cax2, ticks=[0, 20, 40, 60, 80, 100])
    cb2.set_label(label, fontsize=FS_BAR)
    cax3 = fig.add_axes([pos[2].x1 + 0.3 / fig_w, pos[2].y0, bar_w,
                         pos[2].height])
    cb3 = fig.colorbar(mesh, cax=cax3, ticks=MM_TICKS, extend="max",
                       format="%g")
    cb3.set_label("3h Precipitation (mm)", fontsize=FS_BAR)
    for c in (cax, cax2, cax3):
        c.tick_params(labelsize=FS_TICK)
    extra += [cax, cax2, cax3]
    extra.append(fig.text(0.5, 1 - 0.32 / fig_h, title, ha="center",
                          va="center", fontsize=FS_TITLE))
    extra.append(fig.text(0.5, 1 - 0.78 / fig_h, subtitle, ha="center",
                          va="center", fontsize=FS_SUB, color="#333333"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=extra, pad_inches=0.3)
    plt.close(fig)
    return out_path


# =============================================================================
# Driver
# =============================================================================

def plot_swaths(case, thresh_mm=DEFAULT_THRESH_MM, log=print):
    """Driver for `plot-swaths`: [Path] of the figures."""
    log(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-swaths: {case.case_slug} has no complete WoFS "
                         "window; nothing to plot.")
    truth_source = case.plot_truth
    windows, no_truth = [], 0
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        wofs = runs_for(case.wofs_regrid_3h_dir, w.end, "wofs", case.res_label)
        hafs = hafs_members(case, w)
        if not (wofs and hafs):
            continue
        if not case.obs_window_path(truth_source, w.end).exists():
            no_truth += 1
            continue
        windows.append((w, wofs[-1], hafs))
    log(f"Swaths : members at or above {thresh_mm:g} mm per 3 h against "
        f"{LABELS[truth_source]}; {len(windows)} windows with WoFS, HAFS and "
        f"truth" + (f" ({no_truth} more lack truth)" if no_truth else "")
        + f"; {case.eyeball.domain} map extent")
    if not windows:
        log("\nNo window has WoFS, HAFS and truth 3 h files. Run regrid-wofs, "
            "regrid-hafs and window-obs first.")
        return []
    lat, lon, ref = met_regrid.read_regridded(windows[0][2][0][2])
    domain = plotting_domain(case, lat, lon)
    overlays = _overlays(case, lat, lon)
    track = bt.parse_bdeck_status(case.best_track)
    out_dir = swath_dir(case)
    log(f"Output : {out_dir}\n")
    written, started = [], time.monotonic()
    for w, (w_init, w_path), hafs in windows:
        members = ww.read_window(w_path)
        _check_same_grid(ref.shape, members[0], "the WoFS window")
        fields = []
        for _, _, path in hafs:
            field = met_regrid.read_regridded(path)[2]
            _check_same_grid(ref.shape, field, "a HAFS window")
            fields.append(field)
        truth = met_regrid.read_regridded(
            case.obs_window_path(truth_source, w.end))[2]
        _check_same_grid(ref.shape, truth, "the truth window")
        out = out_dir / (f"swaths_{case.case_slug}_{w.end:%Y%m%d%H}"
                         f"_ge{thresh_mm:g}mm{case.plot_domain_suffix}.png")
        mid = w.start + (w.end - w.start) / 2
        swath_figure(
            lat, lon, exceedance_percent(members, thresh_mm),
            exceedance_percent(np.stack(fields), thresh_mm), truth, domain,
            track, storm_position(track, mid),
            (f"WoFS ({members.shape[0]} Members)",
             f"HAFS ({len(hafs)} Members)", f"{LABELS[truth_source]} (Truth)"),
            f"{case.storm_name} {case.deployment}: 3 h rain ending "
            f"{w.end:%Y-%m-%d %HZ}",
            f"Black contour: {LABELS[truth_source]} at {thresh_mm:g} mm / 3h",
            thresh_mm, out, overlays)
        written.append(out)
        log(f"  {w.end:%Y-%m-%d %HZ}  {member_text(members.shape[0], hafs)}  "
            f"(WoFS init {w_init:%d %H%MZ}, {lead_text(w_init, w)})  -> "
            f"{out.name}  (+{time.monotonic() - started:.0f}s)")
    log(f"\nSaved {len(written)} figures under {out_dir}")
    return written
