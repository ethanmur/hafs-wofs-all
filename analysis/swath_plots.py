"""Member-exceedance swath maps, WoFS vs HAFS, one figure per window.

    python analysis/run.py storms/<case>.yaml plot-swaths [--thresh-mm 25]
                                                          [--domain wide|wofs]

At every cell, the percentage of members whose 3 h rain is at or above one
threshold (default 25 mm, about 1 inch). WoFS is the left panel (white to blue,
its colorbar on the left), HAFS the right panel (white to vermillion, its colorbar
on the right); the figure title carries the member counts. Cells where no member
reaches the threshold are left blank, so only the swaths show.

  WoFS   the 18 members of the window's shortest-lead init.
  HAFS   every regridded HFSA and HFSB cycle of the window, pooled as one
         ensemble (the 0-3 h cycle only with hafs.include_spinup / --include-spinup).

Also drawn: the best-track segment of the window and the verification-domain
outline / observation-validity contour, as in the eyeball panels. One figure per
window with both ensembles, flat in figures/ensemble-swaths/, named
swaths_<case>_<window end>_ge<thr>mm.png. No MET; reads the regridded files.
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
import met_regrid
import wofs_windows as ww
from best_track import parse_bdeck
from eyeball_plots import (_check_same_grid, _overlays, lead_text,
                           plotting_domain, runs_for)
from obs_regrid_plots import _panel_frame, track_segment

try:
    import cartopy.crs as ccrs
except ImportError:                                  # pragma: no cover
    ccrs = None

DEFAULT_THRESH_MM = 25.0
WOFS_COLOR = "#0072B2"       # Okabe-Ito blue
HAFS_COLOR = "#D55E00"       # Okabe-Ito vermillion
LEVELS = np.arange(0, 101, 10)
PANEL_WIDTH_IN = 5.0
DPI = 150


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


def white_to(color, name):
    """A 10-step white-to-`color` colormap (the first step is the palest tint,
    not white, so a 0-10 % cell is still visible against blank)."""
    full = LinearSegmentedColormap.from_list(name, ["#ffffff", color], N=256)
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


# =============================================================================
# Drawing
# =============================================================================

def _thresh_text(thresh_mm):
    return f"{thresh_mm:g} mm"


def swath_figure(lat, lon, wofs_pct, hafs_pct, domain, track_line, title,
                 subtitle, thresh_mm, out_path, overlays=None):
    """Two map panels with outer colorbars; returns `out_path`."""
    lat_min, lat_max, lon_min, lon_max = domain
    map_h = PANEL_WIDTH_IN * (lat_max - lat_min) / (lon_max - lon_min)
    left, right, gap, top, bottom = 1.9, 1.5, 1.0, 1.3, 0.9
    fig_w = left + right + 2 * PANEL_WIDTH_IN + gap
    fig_h = map_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for k in range(2):
        x0 = left + k * (PANEL_WIDTH_IN + gap)
        axes.append(fig.add_axes([x0 / fig_w, bottom / fig_h,
                                  PANEL_WIDTH_IN / fig_w, map_h / fig_h],
                                 projection=ccrs.PlateCarree()))
    extra = []
    meshes = []
    for ax, name, pct, color in ((axes[0], "WoFS", wofs_pct, WOFS_COLOR),
                                 (axes[1], "HAFS", hafs_pct, HAFS_COLOR)):
        extra += _panel_frame(ax, domain, track_line, name, overlays)
        cmap = white_to(color, f"white_{name}")
        norm = BoundaryNorm(LEVELS, cmap.N)
        shown = np.ma.masked_where(~(np.nan_to_num(pct, nan=0.0) > 0), pct)
        meshes.append(ax.pcolormesh(lon, lat, shown, cmap=cmap, norm=norm,
                                    shading="nearest", zorder=3,
                                    transform=ccrs.PlateCarree()))
    for ax in axes:
        ax.apply_aspect()
    label = f"% of members at or above {_thresh_text(thresh_mm)} per 3 h"
    for k, (ax, mesh) in enumerate(zip(axes, meshes)):
        pos = ax.get_position()
        if k == 0:        # outside the latitude labels, on the figure's left
            cax = fig.add_axes([0.45 / fig_w, pos.y0, 0.22 / fig_w, pos.height])
        else:             # right of the HAFS map
            cax = fig.add_axes([pos.x1 + 0.3 / fig_w, pos.y0, 0.22 / fig_w,
                                pos.height])
        cb = fig.colorbar(mesh, cax=cax, ticks=[0, 20, 40, 60, 80, 100],
                          label=label if k else "")
        if k == 0:
            cax.yaxis.set_ticks_position("left")
            cax.yaxis.set_label_position("left")
            cb.set_label(label)
        extra.append(cax)
    extra.append(fig.text(0.5, 1 - 0.28 / fig_h, title, ha="center",
                          va="center", fontsize=14))
    extra.append(fig.text(0.5, 1 - 0.62 / fig_h, subtitle, ha="center",
                          va="center", fontsize=11, color="#444444"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=extra, pad_inches=0.3)
    plt.close(fig)
    return out_path


def member_text(n_wofs, hafs):
    """'WoFS 18 members   |   HAFS 14 members (7 HFSA + 7 HFSB)'."""
    counts = {}
    for config, _, _ in hafs:
        counts[config] = counts.get(config, 0) + 1
    split = " + ".join(f"{n} {c}" for c, n in counts.items())
    return (f"WoFS {n_wofs} members   |   HAFS {len(hafs)} members "
            f"({split})")


# =============================================================================
# Driver
# =============================================================================

def plot_swaths(case, thresh_mm=DEFAULT_THRESH_MM, log=print):
    """Driver for `plot-swaths`: [Path] of the figures."""
    log(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-swaths: {case.case_slug} has no complete WoFS "
                         "window; nothing to plot.")
    windows = []
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        wofs = runs_for(case.wofs_regrid_3h_dir, w.end, "wofs", case.res_label)
        hafs = hafs_members(case, w)
        if wofs and hafs:
            windows.append((w, wofs[-1], hafs))
    log(f"Swaths : members at or above {_thresh_text(thresh_mm)} per 3 h; "
        f"{len(windows)} windows with WoFS and HAFS; {case.eyeball.domain} "
        "map extent")
    if not windows:
        log("\nNo window has both WoFS and HAFS 3 h files. Run regrid-wofs and "
            "regrid-hafs first.")
        return []
    first = windows[0][2][0][2]
    lat, lon, ref = met_regrid.read_regridded(first)
    domain = plotting_domain(case, lat, lon)
    overlays = _overlays(case, lat, lon)
    track = parse_bdeck(case.best_track)
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
        out = out_dir / (f"swaths_{case.case_slug}_{w.end:%Y%m%d%H}"
                         f"_ge{thresh_mm:g}mm{case.plot_domain_suffix}.png")
        swath_figure(
            lat, lon, exceedance_percent(members, thresh_mm),
            exceedance_percent(np.stack(fields), thresh_mm), domain,
            track_segment(track, w.start, w.end),
            f"{case.storm_name} {case.deployment}: 3 h rain ending "
            f"{w.end:%Y-%m-%d %HZ}",
            member_text(members.shape[0], hafs), thresh_mm, out, overlays)
        written.append(out)
        log(f"  {w.end:%Y-%m-%d %HZ}  WoFS {members.shape[0]} "
            f"(init {w_init:%d %H%MZ}, {lead_text(w_init, w)}), HAFS "
            f"{len(hafs)}  -> {out.name}  (+{time.monotonic() - started:.0f}s)")
    log(f"\nSaved {len(written)} figures under {out_dir}")
    return written
