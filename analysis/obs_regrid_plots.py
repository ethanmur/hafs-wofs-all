"""Maps of the MET-regridded obs written by regrid-obs (plot-regrid).

For every hour in the scored period, into case.regrid_plot_dir:
  <source>_full_<YYYYMMDDHH>.png
      native vs regridded, 1x2, framed on the deployment grid; written
      straight into regrid_plot_dir, since this is the usual single-product
      check and a subfolder of one thing is noise
  compare-products/products_full_<YYYYMMDDHH>.png
      MRMS | Stage IV on the regrid grid (needs both regridded)
  compare-anomaly/anomaly_full_<YYYYMMDDHH>.png
      truth | truth - other product, the truth product being the case's
      truth_source (needs both regridded)

Reads the obs cache and the regrid cache only; run regrid-obs first.

Usage:
    python analysis/run.py storms/<case>.yaml plot-regrid
"""

import csv
import math
import time
from datetime import timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory
import cartopy.crs as ccrs

from hafs_case import position_on_track
from plot_units import inches
from best_track import parse_bdeck
from plot_common import add_us_geography, QPF_LEVELS, qpf_cmap
import met_regrid
import verification_grid as vg
import obs_cases as oc

LABELS = {"mrms": "MRMS", "stage4": "Stage IV"}
NODATA_COLOR = "#d0d0d0"
# Fixed across hours so a sequence of anomaly maps is directly comparable.
DIFF_LEVELS_IN = np.array([-2, -1, -0.5, -0.25, -0.1, -0.05, -0.01,
                           0.01, 0.05, 0.1, 0.25, 0.5, 1, 2])
# Native grids are subsampled to at most this many cells per axis on
# full-domain maps (display only: ~6x the pixels a panel has anyway).
FULL_DOMAIN_MAX_CELLS = 1500
PANEL_WIDTH_IN = 7.0
DPI = 120


# =============================================================================
# Track and panel helpers
# =============================================================================

def track_segment(track, t0, t1, step_hours=1):
    """[(lat, lon), ...] of the best track sampled hourly over [t0, t1]."""
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    return [position_on_track(track, t0 + timedelta(hours=h))
            for h in range(0, n_hours + 1, step_hours)]


def _draw_track(ax, track_line):
    lats = [p[0] for p in track_line]
    lons = [p[1] for p in track_line]
    ax.plot(lons, lats, color="black", lw=1.4, transform=ccrs.PlateCarree(),
            zorder=5)
    ax.plot(lons[-1], lats[-1], marker="o", color="red", markersize=5,
            transform=ccrs.PlateCarree(), zorder=6)


GRID_EDGE_COLOR = "#444444"
# Blank margin around the regrid grid, as a fraction of its latitude span,
# so the domain outline is visible instead of running along the panel edge.
DOMAIN_MARGIN_FRAC = 0.04
DOMAIN_MARGIN_MIN_DEG = 1.5


def grid_overlays(rlat, rlon):
    """Outline of the regrid grid, for _panel_frame. The grid is the WoFS
    domain, so its outline is the domain boundary."""
    olat, olon = vg._grid_outline(rlat, rlon)
    return {"outline": (olat, olon)}


def centred_domain(rlat, rlon):
    """(lat_min, lat_max, lon_min, lon_max) framing the regrid grid.

    The grid figures are drawn on the grid's own extent with a small margin,
    so the regrid plots use the same framing and the two are comparable.
    """
    margin = max(DOMAIN_MARGIN_MIN_DEG,
                 DOMAIN_MARGIN_FRAC * float(rlat.max() - rlat.min()))
    return (max(float(rlat.min()) - margin, -85.0),
            min(float(rlat.max()) + margin, 85.0),
            float(rlon.min()) - margin, float(rlon.max()) + margin)


def _draw_overlays(ax, overlays):
    if not overlays:
        return
    olat, olon = overlays.get("outline", (None, None))
    if olat is not None:
        ax.plot(olon, olat, color=GRID_EDGE_COLOR, lw=1.6,
                transform=ccrs.PlateCarree(), zorder=7)
    for lat, lon, arr, colour in overlays.get("contours", []):
        ax.contour(lon, lat, arr.astype(float), levels=[0.5], colors=[colour],
                   linewidths=1.4, transform=ccrs.PlateCarree(), zorder=6)


def _panel_frame(ax, domain, track_line, title, overlays=None):
    """Draw one map panel's frame. Returns the text artists, which the
    caller MUST pass to savefig(bbox_extra_artists=...): they sit outside
    the axes, and GeoAxes.get_tightbbox() doesn't report them, so
    bbox_inches="tight" crops them off otherwise."""
    lat_min, lat_max, lon_min, lon_max = domain
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    add_us_geography(ax)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--",
                      alpha=0.5)
    gl.top_labels = gl.right_labels = False
    if track_line:
        _draw_track(ax, track_line)
    _draw_overlays(ax, overlays)
    # ax.set_title() on a GeoAxes is frequently clipped by
    # savefig(bbox_inches="tight"); a plain text artist in axes coordinates
    # is measured correctly and never gets cut off.
    return [
        ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11),
        ax.text(0.5, -0.09, "Longitude", transform=ax.transAxes, ha="center",
                va="top", fontsize=9),
        ax.text(-0.1, 0.5, "Latitude", transform=ax.transAxes, ha="center",
                va="center", rotation=90, fontsize=9),
    ]


def _figure_title(fig, axes, text):
    """Figure-wide title as a text artist just above the panel titles.

    fig.suptitle() places itself in figure coordinates, which on these
    wide, short figures leaves it stranded far above the maps. Blending
    figure-x with axes-y instead keeps it horizontally centred on the
    figure while pinning it to the top of the panels. Returned so the
    caller can include it in bbox_extra_artists.
    """
    transform = blended_transform_factory(fig.transFigure, axes[0].transAxes)
    return axes[0].text(0.5, 1.13, text, transform=transform, ha="center",
                        va="bottom", fontsize=13)


# =============================================================================
# Map figure
# =============================================================================

def _styles():
    qpf, _ = qpf_cmap()
    qpf = qpf.copy()
    qpf.set_bad(NODATA_COLOR)
    qpf_levels = np.asarray(inches(np.asarray(QPF_LEVELS, dtype=float)))
    diff = plt.get_cmap("RdBu", len(DIFF_LEVELS_IN) + 1).copy()
    diff.set_bad(NODATA_COLOR)
    return {
        "qpf": dict(cmap=qpf, norm=mcolors.BoundaryNorm(qpf_levels, qpf.N),
                    ticks=qpf_levels[::2], extend="max", format="%.2g",
                    label="Precipitation (in); grey = no data"),
        "diff": dict(cmap=diff,
                     norm=mcolors.BoundaryNorm(DIFF_LEVELS_IN, diff.N,
                                               extend="both"),
                     ticks=DIFF_LEVELS_IN, extend="both", format="%g",
                     label="Truth minus product (in); blue = truth wetter"),
    }


def crop_slices(lat, lon, domain, max_cells=None):
    """(row_slice, col_slice) bounding every cell inside `domain` plus a
    one-cell margin, strided down to <= max_cells per axis; None if no cell
    of the grid falls inside. Works on curvilinear grids."""
    lat_min, lat_max, lon_min, lon_max = domain
    inside = ((lat >= lat_min) & (lat <= lat_max)
              & (lon >= lon_min) & (lon <= lon_max))
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    if not rows.size:
        return None
    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, lat.shape[0])
    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, lat.shape[1])
    step = 1
    if max_cells:
        step = max(1, math.ceil(max(r1 - r0, c1 - c0) / max_cells))
    return slice(r0, r1, step), slice(c0, c1, step)


def map_figure(panels, domain, track_line, title, out_path,
               overlays=None):
    """panels: [(panel_title, lat2d, lon2d, data_mm_or_None, style)], style
    "qpf" or "diff". One horizontal colorbar under each run of same-style
    panels; data None draws an "unavailable" panel."""
    styles = _styles()
    lat_min, lat_max, lon_min, lon_max = domain
    map_h = PANEL_WIDTH_IN * (lat_max - lat_min) / (lon_max - lon_min)
    n = len(panels)
    top_in, bottom_in, left_in, right_in, gap_in = 1.0, 1.5, 0.9, 0.3, 1.0
    fig_w = left_in + right_in + n * PANEL_WIDTH_IN + (n - 1) * gap_in
    fig_h = map_h + top_in + bottom_in
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = list(axes[0])
    fig.subplots_adjust(left=left_in / fig_w, right=1 - right_in / fig_w,
                        bottom=bottom_in / fig_h, top=1 - top_in / fig_h,
                        wspace=gap_in / PANEL_WIDTH_IN)

    extra = []
    groups = []   # [style, [axes], mappable]
    for ax, (ptitle, lat, lon, data, style) in zip(axes, panels):
        extra += _panel_frame(ax, domain, track_line, ptitle, overlays)
        if data is None:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        s = styles[style]
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(inches(data)),
                             cmap=s["cmap"], norm=s["norm"], shading="nearest",
                             transform=ccrs.PlateCarree())
        if groups and groups[-1][0] == style:
            groups[-1][1].append(ax)
        else:
            groups.append([style, [ax], mesh])

    for style, group_axes, mesh in groups:
        for ax in group_axes:
            ax.apply_aspect()   # GeoAxes shrink to the map's aspect at draw time
        x0 = group_axes[0].get_position().x0
        x1 = group_axes[-1].get_position().x1
        y0 = min(ax.get_position().y0 for ax in group_axes)
        cax = fig.add_axes([x0, y0 - 0.8 / fig_h, x1 - x0, 0.14 / fig_h])
        s = styles[style]
        fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=s["ticks"],
                     extend=s["extend"], label=s["label"], format=s["format"])
        extra.append(cax)
    extra.append(_figure_title(fig, axes, title))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white",
                bbox_extra_artists=extra, pad_inches=0.4)
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

PANELS = ("compare-regrid", "compare-products", "compare-anomaly")


def available_sources(case, cfg, timestamps):
    """The configured products that actually have regridded output.

    A product with no output at all is dropped with a note -- the normal state
    when only the truth product has been regridded. One that is *partly*
    there is an error, since silently plotting a subset of hours would hide a
    failed regrid.
    """
    sources = list(case.regrid_sources)
    present, absent, partial = [], [], []
    for source in sources:
        have = sum(cfg.output_path(source, t).exists() for t in timestamps)
        if have == len(timestamps):
            present.append(source)
        elif have == 0:
            absent.append(source)
        else:
            partial.append((source, have))
    if partial:
        print("\nERROR: regridded output is incomplete -- run regrid-obs "
              "first:\n  python analysis/run.py <yaml> regrid-obs\n")
        for source, have in partial:
            print(f"  {source}: {have} of {len(timestamps)} hours present")
        raise SystemExit(1)
    if absent:
        print(f"Not regridded, skipping: {', '.join(absent)}")
    if not present:
        print(f"\nERROR: no regridded output for {', '.join(sources)} -- run "
              "regrid-obs first:\n  python analysis/run.py <yaml> regrid-obs\n")
        raise SystemExit(1)
    return present


def load_budget(case):
    """{(valid "YYYY-mm-dd HH:MM", source): mean_pct_diff} from the
    regrid-obs conservation CSV, or {} if it hasn't been written."""
    path = case.budget_csv_path
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["valid"], r["source"]): float(r["mean_pct_diff"])
                for r in csv.DictReader(fh)}


def _budget_note(budget, t, source):
    pct = budget.get((f"{t:%Y-%m-%d %H:%M}", source))
    return "" if pct is None else f"   |   area-mean change {pct:+.2f}%"


def _cropped(lat, lon, data, cut):
    if cut is None:
        return lat, lon, None
    return lat[cut], lon[cut], data[cut]


def plot_regrid(case):
    """Native-vs-regridded, product, and anomaly maps for every hour."""
    if not case.wofs_available:
        raise SystemExit(f"plot-regrid: {case.case_slug} has no complete WoFS "
                         "window; nothing to plot.")
    cfg = oc.regrid_config(case)
    timestamps = oc.hourly_timestamps(case.valid_start, case.valid_end)
    sources = available_sources(case, cfg, timestamps)
    oc._print_case_header(case, "Plot regridded obs", sources)
    # The native field is needed for the native-vs-regridded panels, so only
    # the products actually being plotted have to be cached.
    oc._exit_if_cache_incomplete(case, "plot-regrid", sources)

    truth = case.truth_source
    wanted = case.regrid_panels or PANELS
    kinds = list(wanted)
    if len(sources) < 2:
        dropped = [k for k in ("compare-products", "compare-anomaly")
                   if k in kinds]
        if dropped:
            print(f"Only {sources[0]} is regridded, so "
                  f"{' and '.join(dropped)} need a second product and are "
                  "skipped.")
        kinds = [k for k in kinds if k not in dropped]
    elif "compare-anomaly" in kinds and truth not in sources:
        print(f"compare-anomaly uses {LABELS[truth]} as truth and it is not "
              "regridded -- skipped.")
        kinds = [k for k in kinds if k != "compare-anomaly"]
    print(f"Panels:       {', '.join(kinds)}")

    # The regridded files carry the verification grid, so the plots can be
    # framed on it and outline it without re-reading the grid JSON.
    glat, glon, _ = met_regrid.read_regridded(
        cfg.output_path(sources[0], timestamps[0]))
    overlays = grid_overlays(glat, glon)

    out = case.regrid_plot_dir
    # compare-regrid is the common single-product case, so its maps go
    # straight into the plot folder; the multi-product panels keep subfolders.
    dirs = {k: (out if k == "compare-regrid" else out / k) for k in kinds}
    domain = centred_domain(glat, glon)
    grid_desc = f"{case.res_km:g} km deployment grid ({cfg.method})"
    print(f"Regrid cache: {cfg.cache_dir}")
    print(f"Output:       {out}")
    print(f"Domain:       {domain}", flush=True)

    track = parse_bdeck(case.best_track)
    budget = load_budget(case)
    if not budget:
        print("No regrid-obs conservation CSV found; headers will omit it.")
    native_ll = {}   # source -> (lat2d, lon2d), constant across hours
    cuts = {}        # grid key -> crop slices
    started = time.monotonic()
    for i, t in enumerate(timestamps, 1):
        track_line = track_segment(track, case.valid_start, t)
        stamp = f"{t:%Y%m%d%H}"
        valid = f"valid {t:%Y-%m-%d %HZ}"
        regridded = {}
        for source in sources:
            lat, lon, vals, _ = oc._native_hour(
                case, source, t, want_latlon=source not in native_ll)
            if source not in native_ll:
                native_ll[source] = (lat, lon)
            lat, lon = native_ll[source]
            rlat, rlon, rvals = met_regrid.read_regridded(
                cfg.output_path(source, t))
            regridded[source] = (rlat, rlon, rvals)

            for key, la, lo in ((source, lat, lon), ("regrid", rlat, rlon)):
                if key not in cuts:
                    cuts[key] = crop_slices(la, lo, domain,
                                            FULL_DOMAIN_MAX_CELLS)
            if "compare-regrid" not in kinds:
                continue
            label = LABELS[source]
            map_figure(
                [(f"{label} native",
                  *_cropped(lat, lon, vals, cuts[source]), "qpf"),
                 (f"{label} regridded",
                  *_cropped(rlat, rlon, rvals, cuts["regrid"]), "qpf")],
                domain, track_line,
                f"{label} 1-h precipitation, {valid}: native vs "
                f"{grid_desc}{_budget_note(budget, t, source)}",
                dirs["compare-regrid"] / f"{source}_full_{stamp}.png",
                overlays=overlays)

        rlat, rlon, _ = next(iter(regridded.values()))
        cut = cuts["regrid"]
        if "compare-products" in kinds:
            map_figure(
                [(LABELS[s], *_cropped(*regridded[s], cut), "qpf")
                 for s in sources],
                domain, track_line,
                f"Regridded 1-h precipitation on the {grid_desc}, {valid}",
                dirs["compare-products"] / f"products_full_{stamp}.png",
                overlays=overlays)

        if "compare-anomaly" in kinds and truth in regridded:
            truth_field = regridded[truth][2]
            anomaly = [(f"{LABELS[truth]} (truth)",
                        *_cropped(rlat, rlon, truth_field, cut), "qpf")]
            for s in sources:
                if s != truth:
                    anomaly.append((f"{LABELS[truth]} - {LABELS[s]}",
                                    *_cropped(rlat, rlon,
                                              truth_field - regridded[s][2],
                                              cut), "diff"))
            map_figure(anomaly, domain, track_line,
                       f"1-h precipitation anomaly vs {LABELS[truth]}, "
                       f"{grid_desc}, {valid}",
                       dirs["compare-anomaly"] / f"anomaly_full_{stamp}.png",
                       overlays=overlays)
        print(f"  [{i:>3}/{len(timestamps)}] {t:%Y-%m-%d %HZ}  plotted "
              f"(+{time.monotonic() - started:.0f}s)", flush=True)
    print(f"\nSaved maps under {out}")
