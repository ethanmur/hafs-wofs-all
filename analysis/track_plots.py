"""Storm-track maps for each WoFS domain and 3 h window (plan G3).

    python analysis/run.py storms/<case>.yaml plot-hafs-tracks
    python analysis/run.py storms/<case>.yaml plot-tc-location

plot-hafs-tracks   The NHC best track under the HAFS forecast tracks of every
                   cycle in that window's sample (HFSA blues, HFSB oranges,
                   darker = shorter lead), each drawn from its initialization to
                   the window end only. The tracks come from the storm's a-deck
                   (`fetch_bdecks.py --adeck`), interpolated to the window end.
                   The cycles are the ones the window was scored with, so a
                   storm far inland simply has fewer. Shows track errors.
plot-tc-location   The `storm-location` table made visual: best track, WoFS
                   domain outline and center, the best-track fix nearest the
                   window midpoint, the final landfall, and the distances from the
                   domain center to each.

One figure per window, flat in figures/storm-tracks/ (beside the other figure
folders), named hafs-tracks_<case>_<window end>.png and
tc-location_<case>_<window end>.png. Plain matplotlib and cartopy; no MET.
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import adeck
import best_track as bt
import storm_location as sl
import verification_grid as vg
from eyeball_plots import runs_for
from plot_common import add_us_geography

try:
    import cartopy.crs as ccrs
except ImportError:                                  # pragma: no cover
    ccrs = None

HAFS_CMAPS = {"HFSA": "Blues", "HFSB": "Oranges"}
BEST_COLOR = "#111111"
MARGIN_DEG = 1.5
MIN_SPAN_DEG = 6.0


# =============================================================================
# Inputs
# =============================================================================

def adeck_path(case):
    """Where the storm's a-deck is kept: data/adeck/, beside data/bdeck/."""
    name = "a" + Path(case.best_track).name[1:]
    return Path(case.best_track).parent.parent / "adeck" / name


def track_dir(case):
    """figures/storm-tracks/, beside the other figure folders (the eyeball
    folder is figures/eyeball-panels/<case>/)."""
    return Path(case.eyeball_plot_dir).parent.parent / "storm-tracks"


def hafs_cycles(case, window):
    """{config: [init, ...]} of the HAFS cycles the window was scored with,
    oldest first: those with a regridded file for it."""
    return {c: [i for i, _ in runs_for(case.hafs_regrid_dir_for(c), window.end,
                                       c.lower(), case.res_label)]
            for c in case.hafs_models}


def domain_outline(case):
    """(lat, lon) edge of the verification grid and the grid's center."""
    spec = vg.grid_spec_from_json(case.grid_json_path)
    lat, lon = vg.grid_latlon(spec)
    olat, olon = vg._grid_outline(lat, lon)
    return np.asarray(olat), np.asarray(olon), sl.domain_center(case)


def _landfall_test(log=print):
    """The mainland land test, or None (with a note) when Natural Earth is
    unavailable, so the maps still draw without a landfall."""
    try:
        return sl.mainland_land_test()
    except Exception as exc:                          # offline compute node
        log(f"  no landfall markers: Natural Earth land unavailable ({exc})")
        return None


# =============================================================================
# Drawing
# =============================================================================

def _extent(points):
    """(lon_min, lon_max, lat_min, lat_max) around (lat, lon) points, padded and
    made at least MIN_SPAN_DEG across."""
    lat = [p[0] for p in points]
    lon = [p[1] for p in points]
    lo0, lo1 = min(lon) - MARGIN_DEG, max(lon) + MARGIN_DEG
    la0, la1 = min(lat) - MARGIN_DEG, max(lat) + MARGIN_DEG
    for lo, hi, setter in ((lo0, lo1, "lon"), (la0, la1, "lat")):
        pad = (MIN_SPAN_DEG - (hi - lo)) / 2
        if pad > 0:
            if setter == "lon":
                lo0, lo1 = lo - pad, hi + pad
            else:
                la0, la1 = lo - pad, hi + pad
    return lo0, lo1, max(la0, -85.0), min(la1, 85.0)


def _map_axes(extent, title, subtitle):
    fig = plt.figure(figsize=(8.2, 7.0))
    ax = fig.add_axes([0.07, 0.1, 0.66, 0.8], projection=ccrs.PlateCarree())
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    add_us_geography(ax)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    ax.set_title(title, fontsize=11, loc="left")
    fig.text(0.07, 0.04, subtitle, fontsize=7.5, color="#555555")
    return fig, ax


def _draw_best_track(ax, track, end):
    """The best track: bold up to `end`, thin beyond it for context."""
    upto, _ = adeck.track_to(track, end)     # to the window end, interpolated
    after = [(t, la, lo) for t, la, lo, _ in track if t >= end]
    if upto and after:
        ax.plot([upto[-1][2]] + [p[2] for p in after],
                [upto[-1][1]] + [p[1] for p in after], color="#888888",
                lw=1.0, ls=":", transform=ccrs.PlateCarree(), zorder=2)
    fixes = [p for p in upto if p[0] != end or any(
        f[0] == end for f in track)]
    ax.plot([p[2] for p in upto], [p[1] for p in upto], color=BEST_COLOR,
            lw=3.2, solid_capstyle="round", transform=ccrs.PlateCarree(),
            zorder=3, path_effects=[pe.Stroke(linewidth=5, foreground="white"),
                                    pe.Normal()])
    ax.plot([p[2] for p in fixes], [p[1] for p in fixes], ls="none",
            marker="o", ms=3.5, color=BEST_COLOR,
            transform=ccrs.PlateCarree(), zorder=4)


def _draw_domain(ax, olat, olon, center):
    ax.plot(olon, olat, color="#7b3294", lw=1.8, transform=ccrs.PlateCarree(),
            zorder=5)
    ax.plot(center[1], center[0], marker="+", ms=11, mew=2, color="#7b3294",
            ls="none", transform=ccrs.PlateCarree(), zorder=6)


def _legend(ax, handles):
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0))


def _save(fig, path, dpi):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _landfall_marker(ax, ref):
    ax.plot(ref[2], ref[1], marker="^", ms=11, mfc="white", mec="black",
            mew=1.6, ls="none", transform=ccrs.PlateCarree(), zorder=7)


# =============================================================================
# plot-hafs-tracks
# =============================================================================

def plot_window_hafs_tracks(case, window, track, tracks, cycles, olat, olon,
                            center, landfall, dpi=130, out_dir=None):
    """One figure: best track beneath the HAFS tracks of `cycles`
    ({config: [init]}) to the window end. Returns (path, n drawn, n missing)."""
    end = window.end
    drawn, missing, pts, handles = {}, 0, [], []
    lines = []
    for config, inits in cycles.items():
        n = len(inits)
        cmap = plt.get_cmap(HAFS_CMAPS.get(config, "Greys"))
        for k, init in enumerate(inits):               # oldest -> newest
            tr = tracks.get((config, init))
            seg, reached = adeck.track_to(tr, end) if tr else ([], False)
            if len(seg) < 2:
                missing += 1
                continue
            shade = 0.35 + 0.6 * (k / max(n - 1, 1))    # newer = darker
            lines.append((config, shade, cmap, seg, reached))
            drawn[config] = drawn.get(config, 0) + 1
            pts += [(la, lo) for _, la, lo in seg]
    best_pts = [(la, lo) for t, la, lo, _ in track
                if end - timedelta(hours=24) <= t <= end]
    pts += best_pts + list(zip(olat, olon))
    if landfall:
        pts.append((landfall[1], landfall[2]))
    fig, ax = _map_axes(
        _extent(pts), f"HAFS forecast tracks vs best track: {case.storm_name} "
        f"{case.deployment}, window ending {end:%Y-%m-%d %HZ}",
        "Each HAFS track runs from its initialization to the window end "
        "(a-deck, linear between 6 h fixes); darker = shorter lead.")
    _draw_best_track(ax, track, end)
    for config, shade, cmap, seg, reached in lines:
        ax.plot([p[2] for p in seg], [p[1] for p in seg], color=cmap(shade),
                lw=1.4, transform=ccrs.PlateCarree(), zorder=8)
        ax.plot(seg[0][2], seg[0][1], marker="s", ms=3, color=cmap(shade),
                ls="none", transform=ccrs.PlateCarree(), zorder=8)
        ax.plot(seg[-1][2], seg[-1][1], marker="o" if reached else "x", ms=6,
                mfc=cmap(shade), mec="black", mew=0.6, ls="none",
                transform=ccrs.PlateCarree(), zorder=9)
    _draw_domain(ax, olat, olon, center)
    if landfall:
        _landfall_marker(ax, landfall)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BEST_COLOR, lw=3, label="NHC best track")]
    for config in cycles:
        handles.append(Line2D([], [], color=plt.get_cmap(
            HAFS_CMAPS.get(config, "Greys"))(0.8), lw=2,
            label=f"{config} ({drawn.get(config, 0)} cycles)"))
    handles.append(Line2D([], [], color="#7b3294", lw=2, label="WoFS domain"))
    if landfall:
        handles.append(Line2D([], [], marker="^", ls="none", mfc="white",
                              mec="black", ms=9, label="final landfall"))
    handles.append(Line2D([], [], marker="o", ls="none", mfc="#999999",
                          mec="black", ms=6, label="track at window end"))
    _legend(ax, handles)
    path = Path(out_dir or track_dir(case)) / (
        f"hafs-tracks_{case.case_slug}_{end:%Y%m%d%H}.png")
    _save(fig, path, dpi)
    return path, sum(drawn.values()), missing


def plot_hafs_tracks(case, dpi=130, log=print):
    """Driver for `plot-hafs-tracks`: [Path] of the figures."""
    log(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-hafs-tracks: {case.case_slug} has no complete "
                         "WoFS window; nothing to plot.")
    path = adeck_path(case)
    if not path.exists():
        raise SystemExit(
            f"plot-hafs-tracks: no a-deck at {path}; fetch it with\n"
            f"  python analysis/fetch_bdecks.py --adeck --storms "
            f"{case.storm_id}")
    track = bt.parse_bdeck_status(case.best_track)
    tracks = adeck.parse_adeck(path)
    olat, olon, center = domain_outline(case)
    land = _landfall_test(log)
    falls = sl.landfalls(track, land) if land else []
    landfall = falls[-1] if falls else None
    out_dir = track_dir(case)
    log(f"a-deck : {path.name}, {len(tracks)} HFSA/HFSB cycles")
    log(f"Output : {out_dir}\n")
    written, started = [], time.monotonic()
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        cycles = hafs_cycles(case, w)
        if not any(cycles.values()):
            log(f"  {w.end:%Y-%m-%d %HZ}  no HAFS cycles in the sample; "
                "skipped")
            continue
        p, n, missing = plot_window_hafs_tracks(
            case, w, track, tracks, cycles, olat, olon, center, landfall, dpi)
        written.append(p)
        log(f"  {w.end:%Y-%m-%d %HZ}  {n} cycle tracks"
            + (f", {missing} not in the a-deck" if missing else "")
            + f"  -> {p.name}  (+{time.monotonic() - started:.0f}s)")
    log(f"\nSaved {len(written)} figures under {out_dir}")
    return written


# =============================================================================
# plot-tc-location
# =============================================================================

def _geodesic(ax, a, b, color, ls="--"):
    ax.plot([a[1], b[1]], [a[0], b[0]], color=color, lw=1.4, ls=ls,
            transform=ccrs.Geodetic(), zorder=6)


def _label_mid(ax, a, b, text, color, ha="center"):
    ax.text((a[1] + b[1]) / 2, (a[0] + b[0]) / 2, text, fontsize=8,
            color=color, ha=ha, va="bottom", zorder=10,
            transform=ccrs.PlateCarree(),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def plot_window_tc_location(case, row, track, olat, olon, center, landfall,
                            dpi=130, out_dir=None):
    """One figure for a `storm-location` row; returns the path."""
    end = row["window_end"]
    fix = row["fix"]
    pts = list(zip(olat, olon)) + [(la, lo) for t, la, lo, _ in track
                                   if abs((t - end).total_seconds())
                                   <= 36 * 3600]
    if fix:
        pts.append((fix[1], fix[2]))
    if landfall:
        pts.append((landfall[1], landfall[2]))
    sub = ("Nearest best-track fix to the window midpoint "
           f"({row['midpoint']:%m-%d %H:%MZ}); fixes beyond 3 h are not used.")
    fig, ax = _map_axes(
        _extent(pts), f"Best-track location vs WoFS domain: {case.storm_name} "
        f"{case.deployment}, window ending {end:%Y-%m-%d %HZ}", sub)
    _draw_best_track(ax, track, end)
    _draw_domain(ax, olat, olon, center)
    lines = [f"domain center {abs(center[0]):.2f}"
             f"{'N' if center[0] >= 0 else 'S'} {abs(center[1]):.2f}"
             f"{'E' if center[1] >= 0 else 'W'}"]
    if fix:
        t, la, lo, status = fix
        ax.plot(lo, la, marker="*", ms=17, mfc="#E69F00", mec="black",
                mew=1.0, ls="none", transform=ccrs.PlateCarree(), zorder=9)
        _geodesic(ax, center, (la, lo), "#E69F00")
        _label_mid(ax, center, (la, lo), f"{row['tc_km']:.0f} km", "#9a6400",
                   ha="right")
        lines += [f"nearest fix {t:%m-%d %HZ}", f"  {row['offset_h']:+.1f} h "
                  f"from the window midpoint, {status}",
                  f"TC center: {row['tc_km']:.0f} km", "  from the domain center"]
    else:
        lines.append(f"time out of bounds: {abs(row['offset_h']):.1f} h off "
                     f"(limit {sl.TOLERANCE_H:g} h)")
    if landfall:
        _landfall_marker(ax, landfall)
        _geodesic(ax, center, (landfall[1], landfall[2]), "#0072B2")
        _label_mid(ax, center, (landfall[1], landfall[2]),
                   f"{row['landfall_km']:.0f} km", "#004c7a", ha="left")
        lines += [f"landfall {landfall[0]:%m-%d %H:%MZ}: "
                  f"{row['landfall_km']:.0f} km",
                  "  from the domain center",
                  f"  window {row['from_landfall_h']:+.1f} h from landfall"]
    else:
        lines.append("no mainland landfall found in the best track")
    ax.text(1.02, 0.4, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
            va="top")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BEST_COLOR, lw=3, label="NHC best track"),
               Line2D([], [], color="#7b3294", lw=2, label="WoFS domain"),
               Line2D([], [], marker="+", ls="none", color="#7b3294", ms=10,
                      mew=2, label="domain center"),
               Line2D([], [], marker="*", ls="none", mfc="#E69F00", mec="black",
                      ms=13, label="nearest fix"),
               Line2D([], [], marker="^", ls="none", mfc="white", mec="black",
                      ms=9, label="final landfall")]
    _legend(ax, handles)
    path = Path(out_dir or track_dir(case)) / (
        f"tc-location_{case.case_slug}_{end:%Y%m%d%H}.png")
    _save(fig, path, dpi)
    return path


def plot_tc_location(case, dpi=130, log=print):
    """Driver for `plot-tc-location`: [Path] of the figures."""
    log(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-tc-location: {case.case_slug} has no complete "
                         "WoFS window; nothing to plot.")
    track = bt.parse_bdeck_status(case.best_track)
    olat, olon, center = domain_outline(case)
    land = _landfall_test(log)
    result = sl.locate(case, track, center, land or (lambda la, lo: False))
    landfall = result["reference"]
    out_dir = track_dir(case)
    log(f"Output : {out_dir}\n")
    written, started = [], time.monotonic()
    for row in result["rows"]:
        p = plot_window_tc_location(case, row, track, olat, olon, center,
                                    landfall, dpi)
        written.append(p)
        what = (f"{row['tc_km']:.0f} km" if row["fix"]
                else f"time out of bounds, {abs(row['offset_h']):.1f} h off")
        log(f"  {row['window_end']:%Y-%m-%d %HZ}  {what}  -> {p.name}  "
            f"(+{time.monotonic() - started:.0f}s)")
    log(f"\nSaved {len(written)} figures under {out_dir}")
    return written
