"""D5b: HAFS vs Stage IV panels, one per lead, for eyeballing.

Between the obs windows (D5) and the manifest (D6): before any score is
computed, look at the fields. Each figure is one 3 h window from one cycle --
Stage IV on the left, HAFS on the right, same grid, same colour scale -- so a
regridding mistake shows up as a displaced or empty panel rather than as a
strange FSS three steps later, and the forecast quality can be judged by eye.

Restricted to the 00Z cycles by default. Every cycle would be a few thousand
figures per case and they would mostly repeat each other; the 00Z runs give a
clean lead-time sequence across the case, which is what a subjective look
wants.

The map window follows WoFS: that is where the three-way comparison will
happen, so it is the region worth judging. With several deployments, the one
nearest in time to the window is used, since a deployment two days away is
looking at different weather. With none configured, the figure falls back to
the whole verification grid.

Usage:
    python analysis/run.py storms/<case>.yaml plot-hafs
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import accum_windows as aw
import met_regrid
import obs_cases as oc
from best_track import parse_bdeck
from obs_regrid_plots import (LABELS, centred_domain, crop_slices, map_figure,
                              grid_overlays, track_segment,
                              FULL_DOMAIN_MAX_CELLS, _cropped)

DEFAULT_INIT_HOURS = (0,)
# Blank margin around a WoFS box, so the box edge is visible inside the panel.
DOMAIN_PAD_DEG = 0.75


def obs_window_field(case, window, source="stage4"):
    """(lat, lon, 3 h sum) of the regridded obs over `window`, or None.

    The hourly files are stamped at the END of their hour, so the window
    (15Z, 18Z] is the 16Z, 17Z and 18Z files. A missing hour makes the whole
    window missing rather than a partial sum scaled up: a 2 h total is not a
    3 h total, and pretending otherwise biases every threshold.
    """
    n = int(round(window.hours))
    stamps = [window.start + timedelta(hours=h) for h in range(1, n + 1)]
    paths = [case.regrid.output_path(source, t) for t in stamps]
    missing = [t for t, p in zip(stamps, paths) if not p.exists()]
    if missing:
        return None, missing
    lat = lon = total = None
    for path in paths:
        rlat, rlon, vals = met_regrid.read_regridded(path)
        if total is None:
            lat, lon, total = rlat, rlon, np.zeros_like(vals)
        total = total + vals
    return (lat, lon, total), []


def pick_domain(case, window, fallback):
    """The map window for this accumulation window, and a label for it.

    With several WoFS deployments, the nearest in time wins: distance is zero
    while a deployment is live and grows with the gap either side, so a window
    inside a deployment always picks that one.
    """
    domains = case.wofs_domains or []
    if not domains:
        return fallback, "full grid (no WoFS domain configured)"
    mid = window.start + (window.end - window.start) / 2

    def gap(dom):
        if dom.valid_start is None or dom.valid_end is None:
            return timedelta(0)
        if dom.valid_start <= mid <= dom.valid_end:
            return timedelta(0)
        return min(abs(mid - dom.valid_start), abs(mid - dom.valid_end))

    best = min(domains, key=gap)
    lat0, lat1, lon0, lon1 = best.domain
    pad = DOMAIN_PAD_DEG
    hours = gap(best).total_seconds() / 3600.0
    note = best.name if hours == 0 else f"{best.name}, {hours:.0f} h away"
    return ((lat0 - pad, lat1 + pad, lon0 - pad, lon1 + pad), note)


def plot_hafs_obs_case(case):
    """One Stage IV vs HAFS panel per (00Z cycle, window)."""
    hcfg = case.hafs
    if hcfg is None or case.regrid is None:
        raise SystemExit("ERROR: plot-hafs needs both `hafs:` and `regrid:` "
                         "blocks in the YAML")
    if case.init_start is None or case.init_end is None:
        raise SystemExit("ERROR: plot-hafs needs init_start and init_end")

    init_hours = tuple(case.hafs_plot_init_hours or DEFAULT_INIT_HOURS)
    windows = aw.clock_windows(case.valid_start, case.valid_end)
    runs = [r for r in aw.hafs_runs(case.init_start, case.init_end,
                                    hcfg.cycle_h, hcfg.max_lead_h,
                                    model=hcfg.model)
            if r.init.hour in init_hours]
    coverages = aw.coverage(runs, windows)

    out = Path(case.hafs_plot_dir)
    source = case.truth_source
    print(f"Case     : {case.storm_name}  ({case.case_slug})")
    print(f"Cycles   : {len(runs)} at "
          + ", ".join(f"{h:02d}Z" for h in sorted(init_hours))
          + f"  ({hcfg.model})")
    print(f"Pairs    : {len(coverages)} (cycle, window) panels")
    print(f"Truth    : {LABELS.get(source, source)} summed to 3 h windows")
    print(f"HAFS     : {hcfg.cache_dir}")
    print(f"Output   : {out}", flush=True)
    if not coverages:
        raise SystemExit(
            f"ERROR: no {init_hours} cycle covers any window in this case. "
            f"Widen init_start/init_end, or set hafs_plots.init_hours.")

    track = parse_bdeck(case.best_track)
    # Grid geometry from the first HAFS file that exists, for the outline and
    # the fallback map window.
    geom = None
    for cov in coverages:
        path = hcfg.output_path(cov.run.init, int(cov.lead_end_min // 60))
        if path.exists():
            geom = met_regrid.read_regridded(path)[:2]
            break
    if geom is None:
        raise SystemExit(
            f"ERROR: no regridded HAFS output under {hcfg.cache_dir}. Run "
            f"regrid-hafs first:\n  python analysis/run.py <yaml> regrid-hafs")
    glat, glon = geom
    overlays = grid_overlays(glat, glon, case.wofs_domains)
    full = centred_domain(glat, glon)

    cuts, skipped = {}, {"hafs": 0, "obs": 0}
    started = time.monotonic()
    for i, cov in enumerate(coverages, 1):
        lead_h = int(cov.lead_end_min // 60)
        hafs_path = hcfg.output_path(cov.run.init, lead_h)
        if not hafs_path.exists():
            skipped["hafs"] += 1
            continue
        obs, missing = obs_window_field(case, cov.window, source)
        if obs is None:
            skipped["obs"] += 1
            print(f"  [{i:>4}/{len(coverages)}] {cov.window.label}  no "
                  f"{source} for " + ", ".join(f"{t:%HZ}" for t in missing),
                  flush=True)
            continue
        olat, olon, ovals = obs
        hlat, hlon, hvals = met_regrid.read_regridded(hafs_path)
        domain, note = pick_domain(case, cov.window, full)
        key = tuple(round(v, 3) for v in domain)
        if key not in cuts:
            cuts[key] = crop_slices(hlat, hlon, domain, FULL_DOMAIN_MAX_CELLS)
        cut = cuts[key]
        map_figure(
            [(f"{LABELS.get(source, source)} 3 h",
              *_cropped(olat, olon, ovals, cut), "qpf"),
             (f"{hcfg.model.upper()} 3 h  {cov.run.init:%m-%d %HZ} "
              f"{aw.format_lead(cov.lead_end_min)}",
              *_cropped(hlat, hlon, hvals, cut), "qpf")],
            domain, track_segment(track, case.valid_start, cov.window.end),
            f"{case.storm_name}: 3 h precipitation {cov.window.label}   |   "
            f"{hcfg.model.upper()} init {cov.run.init:%Y-%m-%d %HZ}, lead "
            f"{cov.lead_h:g} h   |   {note}",
            out / f"{hcfg.model}_{cov.run.init:%Y%m%d%H}_"
                  f"f{lead_h:03d}_{cov.window.stamp}.png",
            overlays=overlays)
        print(f"  [{i:>4}/{len(coverages)}] {cov.window.label}  "
              f"{cov.run.init:%m-%d %HZ} {aw.format_lead(cov.lead_end_min)}  "
              f"{note}  obs {np.nanmax(ovals):.1f} / fcst "
              f"{np.nanmax(hvals):.1f} mm max"
              f"  (+{time.monotonic() - started:.0f}s)", flush=True)

    drawn = len(coverages) - skipped["hafs"] - skipped["obs"]
    print(f"\n{drawn} panel(s) under {out}")
    if skipped["hafs"]:
        print(f"{skipped['hafs']} skipped: no regridded HAFS window "
              f"(run regrid-hafs, or the cycle is not on disk)")
    if skipped["obs"]:
        print(f"{skipped['obs']} skipped: an hour of {source} is missing, so "
              f"the 3 h window is missing")
