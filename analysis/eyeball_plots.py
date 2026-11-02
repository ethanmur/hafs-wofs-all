"""Eyeball plots of the regridded 3 h fields (plan step D5b), before any score
is computed.

Two commands, both reading only files the regrid steps already wrote and both
limited to a few windows, HAFS cycles and WoFS members (`eyeball:` in the case
YAML; the defaults are 3 windows, 2 cycles per configuration, WoFS members 0
and 9), because every pair would be thousands of figures:

  plot-eyeball      WoFS | HAFS | truth, one figure per (window, HAFS cycle,
                    WoFS member): three panels on the same grid and colour
                    scale, so a regridding or pairing mistake shows up as a
                    displaced or empty panel instead of a strange score later.
                    The WoFS panel is the run with the shortest lead for the
                    window; the HAFS cycles are spread across the lookback
                    (the shortest-lead and the longest-lead, for 2). The
                    deployment outline and the validity region (if build-mask
                    has run) are drawn on every panel, and the fields are not
                    clipped to it.
  plot-hafs-check   HAFS native | HAFS regridded, same windows and cycles: the
                    native 3 h bucket read straight from the parent file next
                    to the field on the deployment grid, framed on the
                    deployment domain, so a shifted or flipped grid shows up
                    by eye. The headline carries the budget-check change from
                    the regrid-hafs CSV.

Figures use plot_common's precipitation scale: a log scale in inches by
default (`eyeball.colorbar_style: true_scale`) or the banded ladder
(`standard`).

Files:
    figures/eyeball-panels/<dep>_<storm>/eyeball_<end>_<model>_c<cycle>_m<NN>.png
    figures/hafs-regrid-checks/<dep>_<storm>/hafs_check_<end>_<model>_c<cycle>.png

Usage:
    python analysis/run.py storms/<case>.yaml plot-eyeball [--res-km N]
    python analysis/run.py storms/<case>.yaml plot-hafs-check [--res-km N]
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import hafs_windows as hw
import masks
import met_regrid
import window_manifest as wm
import wofs_windows as ww
from best_track import parse_bdeck
from obs_regrid_plots import (LABELS, centred_domain, crop_slices,
                              grid_overlays, map_figure, track_segment)
from plot_units import inches

# contourf on the HAFS parent grid is slow past this many cells a side, and
# no panel has more pixels than this anyway.
NATIVE_MAX_CELLS = 700
RUN_RE = re.compile(r"^\d{10}_i(?P<init>\d{12}|\d{10})_")


# =============================================================================
# Choosing what to draw
# =============================================================================

def pick_evenly(items, n):
    """`n` items spread across the list, always including the first and the
    last: 1 -> the last, 2 -> first and last, 3 -> first, middle, last."""
    items = list(items)
    if n >= len(items):
        return items
    if n <= 1:
        return items[-1:]
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def runs_for(directory, end, model, res_label):
    """[(init, path)] of the window ending `end` in `directory`, by init."""
    out = []
    for path in Path(directory).glob(
            f"{end:%Y%m%d%H}_i*_{model}_{res_label}.nc"):
        m = RUN_RE.match(path.name)
        if m:
            text = m.group("init")
            out.append((datetime.strptime(
                text, "%Y%m%d%H%M" if len(text) == 12 else "%Y%m%d%H"), path))
    return sorted(out)


def lead_text(init, window):
    start = (window.start - init).total_seconds() / 3600
    end = (window.end - init).total_seconds() / 3600
    return f"{start:g}-{end:g} h"


def selection(case, need_wofs, need_truth):
    """The windows to draw and, for each, the runs to draw it with:
    [{"window", "wofs": (init, path) or None, "hafs": [(config, init, path)]}].

    A window qualifies when HAFS has a regridded file for it and, as asked, so
    do WoFS and the truth product. `eyeball.windows` of them are spread across
    the period; per configuration `eyeball.cycles` cycles are spread across the
    members."""
    settings = case.eyeball
    res = case.res_label
    found = []
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        hafs = {c: runs_for(case.hafs_regrid_dir_for(c), w.end, c.lower(), res)
                for c in case.hafs_models}
        if not any(hafs.values()):
            continue
        wofs = runs_for(case.wofs_regrid_3h_dir, w.end, "wofs", res)
        if need_wofs and not wofs:
            continue
        if need_truth and not case.obs_window_path(case.truth_source,
                                                   w.end).exists():
            continue
        found.append((w, wofs, hafs))
    out = []
    for w, wofs, hafs in pick_evenly(found, settings.windows):
        picks = [(c, init, path) for c, runs in hafs.items()
                 for init, path in pick_evenly(runs, settings.cycles)]
        out.append({"window": w, "wofs": wofs[-1] if wofs else None,
                    "hafs": picks})
    return out


def _check_same_grid(shape, other, what):
    if other.shape != shape:
        raise ValueError(f"{what} is {other.shape} but the truth grid is "
                         f"{shape}: the fields are not on the same grid")


def _overlays(case, lat, lon):
    """The deployment outline, plus the validity region if build-mask has run."""
    overlays = grid_overlays(lat, lon)
    if case.mask_path.exists():
        mask = masks.read_mask(case.mask_path)
        if mask.shape == lat.shape:
            overlays["contours"] = [(lat, lon, mask.astype(float), "#222222")]
    return overlays


# =============================================================================
# plot-eyeball
# =============================================================================

def plot_eyeball(case):
    """Driver for `plot-eyeball`: WoFS | HAFS | truth for a few windows."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-eyeball: {case.case_slug} has no complete "
                         "WoFS window; nothing to plot.")
    chosen = selection(case, need_wofs=True, need_truth=True)
    s = case.eyeball
    print(f"Truth  : {LABELS[case.truth_source]}; WoFS members "
          f"{', '.join(map(str, s.members))}; up to {s.windows} windows x "
          f"{s.cycles} cycles per HAFS configuration; colour scale "
          f"{s.colorbar_style}")
    if not chosen:
        print("\nNo window has WoFS, HAFS and truth 3 h files together. Run "
              "regrid-wofs, regrid-hafs and window-obs first.")
        return []
    first_truth = case.obs_window_path(case.truth_source,
                                       chosen[0]["window"].end)
    lat, lon, ref = met_regrid.read_regridded(first_truth)
    domain = centred_domain(lat, lon)
    overlays = _overlays(case, lat, lon)
    track = parse_bdeck(case.best_track)
    out_dir = case.eyeball_plot_dir
    print(f"Output : {out_dir}\n", flush=True)

    written, started = [], time.monotonic()
    for item in chosen:
        w = item["window"]
        truth = met_regrid.read_regridded(
            case.obs_window_path(case.truth_source, w.end))[2]
        _check_same_grid(ref.shape, truth, "the truth window")
        w_init, w_path = item["wofs"]
        members = ww.read_window(w_path)
        _check_same_grid(ref.shape, members[0], "the WoFS window")
        track_line = track_segment(track, w.start, w.end)
        for config, h_init, h_path in item["hafs"]:
            hafs = met_regrid.read_regridded(h_path)[2]
            _check_same_grid(ref.shape, hafs, f"the {config} window")
            for m in s.members:
                if m >= members.shape[0]:
                    print(f"  member {m} is not in {w_path.name} "
                          f"({members.shape[0]} members); skipped")
                    continue
                out = out_dir / (f"eyeball_{w.end:%Y%m%d%H}_{config.lower()}_"
                                 f"c{h_init:%Y%m%d%H}_m{m:02d}.png")
                map_figure(
                    [(f"WoFS member {m:02d}, init {w_init:%d %H%MZ} "
                      f"({lead_text(w_init, w)})", lat, lon, members[m],
                      "precip"),
                     (f"{config} cycle {h_init:%d %HZ} "
                      f"({lead_text(h_init, w)})", lat, lon, hafs, "precip"),
                     (f"{LABELS[case.truth_source]} (truth)", lat, lon, truth,
                      "precip")],
                    domain, track_line,
                    f"{case.storm_name} {case.deployment}: 3 h precipitation "
                    f"ending {w.end:%Y-%m-%d %HZ}, {case.res_km:g} km grid",
                    out, overlays=overlays,
                    colorbar_style=s.colorbar_style)
                written.append(out)
                print(f"  {w.end:%Y-%m-%d %HZ}  {config} c{h_init:%d%H} "
                      f"m{m:02d}  -> {out.name}  "
                      f"(+{time.monotonic() - started:.0f}s)", flush=True)
    print(f"\nSaved {len(written)} figures under {out_dir}")
    return written


# =============================================================================
# plot-hafs-check
# =============================================================================

def _budget_text(row):
    if not row or row.get("mean_pct_diff") in (None, ""):
        return ""
    text = f"   |   area-mean change {float(row['mean_pct_diff']):+.2f}%"
    if row.get("native_max_mm") not in (None, ""):
        text += (f", max {inches(float(row['native_max_mm'])):.2f} -> "
                 f"{inches(float(row['regrid_max_mm'])):.2f} in")
    return text


def plot_hafs_check(case, reader=None):
    """Driver for `plot-hafs-check`: native HAFS bucket vs the regridded file.
    `reader` replaces the eccodes read of the native bucket (tests)."""
    reader = reader or hw.read_native_bucket
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"plot-hafs-check: {case.case_slug} has no complete "
                         "WoFS window; nothing to plot.")
    chosen = selection(case, need_wofs=False, need_truth=False)
    s = case.eyeball
    print(f"HAFS   : up to {s.windows} windows x {s.cycles} cycles per "
          f"configuration; colour scale {s.colorbar_style}")
    if not chosen:
        print("\nNo window has a regridded HAFS file. Run regrid-hafs first.")
        return []
    sources = {(j.config, j.init, j.window.end): j.source
               for j in hw.work_list(case, wm.build_manifest(case))}
    first = chosen[0]["hafs"][0][2]
    lat, lon, ref = met_regrid.read_regridded(first)
    domain = centred_domain(lat, lon)
    overlays = _overlays(case, lat, lon)
    track = parse_bdeck(case.best_track)
    prior = {c: hw.load_prior(case.hafs_budget_csv_path(c))
             for c in case.hafs_models}
    out_dir = case.hafs_check_plot_dir
    print(f"Output : {out_dir}\n", flush=True)

    written, cuts, started = [], {}, time.monotonic()
    for item in chosen:
        w = item["window"]
        track_line = track_segment(track, w.start, w.end)
        for config, init, path in item["hafs"]:
            _, _, rvals = met_regrid.read_regridded(path)
            _check_same_grid(ref.shape, rvals, f"the {config} window")
            src = sources.get((config, init, w.end))
            native = reader(src, w.end, hw.BUCKET_H) if src else None
            if native is None:
                print(f"  {w.end:%Y-%m-%d %HZ}  {config} c{init:%d%H}: no "
                      "native 3 h bucket found; skipped")
                continue
            nlat, nlon, nvals = native
            key = (nlat.shape, float(nlat[0, 0]), float(nlon[0, 0]))
            if key not in cuts:
                cuts[key] = crop_slices(nlat, nlon, domain, NATIVE_MAX_CELLS)
            cut = cuts[key]
            if cut is None:     # the parent domain misses the deployment
                native_panel = (f"{config} native (outside the deployment)",
                                lat, lon, None, "precip")
            else:
                native_panel = (f"{config} native 3 h bucket", nlat[cut],
                                nlon[cut], nvals[cut], "precip")
            row = prior[config].get((f"{init:{wm.STAMP}}",
                                     f"{w.end:{wm.STAMP}}"))
            out = out_dir / (f"hafs_check_{w.end:%Y%m%d%H}_{config.lower()}_"
                             f"c{init:%Y%m%d%H}.png")
            map_figure(
                [native_panel,
                 (f"{config} regridded, {case.res_km:g} km grid", lat, lon,
                  rvals, "precip")],
                domain, track_line,
                f"{config} cycle {init:%d %HZ} ({lead_text(init, w)}), 3 h "
                f"precipitation ending {w.end:%Y-%m-%d %HZ}: native vs "
                f"regridded{_budget_text(row)}",
                out, overlays=overlays, colorbar_style=s.colorbar_style)
            written.append(out)
            print(f"  {w.end:%Y-%m-%d %HZ}  {config} c{init:%d%H}  -> "
                  f"{out.name}  (+{time.monotonic() - started:.0f}s)",
                  flush=True)
    print(f"\nSaved {len(written)} figures under {out_dir}")
    return written
