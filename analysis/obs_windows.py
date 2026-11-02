"""Observed 3 h windows on the verification grid (plan step D5).

The hourly truth is already regridded (`regrid-obs`), one MET NetCDF per hour
stamped at the END of the hour, so a window (15Z, 18Z] is the 16Z, 17Z and 18Z
files. This adds those three arrays, in numpy -- no `pcp_combine` -- and writes
the sum as one file per window and product:

    data/mrms-regrid-3h/20241009_d1_milton/2024101000_mrms_6km.nc    (21Z-00Z)
    data/mrms-regrid-3h/20241009_d1_milton/window_sums_<start>_<end>.csv

A window with any hour missing is missing, and no file is written for it: a
2 h total is not a 3 h total, and rescaling it would bias every threshold. A
cell that is missing in any of the three hours is missing in the window, the
same rule the WoFS sums follow. The CSV lists every window with the hours it
found and the ones it did not, so a gap can be lined up with the known obs
outages instead of turning up as a strange score later.

The window file is a copy of its last hourly file with `precip` replaced, so
it keeps MET's grid attributes, lat/lon and variable layout and `grid_stat`
reads it exactly as it reads the hourly obs: `name="precip"; level="(*,*)"`.

Usage:
    python analysis/run.py storms/<case>.yaml window-obs [--res-km N]
                                                         [--sources stage4 mrms]
"""

import csv
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import met_regrid

CSV_FIELDS = ["source", "window_start", "window_end", "status",
              "hours_found", "missing_hours", "mean_mm", "max_mm",
              "nan_cells"]


# =============================================================================
# Pure logic
# =============================================================================

def hour_ends(window):
    """The stamps of the hourly files a window is made of: the END of each
    hour, so (15Z, 18Z] is 16Z, 17Z, 18Z. Not 15Z, 16Z, 17Z."""
    return [window.start + timedelta(hours=k)
            for k in range(1, int(round(window.hours)) + 1)]


def sum_hours(fields):
    """The window total of equal-shaped hourly arrays. NaN in any hour is NaN
    in the total."""
    fields = [np.asarray(f, dtype=float) for f in fields]
    if len({f.shape for f in fields}) != 1:
        raise ValueError(f"hourly fields differ in shape: "
                         f"{sorted({f.shape for f in fields})}")
    total = fields[0].copy()
    for f in fields[1:]:
        total = total + f
    return total


def _stats(values):
    finite = np.isfinite(values)
    return {"mean_mm": round(float(values[finite].mean()), 5)
            if finite.any() else "",
            "max_mm": round(float(values[finite].max()), 3)
            if finite.any() else "",
            "nan_cells": int((~finite).sum())}


# =============================================================================
# Files
# =============================================================================

def hourly_path(case, source, t):
    """Where regrid-obs put one source's hour ending at `t`."""
    cfg = met_regrid.RegridConfig(
        cache_dir=case.regrid_dir_for(source), grid_spec="", grid_name="",
        res_label=case.res_label)
    return cfg.output_path(source, t)


def _usable(path):
    """An hourly file is usable if it exists and has any data in it."""
    return path.exists() and met_regrid.valid_value_count(path) > 0


def write_window(template, out, total, window, source, hours):
    """Copy the last hourly file to `out` and put the window total in it."""
    import netCDF4
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.tmp{out.suffix}")
    shutil.copy(template, tmp)
    try:
        with netCDF4.Dataset(tmp, "a") as nc:
            nc.variables[met_regrid.OUTPUT_NAME][:] = np.where(
                np.isfinite(total), total, met_regrid.MET_MISSING)
            nc.variables[met_regrid.OUTPUT_NAME].long_name = (
                f"{source} {len(hours)} h accumulation, sum of hourly regrids")
            nc.accumulation_hours = len(hours)
            nc.window_start = f"{window.start:%Y%m%d_%H%M%S}"
            nc.valid_time = f"{window.end:%Y%m%d_%H%M%S}"
            nc.hours_summed = ",".join(f"{h:%Y%m%d%H}" for h in hours)
            nc.source_product = source
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


# =============================================================================
# Driver
# =============================================================================

def window_obs_case(case):
    """Driver for `window-obs`: the 3 h sums of every cached source."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"window-obs: {case.case_slug} has no complete WoFS "
                         "window, so there are no windows to build; skipping.")
    windows = aw.clock_windows(case.valid_start, case.valid_end)
    print(f"Windows: {len(windows)} 3 h windows, "
          f"{case.valid_start:%Y-%m-%d %HZ} -> {case.valid_end:%Y-%m-%d %HZ}")
    results = {}
    for source in case.regrid_sources:
        print(f"\n{source}: {case.regrid_dir_for(source)}\n"
              f"  -> {case.regrid_3h_dir_for(source)}", flush=True)
        rows = []
        for w in windows:
            out = case.obs_window_path(source, w.end)
            hours = hour_ends(w)
            paths = [hourly_path(case, source, t) for t in hours]
            row = {"source": source, "window_start": f"{w.start:%Y-%m-%d %H:%M}",
                   "window_end": f"{w.end:%Y-%m-%d %H:%M}"}
            if _usable(out):
                status, total = "cached", met_regrid.read_regridded(out)[2]
                found, missing = len(hours), []
            else:
                missing = [t for t, p in zip(hours, paths) if not _usable(p)]
                found = len(hours) - len(missing)
                if missing:
                    status, total = "missing_hours", None
                else:
                    total = sum_hours([met_regrid.read_regridded(p)[2]
                                       for p in paths])
                    write_window(paths[-1], out, total, w, source, hours)
                    status = "written"
            row.update({"status": status, "hours_found": f"{found}/{len(hours)}",
                        "missing_hours": ";".join(f"{t:%Y%m%d%H}"
                                                  for t in missing)})
            row.update(_stats(total) if total is not None
                       else {"mean_mm": "", "max_mm": "", "nan_cells": ""})
            rows.append(row)
            print(f"  {w.end:%Y-%m-%d %HZ}  {status:<13} {row['hours_found']}"
                  + (f"  missing {row['missing_hours']}" if missing else "")
                  + (f"  mean {row['mean_mm']} max {row['max_mm']} mm"
                     if total is not None else ""), flush=True)
        path = case.obs_window_csv_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w_ = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w_.writeheader()
            w_.writerows(rows)
        have = sum(r["status"] != "missing_hours" for r in rows)
        print(f"{source}: {have} of {len(rows)} windows complete"
              + ("" if have == len(rows) else
                 f"; {len(rows) - have} missing an hour (see the CSV)")
              + f"  ->  {path}")
        results[source] = rows
    return results
