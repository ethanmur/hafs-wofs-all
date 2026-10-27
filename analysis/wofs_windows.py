"""WoFS 3 h windows on the deployment's verification grid (plan step D3).

For every complete (init, window) pair in the manifest:

  1. Sum the 36 five-minute files that make up the window. `rain` is the
     accumulation over the preceding 5 min, in inches, one file per step with
     all members in it, so the window (start, end] is the sum of steps
     s+1 .. s+36, where s is the step at the window start. Step 00 is the init
     itself and is never used.
  2. Convert inches to mm.
  3. Put it on the verification grid. At 6 km that grid is the WoFS grid
     coarsened by whole 2 x 2 blocks, so each 6 km cell is exactly four WoFS
     cells and the area-conserving regrid is the block mean. That is what
     MET's budget method returns onto a nested grid, without 18 members x
     ~160 windows of subprocess calls. At 3 km the grid is the native one and
     nothing is averaged.
  4. Write one NetCDF per (init, window) with an `ne` member dimension, in
     MET's NetCDF convention so grid_stat can read it with
     `level="(<member>,*,*)"`.

A pair that is not complete is never partly summed: a 35-step total is not a
3 h total and would bias every threshold. Existing outputs are reused, so a
rerun only does what is missing.

Usage:
    python analysis/run.py storms/<case>.yaml regrid-wofs [--res-km N]
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import verification_grid as vg
import window_manifest as wm
import wofs_inventory as wi

IN_TO_MM = 25.4
VAR = "rain_mm"
MET_MISSING = -9999.0


# =============================================================================
# Pure logic
# =============================================================================

def window_steps(init, window):
    """The step indices whose files make up `window` for a run from `init`."""
    start = int((window.start - init).total_seconds() // 60) // wi.STEP_MIN
    return list(range(start + 1, start + 1 + wi.STEPS_PER_WINDOW))


def block_mean(field, n):
    """Mean of each n x n block of the last two axes of `field`.

    NaN in any cell of a block makes the block NaN: a coarse cell built from
    partly missing rain is not a smaller version of the right answer.
    """
    if n == 1:
        return field
    *lead, ny, nx = field.shape
    if ny % n or nx % n:
        raise ValueError(f"a {ny} x {nx} field does not divide into "
                         f"{n} x {n} blocks")
    shaped = field.reshape(*lead, ny // n, n, nx // n, n)
    return shaped.mean(axis=(-3, -1))


def sum_steps(files, steps, loader=None):
    """Sum the `rain` field of the files for `steps`, in mm, float64.

    Returns (total, nan_cells): NaN propagates through the sum, so a cell that
    is missing in any step is missing in the window, and is counted.
    """
    loader = loader or load_rain
    missing = [s for s in steps if s not in files]
    if missing:
        raise FileNotFoundError(
            f"steps {missing[:5]}{'...' if len(missing) > 5 else ''} are not "
            "on disk, so this window is not complete")
    total = None
    for step in steps:
        field = np.asarray(loader(files[step]), dtype=np.float64)
        total = field if total is None else total + field
    total = total * IN_TO_MM
    return total, int(np.isnan(total).sum())


def load_rain(path):
    """The (ne, ny, nx) `rain` field of one WoFS file, in inches."""
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        return ds["rain"].values


# =============================================================================
# Output
# =============================================================================

def write_window_netcdf(path, spec, data_mm, lat, lon, init, window,
                        n_steps, source_files):
    """One (init, window): all members, mm, on the grid, MET-readable."""
    import netCDF4
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ne, ny, nx = data_mm.shape
    tmp = path.with_name(path.stem + ".tmp.nc")
    values = np.where(np.isfinite(data_mm), data_mm, MET_MISSING)
    with netCDF4.Dataset(tmp, "w") as nc:
        nc.createDimension("ne", ne)
        nc.createDimension("lat", ny)
        nc.createDimension("lon", nx)
        nc.createVariable("lat", "f4", ("lat", "lon"))[:] = lat
        nc.createVariable("lon", "f4", ("lat", "lon"))[:] = lon
        var = nc.createVariable(VAR, "f4", ("ne", "lat", "lon"),
                                fill_value=MET_MISSING)
        var[:] = values
        var.long_name = "WoFS 3 h accumulated rainfall, one field per member"
        var.units = "mm"
        for key, value in vg.met_grid_attrs(spec).items():
            setattr(nc, key, value)
        nc.init_time = f"{init:%Y%m%d_%H%M%S}"
        nc.window_start = f"{window.start:%Y%m%d_%H%M%S}"
        nc.window_end = f"{window.end:%Y%m%d_%H%M%S}"
        nc.valid_time = f"{window.end:%Y%m%d_%H%M%S}"
        nc.n_steps = n_steps
        nc.source_units = "in (5-min accumulation), summed then x 25.4"
        nc.source_first_file = Path(source_files[0]).name
        nc.source_last_file = Path(source_files[-1]).name
    tmp.replace(path)
    return path


def read_window(path):
    """(ne, ny, nx) mm from a file written here; NaN where missing."""
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        data = np.ma.filled(nc.variables[VAR][:], np.nan)
    return np.asarray(data, dtype=np.float64)


# =============================================================================
# Driver
# =============================================================================

def regrid_wofs_case(case):
    """Driver for `regrid-wofs`."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit("regrid-wofs: no complete WoFS window for this "
                         "deployment, so there is nothing to regrid")
    try:
        spec = vg.grid_spec_from_json(case.grid_json_path)
    except FileNotFoundError as err:
        raise SystemExit(f"ERROR: {err}") from None
    if abs(spec.res_km - case.res_km) > 1e-6:
        raise SystemExit(f"ERROR: {case.grid_json_path.name} is a "
                         f"{spec.res_km:g} km grid but this run is "
                         f"{case.res_km:g} km. Run build-grid first.")
    lat, lon = vg.grid_latlon(spec)

    manifest = wm.build_manifest(case)
    n = vg.aggregation_factor(float(manifest.dep.attrs["d_km"]), case.res_km)
    pairs = wm.wofs_complete_pairs(manifest.rows)
    init_dirs = {i.init_dt: i.path for i in manifest.dep.inits}
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km  "
          f"({n} x {n} WoFS cells each)")
    print(f"Pairs  : {len(pairs)} complete (init, window) pairs in "
          f"{len(manifest.windows)} windows  ->  {case.wofs_regrid_dir}",
          flush=True)
    if not pairs:
        print("Nothing to do.")
        return []

    rows, started = [], time.monotonic()
    for i, (init, window) in enumerate(sorted(pairs, key=lambda p: (p[1].end,
                                                                    p[0])), 1):
        out = case.wofs_window_path(window.end, init)
        steps = window_steps(init, window)
        if out.exists():
            data = read_window(out)
            rows.append(_row(init, window, "cached", steps, data, None))
            status = "cached"
        else:
            files = wi.step_files(init_dirs[init])
            native, nan_cells = sum_steps(files, steps)
            coarse = block_mean(native, n)
            write_window_netcdf(out, spec, coarse, lat, lon, init, window,
                                len(steps), [files[steps[0]], files[steps[-1]]])
            rows.append(_row(init, window, "regridded", steps, coarse,
                             native, nan_cells))
            status = "regridded"
        r = rows[-1]
        print(f"  [{i:>3}/{len(pairs)}] {window.end:%Y-%m-%d %HZ} "
              f"init {init:%d %H%M}  {status:<9} members {r['members']}  "
              f"ens-mean max {r['ens_mean_max_mm']:.1f} mm  "
              f"{r['mean_pct_diff']}  (+{time.monotonic() - started:.0f}s)",
              flush=True)

    csv_path = case.wofs_budget_csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["flag"]]
    print(f"\nSaved conservation check: {csv_path}")
    print(f"{len(rows) - len(bad)} of {len(rows)} windows clean"
          + (f"; {len(bad)} flagged (NaN cells or mass change)" if bad else "."))
    return rows


def _row(init, window, status, steps, coarse, native, nan_cells=0):
    ens_mean = np.nanmean(coarse, axis=0)
    row = {"window_end": f"{window.end:%Y-%m-%d %H:%M}",
           "init": f"{init:%Y-%m-%d %H:%M}", "status": status,
           "n_steps": len(steps), "members": coarse.shape[0],
           "nan_cells": nan_cells,
           "regrid_mean_mm": round(float(np.nanmean(coarse)), 5),
           "ens_mean_max_mm": round(float(np.nanmax(ens_mean)), 3)}
    if native is None:
        row.update({"native_mean_mm": "", "mean_pct_diff": "", "flag": ""})
        return row
    nm, rm = float(np.nanmean(native)), float(np.nanmean(coarse))
    pct = 0.0 if nm == 0 else 100.0 * (rm - nm) / nm
    row.update({"native_mean_mm": round(nm, 5),
                "mean_pct_diff": f"{pct:+.3f}%",
                "flag": "CHECK" if (nan_cells or abs(pct) > 0.1) else ""})
    return row
