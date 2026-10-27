"""WoFS 1 h and 3 h accumulations on the deployment's verification grid
(plan step D3).

For every complete (init, window) pair in the D6 manifest:

  1. Sum the five-minute files natively, at WoFS's own 3 km. `rain` is the
     accumulation over the preceding 5 min, in inches, one file per step with
     all members in it, so a window (start, end] is the sum of steps
     s+1 .. s+36, where s is the step at the window start. Step 00 is the init
     itself and is never used. Each file is read once and feeds the three 1 h
     sums, and the 3 h sum is their total.
  2. Convert inches to mm.
  3. Regrid each accumulation onto the verification grid with MET's
     `regrid_data_plane`, exactly as the obs are regridded: the same method and
     width, the same grid spec. The 3 km sum is written to a MET-readable
     NetCDF (the file layout and grid attributes copied from the WoFS file,
     which MET already reads), and one call regrids all members at once, since
     `-field` may be repeated and `-name` takes one output name per field.
  4. Keep the result: 1 h files in `wofs_regrid_1h_dir` and 3 h files in
     `wofs_regrid_3h_dir`, one per (init, hour or window), all members in it as
     variables `rain_mm_m00` ... `rain_mm_m17`, so `grid_stat` reads a member
     with `name="rain_mm_m03"; level="(*,*)"`.

At 6 km each cell is exactly a 2 x 2 block of WoFS cells, so MET's budget
method should return the plain block mean. That is checked, not assumed: every
file is compared with a numpy block mean of the same sum and the difference is
written to the budget CSV, so a MET result that disagrees with the arithmetic
(a misread grid, a shifted cell edge) is flagged instead of trusted.

A pair that is not complete is never partly summed: a 35-step total is not a
3 h total and would bias every threshold. Existing outputs are reused, so a
rerun only does what is missing.

Usage:
    python analysis/run.py storms/<case>.yaml regrid-wofs [--res-km N]
"""

import csv
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import accum_windows as aw
import met_regrid
import verification_grid as vg
import window_manifest as wm
import wofs_inventory as wi

IN_TO_MM = 25.4
VAR = "rain_mm"
STEPS_PER_HOUR = 60 // wi.STEP_MIN
# MET's budget interpolation and numpy's block mean are the same arithmetic
# on a nested grid, so anything beyond float noise is a real disagreement.
MET_VS_BLOCK_TOL_MM = 0.01
# The WoFS file's own grid description, copied into the staged input so MET
# reads it the way it reads the original files.
GRID_ATTR_KEYS = ("Projection", "scale_lat_1", "scale_lat_2", "lat_pin",
                  "lon_pin", "x_pin", "y_pin", "lon_orient", "d_km", "r_km",
                  "hemisphere", "nx", "ny")


# =============================================================================
# Pure logic
# =============================================================================

def window_steps(init, window):
    """The step indices whose files make up `window` for a run from `init`."""
    start = int((window.start - init).total_seconds() // 60) // wi.STEP_MIN
    n = int(round(window.hours * STEPS_PER_HOUR))
    return list(range(start + 1, start + 1 + n))


def hour_windows(window):
    """The 1 h windows inside a window: (start, start+1], ..."""
    return [aw.Window(window.start + timedelta(hours=k),
                      window.start + timedelta(hours=k + 1))
            for k in range(int(round(window.hours)))]


def member_names(ne):
    return [f"{VAR}_m{m:02d}" for m in range(ne)]


def member_fields(ne):
    """One MET -field string per member of the staged input."""
    return [f'name="{VAR}"; level="({m},*,*)";' for m in range(ne)]


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


def load_rain(path):
    """The (ne, ny, nx) `rain` field of one WoFS file, in inches."""
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        return ds["rain"].values


def accumulate(files, init, window, loader=None):
    """The window's accumulations at WoFS resolution, in mm.

    Returns (hours, total, nan_cells): `hours` is one (ne, ny, nx) array per
    hour of the window, `total` their sum. Each file is read once. NaN
    propagates, so a cell missing in any step is missing in every accumulation
    that includes it, and is counted.
    """
    loader = loader or load_rain
    steps = window_steps(init, window)
    missing = [s for s in steps if s not in files]
    if missing:
        raise FileNotFoundError(
            f"steps {missing[:5]}{'...' if len(missing) > 5 else ''} are not "
            "on disk, so this window is not complete")
    hours = []
    for k in range(0, len(steps), STEPS_PER_HOUR):
        acc = None
        for step in steps[k:k + STEPS_PER_HOUR]:
            field = np.asarray(loader(files[step]), dtype=np.float64)
            acc = field if acc is None else acc + field
        hours.append(acc * IN_TO_MM)
    total = hours[0].copy()
    for h in hours[1:]:
        total = total + h
    nan_cells = int(sum(np.isnan(h).sum() for h in hours))
    return hours, total, nan_cells


# =============================================================================
# MET
# =============================================================================

def native_grid_attrs(attrs):
    """The grid-describing global attributes of a WoFS file."""
    return {k: attrs[k] for k in GRID_ATTR_KEYS if k in attrs}


def write_native_input(path, data_mm, grid_attrs):
    """The 3 km sum as a MET-readable NetCDF: `rain_mm(ne, lat, lon)`.

    Laid out like the WoFS files MET already reads: dimensions lat and lon, no
    coordinate variables, and the grid carried in the global attributes.
    """
    import netCDF4
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ne, ny, nx = data_mm.shape
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("ne", ne)
        nc.createDimension("lat", ny)
        nc.createDimension("lon", nx)
        var = nc.createVariable(VAR, "f4", ("ne", "lat", "lon"),
                                fill_value=np.float32(np.nan))
        var[:] = data_mm.astype("f4")
        var.long_name = "WoFS accumulated rainfall, one field per member"
        var.units = "mm"
        for key, value in grid_attrs.items():
            setattr(nc, key, value)
    return path


def read_window(path, ne=None):
    """(ne, ny, nx) mm from a file written by `regrid_one`; NaN where
    missing."""
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        names = sorted(n for n in nc.variables if n.startswith(f"{VAR}_m"))
        if ne is not None and len(names) != ne:
            raise ValueError(f"{path} has {len(names)} members, expected {ne}")
        fields = [np.ma.filled(nc.variables[n][:].astype(float), np.nan)
                  for n in names]
    out = np.stack(fields)
    out[out <= met_regrid.MET_MISSING + 1] = np.nan
    return out


def regrid_one(tool, cfg, data_mm, grid_attrs, out_path, init, end, hours):
    """Regrid one (ne, ny, nx) accumulation with MET to `out_path`."""
    ne = data_mm.shape[0]
    staged = write_native_input(
        cfg.staging_dir / f"{out_path.stem}_native.nc", data_mm, grid_attrs)
    met_regrid.run_regrid(tool, staged, cfg.grid_spec, out_path,
                          member_fields(ne), cfg, names=member_names(ne))
    # Kept on failure (run_regrid raises first): the staged input is what you
    # inspect when MET rejects a file.
    staged.unlink(missing_ok=True)
    import netCDF4
    with netCDF4.Dataset(out_path, "a") as nc:
        nc.init_time = f"{init:%Y%m%d_%H%M%S}"
        nc.valid_time = f"{end:%Y%m%d_%H%M%S}"
        nc.accumulation_hours = hours
        nc.members = ne
        nc.source_units = "in (5-min accumulation), summed then x 25.4"
    return out_path


# =============================================================================
# Driver
# =============================================================================

def _config(case, hours, spec_string, name):
    s = case.regrid
    return met_regrid.RegridConfig(
        cache_dir=case.wofs_regrid_dir(hours), grid_spec=spec_string,
        grid_name=name or case.case_slug, res_label=case.res_label,
        method=s.method, width=s.width, vld_thresh=s.vld_thresh,
        tolerance_pct=s.tolerance_pct, met_bin_dir=s.met_bin_dir)


def _row(product, init, end, status, data, native, n, nan_cells):
    ens_mean = np.nanmean(data, axis=0)
    row = {"product": product, "end": f"{end:%Y-%m-%d %H:%M}",
           "init": f"{init:%Y-%m-%d %H:%M}", "status": status,
           "members": data.shape[0], "nan_cells": nan_cells,
           "regrid_mean_mm": round(float(np.nanmean(data)), 5),
           "ens_mean_max_mm": round(float(np.nanmax(ens_mean)), 3),
           "native_mean_mm": "", "mean_pct_diff": "",
           "met_vs_block_max_mm": "", "flag": ""}
    if native is None:
        return row
    nm, rm = float(np.nanmean(native)), float(np.nanmean(data))
    pct = 0.0 if nm == 0 else 100.0 * (rm - nm) / nm
    diff = float(np.nanmax(np.abs(data - block_mean(native, n))))
    row.update({"native_mean_mm": round(nm, 5),
                "mean_pct_diff": f"{pct:+.3f}%",
                "met_vs_block_max_mm": round(diff, 5),
                "flag": ("CHECK" if (nan_cells or diff > MET_VS_BLOCK_TOL_MM)
                         else "")})
    return row


def regrid_wofs_case(case):
    """Driver for `regrid-wofs`."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit("regrid-wofs: no complete WoFS window for this "
                         "deployment, so there is nothing to regrid")
    try:
        spec_string, grid_name = met_regrid.read_grid_json(case.grid_json_path)
        spec = vg.grid_spec_from_json(case.grid_json_path)
    except FileNotFoundError as err:
        raise SystemExit(f"ERROR: {err}") from None
    if abs(spec.res_km - case.res_km) > 1e-6:
        raise SystemExit(f"ERROR: {case.grid_json_path.name} is a "
                         f"{spec.res_km:g} km grid but this run is "
                         f"{case.res_km:g} km. Run build-grid first.")

    manifest = wm.build_manifest(case)
    dep = manifest.dep
    n = vg.aggregation_factor(float(dep.attrs["d_km"]), case.res_km)
    grid_attrs = native_grid_attrs(dep.attrs)
    pairs = sorted(wm.wofs_complete_pairs(manifest.rows),
                   key=lambda p: (p[1].end, p[0]))
    init_dirs = {i.init_dt: i.path for i in dep.inits}
    cfgs = {h: _config(case, h, spec_string, grid_name) for h in (1, 3)}
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km  "
          f"({n} x {n} WoFS cells each); MET {cfgs[3].method}, width "
          f"{cfgs[3].width}")
    print(f"Pairs  : {len(pairs)} complete (init, window) pairs in "
          f"{len(manifest.windows)} windows")
    print(f"Output : {case.wofs_regrid_1h_dir}\n         "
          f"{case.wofs_regrid_3h_dir}", flush=True)
    if not pairs:
        print("Nothing to do.")
        return {1: [], 3: []}

    tool = met_regrid.met_tool("regrid_data_plane", cfgs[3].met_bin_dir)
    for h in (1, 3):
        met_regrid.resolve_to_grid(cfgs[h])        # records / checks settings
    rows = {1: [], 3: []}
    started = time.monotonic()
    for i, (init, window) in enumerate(pairs, 1):
        todo = [(3, window.end, window)] + [
            (1, hw.end, hw) for hw in hour_windows(window)]
        fresh = [t for t in todo
                 if not case.wofs_window_path(t[1], init, t[0]).exists()]
        native = hours_native = nan_cells = None
        if fresh:
            files = wi.step_files(init_dirs[init])
            hours_native, native, nan_cells = accumulate(files, init, window)
        statuses = []
        for hrs, end, win in todo:
            out = case.wofs_window_path(end, init, hrs)
            if (hrs, end, win) in fresh:
                src = native if hrs == 3 else hours_native[
                    hour_windows(window).index(win)]
                regrid_one(tool, cfgs[hrs], src, grid_attrs,
                           out, init, end, hrs)
                data = read_window(out, ne=src.shape[0])
                rows[hrs].append(_row(f"{hrs}h", init, end, "regridded", data,
                                      src, n, nan_cells if hrs == 3 else 0))
                statuses.append("regridded")
            else:
                data = read_window(out)
                rows[hrs].append(_row(f"{hrs}h", init, end, "cached", data,
                                      None, n, 0))
                statuses.append("cached")
        last = rows[3][-1]
        print(f"  [{i:>3}/{len(pairs)}] {window.end:%Y-%m-%d %HZ} init "
              f"{init:%d %H%M}  3h {statuses[0]:<9} 1h x{len(statuses) - 1} "
              f"{statuses[1]:<9} ens-mean max {last['ens_mean_max_mm']:.1f} mm"
              f"  {last['mean_pct_diff']}  (+{time.monotonic() - started:.0f}s)",
              flush=True)

    for hrs in (1, 3):
        path = case.wofs_budget_csv_path(hrs)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[hrs][0]))
            w.writeheader()
            w.writerows(rows[hrs])
        bad = [r for r in rows[hrs] if r["flag"]]
        print(f"\n{hrs} h: {len(rows[hrs]) - len(bad)} of {len(rows[hrs])} "
              f"clean" + (f"; {len(bad)} flagged (NaN cells, or MET differing "
                          f"from the block mean by > {MET_VS_BLOCK_TOL_MM} mm)"
                          if bad else ".") + f"  ->  {path}")
    for h in (1, 3):                 # kept only if a MET call failed (raised)
        shutil.rmtree(cfgs[h].cache_dir / "_staging", ignore_errors=True)
    return rows

