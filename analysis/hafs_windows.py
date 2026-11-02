"""HAFS 3 h windows on the deployment's verification grid (plan step D4).

For every HAFS (cycle, window) the D6 manifest lists as `complete` -- the six-
hourly cycles of HFSA and HFSB in the lookback before a window that has a
complete WoFS run -- this regrids the window's 3 h precipitation onto the same
grid as the WoFS and obs files:

  * The file that holds a window is the one named for the bucket's *end*: the
    15-18Z accumulation of an 18Z-prior cycle is `21-24 hour acc fcst` in the
    f024 file. HAFS writes the per-interval bucket and the run total in the
    same file; `level="A3"` picks the 3 h bucket, so no `pcp_combine` and no
    intermediate GRIB are needed. (f003, where bucket and run total coincide,
    is never a window: a cycle is strictly before the window start, so the
    earliest bucket is f006.)
  * `regrid_data_plane` reads the native parent file and writes the 3 h field
    on the grid, with the same method, width and grid spec as the obs and
    WoFS regrids. HAFS tp is kg m-2, which is mm; there is no unit change.

Budget check (`regrid.budget_check`, on by default): the bucket is also read
natively with eccodes, and MET's area-weighted mean over the cells it covers
is compared with an exact box average of the native field over the same
cells -- the check `regrid-obs` makes, so a misread record or a shifted grid
shows up as a flagged row instead of a strange score later. It costs a full
scan of each native file, so it is the one knob to turn off for a production
rerun (`regrid: {budget_check: false}`). With it on, a cycle whose domain does
not overlap the deployment grid, or whose file lacks the 3 h bucket, is a row
with `no_overlap` / `no_bucket` rather than a failed MET call.

Files, one folder per configuration, named like the WoFS files (valid end,
cycle, model, resolution):

    data/hafs-regrid/20241009_d1_milton_HFSA/2024100921_i202410091200_hfsa_6km.nc
    data/hafs-regrid/20241009_d1_milton_HFSA/regrid_budget_<start>_<end>.csv

Each file holds one variable, `precip` (mm, ny x nx), so `grid_stat` reads it
exactly as it reads the obs: `name="precip"; level="(*,*)"`. Existing files are
reused (their earlier budget rows are carried into the new CSV).

Usage:
    python analysis/run.py storms/<case>.yaml regrid-hafs [--res-km N]
"""

import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import accum_windows as aw
import hafs_inventory as hi
import met_regrid
import verification_grid as vg
import window_manifest as wm

HAFS_FIELD = 'name="APCP"; level="A3";'
BUCKET_H = hi.BUCKET_H
ROW_FIELDS = ["model", "init", "window_start", "window_end", "lead_h",
              "status", "source"]
BUDGET_FIELDS = ["common_area_km2", "native_valid_points",
                 "regrid_valid_cells", "native_mean_mm", "regrid_mean_mm",
                 "mean_diff_mm", "mean_pct_diff", "native_max_mm",
                 "regrid_max_mm", "flag"]


@dataclass
class Job:
    """One HAFS cycle's 3 h bucket for one window."""
    config: str
    init: datetime
    window: aw.Window
    lead_h: int
    source: Path


# =============================================================================
# Pure logic
# =============================================================================

def work_list(case, manifest):
    """The (cycle, window) pairs to regrid, in window then cycle order.

    Every complete HAFS row of a window that has a complete WoFS run
    (testable or wofs_only); a window with no WoFS run has nothing to be
    compared with, so its HAFS members are not worth regridding.
    """
    category = {r["window_start"]: r["category"] for r in manifest.summary}
    configs = {c.lower(): c for c in case.hafs_models}
    jobs = []
    for r in manifest.rows:
        if (r["model"] not in configs or r["status"] != "complete"
                or category.get(r["window_start"]) == "no_wofs"):
            continue
        window = aw.Window(datetime.strptime(r["window_start"], wm.STAMP),
                           datetime.strptime(r["window_end"], wm.STAMP))
        jobs.append(Job(config=configs[r["model"]],
                        init=datetime.strptime(r["init"], wm.STAMP),
                        window=window, lead_h=int(r["lead_h"]),
                        source=Path(r["source"])))
    return sorted(jobs, key=lambda j: (j.window.end, j.config, j.init))


def grid_key(lat, lon):
    """What identifies a native grid, so its areas and cell lookup are built
    once even though every cycle's file carries them. A HAFS parent domain can
    move between cycles, so the corners and shape, not the config, key it."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return (lat.shape, round(float(lat[0, 0]), 4), round(float(lon[0, 0]), 4),
            round(float(lat[-1, -1]), 4), round(float(lon[-1, -1]), 4))


def _row(job, status, budget=None):
    row = {"model": job.config,
           "init": f"{job.init:{wm.STAMP}}",
           "window_start": f"{job.window.start:{wm.STAMP}}",
           "window_end": f"{job.window.end:{wm.STAMP}}",
           "lead_h": job.lead_h, "status": status, "source": str(job.source)}
    row.update({k: "" for k in BUDGET_FIELDS})
    row.update(budget or {})
    return row


def _key(job):
    return (f"{job.init:{wm.STAMP}}", f"{job.window.end:{wm.STAMP}}")


def load_prior(path):
    """{(init, window_end): row} from a budget CSV of an earlier run, so a
    cached file keeps the check it was given when it was written."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["init"], r["window_end"]): r for r in csv.DictReader(fh)}


# =============================================================================
# Native read and MET
# =============================================================================

def read_native_bucket(path, valid_end, hours=BUCKET_H):
    """(lat2d, lon2d, mm) of the 3 h bucket ending `valid_end` in a HAFS
    parent file, or None if the file has no such record. Selected by interval
    (the run total is the same variable), taking the finest grid."""
    msg = met_regrid.extract_accumulation([path], valid_end, hours)
    if msg is None:
        return None
    return met_regrid.read_grib_message(msg)


def _config(case, hafs_config, spec_string, name):
    s = case.regrid
    return met_regrid.RegridConfig(
        cache_dir=case.hafs_regrid_dir_for(hafs_config),
        grid_spec=spec_string, grid_name=name or case.case_slug,
        res_label=case.res_label, method=s.method, width=s.width,
        vld_thresh=s.vld_thresh, tolerance_pct=s.tolerance_pct,
        met_bin_dir=s.met_bin_dir)


def _tag(out, job):
    import netCDF4
    with netCDF4.Dataset(out, "a") as nc:
        nc.model = job.config
        nc.init_time = f"{job.init:%Y%m%d_%H%M%S}"
        nc.valid_time = f"{job.window.end:%Y%m%d_%H%M%S}"
        nc.accumulation_hours = BUCKET_H
        nc.source_file = str(job.source)


# =============================================================================
# Driver
# =============================================================================

def regrid_hafs_case(case, reader=None):
    """Driver for `regrid-hafs`. `reader` replaces the eccodes read of a
    native bucket (tests); it takes (path, valid_end, hours)."""
    reader = reader or read_native_bucket
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit("regrid-hafs: no complete WoFS window for this "
                         "deployment, so there are no windows to regrid")
    if not case.hafs_models:
        raise SystemExit("regrid-hafs: the case lists no hafs.models")
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
    jobs = work_list(case, manifest)
    cfgs = {c: _config(case, c, spec_string, grid_name)
            for c in case.hafs_models}
    check = case.regrid.budget_check
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km; "
          f"MET {case.regrid.method}, width {case.regrid.width}")
    print(f"HAFS   : {', '.join(case.hafs_models)}; {len(jobs)} (cycle, "
          f"window) pairs in {len(manifest.windows)} windows; budget check "
          f"{'on' if check else 'off'}")
    for c, cfg in cfgs.items():
        print(f"Output : {c}  {cfg.cache_dir}")
    print(flush=True)
    if not jobs:
        print("Nothing to do.")
        return {c: [] for c in cfgs}

    tool = met_regrid.met_tool("regrid_data_plane",
                               case.regrid.met_bin_dir)
    for cfg in cfgs.values():
        met_regrid.resolve_to_grid(cfg)           # records / checks settings
    prior = {c: load_prior(case.hafs_budget_csv_path(c)) for c in cfgs}
    # The overlap test uses the grid JSON's own cell centres; the budget uses
    # the cells MET actually wrote.
    spec_target = met_regrid.TargetGrid(*vg.grid_latlon(spec))
    target = None
    overlap_grids, budget_grids = {}, {}
    rows = {c: [] for c in cfgs}
    started = time.monotonic()

    for i, job in enumerate(jobs, 1):
        out = case.hafs_window_path(job.config, job.window.end, job.init)
        if out.exists() and met_regrid.valid_value_count(out) == 0:
            out.unlink()               # an all-missing file is never a cache hit
        if out.exists():
            old = prior[job.config].get(_key(job), {})
            row = _row(job, "cached",
                       {k: old.get(k, "") for k in BUDGET_FIELDS})
        else:
            native = reader(job.source, job.window.end, BUCKET_H) if check \
                else None
            if check and native is None:
                row = _row(job, "no_bucket")
            elif check and not _overlaps(native, spec_target, overlap_grids):
                row = _row(job, "no_overlap")
            else:
                met_regrid.run_regrid(tool, job.source, cfgs[job.config].grid_spec,
                                      out, HAFS_FIELD, cfgs[job.config])
                _tag(out, job)
                budget = {}
                if check:
                    rlat, rlon, rvals = met_regrid.read_regridded(out)
                    if target is None:
                        if rvals.shape != (spec.ny, spec.nx):
                            raise ValueError(
                                f"{out} is {rvals.shape}, but the grid JSON "
                                f"says {(spec.ny, spec.nx)}")
                        target = met_regrid.TargetGrid(rlat, rlon)
                    lat, lon, vals = native
                    key = grid_key(lat, lon)
                    if key not in budget_grids:
                        budget_grids[key] = met_regrid.NativeGrid.build(
                            lat, lon, target)
                    budget = met_regrid.budget_row(
                        target, rvals, vals, budget_grids[key],
                        case.regrid.tolerance_pct)
                row = _row(job, "regridded", budget)
        rows[job.config].append(row)
        pct = row["mean_pct_diff"]
        note = (f"  {float(pct):+.2f}%" if pct not in ("", None) else "")
        print(f"  [{i:>3}/{len(jobs)}] {job.config} {job.window.end:%Y-%m-%d %HZ}"
              f"  cycle {job.init:%d %HZ} f{job.lead_h:03d}  "
              f"{row['status']:<10}{note}"
              + (f"  {row['flag']}" if row["flag"] else "")
              + f"  (+{time.monotonic() - started:.0f}s)", flush=True)

    flagged = []
    for c, mine in rows.items():
        if not mine:
            continue
        path = case.hafs_budget_csv_path(c)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_FIELDS + BUDGET_FIELDS)
            w.writeheader()
            w.writerows(mine)
        counts = Counter(r["status"] for r in mine)
        print(f"\n{c}: " + ", ".join(f"{n} {s}" for s, n in sorted(
            counts.items())) + f"  ->  {path}")
        flagged += [r for r in mine if r["flag"]]
    if check:
        if flagged:
            print(f"\n{len(flagged)} flagged (|diff| > "
                  f"{case.regrid.tolerance_pct}% and > "
                  f"{met_regrid.ABS_FLOOR_MM} mm):")
            for r in flagged[:10]:
                print(f"  {r['model']} {r['window_end']} cycle {r['init']}  "
                      f"{r['mean_pct_diff']}%  {r['flag']}")
            if len(flagged) > 10:
                print(f"  ... and {len(flagged) - 10} more in the CSVs")
        else:
            done = sum(1 for m in rows.values() for r in m
                       if r["status"] in ("regridded", "cached")
                       and r["mean_pct_diff"] != "")
            print(f"\nAll {done} checked files within "
                  f"{case.regrid.tolerance_pct}%.")
    return rows


def _overlaps(native, spec_target, cache):
    """Whether any native point of the bucket lies on the deployment grid."""
    lat, lon, _ = native
    key = grid_key(lat, lon)
    if key not in cache:
        cache[key] = met_regrid.NativeGrid.build(lat, lon, spec_target)
    return bool((cache[key].cell >= 0).any())
