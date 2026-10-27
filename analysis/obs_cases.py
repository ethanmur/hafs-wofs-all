"""Observation data for one deployment: MRMS download, cache check, regrid-obs.

Works on a DeploymentCase (deployment_case.py). The truth product is Stage IV
by default and `truth_source: mrms` switches it; `regrid.sources` regrids both,
for the MRMS sensitivity check. Each obs hour in the scored period is regridded
once with MET onto the deployment's own 6 km grid (written by build-grid), so
every later comparison is cell for cell. The maps and distributions built on
the regridded output live in obs_regrid_plots.py and obs_regrid_stats.py.

Usage:
    python analysis/run.py storms/<case>.yaml download-obs
    python analysis/run.py storms/<case>.yaml regrid-obs
"""

import csv
import gzip
import io
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cfgrib
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
import boto3

import met_regrid
import stage4_hourly


# =============================================================================
# MRMS downloader / cache
# =============================================================================

MRMS_BUCKET = "noaa-mrms-pds"
MRMS_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"


def mrms_s3_key(hour_end_dt):
    date_str = hour_end_dt.strftime("%Y%m%d")
    time_str = hour_end_dt.strftime("%Y%m%d-%H%M%S")
    fname = f"MRMS_{MRMS_PRODUCT}_{time_str}.grib2.gz"
    return f"CONUS/{MRMS_PRODUCT}/{date_str}/{fname}", fname


def load_mrms_hour(s3, hour_end_dt, cache_dir, download=True):
    """(lat, lon, data) for one MRMS hour. On a cache miss: downloads from
    S3 when `download` is True (the default); raises FileNotFoundError
    instead of touching `s3` when `download` is False, for callers that
    must not download (e.g. a no-internet compute node)."""
    cache_dir = Path(cache_dir)
    key, fname = mrms_s3_key(hour_end_dt)
    cache_path = cache_dir / fname.replace(".gz", "")
    if not cache_path.exists():
        if not download:
            raise FileNotFoundError(
                f"MRMS hour not cached: {cache_path} -- regrid-obs does "
                f"not download; run 'download-obs' for this case first.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        gz_buf = io.BytesIO()
        s3.download_fileobj(MRMS_BUCKET, key, gz_buf)
        gz_buf.seek(0)
        raw = gzip.decompress(gz_buf.read())
        cache_path.write_bytes(raw)
    datasets = cfgrib.open_datasets(str(cache_path))
    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            data = np.where(da.values < 0, 0.0, da.values)
            return da.latitude.values, da.longitude.values, data
    raise RuntimeError(f"No variable in {cache_path}")


# =============================================================================
# Timestamps and the regrid configuration
# =============================================================================

def hourly_timestamps(t0, t1):
    """End-of-hour stamps of every hour in (t0, t1]. The obs files are stamped
    at the END of their hour, so the window (15Z, 18Z] is the 16Z, 17Z and 18Z
    files."""
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    return [t0 + timedelta(hours=h) for h in range(1, n_hours + 1)]


def regrid_config(case):
    """The MET regrid for this deployment, aimed at the grid build-grid wrote."""
    path = case.grid_json_path
    try:
        spec, name = met_regrid.read_grid_json(path)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: no grid for this deployment ({path}). "
                         "Run build-grid first:\n  python analysis/run.py "
                         "<yaml> build-grid") from None
    # The files this run writes say the resolution in their names, so the grid
    # behind them must really be that resolution.
    res = read_grid_meta(path)["res_km"]
    if abs(res - case.res_km) > 1e-6:
        raise SystemExit(
            f"ERROR: {path.name} is a {res:g} km grid but this run is "
            f"{case.res_km:g} km. Run build-grid at {case.res_km:g} km first "
            "(same --res-km).")
    settings = case.regrid
    return met_regrid.RegridConfig(
        cache_dir=case.regrid_dir, res_label=case.res_label,
        grid_spec=spec, grid_name=name or case.case_slug,
        method=settings.method, width=settings.width,
        vld_thresh=settings.vld_thresh, tolerance_pct=settings.tolerance_pct,
        met_bin_dir=settings.met_bin_dir, fields=dict(settings.fields))


def _require_window(case, command):
    if not case.wofs_available:
        raise SystemExit(f"{command}: {case.case_slug} has no complete WoFS "
                         "window, so there is nothing to verify; skipping.")


# =============================================================================
# Download (login node) -- data only, no regridding, no plotting
# =============================================================================

def download_obs(case):
    """Login-node command: populate the raw MRMS cache.

    Stage IV hourly data is NOT fetched here -- it comes from NCEP's own
    ST4.<day> archive, obtained outside this repo's tooling -- so for a
    Stage IV deployment there is nothing to do beyond saying where the files
    must be. Only products in `regrid_sources` are fetched. Requests are
    sequential, never concurrent, so this stays gentle on a login node.
    regrid-obs (the compute-node command) never downloads on its own; run this
    first.
    """
    _require_window(case, "download-obs")
    sources = case.regrid_sources
    _print_case_header(case, "Download obs", sources)
    if "stage4" in sources:
        print("\nStage IV: not fetched by this command -- place ST4.<day> "
              f"files under {case.stage4_cache_dir} yourself before running "
              "regrid-obs.", flush=True)
    if "mrms" not in sources:
        print("\nMRMS is not a configured source; nothing to fetch.")
        return

    timestamps = hourly_timestamps(case.valid_start, case.valid_end)
    print(f"\nMRMS: fetching {len(timestamps)} hour(s) "
          f"{timestamps[0]:%Y-%m-%d %HZ} -> {timestamps[-1]:%Y-%m-%d %HZ}",
          flush=True)
    s3 = boto3.client("s3", region_name="us-east-1",
                      config=Config(signature_version=UNSIGNED))
    failed = 0
    for t in timestamps:
        try:
            load_mrms_hour(s3, t, case.mrms_cache_dir, download=True)
            status = "ok"
        except Exception as e:
            failed += 1
            status = f"FAILED ({e})"
        print(f"  {t:%Y-%m-%d %HZ}  MRMS={status}", flush=True)
    print(f"\nDownload complete{f' ({failed} failed)' if failed else ''}.")


# =============================================================================
# Cache check -- never touches the network, so safe on a compute node
# =============================================================================

def check_cache_complete(case, sources=None):
    """Human-readable missing-item strings; empty means every raw file
    regrid-obs needs is cached. The Stage IV check parses the cached ST4.<day>
    files' GRIB metadata (stage4_hourly.index_stage4_hourly) rather than
    checking paths, since one file holds many hours and nothing in the
    filename says which; that parsing is memoized and reused by the real read.
    """
    sources = set(sources if sources is not None else case.regrid_sources)
    missing = []
    stage4_idx = (stage4_hourly.index_stage4_hourly(str(case.stage4_cache_dir))
                  if "stage4" in sources else {})
    for t in hourly_timestamps(case.valid_start, case.valid_end):
        if "mrms" in sources:
            _, fname = mrms_s3_key(t)
            path = case.mrms_cache_dir / fname.replace(".gz", "")
            if not path.exists():
                missing.append(f"MRMS hour {t:%Y-%m-%d %HZ} "
                               f"(expected {path})")
        if "stage4" in sources and t not in stage4_idx:
            missing.append(f"Stage IV hourly record for {t:%Y-%m-%d %HZ} "
                           f"(expected inside an ST4.<day> file under "
                           f"{case.stage4_cache_dir})")
    return missing


def _exit_if_cache_incomplete(case, command, sources=None):
    missing = check_cache_complete(case, sources)
    if not missing:
        return
    print(f"\nERROR: required data is not cached, and {command} does not "
          "download it. Run this first (on a node with internet):\n"
          "  python analysis/run.py <yaml> download-obs\n")
    for item in missing[:25]:
        print(f"  missing: {item}")
    if len(missing) > 25:
        print(f"  ... and {len(missing) - 25} more")
    raise SystemExit(1)


def _print_case_header(case, title, sources):
    print(f"{title}: {case.storm_name} {case.deployment}")
    print(f"Window: {case.valid_start:%Y-%m-%d %HZ} -> "
          f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"Sources: {', '.join(sources)}  (truth_source={case.truth_source})",
          flush=True)


# =============================================================================
# Regrid (compute node) -- MET, cached to disk, ahead of any analysis
# =============================================================================

def _native_hour(case, source, t, want_latlon):
    """(lat2d, lon2d, values2d, write_met_input) for one cached obs hour."""
    if source == "mrms":
        _, fname = mrms_s3_key(t)
        path = case.mrms_cache_dir / fname.replace(".gz", "")
        lat, lon, vals = met_regrid.read_grib_field(path, want_latlon)
        return lat, lon, vals, lambda staging: path
    paths = stage4_hourly.index_stage4_hourly(
        str(case.stage4_cache_dir)).get(t, [])
    msg = met_regrid.extract_accumulation(paths, t, hours=1)
    if msg is None:
        raise FileNotFoundError(
            f"no 1h Stage IV GRIB message for {t:%Y-%m-%d %HZ} in "
            f"{[p.name for p in paths]}")
    lat, lon, vals = met_regrid.read_grib_message(msg, want_latlon)
    return lat, lon, vals, lambda staging: met_regrid.write_bytes(
        staging / f"stage4_{t:%Y%m%d%H}.grb2", msg)


def read_grid_meta(path):
    """{ny, nx, res_km} a grid JSON promises, to check against."""
    import json
    payload = json.loads(Path(path).read_text())
    return {"ny": int(payload["ny"]), "nx": int(payload["nx"]),
            "res_km": float(payload["res_km"])}


def grid_shape_from_json(path):
    """(ny, nx) the grid JSON promises, to check MET's output against."""
    meta = read_grid_meta(path)
    return meta["ny"], meta["nx"]


def regrid_obs(case):
    """Regrid every cached obs hour with MET and check it conserves rain."""
    _require_window(case, "regrid-obs")
    cfg = regrid_config(case)
    sources = case.regrid_sources
    _print_case_header(case, "Regrid obs", sources)
    # Only the products being regridded need to be cached.
    _exit_if_cache_incomplete(case, "regrid-obs", sources)

    tool = met_regrid.met_tool("regrid_data_plane", cfg.met_bin_dir)
    grid_arg, grid_desc = met_regrid.resolve_to_grid(cfg)
    expected = grid_shape_from_json(case.grid_json_path)
    print(f"MET tool: {tool}")
    print(f"Grid:     {cfg.grid_name}")
    print(f"  to_grid {grid_desc}")
    print(f"Method:   {cfg.method} (width {cfg.width}, "
          f"vld_thresh {cfg.vld_thresh})")
    print(f"Cache:    {cfg.cache_dir}", flush=True)

    staging = cfg.staging_dir
    timestamps = hourly_timestamps(case.valid_start, case.valid_end)
    target = None
    native_grids = {}
    rows = []
    started = time.monotonic()
    for i, t in enumerate(timestamps, 1):
        notes = []
        for source in sources:
            first = source not in native_grids
            lat, lon, vals, write_met_input = _native_hour(
                case, source, t, want_latlon=first)
            out = cfg.output_path(source, t)
            if out.exists() and met_regrid.valid_value_count(out) == 0:
                out.unlink()   # an all-missing field is never a valid cache entry
            status = "cached"
            if not out.exists():
                staged = write_met_input(staging)
                met_regrid.run_regrid(tool, staged, grid_arg, out,
                                      cfg.field_spec(source), cfg)
                # Kept on failure: the staged input is what you need to
                # inspect with plot_data_plane when MET rejects an hour.
                if staged.parent == staging:
                    staged.unlink(missing_ok=True)
                status = "regridded"
            rlat, rlon, rvals = met_regrid.read_regridded(out)

            if target is None:
                if rvals.shape != expected:
                    raise ValueError(
                        f"{out} is {rvals.shape}, but the grid JSON says "
                        f"{expected}; was the grid rebuilt without clearing "
                        f"{cfg.cache_dir}?")
                target = met_regrid.TargetGrid(rlat, rlon)
                print(f"Target grid: {target.shape[0]} x {target.shape[1]}",
                      flush=True)
            elif rvals.shape != target.shape:
                raise ValueError(f"{out} is {rvals.shape}, expected "
                                 f"{target.shape}")
            if first:
                native_grids[source] = met_regrid.NativeGrid.build(lat, lon,
                                                                   target)
            elif vals.shape != native_grids[source].shape:
                raise ValueError(f"{source} native grid changed shape at {t}")

            row = met_regrid.budget_row(target, rvals, vals,
                                        native_grids[source], cfg.tolerance_pct)
            rows.append({"valid": f"{t:%Y-%m-%d %H:%M}", "source": source,
                         "status": status, **row})
            notes.append(f"{source} {status} {row['mean_pct_diff']:+.2f}%"
                         + (f" {row['flag']}" if row["flag"] else ""))
        print(f"  [{i:>3}/{len(timestamps)}] {t:%Y-%m-%d %HZ}  "
              + "  ".join(notes)
              + f"  (+{time.monotonic() - started:.0f}s)", flush=True)

    out_csv = case.budget_csv_path
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved conservation check: {out_csv}")
    flagged = [r for r in rows if r["flag"]]
    if not flagged:
        print(f"All {len(rows)} source-hours within {cfg.tolerance_pct}%.")
        return
    print(f"{len(flagged)} of {len(rows)} source-hours flagged "
          f"(|diff| > {cfg.tolerance_pct}% and > {met_regrid.ABS_FLOOR_MM} mm):")
    for r in flagged[:10]:
        print(f"  {r['valid']}  {r['source']:<6}  {r['mean_pct_diff']:+.2f}%  "
              f"({r['native_mean_mm']} -> {r['regrid_mean_mm']} mm)  {r['flag']}")
    if len(flagged) > 10:
        print(f"  ... and {len(flagged) - 10} more in the CSV")
