"""Case config and obs data handling for MRMS, Stage IV, and AORC.

Holds the obs-case YAML model shared by every obs command, the login-node
download, the cache-completeness check, and regrid-obs itself (MET
regridding of each cached obs hour onto the HAFS parent grid). The
comparisons built on top of the regridded output live in
obs_regrid_plots.py (maps) and obs_regrid_stats.py (distributions, 1:1).

Usage:
    python analysis/run.py storms/<case>.yaml download-obs
    python analysis/run.py storms/<case>.yaml regrid-obs
"""

import csv
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml
from botocore import UNSIGNED
from botocore.config import Config
import boto3

from hafs_common import load_mrms_hour, mrms_s3_key
import aorc_common
import met_regrid
import verification_grid
import stage4_hourly


# =============================================================================
# Config
# =============================================================================

@dataclass
class ObsCase:
    storm_name: str
    best_track: Path
    valid_start: datetime
    valid_end: datetime
    domain: tuple                 # (lat_min, lat_max, lon_min, lon_max)
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path        # holds ST4.<YYYYMMDD> hourly source files
    aorc_cache_dir: Path
    # Stage IV is the truth product for the multi-case study: the MRMS
    # archive only reaches back to 2020-10, which excludes half the sample.
    truth_source: str = "stage4"
    skip_mrms: bool = False
    skip_stage4: bool = False
    skip_aorc: bool = False
    case_slug: str = "obs_case"
    regrid: Optional[met_regrid.RegridConfig] = None
    regrid_plot_dir: Optional[Path] = None   # plot-regrid output root
    zoom_domain: Optional[tuple] = None      # (lat_min, lat_max, lon_min, lon_max)
    # Model cycles to pull and verify. Independent of the scoring window:
    # init_start may precede valid_start to reach long lead times, and the
    # verification grid still follows the valid window, not init_start.
    init_start: Optional[datetime] = None
    init_end: Optional[datetime] = None
    # WoFS deployments for this case; each keeps its own time window, since
    # WoFS is re-sited between events and a storm can have several.
    wofs_domains: list = field(default_factory=list)
    grid: Optional[verification_grid.GridConfig] = None

    @property
    def wofs_domain(self):
        """Union extent of every WoFS deployment, or None if there are none."""
        if not self.wofs_domains:
            return None
        boxes = [d.domain for d in self.wofs_domains]
        return (min(b[0] for b in boxes), max(b[1] for b in boxes),
                min(b[2] for b in boxes), max(b[3] for b in boxes))

    @property
    def output_slug(self):
        return (f"{self.case_slug}_{self.valid_start:%Y%m%d%H}_"
                f"{self.valid_end:%Y%m%d%H}")


def from_yaml(yaml_path):
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in ("best_track", "valid_start", "valid_end", "domain"):
        if key not in cfg:
            raise KeyError(f"'{key}' is required in an obs-case YAML "
                           f"({yaml_path})")
    out_dir = (Path(cfg["out_dir"]) if cfg.get("out_dir")
              else Path("analysis/output") / yaml_path.stem)
    valid_start = datetime.strptime(str(cfg["valid_start"]), "%Y%m%d%H")
    valid_end = datetime.strptime(str(cfg["valid_end"]), "%Y%m%d%H")
    if valid_end <= valid_start:
        raise ValueError(f"valid_end must be after valid_start in {yaml_path}")
    init_start = verification_grid.parse_stamp(
        cfg.get("init_start"), "init_start", yaml_path)
    init_end = verification_grid.parse_stamp(
        cfg.get("init_end"), "init_end", yaml_path)
    if init_start is not None and init_end is None:
        init_end = valid_end          # cycles past the window score nothing
    if init_start is not None and init_end <= init_start:
        raise ValueError(f"init_end must be after init_start in {yaml_path}")
    if init_start is not None and init_start > valid_end:
        raise ValueError(f"init_start is after valid_end in {yaml_path}; no "
                         "cycle initialized then can reach the window")
    try:
        grid = verification_grid.grid_config_from_dict(
            cfg.get("verification_grid"))
        wofs_domains = verification_grid.wofs_domains_from_cfg(cfg, yaml_path)
    except ValueError as err:
        raise ValueError(f"{err} in {yaml_path}") from None
    truth_source = str(cfg.get("truth_source", "stage4")).lower()
    if truth_source not in met_regrid.SOURCES:
        raise ValueError(f"truth_source must be one of {met_regrid.SOURCES} "
                         f"in {yaml_path}, got {truth_source!r}")
    plots = cfg.get("regrid_plots") or {}
    zoom = plots.get("zoom_domain")
    if zoom is not None and len(zoom) != 4:
        raise ValueError("regrid_plots.zoom_domain must be "
                         f"[lat_min, lat_max, lon_min, lon_max] in {yaml_path}")
    return ObsCase(
        storm_name=cfg.get("storm_name", "Storm"),
        best_track=Path(cfg["best_track"]),
        valid_start=valid_start,
        valid_end=valid_end,
        domain=tuple(cfg["domain"]),
        out_dir=out_dir,
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        stage4_cache_dir=Path(cfg.get("stage4_cache_dir", "/tmp/stage4_cache")),
        aorc_cache_dir=Path(cfg.get("aorc_cache_dir", "/tmp/aorc_cache")),
        truth_source=truth_source,
        skip_mrms=bool(cfg.get("skip_mrms", False)),
        skip_stage4=bool(cfg.get("skip_stage4", False)),
        skip_aorc=bool(cfg.get("skip_aorc", False)),
        case_slug=yaml_path.stem,
        regrid=met_regrid.regrid_config_from_dict(cfg.get("regrid")),
        regrid_plot_dir=(Path(plots["out_dir"]) if plots.get("out_dir")
                         else out_dir / "regrid"),
        zoom_domain=tuple(float(v) for v in zoom) if zoom else None,
        init_start=init_start,
        init_end=init_end,
        wofs_domains=wofs_domains,
        grid=grid,
    )


# =============================================================================
# Timestamps
# =============================================================================

def hourly_timestamps(t0, t1):
    n_hours = int(round((t1 - t0).total_seconds() / 3600))
    return [t0 + timedelta(hours=h) for h in range(1, n_hours + 1)]


# =============================================================================
# Download (login node) -- data only, no regridding, no plotting
# =============================================================================

def download_obs(case):
    """Login-node command: populate the raw MRMS/AORC caches only.

    Stage IV hourly data is NOT fetched here -- it comes from NCEP's own
    ST4.<day> archive, obtained outside this repo's tooling, so this
    command has nothing to do for it beyond what skip_stage4 already
    documents in the YAML. No regridding, no masking, no plotting, no
    matplotlib work -- just sequential (never concurrent) requests to
    MRMS / AORC, so this stays gentle on a login node. regrid-obs (the
    compute-node command) never downloads on its own; run this first.
    """
    print(f"Download obs: {case.storm_name}")
    print(f"Window: {case.valid_start:%Y-%m-%d %HZ} -> "
         f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"skip_mrms={case.skip_mrms}  skip_stage4={case.skip_stage4}  "
         f"skip_aorc={case.skip_aorc}")

    if not case.skip_stage4:
        print("\nStage IV: not fetched by this command -- place ST4.<day> "
             f"files under {case.stage4_cache_dir} yourself before running "
             "regrid-obs.", flush=True)

    if case.skip_mrms and case.skip_aorc:
        print("\nskip_mrms and skip_aorc both set -- nothing more to fetch.")
        return

    timestamps = hourly_timestamps(case.valid_start, case.valid_end)
    print(f"\nMRMS/AORC: fetching {len(timestamps)} hour(s) "
         f"{timestamps[0]:%Y-%m-%d %HZ} -> {timestamps[-1]:%Y-%m-%d %HZ}",
         flush=True)

    s3 = (boto3.client("s3", region_name="us-east-1",
                       config=Config(signature_version=UNSIGNED))
         if not case.skip_mrms else None)
    for t in timestamps:
        if case.skip_mrms:
            mstatus = "skipped"
        else:
            try:
                load_mrms_hour(s3, t, case.mrms_cache_dir, download=True)
                mstatus = "ok"
            except Exception as e:
                mstatus = f"FAILED ({e})"
        if case.skip_aorc:
            astatus = "skipped"
        else:
            try:
                aorc_common.load_aorc_hour(t, case.domain,
                                           case.aorc_cache_dir, download=True)
                astatus = "ok"
            except Exception as e:
                astatus = f"FAILED ({e})"
        print(f"  {t:%Y-%m-%d %HZ}  MRMS={mstatus}  AORC={astatus}",
             flush=True)

    print("\nDownload complete.")


# =============================================================================
# Entry point (compute node) -- reads the cache only, never downloads
# =============================================================================

def check_cache_complete(case):
    """List of human-readable missing-item strings; empty means every file
    regrid-obs will need is already cached. Never touches the network, so
    it's safe on a no-internet compute node. The Stage IV check parses the
    cached ST4.<day> files' GRIB metadata (stage4_hourly.index_stage4_hourly)
    rather than a pure path check, since one file holds many hours and
    there's no filename-only way to know which hours it covers -- that
    parsing is memoized and gets reused by the real read afterwards, so
    it's not wasted work."""
    missing = []

    stage4_idx = ({} if case.skip_stage4
                 else stage4_hourly.index_stage4_hourly(str(case.stage4_cache_dir)))

    for t in hourly_timestamps(case.valid_start, case.valid_end):
        if not case.skip_mrms:
            _, fname = mrms_s3_key(t)
            path = case.mrms_cache_dir / fname.replace(".gz", "")
            if not path.exists():
                missing.append(f"MRMS hour {t:%Y-%m-%d %HZ} "
                               f"(expected {path})")
        if not case.skip_stage4 and t not in stage4_idx:
            missing.append(f"Stage IV hourly record for {t:%Y-%m-%d %HZ} "
                           f"(expected inside an ST4.<day> file under "
                           f"{case.stage4_cache_dir})")
        if not case.skip_aorc:
            path = aorc_common.aorc_cache_path(case.aorc_cache_dir, t)
            if not path.exists():
                missing.append(f"AORC hour {t:%Y-%m-%d %HZ} "
                               f"(expected {path})")
    return missing


def _exit_if_cache_incomplete(case, command):
    missing = check_cache_complete(case)
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


def _print_case_header(case, title):
    print(f"{title}: {case.storm_name}")
    print(f"Window: {case.valid_start:%Y-%m-%d %HZ} -> "
         f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"skip_mrms={case.skip_mrms}  skip_stage4={case.skip_stage4}  "
         f"skip_aorc={case.skip_aorc}", flush=True)


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
    if source == "stage4":
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
    lat1d, lon1d, vals = aorc_common.load_aorc_hour(
        t, case.domain, case.aorc_cache_dir, download=False)
    lat = lon = None
    if want_latlon:
        lon, lat = np.meshgrid(lon1d, lat1d)
    return lat, lon, vals, lambda staging: met_regrid.write_accum_grib2(
        lat1d, lon1d, vals, t, 1, staging / f"aorc_{t:%Y%m%d%H}.grb2")


def regrid_obs(case):
    """Regrid every cached obs hour with MET and check it conserves rain."""
    cfg = case.regrid
    if cfg is None:
        raise SystemExit("ERROR: regrid-obs needs a `regrid:` block in the YAML")
    _print_case_header(case, "Regrid obs")
    _exit_if_cache_incomplete(case, "regrid-obs")

    tool = met_regrid.met_tool("regrid_data_plane", cfg.met_bin_dir)
    grid_file, grid_desc = met_regrid.resolve_to_grid(cfg)
    print(f"MET tool: {tool}")
    print(f"Grid:     {cfg.grid_name}")
    print(f"  to_grid {grid_desc}")
    print(f"Method:   {cfg.method} (width {cfg.width}, "
         f"vld_thresh {cfg.vld_thresh})")
    print(f"Cache:    {cfg.cache_dir}", flush=True)

    # Only the truth product is regridded; the other obs products stay in the
    # obs-vs-obs comparison scripts, which read the native caches.
    skipped = {"mrms": case.skip_mrms, "stage4": case.skip_stage4,
               "aorc": case.skip_aorc}
    sources = [s for s in (cfg.sources or [case.truth_source])
               if not skipped.get(s, False)]
    if not sources:
        print(f"Truth source {case.truth_source!r} is skipped -- "
              "nothing to regrid.")
        return
    print(f"Sources:  {', '.join(sources)}")
    staging = cfg.cache_dir / "_staging"
    timestamps = hourly_timestamps(case.valid_start, case.valid_end)
    target = None
    native_grids = {}
    rows = []
    started = time.monotonic()
    for i, t in enumerate(timestamps, 1):
        notes = []
        for source in sources:
            first = source not in native_grids
            lat, lon, vals, write_met_input = _native_hour(case, source, t,
                                                           want_latlon=first)
            out = cfg.output_path(source, t)
            if out.exists() and met_regrid.valid_value_count(out) == 0:
                out.unlink()   # an all-missing field is never a valid cache entry
            status = "cached"
            if not out.exists():
                staged = write_met_input(staging)
                try:
                    met_regrid.run_regrid(tool, staged, grid_file, out,
                                          cfg.field_spec(source), cfg)
                finally:
                    if staged.parent == staging:
                        staged.unlink(missing_ok=True)
                status = "regridded"
            rlat, rlon, rvals = met_regrid.read_regridded(out)

            if target is None:
                target = met_regrid.TargetGrid(rlat, rlon)
                print(f"Target grid: {target.shape[0]} x {target.shape[1]}",
                     flush=True)
                outside = target.fraction_outside(*case.domain)
                if outside > 0:
                    print(f"WARNING: {100 * outside:.0f}% of the case domain "
                         "lies outside the grid template -- obs there are "
                         "not regridded.", flush=True)
            elif rvals.shape != target.shape:
                raise ValueError(
                    f"{out} is {rvals.shape}, expected {target.shape} -- was "
                    "the template changed without changing regrid.grid_name?")
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

    case.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = case.out_dir / f"regrid_budget_{case.output_slug}.csv"
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
