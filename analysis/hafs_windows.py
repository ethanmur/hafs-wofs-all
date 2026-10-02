"""D3+D4: HAFS 3 h accumulation windows, regridded to the case grid.

The two plan steps are one pass. D1 established that HAFS writes a per-interval
3 h APCP bucket alongside the run total, so the window needs no accumulation
arithmetic: it already exists in the file, and `regrid_data_plane` can select
it in place with

    -field 'name="APCP"; level="A3";'

Extracting the bucket to its own GRIB2 first would be strictly more I/O --
read-big, write-small, read-small instead of read-big -- because each source
file contributes exactly one bucket to exactly one window. So nothing is
staged: MET reads the native file and writes the 6 km field directly.

`level="A3"` rather than the record name is what keeps the run total out. Both
records are APCP at the surface and differ only in accumulation interval, and
picking the wrong one is silent: a run-total field decodes cleanly and just
looks like a very wet forecast.

Which (cycle, lead) covers which window comes from accum_windows, including
the rule that a HAFS run can only serve windows built from whole 3 h buckets.

Usage:
    python analysis/run.py storms/<case>.yaml regrid-hafs
"""

import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import accum_windows as aw
import met_regrid

DEFAULT_FIELD = 'name="APCP"; level="A3";'
DEFAULT_MODEL = "hfsa"
# 00l.2024092412.hfsb_multistorm.parent.atm.f000.grb2 and friends.
PARENT_RE = re.compile(r"(?P<storm>[0-9]{2}[a-z])\.(?P<init>[0-9]{10})\."
                       r"(?P<model>[a-z_]+)\.parent\.atm\."
                       r"f(?P<lead>[0-9]{3})\.grb2$", re.IGNORECASE)


@dataclass
class HafsConfig:
    """Where the HAFS files are, and which of them belong to this case."""
    root: Path
    cache_dir: Path
    model: str = DEFAULT_MODEL
    storm_id: Optional[str] = None          # e.g. '10l'; None keeps every storm
    field_spec: str = DEFAULT_FIELD
    cycle_h: int = aw.HAFS_CYCLE_H
    max_lead_h: int = aw.HAFS_MAX_LEAD_H
    budget_check: bool = True
    tolerance_pct: float = 2.0

    def output_path(self, init, lead_h):
        return self.cache_dir / f"{self.model}_{init:%Y%m%d%H}_f{lead_h:03d}.nc"


def hafs_config_from_dict(cfg, where=""):
    if not cfg:
        return None
    known = {"root", "cache_dir", "model", "storm_id", "field_spec", "cycle_h",
             "max_lead_h", "budget_check", "tolerance_pct"}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError("unknown hafs keys: " + ", ".join(sorted(unknown)))
    for required in ("root", "cache_dir"):
        if not cfg.get(required):
            raise ValueError(f"hafs.{required} is required{where}")
    return HafsConfig(
        root=Path(cfg["root"]),
        cache_dir=Path(cfg["cache_dir"]),
        model=str(cfg.get("model", DEFAULT_MODEL)),
        storm_id=(str(cfg["storm_id"]).lower() if cfg.get("storm_id")
                  else None),
        field_spec=str(cfg.get("field_spec", DEFAULT_FIELD)),
        cycle_h=int(cfg.get("cycle_h", aw.HAFS_CYCLE_H)),
        max_lead_h=int(cfg.get("max_lead_h", aw.HAFS_MAX_LEAD_H)),
        budget_check=bool(cfg.get("budget_check", True)),
        tolerance_pct=float(cfg.get("tolerance_pct", 2.0)))


# =============================================================================
# Finding the files
# =============================================================================

def index_parent_files(cfg):
    """{(init_str, lead_h): path} for every parent file under cfg.root.

    Built in one recursive pass and reused, rather than globbing per lead: on
    a parallel filesystem the directory walk dominates, and a case can ask for
    several hundred (cycle, lead) pairs.
    """
    index = {}
    for path in sorted(Path(cfg.root).rglob("*parent.atm.f*.grb2")):
        m = PARENT_RE.search(path.name)
        if not m:
            continue
        if cfg.model and cfg.model.lower() not in m.group("model").lower():
            continue
        if cfg.storm_id and m.group("storm").lower() != cfg.storm_id:
            continue
        index[(m.group("init"), int(m.group("lead")))] = path
    return index


def source_file(index, init, lead_h):
    """The file holding the bucket that *ends* at this lead, or None.

    The bucket is named for its end: the 15-18Z accumulation from an 18Z-prior
    init lives in the f024 file as `21-24 hour acc fcst`, not in f021.
    """
    return index.get((f"{init:%Y%m%d%H}", int(lead_h)))


# =============================================================================
# Driver
# =============================================================================

def _native_bucket(path, window):
    """(lat, lon, values) of the 3 h bucket ending at the window end.

    Only read when the conservation check is on: it costs a full scan of the
    source file, which is exactly the I/O the no-staging design avoids.
    """
    msg = met_regrid.extract_accumulation([path], window.end,
                                          int(round(window.hours)))
    if msg is None:
        raise RuntimeError(
            f"{path} has no {window.hours:g} h accumulation ending "
            f"{window.end:%Y-%m-%d %HZ}. If the per-lead bucket-width check "
            f"(D1) shows this lead is not a clean 3 h bucket, that lead needs "
            f"pcp_combine.")
    return met_regrid.read_grib_message(msg)


def regrid_hafs_case(case):
    """Regrid every HAFS 3 h window that a cycle can fully span."""
    cfg = case.hafs
    if cfg is None:
        raise SystemExit("ERROR: regrid-hafs needs a `hafs:` block in the YAML")
    if case.regrid is None:
        raise SystemExit("ERROR: regrid-hafs needs the `regrid:` block too "
                         "(it supplies the grid and the method)")
    if case.init_start is None or case.init_end is None:
        raise SystemExit("ERROR: regrid-hafs needs init_start and init_end in "
                         "the YAML to know which cycles to pull")

    windows = aw.clock_windows(case.valid_start, case.valid_end)
    runs = aw.hafs_runs(case.init_start, case.init_end, cfg.cycle_h,
                        cfg.max_lead_h, model=cfg.model)
    coverages = aw.coverage(runs, windows)

    rcfg = case.regrid
    tool = met_regrid.met_tool("regrid_data_plane", rcfg.met_bin_dir)
    grid_file, grid_desc = met_regrid.resolve_to_grid(rcfg)
    print(f"Case     : {case.storm_name}  ({case.case_slug})")
    print(f"Window   : {case.valid_start:%Y-%m-%d %HZ} -> "
          f"{case.valid_end:%Y-%m-%d %HZ}  ({len(windows)} x 3 h windows)")
    print(f"Cycles   : {case.init_start:%Y-%m-%d %HZ} -> "
          f"{case.init_end:%Y-%m-%d %HZ} every {cfg.cycle_h} h, "
          f"to f{cfg.max_lead_h:03d}")
    print(f"Pairs    : {len(coverages)} (run, window) pairs to regrid")
    print(f"Source   : {cfg.root}  [{cfg.model}"
          + (f", storm {cfg.storm_id}" if cfg.storm_id else "") + "]")
    print(f"Field    : {cfg.field_spec}")
    print(f"Grid     : {rcfg.grid_name}  to_grid {grid_desc}")
    print(f"Method   : {rcfg.method} (width {rcfg.width}, "
          f"vld_thresh {rcfg.vld_thresh})")
    print(f"Cache    : {cfg.cache_dir}", flush=True)

    index = index_parent_files(cfg)
    print(f"Found    : {len(index)} parent files under the source root",
          flush=True)
    if not index:
        raise SystemExit(
            f"ERROR: no files matching *parent.atm.f*.grb2 for model "
            f"{cfg.model!r} under {cfg.root}. Check hafs.root, hafs.model and "
            f"hafs.storm_id.")

    target = native_grid = None
    rows, missing = [], []
    started = time.monotonic()
    for i, cov in enumerate(coverages, 1):
        init, lead_h = cov.run.init, int(cov.lead_end_min // 60)
        src = source_file(index, init, lead_h)
        row = {"model": cfg.model, "init": f"{init:%Y-%m-%d %H:%M}",
               "lead_h": lead_h,
               "window_start": f"{cov.window.start:%Y-%m-%d %H:%M}",
               "window_end": f"{cov.window.end:%Y-%m-%d %H:%M}",
               "source_file": str(src) if src else "",
               "output": "", "status": ""}
        if src is None:
            row["status"] = "missing_source"
            missing.append((init, lead_h))
            rows.append(row)
            continue

        out = cfg.output_path(init, lead_h)
        row["output"] = str(out)
        if out.exists() and met_regrid.valid_value_count(out) == 0:
            out.unlink()     # an all-missing field is never a valid cache entry
        status = "cached"
        if not out.exists():
            met_regrid.run_regrid(tool, src, grid_file, out, cfg.field_spec,
                                  rcfg)
            status = "regridded"
        row["status"] = status
        rlat, rlon, rvals = met_regrid.read_regridded(out)
        if target is None:
            target = met_regrid.TargetGrid(rlat, rlon)
            print(f"Target   : {target.shape[0]} x {target.shape[1]}",
                  flush=True)
        elif rvals.shape != target.shape:
            raise ValueError(
                f"{out} is {rvals.shape}, expected {target.shape} -- the "
                f"grid changed between runs; clear {cfg.cache_dir} and redo.")

        note = f"{status} {np.nanmax(rvals):.1f} mm max"
        if cfg.budget_check:
            lat, lon, vals = _native_bucket(src, cov.window)
            if native_grid is None:
                native_grid = met_regrid.NativeGrid.build(lat, lon, target)
            row.update(met_regrid.budget_row(target, rvals, vals, native_grid,
                                             cfg.tolerance_pct))
            note += (f"  {row['mean_pct_diff']:+.2f}%"
                     + (f" {row['flag']}" if row["flag"] else ""))
        rows.append(row)
        print(f"  [{i:>4}/{len(coverages)}] {cov.window.label}  "
              f"{init:%m-%d %HZ} {aw.format_lead(cov.lead_end_min)}  {note}"
              f"  (+{time.monotonic() - started:.0f}s)", flush=True)

    case.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = case.out_dir / f"hafs_windows_{case.case_slug}.csv"
    with open(out_csv, "w", newline="") as fh:
        names = list(rows[0].keys())
        names += [k for r in rows for k in r if k not in names]
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved window manifest: {out_csv}")

    done = [r for r in rows if r["status"] in ("cached", "regridded")]
    print(f"{len(done)} of {len(coverages)} pairs on the case grid")
    if missing:
        print(f"{len(missing)} pair(s) had no source file -- these windows "
              f"lose that cycle:")
        for init, lead in missing[:10]:
            print(f"  {init:%Y-%m-%d %HZ}  f{lead:03d}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    flagged = [r for r in done if r.get("flag")]
    if cfg.budget_check and flagged:
        print(f"{len(flagged)} pair(s) outside {cfg.tolerance_pct}% on the "
              f"conservation check; see the manifest.")
    elif cfg.budget_check:
        print(f"All {len(done)} within {cfg.tolerance_pct}% on mass.")
    return rows
