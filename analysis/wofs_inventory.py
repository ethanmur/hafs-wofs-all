"""WoFS deployment inventory (plan step A1).

Walks a WoFS archive laid out as

    <root>/<YYYY>_<STORM>/<YYYYMMDD>_d<N>/<HHMM init>/wofs_ALL_<step>_<YYYYMMDD>_<HHMM init>_<HHMM valid>.nc

and prints one row per deployment folder: init range, valid range, number of
init cycles and steps, and the grid each folder's files carry. Folder names
only organise the runs; init and valid times and the grid come from the files'
own attributes, and the file name is used as a cross-check.

Each file holds all ensemble members (``ne``) at one 5-minute step. ``rain`` is
the accumulation over the preceding 5 minutes, in inches.

Usage:
    python analysis/wofs_inventory.py <wofs_root> [--csv out.csv] [--deep]

--deep opens one init per deployment and also reports the ``rain`` units,
whether step 00 holds rainfall, and whether the lat/lon arrays agree with the
grid attributes.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

NAME_RE = re.compile(
    r"^wofs_ALL_(\d{2,3})_(\d{8})_(\d{4})_(\d{4})\.nc$")
STEP_MIN = 5
CYCLE_MIN = 30
TIME_FMT = "%Y%m%d_%H%M%S"

# Global attributes that define a grid. Two files with equal values here are
# on the same grid.
GRID_KEYS = ("start_lat", "start_lon", "lon_orient", "scale_lat_1",
             "scale_lat_2", "d_km", "r_km", "nx", "ny")


# =============================================================================
# File and cycle bookkeeping (no file I/O)
# =============================================================================

def parse_wofs_name(name):
    """(step, init_dt, valid_hhmm) from a WoFS file name, or None. The valid
    time in the name is HHMM only, so it is a cross-check, not a source."""
    m = NAME_RE.match(name)
    if not m:
        return None
    step = int(m.group(1))
    init_dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M")
    return step, init_dt, m.group(4)


def expected_valid_hhmm(init_dt, step):
    """The valid HHMM implied by the init time and step index."""
    return (init_dt + timedelta(minutes=STEP_MIN * step)).strftime("%H%M")


def missing_steps(steps):
    """Step indices absent between the lowest and highest present step."""
    steps = sorted(set(steps))
    if not steps:
        return []
    return sorted(set(range(steps[0], steps[-1] + 1)) - set(steps))


def cycle_gaps(inits, cycle_min=CYCLE_MIN):
    """[(from_dt, to_dt)] wherever consecutive inits are more than one cycle
    apart. Informational: WoFS is not run around the clock, so overnight gaps
    are normal."""
    inits = sorted(set(inits))
    return [(a, b) for a, b in zip(inits, inits[1:])
            if b - a > timedelta(minutes=cycle_min)]


def grid_id(attrs):
    """Hashable identity of the grid a file carries."""
    return tuple(round(float(attrs[k]), 4) for k in GRID_KEYS)


# =============================================================================
# Reading
# =============================================================================

def read_global_attrs(path):
    """Global attributes of one file; no variable data is read."""
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        return dict(ds.attrs), dict(ds.sizes)


@dataclass
class InitDir:
    path: Path
    init_dt: datetime
    steps: list
    first_file: Path
    mismatches: list = field(default_factory=list)


@dataclass
class Deployment:
    storm: str
    name: str
    path: Path
    inits: list = field(default_factory=list)
    grids: dict = field(default_factory=dict)   # grid_id -> [init HHMM, ...]
    attrs: dict = field(default_factory=dict)   # first file's attributes
    sizes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def scan_init_dir(path):
    """InitDir for one <HHMM> folder, or None if it holds no WoFS files."""
    files = {}
    for f in sorted(path.glob("wofs_ALL_*.nc")):
        parsed = parse_wofs_name(f.name)
        if parsed:
            files[parsed[0]] = (f, parsed[1], parsed[2])
    if not files:
        return None
    first_step = min(files)
    first_file, init_dt, _ = files[first_step]
    mismatches = [
        f"step {s}: name says valid {v}, init+step says "
        f"{expected_valid_hhmm(i, s)}"
        for s, (_, i, v) in sorted(files.items())
        if v != expected_valid_hhmm(i, s)]
    return InitDir(path, init_dt, sorted(files), first_file, mismatches)


def scan_deployment(storm, dep_path):
    dep = Deployment(storm=storm, name=dep_path.name, path=dep_path)
    for sub in sorted(p for p in dep_path.iterdir() if p.is_dir()):
        info = scan_init_dir(sub)
        if info is None:
            dep.notes.append(f"{sub.name}: no wofs_ALL_*.nc files")
            continue
        attrs, sizes = read_global_attrs(info.first_file)
        file_init = datetime.strptime(attrs["init_time"], TIME_FMT)
        if file_init != info.init_dt:
            info.mismatches.append(
                f"file init_time {file_init:%Y%m%d %H%M} != name "
                f"{info.init_dt:%Y%m%d %H%M}")
        # The file's own init time is the one that counts.
        info.init_dt = file_init
        dep.inits.append(info)
        dep.grids.setdefault(grid_id(attrs), []).append(sub.name)
        if not dep.attrs:
            dep.attrs, dep.sizes = attrs, sizes
    dep.inits.sort(key=lambda i: i.init_dt)
    return dep


def find_deployments(root):
    """[(storm, deployment_path)] under the archive root."""
    root = Path(root)
    out = []
    for storm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for dep in sorted(storm_dir.glob("*_d*")):
            if dep.is_dir():
                out.append((storm_dir.name, dep))
    return out


# =============================================================================
# Reporting
# =============================================================================

def deployment_row(dep, grid_names):
    """One flat dict per deployment, for the table and the CSV."""
    if not dep.inits:
        return {"storm": dep.storm, "deployment": dep.name,
                "n_inits": 0, "notes": "; ".join(dep.notes)}
    first, last = dep.inits[0], dep.inits[-1]
    max_step = max(i.steps[-1] for i in dep.inits)
    last_valid = last.init_dt + timedelta(minutes=STEP_MIN * last.steps[-1])
    steps_per_init = sorted({len(i.steps) for i in dep.inits})
    holes = {i.init_dt.strftime("%H%M"): missing_steps(i.steps)
             for i in dep.inits if missing_steps(i.steps)}
    mism = sum(len(i.mismatches) for i in dep.inits)
    gid = next(iter(dep.grids))
    notes = list(dep.notes)
    if len(dep.grids) > 1:
        notes.append(f"{len(dep.grids)} DIFFERENT GRIDS in one folder")
    if holes:
        notes.append(f"missing steps in {len(holes)} inits")
    if mism:
        notes.append(f"{mism} name/attribute time mismatches")
    return {
        "storm": dep.storm,
        "deployment": dep.name,
        "grid": grid_names[gid],
        "n_inits": len(dep.inits),
        "first_init": f"{first.init_dt:%Y-%m-%d %H%MZ}",
        "last_init": f"{last.init_dt:%Y-%m-%d %H%MZ}",
        "last_valid": f"{last_valid:%Y-%m-%d %H%MZ}",
        "max_step": max_step,
        "steps_per_init": "/".join(str(n) for n in steps_per_init),
        "ne": dep.sizes.get("ne", ""),
        "nx": dep.attrs.get("nx", ""),
        "ny": dep.attrs.get("ny", ""),
        "d_km": dep.attrs.get("d_km", ""),
        "start_lat": f"{float(dep.attrs['start_lat']):.4f}",
        "start_lon": f"{float(dep.attrs['start_lon']):.4f}",
        "cen_lat": f"{float(dep.attrs['CEN_LAT']):.4f}",
        "cen_lon": f"{float(dep.attrs['CEN_LON']):.4f}",
        "notes": "; ".join(notes),
    }


def deep_check(dep):
    """Open one init: rain units, step-00 rainfall, lat/lon vs attributes."""
    import numpy as np
    import xarray as xr
    info = dep.inits[len(dep.inits) // 2]
    out = [f"  {dep.storm}/{dep.name} init {info.init_dt:%H%MZ}:"]
    files = {parse_wofs_name(p.name)[0]: p
             for p in info.path.glob("wofs_ALL_*.nc")}
    for step in sorted(files)[:3]:
        with xr.open_dataset(files[step], decode_times=False) as ds:
            rain = ds["rain"]
            vals = rain.values
            out.append(
                f"    step {step:02d}: units={rain.attrs.get('units')!r} "
                f"nan_frac={np.isnan(vals).mean():.3f} "
                f"mean={np.nanmean(vals):.5f} max={np.nanmax(vals):.4f} "
                f"dims={rain.dims}")
            if step == min(files):
                lat, lon = ds["xlat"].values, ds["xlon"].values
                out.append(
                    f"    xlat[0,0]={lat[0, 0]:.4f} (start_lat "
                    f"{float(ds.attrs['start_lat']):.4f}), "
                    f"xlon[0,0]={lon[0, 0]:.4f} (start_lon "
                    f"{float(ds.attrs['start_lon']):.4f}), "
                    f"xlon range [{lon.min():.2f}, {lon.max():.2f}]")
    return "\n".join(out)


def print_table(rows):
    cols = ["storm", "deployment", "grid", "n_inits", "first_init",
            "last_init", "last_valid", "steps_per_init", "ne", "nx", "ny"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
        if r.get("notes"):
            print(f"    ! {r['notes']}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--csv", help="also write the table to this CSV")
    ap.add_argument("--deep", action="store_true",
                    help="open one init per deployment for variable checks")
    args = ap.parse_args(argv)

    deps = [scan_deployment(s, p) for s, p in find_deployments(args.root)]

    # Name grids G1, G2, ... in first-seen order, shared across deployments.
    grid_names = {}
    for dep in deps:
        for gid in dep.grids:
            grid_names.setdefault(gid, f"G{len(grid_names) + 1}")
    rows = [deployment_row(d, grid_names) for d in deps]
    print_table(rows)
    print(f"\n{len(deps)} deployment folders, {len(grid_names)} distinct grids")

    gaps = defaultdict(list)
    for d in deps:
        for a, b in cycle_gaps(i.init_dt for i in d.inits):
            gaps[f"{d.storm}/{d.name}"].append(f"{a:%d %H%M}->{b:%d %H%M}")
    if gaps:
        print("\nInit gaps longer than one cycle (informational):")
        for k, v in gaps.items():
            print(f"  {k}: {', '.join(v)}")

    if args.deep:
        print("\nDeep checks:")
        for d in deps:
            if d.inits:
                print(deep_check(d))

    if args.csv:
        fields = list(rows[0])
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
