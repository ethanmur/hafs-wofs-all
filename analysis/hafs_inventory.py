"""HAFS cycle inventory and per-window member availability (plan step A1b).

Walks a HAFS archive laid out as

    <root>/<YYYY>-<id>/<HFSA|HFSB>/<YYYYMMDDHH>/<id>.<YYYYMMDDHH>.<model>.parent.atm.f<LLL>.grb2

and reports, per storm and configuration, which cycles exist and which forecast
hours each has. HAFS stops cycling a storm once it is over land or
post-tropical, so late WoFS windows often have few or no recent HAFS cycles;
this makes that visible before anything is regridded.

A 3 h window is served by one file: the 3 h bucket is named for its end, so the
15-18Z accumulation from an 18Z-prior init lives in the f024 file. A window is
therefore ``complete`` for a cycle when that one file is on disk.

With --wofs-windows (the table from wofs_inventory.py) the script also counts,
for every complete WoFS window, the HAFS members available under the member
rule, per configuration. HFSA and HFSB are never pooled.

Member rule: cycles of one configuration initialized no more than
``lookback_h`` (48) before the window start and strictly before it. A cycle
initialized at the window start is spin-up, not a forecast of that window; the
cycle exactly 48 h back is included. Six-hourly cycles then give 8 members per
configuration, 16 with HFSA and HFSB.

Usage:
    python analysis/hafs_inventory.py <hafs_root> [--wofs-windows windows.csv]
        [--members-csv members.csv] [--deep]

--deep lists the ``tp`` accumulation intervals in a few files per configuration
(needs eccodes), to confirm that the 3 h bucket exists at every lead.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_case_yamls import STORMS, parse_time

PARENT_RE = re.compile(
    r"^(?P<storm>\d{2}[a-z])\.(?P<init>\d{10})\.(?P<model>[a-z_]+)"
    r"\.parent\.atm\.f(?P<lead>\d{3})\.grb2$", re.IGNORECASE)
CONFIGS = ("HFSA", "HFSB")
CYCLE_H = 6
BUCKET_H = 3
LOOKBACK_H = 48
DEEP_LEADS = (3, 6, 12, 24, 48, 126)


# =============================================================================
# Pure logic (no file I/O)
# =============================================================================

def parse_init(text):
    return datetime.strptime(text, "%Y%m%d%H")


def hafs_window_status(init_dt, leads, window):
    """Status of one (cycle, 3 h window): complete, missing_file, short_run
    or no_files. The bucket ending at the window end is in the file named for
    that lead, so one file decides it."""
    if not leads:
        return "no_files"
    lead_start = (window.start - init_dt).total_seconds() / 3600
    lead_end = (window.end - init_dt).total_seconds() / 3600
    if lead_start < 0 or lead_end % BUCKET_H or lead_start % BUCKET_H:
        return "not_on_bucket_grid"
    if int(lead_end) in leads:
        return "complete"
    if lead_end > max(leads):
        return "short_run"
    return "missing_file"


def members_for_window(window_start, inits, lookback_h=LOOKBACK_H):
    """Cycles that count as members for a window: initialized at most
    lookback_h before the window start and strictly before it."""
    lo = window_start - timedelta(hours=lookback_h)
    return sorted(i for i in inits if lo <= i < window_start)


def cycle_gaps(inits, cycle_h=CYCLE_H):
    """[(from, to)] where consecutive cycles are more than one cycle apart."""
    inits = sorted(set(inits))
    return [(a, b) for a, b in zip(inits, inits[1:])
            if b - a > timedelta(hours=cycle_h)]


def window_members(window, cycles, lookback_h=LOOKBACK_H):
    """(n_members, [init, ...]) usable for a window, from {init: leads}."""
    usable = [i for i in members_for_window(window.start, cycles, lookback_h)
              if hafs_window_status(i, cycles[i], window) == "complete"]
    return len(usable), usable


# =============================================================================
# Reading the archive
# =============================================================================

def scan_config(config_dir):
    """({init_dt: sorted lead list}, notes) for one <storm>/<config> folder."""
    cycles, notes = {}, []
    if not config_dir.is_dir():
        return cycles, [f"{config_dir} not found"]
    for sub in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        try:
            folder_init = parse_init(sub.name)
        except ValueError:
            continue
        leads = []
        for f in sub.iterdir():
            m = PARENT_RE.match(f.name)
            if not m:
                continue
            if parse_init(m.group("init")) != folder_init:
                notes.append(f"{sub.name}: file {f.name} has another init")
                continue
            leads.append(int(m.group("lead")))
        if leads:
            cycles[folder_init] = sorted(leads)
        else:
            notes.append(f"{sub.name}: no parent.atm files")
    return cycles, notes


def scan_archive(root):
    """{(storm_id, config): ({init: leads}, notes)} for the mapped storms."""
    root = Path(root)
    out = {}
    for sid in sorted({v[0] for v in STORMS.values()}):
        for config in CONFIGS:
            out[(sid, config)] = scan_config(root / sid / config)
    return out


def deep_check(path):
    """tp accumulation intervals in one file: [(stepRange, stepType), ...]."""
    import eccodes
    out = []
    with open(path, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                if eccodes.codes_get(gid, "shortName") == "tp":
                    out.append((eccodes.codes_get(gid, "stepRange"),
                                eccodes.codes_get(gid, "stepType")))
            finally:
                eccodes.codes_release(gid)
    return out


# =============================================================================
# Reporting
# =============================================================================

def cycle_table(archive):
    rows = []
    for (sid, config), (cycles, notes) in archive.items():
        if not cycles:
            rows.append({"storm_id": sid, "config": config, "n_cycles": 0,
                         "notes": "; ".join(notes) or "no cycles"})
            continue
        inits = sorted(cycles)
        lead_sets = sorted({(l[0], l[-1], len(l)) for l in cycles.values()})
        rows.append({
            "storm_id": sid, "config": config, "n_cycles": len(inits),
            "first_init": f"{inits[0]:%Y-%m-%d %HZ}",
            "last_init": f"{inits[-1]:%Y-%m-%d %HZ}",
            "leads": "; ".join(f"f{a:03d}-f{b:03d} x{n}"
                               for a, b, n in lead_sets),
            "gaps": ", ".join(f"{a:%d %HZ}->{b:%d %HZ}"
                              for a, b in cycle_gaps(inits)),
            "notes": "; ".join(notes),
        })
    return rows


def print_cycles(rows):
    print("HAFS cycles on disk:")
    for r in rows:
        if not r["n_cycles"]:
            print(f"  {r['storm_id']} {r['config']}: NONE  {r['notes']}")
            continue
        print(f"  {r['storm_id']} {r['config']}: {r['n_cycles']} cycles "
              f"{r['first_init']} .. {r['last_init']}   leads {r['leads']}")
        if r["gaps"]:
            print(f"      gaps: {r['gaps']}")
        if r["notes"]:
            print(f"      ! {r['notes']}")


def member_rows(archive, wofs_rows, lookback_h=LOOKBACK_H):
    """One row per (deployment, complete WoFS window, config): the HAFS
    members available for it."""
    from accum_windows import Window
    folder_to_sid = {k: v[0] for k, v in STORMS.items()}
    seen, rows = set(), []
    for r in wofs_rows:
        if r["status"] != "complete":
            continue
        key = (r["storm"], r["deployment"], r["window_start"],
               r["window_end"])
        if key in seen:
            continue
        seen.add(key)
        win = Window(parse_time(r["window_start"]), parse_time(r["window_end"]))
        sid = folder_to_sid[r["storm"]]
        for config in CONFIGS:
            cycles = archive.get((sid, config), ({}, []))[0]
            n, usable = window_members(win, cycles, lookback_h)
            rows.append({
                "storm": r["storm"], "deployment": r["deployment"],
                "window_start": r["window_start"],
                "window_end": r["window_end"], "config": config,
                "n_members": n,
                "members": ";".join(f"{i:%Y%m%d%H}" for i in usable),
            })
    return rows


def summarize_members(rows):
    """{deployment: counts} -- how many WoFS windows have HAFS members."""
    by_win = defaultdict(dict)
    for r in rows:
        by_win[(r["storm"], r["deployment"], r["window_start"])][
            r["config"]] = r["n_members"]
    out = defaultdict(lambda: defaultdict(int))
    for (storm, dep, _), cfg in by_win.items():
        d = out[(storm, dep)]
        d["windows"] += 1
        a, b = cfg.get("HFSA", 0), cfg.get("HFSB", 0)
        d["both"] += bool(a and b)
        d["hfsa_only"] += bool(a and not b)
        d["hfsb_only"] += bool(b and not a)
        d["none"] += not (a or b)
        d["members_sum"] += a + b
    return out


def print_members(summary):
    print("\nHAFS members per complete WoFS window "
          "(lookback rule: 48 h, strictly before the window):")
    cols = ["windows", "both", "hfsa_only", "hfsb_only", "none"]
    print("  " + "deployment".ljust(34)
          + "".join(c.rjust(11) for c in cols) + "   mean members")
    for (storm, dep), d in summary.items():
        mean = d["members_sum"] / d["windows"] if d["windows"] else 0
        flag = "   <-- WoFS-only" if d["windows"] and d["none"] == d["windows"] else ""
        print("  " + f"{storm}/{dep}".ljust(34)
              + "".join(str(d[c]).rjust(11) for c in cols)
              + f"   {mean:5.1f}{flag}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="HAFS archive root, e.g. .../data/hafs")
    ap.add_argument("--wofs-windows",
                    help="windows CSV from wofs_inventory.py")
    ap.add_argument("--members-csv",
                    help="write the per-window member table here")
    ap.add_argument("--lookback-h", type=int, default=LOOKBACK_H)
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args(argv)

    archive = scan_archive(args.root)
    print_cycles(cycle_table(archive))

    if args.wofs_windows:
        with open(args.wofs_windows, newline="") as fh:
            wofs_rows = list(csv.DictReader(fh))
        rows = member_rows(archive, wofs_rows, args.lookback_h)
        print_members(summarize_members(rows))
        if args.members_csv and rows:
            with open(args.members_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {args.members_csv} ({len(rows)} rows)")

    if args.deep:
        print("\nDeep checks (tp intervals):")
        for (sid, config), (cycles, _) in archive.items():
            if not cycles:
                continue
            init = sorted(cycles)[len(cycles) // 2]
            for lead in DEEP_LEADS:
                if lead not in cycles[init]:
                    continue
                p = (Path(args.root) / sid / config / f"{init:%Y%m%d%H}")
                f = next(p.glob(f"*parent.atm.f{lead:03d}.grb2"), None)
                if f:
                    print(f"  {sid} {config} {init:%Y%m%d%H} f{lead:03d}: "
                          f"{deep_check(f)}")


if __name__ == "__main__":
    main(sys.argv[1:])
