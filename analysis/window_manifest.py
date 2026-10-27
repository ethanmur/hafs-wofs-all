"""The window manifest: every 3 h window of a deployment and what is on disk
for it (plan step D6).

For each deployment this lists, per window of the scored period:

  * every WoFS init that could cover it, with its status (complete,
    missing_steps, short_run, no_files) from the exact steps on disk;
  * every HAFS cycle that should be a member -- the six-hourly cycles of each
    configuration initialized in the `lookback_h` before the window starts and
    strictly before it -- with its status, including cycles HAFS never ran
    (`no_cycle`: HAFS stops cycling a storm after landfall).

Nothing is dropped. A pair that cannot be used stays in the table with the
reason, so a window that is missing data shows up here, cheaply, instead of as
a strange score three steps later.

Two files per deployment, both in `manifest_dir` and named for the case:

  <case>_manifest.csv   one row per (window, model run): status, lead, source
  <case>_windows.csv    one row per window: WoFS runs and HAFS members counted,
                        and whether the window can be verified

A window is **testable** when at least one WoFS run is complete and the HAFS
members, HFSA and HFSB together, reach `min_members` (8). With WoFS but too
few HAFS members it is **wofs_only**: WoFS can still be scored against the
obs, but the window is kept out of head-to-head comparison. With no complete
WoFS run it is **no_wofs** and nothing can be verified.

Usage:
    python analysis/run.py storms/<case>.yaml build-manifest
"""

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import hafs_inventory as hi
import wofs_inventory as wi

STAMP = "%Y-%m-%d %H%MZ"
MANIFEST_FIELDS = ["deployment", "window_start", "window_end", "model", "run",
                   "init", "lead", "lead_h", "status", "n_members", "source"]


@dataclass
class Manifest:
    rows: list
    summary: list
    windows: list
    dep: object            # the scanned WoFS Deployment
    notes: list


# =============================================================================
# Pure logic
# =============================================================================

def expected_cycles(window_start, lookback_h=hi.LOOKBACK_H,
                    cycle_h=hi.CYCLE_H):
    """The cycle times that should be members for a window: every `cycle_h`
    boundary from `lookback_h` before the window start, up to but not
    including the start (a cycle initialized at the start is spin-up)."""
    t = (window_start - timedelta(hours=lookback_h)).replace(
        minute=0, second=0, microsecond=0)
    t += timedelta(hours=(-t.hour) % cycle_h)
    out = []
    while t < window_start:
        out.append(t)
        t += timedelta(hours=cycle_h)
    return out


def _lead_label(init, window):
    start = int((window.start - init).total_seconds() // 60)
    end = int((window.end - init).total_seconds() // 60)
    return f"{aw.format_lead(start)}-{aw.format_lead(end)}", end / 60.0


def wofs_rows(case, dep, windows):
    """One row per (WoFS init, window) the init could reach."""
    ne = dep.sizes.get("ne", "")
    entries = [(i.init_dt, set(i.steps), i.path.name) for i in dep.inits]
    entries += [(t, set(), h) for h, t in dep.empty_inits.items()]
    rows = []
    for init, steps, folder in sorted(entries, key=lambda e: e[0]):
        reach = set(wi.init_windows(init, steps))
        for w in windows:
            if w not in reach:
                continue
            status = wi.window_status(init, steps, w)[0]
            lead, lead_h = _lead_label(init, w)
            rows.append({
                "deployment": case.case_slug,
                "window_start": f"{w.start:{STAMP}}",
                "window_end": f"{w.end:{STAMP}}",
                "model": "wofs", "run": f"{init:%Y%m%d%H%M}",
                "init": f"{init:{STAMP}}", "lead": lead, "lead_h": lead_h,
                "status": status, "n_members": ne,
                "source": str(dep.path / folder)})
    return rows


def hafs_rows(case, cycles, files, windows):
    """One row per (window, configuration, expected cycle).

    `cycles` is {config: {init: leads}} and `files` {config: {(init, lead_h):
    path}}, both from hafs_inventory.
    """
    rows = []
    for w in windows:
        for config in case.hafs_models:
            have = cycles.get(config, {})
            fmap = files.get(config, {})
            for init in expected_cycles(w.start, case.hafs_lookback_h):
                leads = have.get(init)
                status = ("no_cycle" if leads is None
                          else hi.hafs_window_status(init, leads, w))
                lead, lead_h = _lead_label(init, w)
                source = ""
                if status == "complete":
                    source = str(fmap.get((init, int(lead_h)), ""))
                rows.append({
                    "deployment": case.case_slug,
                    "window_start": f"{w.start:{STAMP}}",
                    "window_end": f"{w.end:{STAMP}}",
                    "model": config.lower(), "run": f"{init:%Y%m%d%H}",
                    "init": f"{init:{STAMP}}", "lead": lead,
                    "lead_h": lead_h, "status": status, "n_members": 1,
                    "source": source})
    return rows


def summarize(case, rows, windows):
    """One row per window: the counts that decide whether it can be verified."""
    configs = [c.lower() for c in case.hafs_models]
    out = []
    for w in windows:
        start = f"{w.start:{STAMP}}"
        mine = [r for r in rows if r["window_start"] == start]
        wofs = sum(r["model"] == "wofs" and r["status"] == "complete"
                   for r in mine)
        members = {c: sum(r["model"] == c and r["status"] == "complete"
                          for r in mine) for c in configs}
        total = sum(members.values())
        hafs = hi.hafs_status(total, case.hafs_min_members)
        category = ("no_wofs" if wofs == 0
                    else "testable" if hafs == "ok" else "wofs_only")
        row = {"deployment": case.case_slug, "window_start": start,
               "window_end": f"{w.end:{STAMP}}", "wofs_complete_runs": wofs}
        row.update({f"{c}_members": n for c, n in members.items()})
        row.update({"hafs_total": total, "hafs_status": hafs,
                    "category": category})
        out.append(row)
    return out


def wofs_complete_pairs(rows):
    """[(init datetime, Window)] for every complete WoFS run and window."""
    out = []
    for r in rows:
        if r["model"] == "wofs" and r["status"] == "complete":
            out.append((datetime.strptime(r["run"], "%Y%m%d%H%M"),
                        aw.Window(datetime.strptime(r["window_start"], STAMP),
                                  datetime.strptime(r["window_end"], STAMP))))
    return out


# =============================================================================
# Building from disk
# =============================================================================

def build_manifest(case):
    """Scan the WoFS and HAFS archives for this deployment's windows."""
    dep = wi.scan_deployment(case.wofs_storm_folder, case.wofs_root)
    windows = aw.clock_windows(case.valid_start, case.valid_end)
    cycles, files, notes = {}, {}, list(dep.notes)
    for config, root in case.hafs_models.items():
        cycles[config], scan_notes = hi.scan_config(Path(root))
        files[config] = hi.index_config_files(Path(root))
        notes += [f"{config}: {n}" for n in scan_notes]
    rows = wofs_rows(case, dep, windows) + hafs_rows(case, cycles, files,
                                                      windows)
    return Manifest(rows=rows, summary=summarize(case, rows, windows),
                    windows=windows, dep=dep, notes=notes)


def _write(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def build_manifest_case(case):
    """Driver for `build-manifest`: scan, report, write both tables."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        print("WoFS   : no complete 3 h window; nothing to put in a manifest")
        return None
    print(f"Window : {case.valid_start:%Y-%m-%d %HZ} -> "
          f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"HAFS   : {', '.join(case.hafs_models) or 'none configured'}; "
          f"members = cycles up to {case.hafs_lookback_h} h back, strictly "
          f"before the window; at least {case.hafs_min_members} to be eligible",
          flush=True)
    m = build_manifest(case)
    for note in m.notes:
        print(f"  note: {note}")

    cats = Counter(r["category"] for r in m.summary)
    print(f"\n{len(m.windows)} windows between valid_start and valid_end:")
    print(f"  testable   {cats['testable']:>4}   (a complete WoFS run and "
          f">= {case.hafs_min_members} HAFS members)")
    print(f"  wofs_only  {cats['wofs_only']:>4}   (WoFS run, too few HAFS "
          "members)")
    print(f"  no_wofs    {cats['no_wofs']:>4}   (no complete WoFS run)")
    configs = [c.lower() for c in case.hafs_models]
    header = ("  window_start        wofs " + " ".join(f"{c:>6}" for c in configs)
              + "  hafs      category")
    print("\n" + header)
    for r in m.summary:
        print(f"  {r['window_start']:<19} {r['wofs_complete_runs']:>4} "
              + " ".join(f"{r[c + '_members']:>6}" for c in configs)
              + f"  {r['hafs_status']:<8}  {r['category']}")

    summary_fields = (["deployment", "window_start", "window_end",
                       "wofs_complete_runs"]
                      + [f"{c}_members" for c in configs]
                      + ["hafs_total", "hafs_status", "category"])
    p1 = _write(case.manifest_path, MANIFEST_FIELDS, m.rows)
    p2 = _write(case.window_summary_path, summary_fields, m.summary)
    print(f"\nWrote  : {p1}  ({len(m.rows)} rows)")
    print(f"Wrote  : {p2}")
    return m
