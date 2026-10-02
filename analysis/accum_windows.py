"""D2: absolute-clock accumulation windows, and the model runs covering them.

Verification happens over 3 h windows pinned to the wall clock -- 00-03Z,
03-06Z, ... -- not to any model's initialization. That choice is what lets a
HAFS cycle, a WoFS deployment and Stage IV be compared at all: each produces
accumulations on its own schedule, and the clock is the only thing they share.

Two consequences follow, and they drive everything in this module:

  * A model run is used for a window only when it *fully spans* it. A run that
    covers 19-21Z of the 18-21Z window contributes nothing; a partial
    accumulation is not a smaller version of the right answer, it is the wrong
    answer. So windows a run cannot span are dropped rather than scaled.

  * The same window is forecast by several runs at different leads, and that
    is the point. Each (run, window) pair carries its own lead, which is what
    turns a pile of forecasts into a lead-time dependence without any extra
    bookkeeping.

Everything is computed in whole minutes. WoFS cycles every 30 min and writes
every 5, so leads like f02:30 are routine, and float hours would make
"is this lead a multiple of the output step" a tolerance question instead of
an exact one.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_WINDOW_H = 3
# HAFS: 6-hourly cycles, hourly output, 3 h APCP buckets (D1), out to 126 h.
HAFS_CYCLE_H = 6
HAFS_MAX_LEAD_H = 126
HAFS_STEP_MIN = 180        # accumulation granularity, not output frequency
# WoFS: 30 min cycling, 5 min output, 0-6 h runs.
WOFS_CYCLE_MIN = 30
WOFS_MAX_LEAD_MIN = 360
WOFS_STEP_MIN = 5


def _minutes(delta):
    return int(round(delta.total_seconds() / 60.0))


@dataclass(frozen=True)
class Window:
    """One accumulation window, (start, end], on the absolute clock."""
    start: datetime
    end: datetime

    @property
    def hours(self):
        return _minutes(self.end - self.start) / 60.0

    @property
    def stamp(self):
        """Stamp of the window end, which is how MET labels an accumulation."""
        return f"{self.end:%Y%m%d%H}"

    @property
    def label(self):
        return f"{self.start:%Y-%m-%d %HZ}-{self.end:%HZ}"


@dataclass(frozen=True)
class ModelRun:
    """One initialization of a model, and what it can accumulate.

    step_min is the accumulation granularity: the smallest interval the model
    can give a clean accumulation over. For HAFS that is the 3 h APCP bucket
    (180), not the hourly output frequency -- a 1 h slice of a 3 h bucket does
    not exist. For WoFS it is the 5 min output step.
    """
    model: str
    init: datetime
    max_lead_min: int
    step_min: int

    @property
    def label(self):
        return f"{self.model} {self.init:%Y-%m-%d %H:%MZ}"


@dataclass(frozen=True)
class Coverage:
    """A (run, window) pair that works, with the leads spanning it."""
    run: ModelRun
    window: Window
    lead_start_min: int
    lead_end_min: int

    @property
    def lead_h(self):
        """Lead at the *end* of the window: how far out the forecast had run
        by the time the accumulation was complete. This is the x-axis of every
        lead-time plot, so it is defined once, here."""
        return self.lead_end_min / 60.0

    @property
    def lead_label(self):
        return f"{format_lead(self.lead_start_min)}-{format_lead(self.lead_end_min)}"


def format_lead(minutes):
    """f003 for whole hours, f02:30 otherwise -- how the leads read in logs."""
    h, m = divmod(int(minutes), 60)
    return f"f{h:03d}" if m == 0 else f"f{h:02d}:{m:02d}"


# =============================================================================
# Windows
# =============================================================================

def clock_windows(valid_start, valid_end, window_h=DEFAULT_WINDOW_H):
    """Every `window_h` window on the absolute clock inside the valid period.

    Only windows *fully contained* in [valid_start, valid_end] are returned.
    A window hanging off either end would be verified against a partial obs
    accumulation, which is a different quantity -- better to lose an hour of
    record than to score a 2 h sum against a 3 h forecast.

    The clock is anchored at 00Z, so with window_h=3 the edges are
    00/03/06/09/12/15/18/21Z regardless of when the case window starts.
    """
    if window_h <= 0 or 24 % window_h:
        raise ValueError("window_h must divide 24 evenly")
    step = timedelta(hours=window_h)
    day = valid_start.replace(hour=0, minute=0, second=0, microsecond=0)
    n = _minutes(valid_start - day) // (window_h * 60)
    start = day + n * step
    if start < valid_start:
        start += step               # first edge at or after valid_start
    windows = []
    while start + step <= valid_end:
        windows.append(Window(start, start + step))
        start += step
    return windows


# =============================================================================
# Runs
# =============================================================================

def hafs_runs(init_start, init_end, cycle_h=HAFS_CYCLE_H,
              max_lead_h=HAFS_MAX_LEAD_H, step_min=HAFS_STEP_MIN,
              model="hafs"):
    """HAFS cycles over an init range. The cycles that actually exist on disk
    are a separate question (D6); this is the nominal schedule."""
    runs, init = [], init_start
    while init <= init_end:
        runs.append(ModelRun(model, init, int(max_lead_h * 60), step_min))
        init += timedelta(hours=cycle_h)
    return runs


def wofs_runs(domain, cycle_min=WOFS_CYCLE_MIN, max_lead_min=WOFS_MAX_LEAD_MIN,
              step_min=WOFS_STEP_MIN, model="wofs"):
    """WoFS cycles across one deployment's own time window.

    The deployment's valid_start/valid_end bound when WoFS was *running*, so
    initializations are generated inside it; a run initialized at valid_end
    still contributes its full 6 h forecast.
    """
    if domain.valid_start is None or domain.valid_end is None:
        raise ValueError(f"WoFS deployment {domain.name} has no valid_start/"
                         "valid_end; cycles cannot be enumerated without them")
    runs, init = [], domain.valid_start
    while init <= domain.valid_end:
        runs.append(ModelRun(f"{model}:{domain.name}", init, max_lead_min,
                             step_min))
        init += timedelta(minutes=cycle_min)
    return runs


# =============================================================================
# Pairing
# =============================================================================

def covers(run, window):
    """Coverage if `run` can accumulate exactly over `window`, else None.

    Three conditions, all necessary:
      1. the window starts at or after the init (no negative leads)
      2. the window ends at or before the end of the forecast
      3. both leads fall on the model's accumulation granularity, so the
         window is a whole number of the model's own accumulation intervals
    """
    lead_start = _minutes(window.start - run.init)
    lead_end = _minutes(window.end - run.init)
    if lead_start < 0 or lead_end > run.max_lead_min:
        return None
    if lead_start % run.step_min or lead_end % run.step_min:
        return None
    return Coverage(run, window, lead_start, lead_end)


def coverage(runs, windows):
    """Every (run, window) pair that works, ordered window then lead."""
    out = [c for w in windows for r in runs
           if (c := covers(r, w)) is not None]
    out.sort(key=lambda c: (c.window.start, c.lead_end_min, c.run.init))
    return out


def by_window(coverages):
    """{Window: [Coverage, ...]} preserving the lead ordering."""
    grouped = {}
    for c in coverages:
        grouped.setdefault(c.window, []).append(c)
    return grouped


def case_windows(case, window_h=DEFAULT_WINDOW_H):
    """The windows for a case, plus the HAFS and WoFS runs covering each.

    WoFS runs are only paired with windows inside their own deployment, which
    `wofs_runs` already enforces by construction: a deployment's cycles cannot
    reach a window outside its own period, since the 6 h forecast is the only
    thing extending past valid_end.
    """
    windows = clock_windows(case.valid_start, case.valid_end, window_h)
    runs = []
    if case.init_start is not None and case.init_end is not None:
        runs += hafs_runs(case.init_start, case.init_end)
    for dom in (case.wofs_domains or []):
        if dom.valid_start is not None and dom.valid_end is not None:
            runs += wofs_runs(dom)
    return windows, runs, coverage(runs, windows)


def print_windows(case, window_h=DEFAULT_WINDOW_H):
    """Driver for `list-windows`: the window table, before any data is read."""
    windows, runs, cov = case_windows(case, window_h)
    grouped = by_window(cov)
    print(f"Case   : {case.storm_name}  ({case.case_slug})")
    print(f"Window : {case.valid_start:%Y-%m-%d %HZ} -> "
          f"{case.valid_end:%Y-%m-%d %HZ}")
    print(f"Clock  : {window_h} h windows anchored at 00Z  "
          f"({len(windows)} fully inside the valid period)")
    if case.init_start is not None:
        print(f"HAFS   : cycles {case.init_start:%Y-%m-%d %HZ} -> "
              f"{case.init_end:%Y-%m-%d %HZ} every {HAFS_CYCLE_H} h, "
              f"to f{HAFS_MAX_LEAD_H:03d}")
    else:
        print("HAFS   : no init_start/init_end in the YAML; no cycles paired")
    for dom in (case.wofs_domains or []):
        n = sum(1 for r in runs if r.model.endswith(dom.name))
        print(f"WoFS   : {dom.label}  ({n} cycles every "
              f"{WOFS_CYCLE_MIN} min)")
    print()
    for w in windows:
        here = grouped.get(w, [])
        models = {}
        for c in here:
            models.setdefault(c.run.model.split(":")[0], []).append(c)
        parts = []
        for name, cs in models.items():
            leads = f"{cs[0].lead_h:g}-{cs[-1].lead_h:g} h"
            parts.append(f"{name} x{len(cs)} (lead {leads})")
        print(f"  {w.label}   " + ("  |  ".join(parts) if parts
                                   else "no run spans this window"))
    print(f"\n{len(cov)} usable (run, window) pairs over {len(windows)} "
          f"windows")
    missing = [w for w in windows if w not in grouped]
    if missing:
        print(f"{len(missing)} window(s) with no run at all -- these drop out "
              f"of every statistic")
    return windows, cov
