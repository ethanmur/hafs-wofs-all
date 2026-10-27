"""Absolute-clock accumulation windows.

Verification happens over 3 h windows pinned to the wall clock -- 00-03Z,
03-06Z, ... -- not to any model's initialization. That choice is what lets a
HAFS cycle, a WoFS run and Stage IV be compared at all: each produces
accumulations on its own schedule, and the clock is the only thing they share.

A model run is used for a window only when it *fully* covers it. A run that
covers 19-21Z of the 18-21Z window contributes nothing; a partial accumulation
is not a smaller version of the right answer, it is the wrong answer. So
windows a run cannot cover are dropped rather than scaled.

Which runs exist for a window is not decided here: the WoFS and HAFS
availability tables (wofs_inventory, hafs_inventory) say what is actually on
disk, and window_manifest combines them. This module is just the clock.

Everything is computed in whole minutes. WoFS cycles every 30 min and writes
every 5, so leads like f02:30 are routine, and float hours would make "is this
lead a multiple of the output step" a tolerance question instead of an exact
one.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_WINDOW_H = 3


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
