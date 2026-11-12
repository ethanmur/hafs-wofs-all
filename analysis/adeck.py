"""HAFS forecast tracks from NHC a-decks.

An a-deck (aal142024.dat, aep092023.dat ...) holds model guidance in ATCF
format: one line per (cycle, model, forecast hour) with the cycle's initial time
in column 3, the model ("tech") in column 5, the forecast hour (tau) in column 6
and the position in columns 7-8, like a b-deck's. HAFS-A and HAFS-B are `HFSA`
and `HFSB`. Wind-radii lines repeat a time, so each (tech, cycle, tau) is kept
once. Fetch the files with `fetch_bdecks.py --adeck`.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from best_track import decode_latlon

TECHS = ("HFSA", "HFSB")


def parse_adeck(path, techs=TECHS):
    """{(tech, init): [(valid, lat, lon, vmax_kt or None), ...]} sorted by
    forecast hour, for the wanted techs. Lines that do not parse are skipped."""
    seen, tracks = set(), {}
    wanted = {t.upper() for t in techs}
    with open(path) as fh:
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8 or cols[4].upper() not in wanted:
                continue
            try:
                init = datetime.strptime(cols[2], "%Y%m%d%H")
                tau = int(cols[5])
                lat, lon = decode_latlon(cols[6]), decode_latlon(cols[7])
            except (ValueError, IndexError):
                continue
            key = (cols[4].upper(), init, tau)
            if key in seen:
                continue
            seen.add(key)
            try:
                vmax = float(cols[8]) if len(cols) > 8 else None
            except ValueError:
                vmax = None
            tracks.setdefault(key[:2], []).append(
                (tau, init + timedelta(hours=tau), lat, lon, vmax))
    return {k: [(t, la, lo, v) for _, t, la, lo, v in sorted(rows)]
            for k, rows in tracks.items()}


def adeck_summary(path):
    """{tech: (n cycles, first init, last init)} of an a-deck."""
    tracks = parse_adeck(path)
    out = {}
    for tech in TECHS:
        inits = sorted(i for (t, i) in tracks if t == tech)
        if inits:
            out[tech] = (len(inits), inits[0], inits[-1])
    return out


def track_to(track, end):
    """([(time, lat, lon)], reached): the forecast track from its first fix up to
    `end`, with a last point at `end` interpolated linearly between the two fixes
    around it. `reached` is False when the track stops before `end` (what is
    there is returned)."""
    pts = [(t, la, lo) for t, la, lo, _ in track]
    out = [p for p in pts if p[0] <= end]
    if not out:
        return [], False
    if out[-1][0] == end:
        return out, True
    later = [p for p in pts if p[0] > end]
    if not later:
        return out, False
    (t0, la0, lo0), (t1, la1, lo1) = out[-1], later[0]
    f = (end - t0).total_seconds() / (t1 - t0).total_seconds()
    return out + [(end, la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0))], True
