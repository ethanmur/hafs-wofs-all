"""Where each WoFS domain sits relative to the NHC best track (plan G3, step 1).

    python analysis/run.py storms/<case>.yaml storm-location

Prints two numbers per 3 h window of the domain, nothing else (no plots yet):

  1. the distance from the center of the WoFS domain to the TC center, taken
     from the best-track fix nearest in time to the window midpoint, used only
     when that fix is within 3 h of it (no interpolation). A track that ends
     before the domain is formed, or starts after it, prints
     "time out of bounds: <h> h off" instead.
  2. the distance from the domain center to the storm's final mainland landfall,
     the same number for every window of the domain, with the hours from the
     landfall to the window midpoint (negative: before landfall).

The domain center is the center of the verification grid (the mean of its four
central cells). A b-deck does not flag landfall, so landfalls are found from the
track: a step from ocean to land on a mainland landmass (Natural Earth 50 m land
polygons larger than MAINLAND_KM2, which keeps Baja California and the Yucatan
and drops island chains), placed where the straight line between the two fixes
crosses the coast. All landfalls found are printed; the last one is the
reference. Check them against the NHC reports: a storm that moves from one
landmass to another over land (Hilary from Baja California into California) has
no ocean-to-land step for the second, so its reference is the first.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import best_track as bt
import verification_grid as vg

TOLERANCE_H = 3.0            # a fix further than this from the window is unused
MAINLAND_KM2 = 100_000.0     # land polygons smaller than this are islands
BISECT_STEPS = 30


# =============================================================================
# Geometry
# =============================================================================

def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle (WGS84 geodesic) distance in km."""
    from pyproj import Geod
    return Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)[2] / 1000.0


def domain_center(case):
    """(lat, lon) of the center of the verification grid."""
    spec = vg.grid_spec_from_json(case.grid_json_path)
    lat, lon = vg.grid_latlon(spec)
    ny, nx = lat.shape
    rows = slice(ny // 2 - (1 - ny % 2), ny // 2 + 1)
    cols = slice(nx // 2 - (1 - nx % 2), nx // 2 + 1)
    return float(lat[rows, cols].mean()), float(lon[rows, cols].mean())


def mainland_land_test(min_km2=MAINLAND_KM2):
    """A function (lat, lon) -> True over mainland land, from Natural Earth's
    50 m land polygons (cartopy downloads them once)."""
    import cartopy.io.shapereader as shpreader
    from pyproj import Geod
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from shapely.geometry import Point
    path = shpreader.natural_earth(resolution="50m", category="physical",
                                   name="land")
    geod = Geod(ellps="WGS84")
    polys = []
    for geom in shpreader.Reader(path).geometries():
        for part in getattr(geom, "geoms", [geom]):
            if abs(geod.geometry_area_perimeter(part)[0]) / 1e6 >= min_km2:
                polys.append(part)
    land = prep(unary_union(polys))
    return lambda lat, lon: land.contains(Point(lon, lat))


# =============================================================================
# Track
# =============================================================================

def nearest_fix(track, when, tolerance_h=TOLERANCE_H):
    """((time, lat, lon, status), offset_h) of the fix nearest `when`, or
    (None, offset_h) when even that fix is further than `tolerance_h`. The
    offset is fix minus `when` in hours."""
    best = min(track, key=lambda f: abs((f[0] - when).total_seconds()))
    offset = (best[0] - when).total_seconds() / 3600.0
    return (best if abs(offset) <= tolerance_h else None), offset


def landfalls(track, is_land):
    """[(time, lat, lon)] of every ocean-to-land step of the track, placed where
    the line between the two fixes crosses the coast (bisection)."""
    out = []
    for (t0, la0, lo0, _), (t1, la1, lo1, _) in zip(track, track[1:]):
        if is_land(la0, lo0) or not is_land(la1, lo1):
            continue
        lo_f, hi_f = 0.0, 1.0
        for _ in range(BISECT_STEPS):
            mid = (lo_f + hi_f) / 2
            if is_land(la0 + mid * (la1 - la0), lo0 + mid * (lo1 - lo0)):
                hi_f = mid
            else:
                lo_f = mid
        out.append((t0 + (t1 - t0) * hi_f, la0 + hi_f * (la1 - la0),
                    lo0 + hi_f * (lo1 - lo0)))
    return out


# =============================================================================
# Command
# =============================================================================

def locate(case, track, center, is_land, tolerance_h=TOLERANCE_H):
    """{"landfalls": [...], "rows": [dict per window]} for one domain."""
    lat_c, lon_c = center
    falls = landfalls(track, is_land)
    ref = falls[-1] if falls else None
    rows = []
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        mid = w.start + (w.end - w.start) / 2
        fix, offset = nearest_fix(track, mid, tolerance_h)
        row = {"window_end": w.end, "midpoint": mid, "fix": fix,
               "offset_h": offset, "tc_km": None, "landfall_km": None,
               "from_landfall_h": None}
        if fix is not None:
            row["tc_km"] = distance_km(lat_c, lon_c, fix[1], fix[2])
        if ref is not None:
            row["landfall_km"] = distance_km(lat_c, lon_c, ref[1], ref[2])
            row["from_landfall_h"] = (mid - ref[0]).total_seconds() / 3600.0
        rows.append(row)
    return {"landfalls": falls, "reference": ref, "rows": rows}


def _ns(lat):
    return f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"


def _ew(lon):
    return f"{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"


def print_location(case, track, center, result, tolerance_h=TOLERANCE_H):
    print(f"Domain center : {_ns(center[0])} {_ew(center[1])}  "
          "(center of the verification grid)")
    print(f"Best track    : {Path(case.best_track).name}, {len(track)} fixes "
          f"{track[0][0]:%Y-%m-%d %HZ} to {track[-1][0]:%Y-%m-%d %HZ}")
    falls = result["landfalls"]
    if falls:
        for t, la, lo in falls:
            print(f"Landfall      : {t:%Y-%m-%d %H:%MZ} at {_ns(la)} {_ew(lo)}"
                  "  (ocean to mainland)")
        ref = result["reference"]
        print(f"Reference     : the last, {ref[0]:%Y-%m-%d %H:%MZ}; the domain "
              f"center is {result['rows'][0]['landfall_km']:.0f} km from it")
    else:
        print("Landfall      : none found (no ocean-to-mainland step in the "
              "best track)")
    print(f"\n{'window end':<14} {'nearest fix':<14} {'off':>6} {'status':<6} "
          f"{'TC center':<17} {'dist to TC':>11} {'to landfall':>12}")
    for r in result["rows"]:
        end = f"{r['window_end']:%m-%d %HZ}"
        if r["fix"] is None:
            print(f"{end:<14} time out of bounds: {abs(r['offset_h']):.1f} h "
                  f"off (limit {tolerance_h:g} h)")
            continue
        t, la, lo, status = r["fix"]
        land = ("" if r["from_landfall_h"] is None
                else f"{r['from_landfall_h']:+.1f} h")
        print(f"{end:<14} {t:%m-%d %HZ}      {r['offset_h']:+5.1f}h "
              f"{status:<6} {_ns(la) + ' ' + _ew(lo):<17} "
              f"{r['tc_km']:>8.0f} km {land:>12}")


def storm_location_case(case, is_land=None):
    """Driver for `storm-location`: print the two numbers per window."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"storm-location: {case.case_slug} has no complete "
                         "WoFS window, so there is no domain to locate.")
    track = bt.parse_bdeck_status(case.best_track)
    center = domain_center(case)
    is_land = is_land or mainland_land_test()
    result = locate(case, track, center, is_land)
    print_location(case, track, center, result)
    return result
