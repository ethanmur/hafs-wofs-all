#!/usr/bin/env python3
"""Report the true grid spacing of any GRIB file, in km.

Degree increments are misleading: a 0.06-degree grid is ~6.7 km east-west at
the equator but ~4.7 km at 45N, while north-south stays ~6.7 km everywhere.
This prints both, so a common verification grid can be chosen against the
coarsest input rather than against a nominal degree value.

Usage:
    python analysis/grid_spacing.py '/work2/.../HFSA/*/​*parent.atm.f000.grb2'
    python analysis/grid_spacing.py file.grb2 --lats 15 25 35 45
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import eccodes

R_EARTH_KM = 6371.0


def grid_info(path):
    """Grid geometry of a GRIB file's first message."""
    with open(path, "rb") as fh:
        gid = eccodes.codes_grib_new_from_file(fh)
    if gid is None:
        raise ValueError(f"{path} contains no GRIB messages")
    try:
        keys = ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                "longitudeOfFirstGridPointInDegrees",
                "latitudeOfLastGridPointInDegrees",
                "longitudeOfLastGridPointInDegrees")
        info = {k: eccodes.codes_get(gid, k) for k in keys}
        for key, name in (("iDirectionIncrementInDegrees", "dlon"),
                          ("jDirectionIncrementInDegrees", "dlat")):
            try:
                info[name] = float(eccodes.codes_get(gid, key))
            except Exception:
                info[name] = np.nan
        return info
    finally:
        eccodes.codes_release(gid)


def spacing_km(info, lats):
    """(dy_km, {lat: dx_km}) for the given latitudes."""
    dlat, dlon = info["dlat"], info["dlon"]
    if not np.isfinite(dlat):     # derive from the corners when absent
        dlat = abs(info["latitudeOfLastGridPointInDegrees"]
                   - info["latitudeOfFirstGridPointInDegrees"]) / (info["Nj"] - 1)
    if not np.isfinite(dlon):
        dlon = abs(info["longitudeOfLastGridPointInDegrees"]
                   - info["longitudeOfFirstGridPointInDegrees"]) / (info["Ni"] - 1)
    deg_km = np.pi * R_EARTH_KM / 180.0
    return dlat * deg_km, {lat: dlon * deg_km * np.cos(np.radians(lat))
                           for lat in lats}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="GRIB files or glob patterns")
    parser.add_argument("--lats", type=float, nargs="+",
                        default=[10, 20, 30, 40, 50],
                        help="latitudes at which to report east-west spacing")
    args = parser.parse_args(argv)

    files = [Path(p) for pattern in args.paths
             for p in sorted(glob.glob(pattern, recursive=True))]
    if not files:
        sys.exit(f"no files matched: {args.paths}")
    print(f"{'file':<48} {'Ni x Nj':>12} {'d(deg)':>8} {'dy km':>7} "
          + " ".join(f"{f'dx@{int(l)}N':>9}" for l in args.lats))
    for path in files:
        info = grid_info(path)
        dy, dx = spacing_km(info, args.lats)
        print(f"{path.name[:48]:<48} {info['Ni']:>5} x {info['Nj']:<4} "
              f"{info['dlon']:>8.4f} {dy:>7.2f} "
              + " ".join(f"{dx[l]:>9.2f}" for l in args.lats))
    print("\nA common verification grid should be no finer than the coarsest "
          "input's spacing above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
