#!/usr/bin/env python3
"""Download the NHC b-decks for a list of storms, skipping those on disk.

Storms may come from an explicit list (--storm-list / --storms) or, with
neither given, from the subdirectory names of the HAFS run directory. Any of
these spellings is accepted for a storm:

    2024-02l        HAFS run-directory style
    AL022024        ATCF storm id
    bal022024.dat   b-deck filename

all of which map to b-deck bal022024.dat in the NHC ATCF archive for that
year. Files are fetched gzipped and written out decompressed, since that is
what analysis/best_track.py reads.

After the fetch, every b-deck on disk is summarised -- fix count, first and
last valid time, lat/lon extent, and the status codes present -- so the track
window feeding the verification grid can be eyeballed before use.

Usage:
    python analysis/fetch_bdecks.py                      # scan --hafs-dir
    python analysis/fetch_bdecks.py --storms 2024-02l,2024-09l
    python analysis/fetch_bdecks.py --storm-list storms.txt
    python analysis/fetch_bdecks.py --inventory          # report only
    python analysis/fetch_bdecks.py --dry-run
    python analysis/fetch_bdecks.py --hafs-dir DIR --bdeck-dir DIR
"""

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from best_track import bdeck_summary

HAFS_DIR = Path("/work2/noaa/aoml-hafs1/emurray/ppgc-hafs/data/hafs")
BDECK_DIR = Path("/work2/noaa/aoml-hafs1/emurray/ppgc-hafs/data/bdeck")
ARCHIVE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/{name}.gz"
BASINS = {"l": "al", "e": "ep", "c": "cp", "w": "wp"}
BASIN_IDS = {v: v for v in BASINS.values()}
TIMEOUT_S = 60


def bdeck_name(storm):
    """Storm identifier -> b-deck filename.

    Accepts '2023-10l', 'AL102023' and 'bal102023.dat' alike, so a storm list
    can be written in whichever style is already to hand.
    """
    token = storm.strip().lower()
    if token.endswith(".dat"):
        token = token[:-4]
    if token.startswith("b") and len(token) == 9 and token[1:3] in BASIN_IDS \
            and token[3:].isdigit():                      # bal102023
        return f"{token}.dat"
    if len(token) == 8 and token[:2] in BASIN_IDS and token[2:].isdigit():
        return f"b{token[:2]}{token[2:4]}{token[4:]}.dat"  # al102023
    year, _, tag = token.partition("-")                    # 2023-10l
    if year.isdigit() and len(year) == 4 and len(tag) >= 2 \
            and tag[-1] in BASINS and tag[:-1].isdigit():
        return f"b{BASINS[tag[-1]]}{int(tag[:-1]):02d}{year}.dat"
    raise ValueError(f"unrecognized storm identifier {storm!r} (expected e.g. "
                     "2023-10l, AL102023 or bal102023.dat)")


def bdeck_year(name):
    """'bal022024.dat' -> '2024', the ATCF archive subdirectory."""
    return name[5:9]


def read_storm_list(path):
    """Storm identifiers from a text file, one per line; # comments ignored."""
    storms = []
    for raw in Path(path).read_text().splitlines():
        token = raw.split("#", 1)[0].strip()
        if token:
            storms.append(token)
    if not storms:
        raise ValueError(f"no storms listed in {path}")
    return storms


def print_inventory(bdeck_dir, storms):
    """Summarise each storm's b-deck: extent, status codes, fix count."""
    print(f"\n{'storm':<14}{'b-deck':<16}{'fixes':>6}  "
          f"{'first':<15}{'last':<15}{'lat':<13}{'lon':<17}statuses")
    for storm in storms:
        try:
            name = bdeck_name(storm)
        except ValueError:
            continue
        path = bdeck_dir / name
        if not path.exists():
            print(f"  {storm:<12}{name:<16}{'-':>6}  (not on disk)")
            continue
        try:
            info = bdeck_summary(path)
        except (OSError, ValueError) as err:
            print(f"  {storm:<12}{name:<16}{'-':>6}  UNREADABLE: {err}")
            continue
        print(f"  {storm:<12}{name:<16}{info['n']:>6}  "
              f"{info['first']:%Y-%m-%d %HZ} {info['last']:%Y-%m-%d %HZ} "
              f"{info['lat_min']:5.1f}-{info['lat_max']:5.1f}  "
              f"{info['lon_min']:7.1f}-{info['lon_max']:7.1f}  "
              f"{','.join(info['statuses'])}")


def download_bdeck(url, dest):
    """Fetch a gzipped b-deck and write it out decompressed."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        payload = gzip.decompress(response.read())
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.rename(dest)   # never leave a half-written b-deck behind
    return len(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hafs-dir", type=Path, default=HAFS_DIR,
                        help="directory of <year>-<storm> run folders")
    parser.add_argument("--bdeck-dir", type=Path, default=BDECK_DIR,
                        help="where b-decks are kept")
    parser.add_argument("--storms", help="comma-separated storm identifiers, "
                        "e.g. 2024-02l,AL092024")
    parser.add_argument("--storm-list", type=Path,
                        help="file of storm identifiers, one per line")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing without downloading")
    parser.add_argument("--inventory", action="store_true",
                        help="only summarise the b-decks already on disk")
    args = parser.parse_args(argv)

    if args.storms:
        storms = [s for s in args.storms.split(",") if s.strip()]
        source = "--storms"
    elif args.storm_list:
        try:
            storms = read_storm_list(args.storm_list)
        except (OSError, ValueError) as err:
            sys.exit(f"ERROR: {err}")
        source = str(args.storm_list)
    else:
        if not args.hafs_dir.is_dir():
            sys.exit(f"ERROR: no such HAFS directory: {args.hafs_dir}")
        storms = sorted(p.name for p in args.hafs_dir.iterdir() if p.is_dir())
        if not storms:
            sys.exit(f"ERROR: no storm directories under {args.hafs_dir}")
        source = str(args.hafs_dir)

    print(f"Storms ({len(storms)}) from {source}: {', '.join(storms)}")
    print(f"B-deck dir:  {args.bdeck_dir}\n")

    if args.inventory:
        print_inventory(args.bdeck_dir, storms)
        return 0

    if not args.dry_run:
        args.bdeck_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    fetched = skipped = 0
    for storm in storms:
        try:
            name = bdeck_name(storm)
        except ValueError as err:
            print(f"  {storm:<12} SKIPPED  {err}")
            failed.append(storm)
            continue
        dest = args.bdeck_dir / name
        if dest.exists():
            skipped += 1
            print(f"  {storm:<12} {name:<16} already present")
            continue
        if args.dry_run:
            print(f"  {storm:<12} {name:<16} MISSING (would download)")
            continue
        url = ARCHIVE_URL.format(year=bdeck_year(name), name=name)
        try:
            size = download_bdeck(url, dest)
            fetched += 1
            print(f"  {storm:<12} {name:<16} downloaded ({size:,} bytes)")
        except (urllib.error.URLError, OSError, EOFError) as err:
            failed.append(storm)
            print(f"  {storm:<12} {name:<16} FAILED: {err}")
            print(f"               tried {url}")

    print(f"\n{fetched} downloaded, {skipped} already present, "
          f"{len(failed)} failed")
    if not args.dry_run:
        print_inventory(args.bdeck_dir, storms)
    if failed:
        print("Failed storms: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
