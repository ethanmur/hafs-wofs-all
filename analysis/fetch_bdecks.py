#!/usr/bin/env python3
"""Download the NHC b-decks for whichever storms are present in the HAFS
run directory, skipping the ones already on disk.

Storm directories are named <year>-<number><basin>, e.g. 2023-10l, which
maps to b-deck bal102023.dat in the NHC ATCF archive for that year. Files
are fetched gzipped and written out decompressed, since that is what
analysis/best_track.py reads.

Usage:
    python analysis/fetch_bdecks.py                 # fetch what's missing
    python analysis/fetch_bdecks.py --dry-run       # just report
    python analysis/fetch_bdecks.py --hafs-dir DIR --bdeck-dir DIR
"""

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from pathlib import Path

HAFS_DIR = Path("/work2/noaa/aoml-hafs1/emurray/ppgc-hafs/data/hafs")
BDECK_DIR = Path("/work2/noaa/aoml-hafs1/emurray/ppgc-hafs/data/bdeck")
ARCHIVE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/{name}.gz"
BASINS = {"l": "al", "e": "ep", "c": "cp", "w": "wp"}
TIMEOUT_S = 60


def bdeck_name(storm_dir):
    """'2023-10l' -> 'bal102023.dat'."""
    year, _, tag = storm_dir.partition("-")
    tag = tag.strip().lower()
    if not (year.isdigit() and len(year) == 4 and len(tag) >= 2
            and tag[-1] in BASINS and tag[:-1].isdigit()):
        raise ValueError(f"unrecognized storm directory name {storm_dir!r} "
                         "(expected e.g. 2023-10l)")
    return f"b{BASINS[tag[-1]]}{int(tag[:-1]):02d}{year}.dat"


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
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is missing without downloading")
    args = parser.parse_args(argv)

    if not args.hafs_dir.is_dir():
        sys.exit(f"ERROR: no such HAFS directory: {args.hafs_dir}")
    storms = sorted(p.name for p in args.hafs_dir.iterdir() if p.is_dir())
    if not storms:
        sys.exit(f"ERROR: no storm directories under {args.hafs_dir}")
    print(f"HAFS storms ({len(storms)}): {', '.join(storms)}")
    print(f"B-deck dir:  {args.bdeck_dir}\n")

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
        url = ARCHIVE_URL.format(year=storm.split("-")[0], name=name)
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
    if failed:
        print("Failed storms: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
