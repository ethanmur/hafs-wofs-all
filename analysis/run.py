"""Single entry point for the HAFS/WoFS rainfall verification framework.

    python analysis/run.py <case.yaml> <command> [--res-km N]

Loads a per-deployment case (deployment_case.py) and runs one command:
  build-grid    build this deployment's verification grid: the WoFS domain
                read from the WoFS files and coarsened to grid.res_km (6 km),
                written to <grid_json_dir>/<case>_grid.json with the MET grid
                spec, plus a sanity map in grid_plot_dir (grid outline, best
                track by ATCF status)
  build-mask    the observation-validity mask on that grid: where Stage IV /
                MRMS can be trusted (US land plus 150 km, detached land cut),
                written as a MET NetCDF to <mask_dir> with a map in
                <mask_plot_dir>; needs build-grid to have run
  build-manifest
                every 3 h window of the scored period and what is on disk for
                it: each WoFS init's status, each expected HAFS cycle's status
                (including cycles HAFS never ran), and per window whether it is
                testable, wofs_only or no_wofs. Writes two CSVs to
                <manifest_dir>; reads only directory listings and file headers
  regrid-wofs   for every complete (WoFS init, window): sum the 36 five-minute
                files, convert in -> mm, coarsen to the verification grid, and
                write one NetCDF per (init, window) with all members to
                <wofs_regrid_dir>; needs build-grid to have run
  download-obs  fetch/cache MRMS only -- no regridding or plotting; run this
                on a login node. Stage IV hourly data is NOT fetched here (see
                analysis/stage4_hourly.py); place it in the cache yourself
  regrid-obs    regrid every cached obs hour in the scored period onto the
                deployment grid with MET regrid_data_plane, cached as NetCDF,
                plus a per-hour conservation check CSV; needs build-grid to
                have run and `module load met`. Regrids the truth product, or
                both products if regrid.sources lists them
  plot-regrid   hourly maps of the regrid-obs output: native vs regridded for
                each product and, with both regridded, the products side by
                side and the anomaly against the truth product. Reads the
                caches only
  stats-regrid  distributions and cell-by-cell 1:1 comparison of the two
                regridded products, per hour and over the whole period, plus
                stats CSVs (needs both products regridded); adds a variant
                restricted to the validity mask when build-mask has run

Resolution is 6 km unless a case sets grid.res_km or the run passes --res-km;
only a non-default resolution adds its label to output names, so the default
run's files never move.

The HAFS window regrid and the 3 h truth windows follow in the rest of plan
phase D (see docs/analysis_plan.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("build-grid", "build-mask", "build-manifest", "regrid-wofs",
            "download-obs", "regrid-obs", "plot-regrid", "stats-regrid")


def parse_args(argv):
    """(yaml_path, command, res_km) from argv. The command is required;
    --res-km N overrides the case's grid.res_km (None keeps it)."""
    argv = list(argv)
    res_km = None
    if "--res-km" in argv:
        i = argv.index("--res-km")
        try:
            res_km = float(argv[i + 1])
        except (IndexError, ValueError):
            print("--res-km needs a number, e.g. --res-km 3")
            raise SystemExit(2)
        del argv[i:i + 2]
    if len(argv) < 2:
        print("usage: run.py <case.yaml> <command> [--res-km N]\n"
              f"commands: {' | '.join(COMMANDS)}")
        raise SystemExit(2)
    yaml_path, command = argv[0], argv[1]
    if command not in COMMANDS:
        print(f"unknown command '{command}'; choose from {COMMANDS}")
        raise SystemExit(2)
    return yaml_path, command, res_km


def dispatch(case, command):
    """Run one command for a loaded DeploymentCase."""
    if command == "build-grid":
        from verification_grid import build_grid_case
        return build_grid_case(case)
    if command == "build-mask":
        from masks import build_mask_case
        return build_mask_case(case)
    if command == "build-manifest":
        from window_manifest import build_manifest_case
        return build_manifest_case(case)
    if command == "regrid-wofs":
        from wofs_windows import regrid_wofs_case
        return regrid_wofs_case(case)
    if command == "download-obs":
        import obs_cases
        return obs_cases.download_obs(case)
    if command == "regrid-obs":
        import obs_cases
        return obs_cases.regrid_obs(case)
    if command == "plot-regrid":
        from obs_regrid_plots import plot_regrid
        return plot_regrid(case)
    from obs_regrid_stats import stats_regrid
    return stats_regrid(case)


def main(argv):
    yaml_path, command, res_km = parse_args(argv)
    import deployment_case
    import run_log
    case = deployment_case.from_yaml(yaml_path, res_km=res_km)
    with run_log.tee(case, command):
        dispatch(case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
