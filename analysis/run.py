"""Single entry point for the HAFS/WoFS rainfall verification framework.

    python analysis/run.py <case.yaml> <command>

Loads an ObsCase from the YAML case file and runs one command:
  download-obs  fetch/cache MRMS and AORC obs only -- no regridding or
                plotting; run this on a login node. Stage IV hourly data is
                NOT fetched here (see analysis/stage4_hourly.py); place it
                in the cache yourself
  regrid-obs    regrid every cached obs hour onto the regrid.grid_template
                grid with MET regrid_data_plane, cached as NetCDF, plus a
                per-hour conservation check CSV; needs `module load met`
                (needs a `regrid:` block)
  plot-regrid   hourly maps of the regrid-obs output: native vs regridded
                (full + zoom domain) for each regridded product, plus, when
                two or more are regridded, all products side by side and
                anomalies vs AORC; `regrid_plots.panels` restricts which.
                Reads the caches only
  stats-regrid  distributions and cell-by-cell 1:1 comparisons of the
                regridded products, per hour and over the whole window,
                land-only and including ocean, plus stats CSVs
  build-grid    build this deployment's verification grid: the WoFS domain
                read from the WoFS files and coarsened to grid.res_km (6 km),
                written to <out_dir>/<case>_grid.json with the MET grid spec,
                plus a sanity map (grid outline, best track by ATCF status).
                Takes a per-deployment case YAML; the commands below still
                read the older case format until their plan steps are done
  build-masks   every Phase C region on this case's grid: observation
                validity (US land plus a fixed buffer beyond it),
                the track swath, the WoFS boxes, and their intersection.
                Writes <out_dir>/<case>_mask_*.nc plus two figures; needs
                build-grid to have run
  list-windows  the 3 h absolute-clock verification windows for this case and
                the HAFS/WoFS cycles that fully span each one, with leads;
                reads no data, so it can be run before anything is staged
  regrid-hafs   the HAFS 3 h APCP buckets for every window a cycle fully
                spans, read straight from the parent files and put on the
                case grid in one pass; writes a window manifest CSV
                and, unless hafs.budget_check is false, a mass check
  plot-hafs     Stage IV vs HAFS 3 h precipitation, one panel per lead for
                each 00Z cycle (hafs_plots.init_hours to change), zoomed to
                the WoFS deployment nearest in time; needs regrid-hafs and
                regrid-obs to have run
  plot-masked   the regridded obs clipped to the mask intersection, one map
                per hour, with the three region boundaries drawn over it;
                needs regrid-obs and build-masks to have run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("download-obs", "regrid-obs", "plot-regrid", "stats-regrid",
            "build-grid", "build-masks", "plot-masked", "list-windows",
            "regrid-hafs", "plot-hafs")


def parse_args(argv):
    """(yaml_path, command) from argv; the command is required."""
    if len(argv) < 2:
        print("usage: run.py <case.yaml> <command>\n"
              f"commands: {' | '.join(COMMANDS)}")
        raise SystemExit(2)
    yaml_path, command = argv[0], argv[1]
    if command not in COMMANDS:
        print(f"unknown command '{command}'; choose from {COMMANDS}")
        raise SystemExit(2)
    return yaml_path, command


def _run_obs(obs_cases, obs_case, command):
    if command == "plot-regrid":
        from obs_regrid_plots import plot_regrid
        plot_regrid(obs_case)
        return
    if command == "stats-regrid":
        from obs_regrid_stats import stats_regrid
        stats_regrid(obs_case)
        return
    if command == "plot-hafs":
        from hafs_obs_plots import plot_hafs_obs_case
        plot_hafs_obs_case(obs_case)
        return
    if command == "regrid-hafs":
        from hafs_windows import regrid_hafs_case
        regrid_hafs_case(obs_case)
        return
    if command == "list-windows":
        from accum_windows import print_windows
        print_windows(obs_case)
        return
    if command == "plot-masked":
        from obs_regrid_plots import plot_masked
        plot_masked(obs_case)
        return
    if command == "build-masks":
        from masks import build_masks_case
        build_masks_case(obs_case)
        return
    {"download-obs": obs_cases.download_obs,
     "regrid-obs": obs_cases.regrid_obs}[command](obs_case)


def main(argv):
    yaml_path, command = parse_args(argv)
    import run_log
    if command == "build-grid":
        import deployment_case
        from verification_grid import build_grid_case
        case = deployment_case.from_yaml(yaml_path)
        with run_log.tee(case, command):
            build_grid_case(case)
        return
    import obs_cases
    obs_case = obs_cases.from_yaml(yaml_path)
    with run_log.tee(obs_case, command):
        _run_obs(obs_cases, obs_case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
