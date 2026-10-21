"""Single entry point for the HAFS/WoFS rainfall verification framework.

    python analysis/run.py <case.yaml> <command>

Loads a per-deployment case (deployment_case.py) and runs one command:
  build-grid    build this deployment's verification grid: the WoFS domain
                read from the WoFS files and coarsened to grid.res_km (6 km),
                written to <out_dir>/<case>_grid.json with the MET grid spec,
                plus a sanity map (grid outline, best track by ATCF status)
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
                stats CSVs (needs both products regridded)

The masking, window-listing and HAFS commands return with the plan steps that
rebuild them on this case format (see docs/analysis_plan.md, phases C and D).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = ("build-grid", "download-obs", "regrid-obs", "plot-regrid",
            "stats-regrid")


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


def dispatch(case, command):
    """Run one command for a loaded DeploymentCase."""
    if command == "build-grid":
        from verification_grid import build_grid_case
        return build_grid_case(case)
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
    yaml_path, command = parse_args(argv)
    import deployment_case
    import run_log
    case = deployment_case.from_yaml(yaml_path)
    with run_log.tee(case, command):
        dispatch(case, command)


if __name__ == "__main__":
    main(sys.argv[1:])
