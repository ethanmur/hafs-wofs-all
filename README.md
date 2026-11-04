# HAFS & WoFS rainfall verification

Case-driven tools for verifying 3 h tropical-cyclone rainfall from HAFS (HFSA,
HFSB cycles) and the WoFS 18-member ensemble against MRMS (Stage IV is the
sensitivity product). Each WoFS deployment is one case: every product is
regridded to that deployment's 900 x 900 km domain on a common grid (3 km is
the working resolution; 6 km is the default and `--res-km N` changes it), and
scoring uses absolute-clock 3 h windows. `docs/analysis_plan.md` is the
authoritative design, phase by phase.

## Setup

On Orion or Hercules:

```bash
module load miniconda3
conda env create -f environment.yml
conda activate hafs
```

The main dependencies are NumPy, SciPy, PyYAML, boto3, cfgrib, ecCodes,
xarray, Matplotlib, Cartopy, Seaborn, and Pillow. `environment.yml` is preferred because it
installs the required native GRIB and mapping libraries.

## Run one deployment

One YAML per deployment lives in `storms/` (regenerate with
`analysis/make_case_yamls.py`; do not edit scoring windows by hand). Run the
steps in order; global flags `--res-km N`, `--sources stage4 mrms` and `--coast-km N`
(mask buffer, 150 km by default) apply to any of them.

```bash
python analysis/run.py storms/<case>.yaml build-grid       # verification grid
python analysis/run.py storms/<case>.yaml build-mask       # observation-validity mask
python analysis/run.py storms/<case>.yaml build-manifest   # windows and what is on disk
python analysis/run.py storms/<case>.yaml regrid-wofs      # WoFS 1 h / 3 h on the grid
python analysis/run.py storms/<case>.yaml regrid-hafs      # HAFS 3 h buckets on the grid
python analysis/run.py storms/<case>.yaml download-obs     # login node: fetch MRMS
python analysis/run.py storms/<case>.yaml regrid-obs       # hourly obs on the grid
python analysis/run.py storms/<case>.yaml window-obs       # obs 3 h windows
python analysis/run.py storms/<case>.yaml plot-eyeball     # WoFS | HAFS | truth panels
python analysis/run.py storms/<case>.yaml plot-hafs-check  # native vs regridded HAFS
python analysis/run.py storms/<case>.yaml plot-mask-check    # validity mask vs truth windows
python analysis/run.py storms/<case>.yaml grid-stat-test     # E1: grid_stat on one window
python analysis/run.py storms/<case>.yaml grid-stat          # E2: every window, every run (one SLURM job per deployment)
python analysis/run.py storms/<case>.yaml parse-stat         # E3: .stat files -> data/grid-stat/csvs/stat_<linetype>_<case>.csv
```

`plot-regrid` and `stats-regrid` are the obs-product checks. The regrid steps
need `module load met` (and eccodes for `regrid-hafs`). Outputs go to flat
`data/<kind>/` and `figures/<kind>/` folders with the case in the file name; a
non-default resolution adds `_<N>km`. Large model data and generated products
are excluded from Git.

## Tests

```bash
python -m pytest analysis/tests -q
```

Tests use synthetic fields and stand-ins for MET, so they need no HPC archive or
downloads.

## Repository layout

```text
analysis/
  run.py              command dispatcher
  deployment_case.py  per-deployment YAML loading and every output path
  make_case_yamls.py  writes the case YAMLs from the inventories
  wofs_inventory.py   WoFS archive inventory (A1)
  hafs_inventory.py   HAFS cycle inventory (A1b)
  accum_windows.py    absolute-clock 3 h windows
  verification_grid.py grid builder       masks.py   observation-validity mask
  window_manifest.py  window manifest (D6)
  wofs_windows.py     WoFS 1 h / 3 h on the grid (D3)
  hafs_windows.py     HAFS 3 h on the grid (D4)
  obs_cases.py        obs download, cache and MET regrid
  obs_windows.py      obs 3 h windows (D5)
  eyeball_plots.py    eyeball and HAFS check panels (D5b)
  mask_check.py       validity mask vs truth windows (E0b)
  grid_stat_windows.py grid_stat configuration, one-window test (E1), all-window run (E2)
  stat_tables.py       .stat files -> one tidy CSV per line type per case (E3)
  obs_regrid_plots.py / obs_regrid_stats.py / plot_common.py / plot_units.py
  met_regrid.py       MET regrid_data_plane wrapper and budget check
  stage4_hourly.py    Stage IV hourly access
  best_track.py       NHC b-deck parsing
  fetch_bdecks.py     b-deck downloader        grid_spacing.py  GRIB spacing check
  tests/              unit and plotting tests
storms/               one YAML per deployment
docs/analysis_plan.md the plan
```
