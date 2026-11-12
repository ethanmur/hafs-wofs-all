# HAFS / WoFS precipitation verification — condensed plan

One bullet per bullet or step of [analysis_plan.md](analysis_plan.md), for a quick
view of the pipeline and what is left. **Kept in sync with the full plan: every
change to `analysis_plan.md` is made here too.** Detail, numbers and caveats live
in the full plan; ✅ done, 🔧 built but unchecked on real data, ⏳ to do.

## Next up (Phase F work queue, in order)

1. 🔧 `plot-swaths` built: WoFS (blue, left) vs HAFS (vermillion, right, HFSA+HFSB pooled), percent of members at or above `--thresh-mm` (default 25 mm / 3 h), counts in the subtitle; run on Orion and check.
2. 🔧 0-3 h HAFS period: built (`hafs.include_spinup` / `--include-spinup`, `--first-bucket exclude|include`, `--matched fixed-spinup`, `--spinup-only` eyeball flag); run on Orion, default stays exclude until you see the plots.
3. ⏳ WoFS and HAFS ensembles (HFSA, HFSB, all HAFS): ensemble mean and PMM scored like members, then paired comparison with members.
4. ⏳ Inland vs landfalling verification: you classify domains/windows from the storm-location tables, then repeat the statistics per class.
- ⏳ G3 stratification (storm distance): step 1 (`storm-location` printout) done; check all 11 domains, then the stratification.
- 🔧 `plot-tc-location`: built on synthetic data; check on the real domains (`plot-hafs-tracks` is confirmed).

## Settled design decisions

- ✅ One grid per WoFS deployment; the unit of work is (storm, deployment).
- ✅ The WoFS native domain is the verification domain (900 x 900 km, WoFS projection), all rainfall in the box.
- ✅ Resolution is configurable (6 km default, 3 km used for the statistics), built by block-averaging the 3 km WoFS grid.
- ✅ No track swath; storm proximity is a per-window diagnostic, not a mask (G3).
- ✅ Observation validity mask (150 km US-land buffer, `--coast-km` for sensitivity) excludes cells with no truth.
- ✅ HAFS cycle range (`init_start`/`init_end`) is independent of the scored period (`valid_start`/`valid_end`).
- ✅ 3 h accumulation windows on an absolute 00/03/06Z clock, not init-relative.
- ✅ One comparison scope, WoFS vs HAFS vs obs, on the same cells and windows; individual runs first, ensembles after.
- ✅ Flat `data/` and `figures/` folders by kind, case in the file name, no `output/` tree.
- ✅ Simultaneous deployments of a storm are separate cases.
- ✅ Data gaps are recorded, not assumed away (manifest reasons `short_run`, `missing_steps`, `no_files`).
- ✅ HAFS-A and HAFS-B are separate models, never pooled by default.
- ✅ Truth is MRMS at 3 km, Stage IV the backup, switched by `truth_source`.
- ✅ The resolution advantage is reported, not hidden (FSS across scales).
- ✅ HAFS 0-3 h spin-up bucket excluded (decided 2026-11-03); WoFS at the window start is kept (lead 0-3 h); 0-3 h toggle built (`--first-bucket`, default exclude); default to be re-decided after the example plots.

### Open items

- ⏳ Idalia has no complete WoFS window; it stays in the inventory as unavailable.
- ✅ Truth choice settled: MRMS, with Stage IV as backup and sensitivity check.
- ⏳ Whether `stats-regrid` earns its place after the Stage IV check (decided in H1).
- ⏳ Which WoFS members to verify and how to label them (D1b).

### Deferred

- ⏳ Probability-matched means and ensemble verification (now queued: item 3 above; probabilistic scores stay in G2).
- ⏳ Storm-proximity stratification (now G3 in Phase F, To Complete).
- ⏳ Object-based verification (MODE/MTD).
- ⏳ TC-RMW storm-relative composites.
- ⏳ Track and intensity error (TC-Pairs).
- ⏳ Data-derived validity mask for inland radar gaps and coastal outages.

## Phase A — Domains and the verification grid

- ✅ A0 Delete the best-track grid machinery and HAFS-only products.
- ✅ A1 WoFS deployment inventory (`wofs_inventory.py`): deployments, grids, runs, window availability.
- ✅ A1b HAFS cycle inventory (`hafs_inventory.py`): cycles per storm and their forecast hours.
- ✅ A2 Grid builder (`verification_grid.py`): grid read from the WoFS files, `build-grid`.
- ✅ A3 Grid sanity plot: domain outline, best track for context.
- ✅ A4 Grid-spacing check: true ground spacing via `pyproj.Geod`.

## Phase B — Observations on the deployment grid

- ✅ B0 Delete AORC and the per-case regrid paths.
- ✅ B1 Cache layout: obs regrids in one folder per deployment (`<deployment>_<storm>`).
- ✅ B2 Regrid the truth product to the deployment grid (`regrid-obs`, MET).
- ✅ B3 Visual QC of the regridded obs (`plot-regrid`, `stats-regrid`).

## Phase C — Observation-validity mask

- ✅ C0 Delete the swath masks, WoFS boxes and the old grid configuration.
- ✅ C1 Validity mask (`build-mask`): US-land buffer, written as NetCDF with MET version attribute.

## Phase D — Forecast-side 3 h accumulations

- ✅ D0 Delete the old HAFS case, window and plot modules.
- ✅ D1 Accumulation convention: HAFS accumulations from APCP buckets; WoFS `rain` is a 5 min accumulation in inches.
- ✅ D1b WoFS member labeling: each member is its own run (`m00`...`m17`).
- ✅ D2 Absolute-clock windows (`accum_windows.py`): `clock_windows`, whole-minute arithmetic.
- ✅ D3 WoFS 1 h and 3 h accumulations (`regrid-wofs`): sum natively at 3 km, regrid with MET.
- ✅ D4 HAFS 3 h windows (`regrid-hafs`): a cycle counts only if initialized strictly before the window start.
- ✅ D5 Obs 3 h windows (`window-obs`): hourly truth summed into the same clock windows.
- ✅ D5b Eyeball panels WoFS | HAFS | truth (`plot-eyeball`, `plot-hafs-check`) with `--members --windows --cycles --cycle-every-h --domain`.
- ✅ D6 Window manifest (`build-manifest`): every (window, run) and what is on disk.

## Phase E — Statistics

- ✅ E0 Delete remaining swath/radius references.
- ✅ E0b Mask check plots (`plot-mask-check`): kept / discarded / gap windows.
- ✅ E1 `grid_stat` configuration and one-window test (`grid-stat-test`): 11 thresholds, 5 neighborhood widths, all and wet cells.
- ✅ E2 `grid-stat` over every window and run, resumable, parallel, with a runs CSV.
- ✅ E3 `parse-stat`: one tidy CSV per line type per case, times from the MODEL tag and folder.
- ✅ E4 Pooling (`pool_stats.py`): sum counts and partial sums, then derive scores; levels case / storm / all; `--matched windows|leads|fixed`, `--pool-leads`; Beryl 20240710_d1 excluded by default.
- ✅ E4b Uncertainty (`bootstrap_stats.py`): window bootstrap intervals, paired differences, leave-one-storm-out.
- ⏳ E5 Accumulation diagnostics: Python distribution and Q-Q summaries of the 3 h fields.

## Phase F — Figures

### Completed

- ✅ Eyeball panels WoFS | HAFS | truth, with `--domain wide|wofs`.
- ✅ Validity-mask check and HAFS regrid check plots.
- ✅ Score figures (`plot_stats.py`): lead and threshold curves, RMSE, performance diagrams; `--no-bands`, `--max-thresh 75`, shared y-limits per score.
- ✅ All-available-data set of the same figures (`--sample-set all|matched|both`, `_allwindows` names).
- ✅ Samples documented: lead 6 = 12 windows in 7 domains; fixed lead curves = 11; all leads matched = 43; unmatched = 47 windows in 10 cases.
- ✅ Beryl 20240710_d1 excluded from all pooled tables (domain over Iowa, not Beryl-related).
- ✅ `plot-hafs-tracks` with the best track over the HAFS tracks and final-position circles on top (confirmed on real data).

### To Complete

- 🔧 G3 storm-location stratification: `storm-location` printout of distances to the TC and to landfall per window; stratification pending.
- 🔧 `plot-tc-location` per domain and window (a-decks via `fetch_bdecks.py --adeck`), to be checked on real data.
- ⏳ Work queue (see Next up): 0-3 h period 🔧 built (in progress); swath maps, ensembles and inland vs landfalling to do.

### To Consider

- ⏳ Twelve initial findings from the pooled and bootstrap tables (ETS within noise, WoFS light-rain under-forecast, WoFS lead drift, HAFS lead decay mostly sampling, storm dependence, member spread, continuous-error gap, HFSB mid-lead edge, useful FSS scale, wet-storm dominance, thin 100-150 mm, data checks).
- ⏳ Suggested plots by finding: forest plot of paired differences, performance diagram with WoFS lead trajectory, fixed-window lead curves, FSS vs scale with the useful line, per-storm dot plot, per-member strip plot, leave-one-storm-out dumbbell, FSS-difference heatmap, coverage matrix.
- ⏳ Earlier F1-F4: FSS vs scale per lead, ETS/CSI vs threshold per lead, lead series of bias/RMSE/FSS, spatial panels with differences (obs/WoFS/HAFS part done).

## Phase G — Later analyses

- ⏳ G1 Probability-matched mean of WoFS members (queued in the work queue, item 3).
- ⏳ G2 WoFS and lagged HAFS ensembles; deterministic part queued (item 3), probabilistic scores (`ensemble_stat`) remain.
- 🔧 G3 Storm-location stratification: moved to Phase F, To Complete.
- ⏳ G4 Stage IV sensitivity: repeat E on `truth_source: stage4`.
- ⏳ G5 Mixing HAFS-A and HAFS-B as an explicit experiment.

## Phase H — Final cleanup

- ⏳ H1 Delete dead code and configs; re-decide `obs_regrid_stats`.
- ⏳ H2 Prune `environment.yml`, `requirements.txt` and the Orion job scripts.
- ⏳ H3 Rewrite `README.md` and make the plan match what was built.

## MET programs

- ✅ Used: `regrid_data_plane`, `grid_stat`, `gen_vx_mask`, `plot_data_plane` for QC; pooling is done in Python, so `stat_analysis` is not needed.
- ⏳ Not used: `MODE`/`MTD`, `TC-RMW`, `TC-Pairs`/`TC-Stat`, `Grid-Diag`, `Series-Analysis`, `Point-Stat`, `Ensemble-Stat`/`gen_ens_prod` (the last two gated on ensemble work).
