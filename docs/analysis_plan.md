# HAFS / WoFS precipitation verification — analysis plan

Verification of HAFS and WoFS rainfall forecasts against gridded observations
inside each WoFS deployment domain, for the 2023–2024 TC cases. Each step
below is sized to be implemented and checked in one sitting. "Check" is what to
look at before moving on. Steps named **Delete** remove code the new scope
makes obsolete; they are part of the plan, not an afterthought, and each one
ends with the test suite and the command list still passing.

Revised 2026-10-19 for the WoFS-domain scope (previous plan: best-track grid,
500 km swath, HAFS-vs-obs everywhere). The previous scope is preserved on the
`hafs-only` branch.

## Settled design decisions

- **One grid per WoFS deployment.** All members of a deployment share one
  domain, so they share one grid. A storm can have several deployments (WoFS is
  re-sited between events), so a storm can have several grids. The unit of work
  is (storm, deployment).
- **The WoFS native domain is the verification domain.** No best-track-derived
  grid, no pad, no Lambert parameters chosen by rule: the grid is read from the
  WoFS files and keeps their projection (900 x 900 km). The test region is the
  full grid, so the statistics measure WoFS and HAFS skill for *all* rainfall
  in the box, not only TC rainfall.
- **6 km resolution** (configurable), built by 2 x 2 aggregation of the native
  3 km WoFS grid, so cell edges coincide with WoFS cells. HAFS parent is
  6.67 km north-south and 5.1–5.8 km east-west at landfall latitudes, so 6 km
  asks the least of HAFS's interpolation. The upscaling is of WoFS (3 -> 6 km)
  and happens only inside the deployment domain.
- **No track swath.** The only mask is observation validity (below). Storm
  proximity is *not* a mask but a per-window diagnostic: distance of the NHC
  best-track center from the domain, whether it is inside, and the percent of
  domain cells within 500 km of it. Later analysis splits "mostly TC" from
  "little TC" windows on these columns; none of that splits the statistics now.
- **Observation validity mask.** Stage IV and MRMS have no truth offshore or
  where radar coverage fades, and those cells would count as zero rainfall.
  Cells outside the validity region are excluded from every statistic. The
  region is truth-dependent: the 150 km US-land buffer is the Stage IV rule,
  and an MRMS run may use a different one.
- **Cycle range is independent of the scoring window.** `init_start` /
  `init_end` select which HAFS cycles to pull, so old and recent cycles can be
  compared against the same WoFS output. `valid_start` / `valid_end` come from
  the deployment's active WoFS period and set what is scored. Nothing is scored
  outside it.
- **3 h accumulation windows on an absolute clock** (00/03/06...Z), not
  init-relative. WoFS cycles every 30 min, so init-relative windows would
  compare 1730–2030Z against 1800–2100Z. A run is used for a window if its span
  fully contains that window, whatever its init. Every comparison is between
  full 3 h chunks.
- **One comparison scope: WoFS vs HAFS vs obs**, over the same cells and the
  same windows. Individual runs first (one WoFS member against one HAFS cycle at
  the same valid window); probability-matched means and ensembles come after
  the individual-run pipeline is stable.
- **Simultaneous deployments are a feature.** A storm can have two WoFS
  domains running at once (separate `dN` runs, different grids), and each is
  verified on its own grid. The same valid window therefore appears once per
  deployment, and a HAFS cycle is regridded once per grid.
- **Data gaps are the norm, and are recorded, not assumed away.** The archive
  has runs of different lengths (3 h and 6 h), runs with holes, empty init
  folders, and one storm (Idalia) with a single file per init. Availability is
  always decided from the steps actually on disk, per (init, window), and every
  unavailable pair stays in the manifest with a reason (`short_run`,
  `missing_steps`, `no_files`) so it is visible instead of silently absent. A
  deployment with nothing to verify is reported, not dropped.
- **HAFS-A and HAFS-B are separate models**, never pooled, because their
  physics differ. A later HAFS "ensemble" is built from several cycles of one
  configuration. Mixing configurations is an experiment, not a default.
- **Truth product: Stage IV by default, switchable to MRMS** by one config key
  (`truth_source`) with no other change. The `noaa-mrms-pds` archive of
  `MultiSensor_QPE_01H_Pass2` begins 2020-10-14, so it covers every 2023–2024
  case; MRMS matters here because the small sample makes a second truth product
  a worthwhile sensitivity check. Only the truth product is regridded unless
  `regrid.sources` lists more.
- **Resolution advantage is reported, not hidden.** FSS across a range of
  neighborhood scales, since WoFS was run at 3 km and budget interpolation from
  HAFS's 6 km adds no information.

### Open items

- Idalia (2023) has one file per init and 11 empty init folders, so it has no
  complete 3 h WoFS window as downloaded. It stays in the inventory and
  manifest as unavailable; whether more files can be obtained is outside the
  code.
- Stage IV coverage over the mountain West (Hilary's three California and
  Nevada domains) is verified by hand; `truth_source: mrms` is the fallback.
- Whether AORC and the obs-vs-obs comparison plots stay. AORC is no longer a
  truth candidate; B3 and H1 decide whether the comparison tooling survives as
  QC or is deleted.
- Which WoFS members to verify and how to label them (Phase D1).

### Deferred

- Probability-matched means and ensemble verification (WoFS 18-member, lagged
  HAFS ensembles from several cycles of one configuration).
- Splitting statistics by storm proximity (the diagnostics are collected from
  the start; the analysis is later).
- Object-based verification (MODE/MTD), designed deliberately later.
- TC-RMW storm-relative composites.
- Track and intensity error (TC-Pairs).
- Data-derived obs validity mask for inland radar gaps and coastal radar
  outages during landfall (these appear as missing, not zero).

---

## Phase A — Deployment domains and the verification grid

### A0. Delete: best-track grid machinery and HAFS-only products
Remove what the new scope no longer calls. First confirm with an import graph
which helper functions in the modules below are still used by modules that
stay, and move those helpers rather than leaving a dependency on dead code.

- HAFS-only multi-cycle and comparison products: `cycles.py`,
  `cycles_compare.py`, `compare.py`, `parent_qpf.py`, `ets_full.py`,
  `ets_score.py`, `rmse_scatter.py`, `precip_structure.py`, `track_skill.py`.
- `skill_metrics.py` (swath and FSS helpers), once its remaining callers are
  known. Scores come from MET.
- `hafs_common.py`, keeping only what a surviving module imports.
- The matching tests, and the `run.py` subcommands `parent`, `ets`, `rmse`,
  `cycles`, `cycles-compare`, `all`, `compare`, `replot`.
- The HAFS-only configs in `storms/helene_only/`, and the old per-storm
  YAMLs (`build_grid_*`, `obs_regrid_plot_*`), replaced by the per-deployment
  files written in A1.
- Helpers that surviving modules used move out first: the US-geography overlay
  and QPF colormap to `plot_common.py`, the MRMS download functions to
  `obs_cases.py`.

*Check:* `pytest` passes; `run.py` with no command lists only the commands that remain;
no surviving module imports a deleted one.

### A1. WoFS deployment inventory (`wofs_inventory.py`)

    python analysis/wofs_inventory.py <wofs_root> --csv inventory.csv \
        --windows-csv windows.csv [--deep]

Walks `<root>/<year>_<STORM>/<YYYYMMDD>_d<N>/<HHMM>/wofs_ALL_<step>_...nc` and
reports one row per deployment folder: first and last init, last valid time,
init and step counts, the grid its files carry, and notes for anything odd.
Init and valid times come from the files' attributes; folder and file names are
only cross-checked, because a folder named for one date holds inits on the
next (a `0300` init in a folder dated the day before).

Facts established from the 2023–2024 archive:

- 12 deployment folders in 5 storms (Hilary, Idalia, Beryl, Helene, Milton),
  **12 distinct grids**: each deployment has its own center and corner. Shared
  across all: 300 x 300 cells at 3 km, Lambert conformal, standard parallels
  30/60, sphere radius 6371.2 km.
- One file is one 5 min step holding all 18 members. `rain` is the
  accumulation over the preceding 5 min, in inches; step 00 is exactly zero.
- `xlat`/`xlon` agree with the grid attributes to ~0.004 degrees, longitudes
  are east-negative. The grid is built from the attributes (A2); the arrays
  are only a check.
- Runs differ in length (37 or 73 steps, i.e. 3 h or 6 h, and shorter in
  Hilary), some have holes, some init folders are empty, and Idalia has a
  single file per init.

**Window availability table** (`--windows-csv`): one row per (init, 3 h clock
window) from the init to 6 h later, with a status judged from the exact steps
the window needs. The window (start, end] is the sum of steps `s+1 .. s+36`,
where `s` is the step at the window start, so step 00 is never needed:

| status | meaning |
|---|---|
| `complete` | all 36 steps are on disk |
| `missing_steps` | the run reaches the window but a step below its last file is absent |
| `short_run` | the run ends before the window does |
| `no_files` | the init folder is empty (init time inferred from the folder names, flagged `init_inferred`) |

This table, not a fixed "6 h run" constant, is the source for D2. The script
also prints a per-deployment summary (windows, windows with at least one
complete run, a flag for deployments with nothing to verify) and the pairs of
deployments that overlap in time.

**Per-deployment case YAMLs** (replace the per-storm files):

    python analysis/make_case_yamls.py wofs_inventory.csv wofs_windows.csv

One file per deployment in `storms/`, named
`<year>_<id>_<storm>_<deployment>.yaml`. The scoring window is not typed in: it
runs from the first complete 3 h window to the last, read from the window table
above. Each file holds the storm and HAFS id, the best-track path, `truth_source`,
the WoFS root with `available` and `complete_windows`, `valid_start` /
`valid_end`, and the HAFS block: `lookback_h: 48`, the `init_start` / `init_end`
range, and one root per configuration (`HFSA`, `HFSB`, each
`.../data/hafs/<id>/<config>/<YYYYMMDDHH>`). A deployment with no complete
window (Idalia) is still written, with `available: false` and no window.

HAFS members for a window are the cycles of one configuration initialized
no more than `lookback_h` (48) before the window start and strictly before it:
the cycle exactly 48 h back counts, the cycle initialized at the window start
does not (its first bucket is spin-up, not a forecast of that window).
Six-hourly cycles then give 8 members per configuration, 16 with HFSA and HFSB
kept as separate models, close to WoFS's 18 members. The case YAML only bounds
the cycle range; D2 picks the members per window.

### A1b. HAFS cycle inventory (`hafs_inventory.py`)

    python analysis/hafs_inventory.py <hafs_root> \
        --wofs-windows wofs_windows.csv --members-csv hafs_members.csv [--deep]

HAFS stops cycling a storm once it is over land or post-tropical, so the
members the 48 h rule asks for are often not on disk, especially for deployments
that run after landfall. This step makes that visible before anything is
regridded. It walks `<root>/<year>-<id>/<HFSA|HFSB>/<YYYYMMDDHH>/`, reports the
cycles and forecast hours per storm and configuration, and, for every complete
WoFS window, counts the HAFS members per configuration under the rule above.

A 3 h window is served by one file: the bucket is named for its end, so the
15-18Z accumulation from an 18Z-prior init is in the f024 file. A (cycle,
window) pair is `complete` when that file exists, `short_run` when the run ends
before it, `missing_file` when the run reaches it but the file is absent, and
`no_files` for an empty cycle folder. `--deep` lists the `tp` intervals in a few
files per configuration, to close D1's open question that the bucket is 3 h at
every lead.

Windows where neither configuration has a member are not dropped. They stay in
the manifest with `hafs_available: false` as **WoFS-only** windows: WoFS vs obs
can still be scored, but they are excluded from head-to-head WoFS/HAFS
comparison and pooling.

*Check:* the cycles on disk match the archive by eye; the per-deployment
summary shows, for each deployment, how many windows have both
configurations, one, or none.

*Check:* the table matches the archive by eye; a deployment with nothing
complete is flagged; the empty-folder date inference reproduces the dates the
files give for the non-empty inits.

### A2. Grid builder (`verification_grid.py`)

    python analysis/run.py <deployment>.yaml build-grid

Read the native grid (projection, corners, 3 km spacing) from a WoFS file,
aggregate 2 x 2 to `res_km` (6 km), write `grids/<deployment>_grid.json` with
the MET grid spec. Print the cell counts and which file defined the grid.

The plan's earlier precedence rules and the rule-derived Lambert parameters
are gone; there is nothing to choose. `domain_override` is gone with them.

*Lesson kept:* MET's grid-specification strings take longitudes
**east**-positive (negative for west) and negate them into its internal
west-positive form (`MET_LON_WEST_POSITIVE = False`). A west-positive spec
fails silently: the grid lands in the wrong hemisphere and every output cell
comes back missing, with no error. MET's debug output gives the check. It
prints the HRAP input grid as `Lon_orient: 105.000` (positive, for 105 W), so
positive means west, and opposite signs on two grids in the same hemisphere
means the spec is wrong.

*Lesson kept:* MET silently clamps the BUDGET width
(`RegridInfo::validate() -> Resetting the regridding width from 5 to 2`).
Tuning `width` upward for BUDGET does nothing.

*Check:* cell counts (150 x 150 expected for 900 km at 6 km); corners match the
WoFS file; a regrid of a test field produces no all-missing output.

*MET:* none for construction; the output is the grid spec consumed downstream
(`regrid_data_plane -to_grid`, `grid_stat` `regrid.to_grid`).

### A3. Grid sanity plot
One cartopy figure: domain outline, best-track fixes inside the valid window
colored by status, US geography, obs coverage extent. The track is context
here, not geometry.

*Check:* by eye — domain where expected, storm positions plausible relative to
it.

### A4. Grid-spacing verification
True ground spacing between adjacent grid points via `pyproj.Geod`: min/max
mean dx, dy and worst deviation from `res_km`. Native WoFS projection means
this should be small and stable; the check catches a misread projection.

*Check:* deviation of a few percent at most, the same for every deployment.

---

## Phase B — Observations on the deployment grid

### B0. Delete: AORC and per-case regrid paths (pending open item)
Remove the pieces the new truth model does not use: AORC as a regrid
candidate, `regrid.grid_template` pointing at the HAFS parent, and any default
that assumes one grid per storm. Delete `aorc_common.py` and its tests if the
decision on AORC is to drop it.

*Check:* `download-obs` and `regrid-obs` run for `stage4` and `mrms` only;
tests pass.

### B1. Repoint and clear the cache
Add the grid identity to the regridded cache path
(`<deployment>/<grid_name>_<res>km/`). Delete old regridded obs; **keep the
native downloads**. MRMS re-download is the expensive part and the files are
grid-independent.

### B2. Regrid the truth product to the deployment grid

    python analysis/run.py <deployment>.yaml regrid-obs

`truth_source` defaults to `stage4`; `regrid.sources: [stage4, mrms]`
regrids both for the sensitivity check. Every command tees its console output
to `<out_dir>/logs/<case>_<command>_<stamp>.log` (override with `log_dir:`),
flushed on every write, so a killed job still leaves a usable record.

*Check:* per-hour conservation CSV — domain-mean percent difference within a
few tenths of a percent.

*MET:* `regrid_data_plane` (BUDGET, width 2, `vld_thresh 0.5`).

### B3. Visual QC of the regridded obs
Run `plot-regrid` on the regridded truth. If a second product was regridded,
the product-vs-product map comes with it. Distribution and 1:1 comparisons
stay only if the AORC/obs-vs-obs decision keeps them.

*Check:* native-vs-regridded pairs look right; distributions have not shifted
in a way 6 km smoothing cannot explain.

*MET:* `plot_data_plane` for a quick raster check on a single NetCDF.

---

## Phase C — Validity region and storm-proximity diagnostics

### C0. Delete: track swath and WoFS-box masks
Remove `track_swath_mask`, `wofs_box_mask`, `WOFS_ANY`, `active_verify_mask`
and the multi-region composition in `masks.py`, with their tests and the
`build-masks` figures that drew them. The domain is the grid, so these are
empty operations.

*Check:* `masks.py` contains only the validity mask; tests pass.

### C1. Observation validity mask
Rasterize Natural Earth US-state polygons onto the deployment grid,
distance-transform, keep everything within 150 km of US land, then cut land
masses not connected to the US. The buffer reaches 150 km into Canada and
Mexico and stops, since radar range does not end at a political line but
coverage fades inland. The connectivity test (not a country list) keeps Cuba,
Hispaniola and the Bahamas out while keeping Long Island, the Keys and the
barrier islands in.

The rule is a function of `truth_source`, so MRMS can substitute its own
region without touching callers. The cell count of the valid fraction prints
once per deployment: a WoFS box can be mostly ocean, and `FULL` in Phase E is
`full domain ∩ obs_valid`, or the statistics are computed over cells with no
truth.

*Caveat:* the NetCDF's MET Lambert attributes are written from MET's
convention, east-positive. Verified readable with
`plot_data_plane <mask>.nc /tmp/m.ps 'name="obs_valid"; level="(*,*)";'`; the
fallback if MET rejects it is to write the field as GRIB2 through
`regrid_data_plane`.

*Check:* the valid fraction per deployment; map of the valid region.

*MET:* `gen_vx_mask` has no polygon-buffer type; Python is simpler and the
output feeds `grid_stat` natively.

### C2. Storm-proximity diagnostics
For each valid window, from the NHC best track (b-deck): distance from the TC
center to the domain center, a flag for whether the center is inside the
domain, and the percent of obs-valid domain cells within 500 km of the center.
The track is interpolated to the window midpoint; the fraction is computed in
the deployment's projected plane so 500 km is a true constant radius.

These are a diagnostic table consumed by D6. They never mask anything.

*Check:* storm clearly in the box -> high fraction, storm far away -> near 0;
the fraction varies smoothly as the storm crosses a deployment.

*MET:* none.

---

## Phase D — WoFS and forecast-side 3 h accumulations

### D0. Delete: HAFS case configuration and ATCF parsing for the old scope
Replace `hafs_case.py` (StormCase with `domain`, `mask_radius_km`,
per-initialization fields) by the per-deployment case from A1. Keep only the
ATCF/b-deck parsing that A3 and C2 use, and move it beside `best_track.py`.

*Check:* one case class; tests pass.

### D1. Accumulation convention

**HAFS (established).** HAFS writes *both* forms in the same file:

```
702: Total Precipitation:kg m**-2 (accum):surface:fcst time 21-24 hrs (accum):from 202308171800
703: Total Precipitation:kg m**-2 (accum):surface:fcst time  0-24 hrs (accum):from 202308171800
```

- `21-24` is a per-interval 3 h bucket — the one to use.
- `0-24` is the run total since init — the one to avoid.

Consequences: no `pcp_combine` for HAFS 3 h windows (the bucket already is the
window), and the buckets land on the absolute clock for free, because HAFS
initializes at 00/06/12/18Z, all multiples of 3. The selection must be by
interval, not by name, since both records are APCP at surface:

```
name="APCP"; level="A3";      # the 3 h bucket
name="APCP"; level="A24";     # the run total -- not this
```

Getting this wrong is silent: both decode fine, and a run-total field just
looks like a very wet forecast.

*Still to confirm:* that the bucket is 3 h at **every** lead. Some models emit
`0-1`/`0-2` style buckets at the first leads or widen the bucket past a certain
lead.

```bash
for f in <case>/*.parent.atm.f*.grb2; do
  echo -n "$(basename $f)  "
  grib_get -p stepRange -w shortName=tp $f | tr '\n' ' '; echo
done
```

Every line should show one bucket of width 3 ending at that lead, plus the
`0-N` run total. *Check:* at f024, the eight 3 h buckets sum to the `0-24`
field.

**WoFS (to establish from the data).** Record: precipitation variable and
units, since-init vs per-interval accumulation, output frequency (expected
5 min), cycle spacing (expected 30 min), run length, member count and naming.
Document the member selection rule (D1b) so it cannot be mistaken for
cherry-picking.

*Check:* a WoFS accumulation over one known 3 h window reproduces a hand
calculation from two raw output times.

### D1b. WoFS member labeling
Individual runs come first: each member is a separate "run" in the manifest
with its own id, so member -> ensemble is later a group-by, not a refactor.

### D2. Absolute-clock window definition (`accum_windows.py`)
Produces the list of 3 h windows spanning `valid_start` -> `valid_end` on the
00/03/06...Z clock and, for each model run, the lead offsets covering each
window. Drops windows a run cannot fully span. HAFS runs come from the nominal
cycle schedule and need no data. **WoFS runs come from the A1 availability
table**, not from a nominal 30 min schedule with a fixed 6 h length: only
`complete` (init, window) pairs are paired, and the others are carried forward
with their status for D6. The old `wofs_runs` constants (`WOFS_MAX_LEAD_MIN`,
cycle spacing) are replaced by what is on disk. **HAFS members per window come
from the A1b rule** (cycles within `lookback_h` before the window start, strictly
before it), checked against the cycles actually on disk; a window with no member
is carried forward as WoFS-only.

Three decisions worth recording:

- **Whole minutes everywhere.** WoFS cycles every 30 min and writes every
  5, so f02:30 is routine; in float hours "is this lead on the model's
  accumulation granularity" becomes a tolerance question instead of an exact
  one.
- **`step_min` is accumulation granularity, not output frequency.** For HAFS
  that is the 3 h bucket from D1 (180), not the hourly output — a 1 h slice of
  a 3 h bucket does not exist. This is what makes a 20–23Z window correctly
  unreachable from an 18Z HAFS init while 21–00Z is f003–f006. For WoFS it is
  the 5 min output step.
- **Lead is measured at the window end.** Defined once on `Coverage.lead_h`,
  so every lead-time plot means the same thing: how far out the forecast had
  run by the time the accumulation was complete.

Worked example for the 18–21Z window, WoFS 6 h runs at 30 min cycling:

| WoFS init | lead covering 18–21Z |
|---|---|
| 1500Z | f03–f06 |
| 1530Z | f02:30–f05:30 |
| 1600Z | f02–f05 |
| ... | ... |
| 1800Z | f00–f03 |

A 1430Z init ends at 2030Z (short of 21Z) and an 1830Z init starts after 18Z;
both are dropped. Every usable run carries its own `wofs_lead_h`, which gives
WoFS's internal lead-time dependence for free.

*Check:* unit tests — a 1730Z 6 h run yields exactly the 18–21Z window at
f00:30–f03:30; a 1430Z run yields none; HAFS bucket-granularity cases.

### D3. WoFS 3 h windows on the deployment grid
For each (member, init, window) the A1 table marks `complete`: accumulate the
36 five-minute steps into the 3 h window and put it on the 6 km grid. `rain` is
a per-interval accumulation in inches, so the window is a sum of steps
`s+1 .. s+36`, converted to mm (x 25.4); no differencing is needed. A pair that
is not `complete` is never partially summed: a 35-step total is not a 3 h total
and would bias every threshold, the same reason a missing obs hour makes an obs
window missing. Each deployment's windows are put on that deployment's own grid. The 3 -> 6 km step is a budget regrid to the A2 grid, not a block mean
written by hand, so WoFS goes through the same operator as everything else.

*Check:* conservation CSV, same form as obs; domain-mean preserved to a few
tenths of a percent.

*MET:* `pcp_combine`, `regrid_data_plane` (BUDGET).

### D4. HAFS 3 h windows on the deployment grid (`hafs_windows.py`)
`regrid_data_plane` takes `-field 'name="APCP"; level="A3";'` and selects the
bucket out of the native parent file, so the pass is: read the HAFS file,
write the 6 km field. No `pcp_combine`, and no intermediate APCP-only GRIB2:
each source file holds exactly one bucket for exactly one window, so
extracting first is read-big, write-small, read-small instead of read-big.
(Extraction is still worth it for archiving; the
`wgrib2 -match ":APCP:surface:${start}-${lead} hour acc fcst:"` recipe matches
the literal interval so the run total is excluded.)

Which cycle covers which window comes from D2, including the rule that a HAFS
run can only serve windows made of whole 3 h buckets. The file holding a window
is the one named for the bucket's *end*: the 15–18Z accumulation from an
18Z-prior init is `21-24 hour acc fcst` in the **f024** file, not f021.

`budget_check: true` (default) runs the mass comparison; it costs a full scan
of each native file, so it is the one knob to turn off for a production re-run.
HAFS configurations (A, B) are separate model labels.

*Check:* `hafs_windows_<case>.csv`, one row per (cycle, window) with its source
file, status and mass columns. Rows marked `missing_source` are cycles D2
expected but that are not on disk.

*MET:* `regrid_data_plane` (BUDGET).

### D5. Obs 3 h windows
Sum the already-regridded hourly truth into the same clock windows; a missing
hour makes the window missing. No rescaling: a 2 h total is not a 3 h total,
and pretending otherwise biases every threshold. The hourly files are stamped
at the END of their hour, so the window (15Z, 18Z] is the 16Z, 17Z and 18Z
files. An off-by-one here shifts the whole verification an hour.

Done in numpy rather than with `pcp_combine`: the hourly obs are MET NetCDF on
the grid already, so the window is three arrays added.

*Check:* count of valid obs windows vs expected; missing ones line up with
known outages.

### D5b. Eyeball panels: WoFS | HAFS | truth
Before any score is computed, look at the fields. One figure per (WoFS member
or a fixed example member, HAFS cycle, window): three panels on the same grid
and colour scale, so a regridding or pairing mistake shows up as a displaced
or empty panel rather than as a strange FSS three steps later. Restricted to a
few cycles and members by default; every pair would be thousands of figures.
Panels are drawn unclipped with the validity region outlined.

*Check:* subjective. A panel set that disagrees in placement means the
regridding or window pairing is wrong; one that disagrees in amount is the
forecast being wrong, which is the thing being measured.

### D6. Window manifest
A CSV inventory of every `(model, deployment, run id, member, init,
window_start, window_end, lead_h, status)` plus the C2 proximity columns,
before any statistics run. WoFS rows carry the A1 status (`complete`,
`missing_steps`, `short_run`, `no_files`) and HAFS rows carry the D4 source
status, so every window a deployment was expected to supply is accounted for:
served, or missing with a reason. HAFS rows carry the member count per
configuration, and a window with no HAFS member is marked `hafs_available:
false` (WoFS-only) instead of being dropped. Deployments with no served windows
(Idalia as downloaded) appear here explicitly. Where deployments overlap in time, the same
valid window appears once per deployment.

*Check:* read it — this is where silent data gaps surface cheaply rather than
as strange scores later.

---

## Phase E — Statistics

### E0. Delete: remaining swath/radius references
Remove `mask_radius_km`, `swath` and the `SWATH`/`WOFS` mask names from every
config, schema and plotting default. Search the repo for each.

*Check:* grep finds none; tests pass.

### E1. `grid_stat` configuration
- thresholds: 0.5 / 1 / 2.5 / 5 / 10 / 25 / 50 mm per 3 h
- mask: `FULL` = whole deployment grid ∩ `obs_valid` (one region now)
- neighborhood widths: 1 / 3 / 5 / 9 / 21 cells = 6 / 18 / 30 / 54 / 126 km
- line types: `CTC CTS CNT SL1L2 NBRCTC NBRCNT`
- `interp` none (already on a common grid)

*Check:* run one window, read the ASCII output by hand, confirm POD/FAR/ETS
are sane at a low threshold.

*MET:* `grid_stat`.

### E2. Run across all windows, individual runs
Three pairings per window, all on the same cells:

1. each WoFS member vs truth
2. each HAFS cycle vs truth
3. each WoFS member vs each HAFS cycle at the same valid window, with the
   same truth, so a head-to-head difference needs no extra definitions

*Check:* runtime per window, total output size.

### E3. Parse into the long CSV schema
Columns: `storm, deployment, model, run_id, member, init, valid_time,
window_h, lead_h, mask, threshold_mm, scale_km, stat_name, value`, plus the C2
proximity columns carried on each row (or joined by window).

`lead_h` per model makes matched-lead comparison and the HAFS
initialization-degradation test (HAFS f24–f27 vs WoFS f00–f03, same valid
window) filters on this table, not new code.

*Check:* row counts match windows x thresholds x scales per run.

### E4. Pooling
Aggregate contingency counts and partial sums *before* deriving scores. Never
average per-run ratios: ETS = (H - H_rand)/(H + M + F - H_rand) with H_rand
depending on each run's base rate and N, so the mean of ratios is not the
ratio of means. Pool across windows within a deployment first; pooling across
deployments is a choice made when the question requires it.

*MET:* `stat_analysis -job aggregate_stat` — CTC->CTS for ETS/CSI/POD/FAR,
SL1L2->CNT for RMSE/MAE/ME/bias, NBRCTC->NBRCTS and NBRCNT for FSS.

*Check:* a pooled ETS recomputed by hand from summed counts for one threshold.

### E5. Accumulation diagnostics
Keep the Python distribution and Q-Q summaries on the 3 h fields. Along/across
track error is out of scope for now (see Deferred).

---

## Phase F — Figures

Colorblind-safe palettes throughout (Okabe-Ito; no red/green pairings).

- **F1.** FSS vs neighborhood scale, one curve per lead — the fairness plot for
  the HAFS/WoFS resolution difference.
- **F2.** ETS/CSI vs threshold, per lead.
- **F3.** Lead-time series of bias, RMSE, and FSS at a fixed scale.
- **F4.** Spatial 3 h panels: obs, WoFS, HAFS, differences for selected windows.

*MET:* none — matplotlib reading the Phase E CSV.

---

## Phase G — Later analyses (design when Phase E is stable)

- **G1.** Probability-matched mean of WoFS members; verify it as a field.
- **G2.** WoFS ensemble and lagged HAFS ensemble (several cycles of one
  configuration, valid at the same window); `gen_ens_prod`, `ensemble_stat`.
- **G3.** Stratify by the C2 proximity columns: do "mostly TC" and "little TC"
  windows diverge?
- **G4.** MRMS sensitivity: repeat E on `truth_source: mrms`.
- **G5.** Mixing HAFS-A and HAFS-B as an explicit experiment.

---

## Phase H — Final cleanup

### H1. Delete: dead code and configs
Run a dead-code pass (unused functions, unused config keys, unused imports) and
remove what it finds. Re-decide the obs-vs-obs tooling (`obs_regrid_plots`,
`obs_regrid_stats`) and `aorc_common` against what the finished pipeline uses.

### H2. Prune the environment and job scripts
Remove dependencies only the deleted code needed from `environment.yml` and
`requirements.txt`. Update the Orion job scripts to the new commands and the
per-deployment YAMLs.

### H3. Documentation
Rewrite `README.md` for the new scope and command list; make this plan reflect
what was built.

*Check:* fresh clone + `conda env create` + `pytest` passes; `run.py --help`
matches the README.

---

## MET programs

**Used**

| Program | Role |
|---|---|
| `regrid_data_plane` | budget regridding of obs and forecasts to the deployment grid |
| `pcp_combine` | WoFS 3 h absolute-clock accumulation |
| `grid_stat` | categorical, continuous, and neighborhood scores |
| `stat_analysis` | pooling counts and partial sums across runs and windows |
| `plot_data_plane` | quick raster QC of MET NetCDF output |

**Available but not used, and why**

| Program | Why not |
|---|---|
| `gen_vx_mask` | the only mask is the obs-validity region, built in Python |
| `MODE` / `MTD` | object-based verification; to be designed deliberately later |
| `TC-RMW` | storm-relative radial composites; deferred |
| `TC-Pairs` / `TC-Stat` | track error; deferred |
| `Grid-Diag` | joint histograms across cases; existing accumulators cover it |
| `Series-Analysis` | per-grid-point statistics across cases; useful for "where does the model fail" maps, a separate question once E is stable |
| `Point-Stat` | needs point observations; everything here is gridded |
| `Ensemble-Stat` / `gen_ens_prod` | probabilistic verification; gated on Phase G |
