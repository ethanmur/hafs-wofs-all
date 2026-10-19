# HAFS / WoFS precipitation verification — analysis plan

Common-grid verification of HAFS and WoFS rainfall forecasts against gridded
observations, per TC case. Each step below is sized to be implemented and
checked in one sitting. "Check" is what to look at before moving on.

## Settled design decisions

- **One grid per case**, not a continental grid. Fairness comes from identical
  cell size in km everywhere, not from all cases sharing one grid object.
- **Lambert conformal**, standard parallels chosen per case, keeping cell-size
  error ~1-2% across the domain. Mercator varies ~+/-25% over 10-50N; a
  degree-based grid has the same cos(lat) defect in the zonal direction.
- **6 km resolution** (configurable). HAFS parent is 6.67 km north-south and
  5.1-5.8 km east-west at landfall latitudes, so 6 km is honest in both
  directions; it is also an exact 2x aggregation of WoFS's 3 km.
- **Best track (b-deck) defines the domain** for every model, so no model is
  scored over a different region than another.
- **Grid is a rectangle; the track buffer is a mask.** Two separate radii:
  `pad_km` (750 km) is how far the grid extends past the trimmed track, and
  `mask_radius_km` (500 km) is how far the statistics swath extends. The grid
  pad is deliberately larger, because a forecast initialized days before
  landfall can be several hundred km off track and any of its rain outside the
  grid is silently uncounted -- an asymmetric bias that loses false alarms
  while keeping misses.
- **Cycle range is independent of the scoring window.** `init_start` /
  `init_end` select which model cycles to pull (e.g. from 3 days before
  landfall); `valid_start` / `valid_end` set what is scored, and the grid
  follows the *valid* window. Long-lead verification therefore costs no extra
  observations.
- **Track trimmed by time, not geography** — only points inside
  `valid_start`/`valid_end` (+/-6 h margin) shape the grid. A long
  Cape Verde track collapses to the landfall segment automatically.
- **3 h accumulation windows on an absolute clock** (00/03/06...Z), not
  init-relative. WoFS cycles every 30 min, so init-relative windows would
  compare 1730-2030Z against 1800-2100Z. A WoFS run is used for a window if
  its 6 h span fully contains that window, whatever its init.
- **Two comparison scopes**: HAFS vs obs (always available) and
  HAFS vs WoFS vs obs (inside the WoFS domain masks).
- **WoFS deployments are plural and time-bounded.** WoFS is re-sited between
  events, so one storm can have several 900x900 km domains on different days.
  The grid unions them; each remains a separate statistics mask applied only
  within its own valid window, so a box live on day 1 never masks day-3
  statistics.
- **Truth product**: Stage IV. The `noaa-mrms-pds` archive of
  `MultiSensor_QPE_01H_Pass2` begins 2020-10-14, which excludes 9 of the 18
  cases (Fay, Hanna, Isaias, Laura, Sally, Delta, Barry, Dorian, Michael);
  Stage IV reaches back to 2002 and covers all of them. MRMS stays available
  as a sensitivity check on the 9 post-2020 cases, where its offshore coverage
  is better.
- **Resolution advantage is reported, not hidden**: FSS across a range of
  neighborhood scales, since WoFS was run at 3 km and budget interpolation
  from HAFS's 6 km adds no information.

### Deferred

- Object-based verification (MODE/MTD) — to be designed deliberately later.
- TC-RMW storm-relative composites.
- Data-derived obs validity mask for inland radar gaps and **coastal radar
  outages during landfall** (these appear as missing, not zero).
- WoFS member treatment: verify individual members and pool, dropping
  near-duplicate physics; probabilistic scoring is a later option.
- Automatic multi-storm batch processing.

---

## Phase A — Build the verification grid

### A1. B-deck ingest from a storm list
Extend `fetch_bdecks.py` to take a storm-list file in addition to the HAFS
folder scan; add a parser returning `(datetime, lat, lon, status)` accepting
*all* status codes (`TD/TS/HU/EX/SD/SS/LO/DB`), deduplicated to synoptic times.

*Check:* print first/last track time and the set of status codes per storm;
confirm remnant/extratropical tails survive.

*MET:* none. TC-Pairs reads ATCF but emits pair statistics, not geometry.

### A2. Grid builder (`verification_grid.py`) -- BUILT

    python analysis/run.py storms/build_grid_<year>_<storm>.yaml build-grid

Trim track to `valid_start`/`valid_end` +/-`margin_h` -> pad `pad_km` -> union the WoFS
box -> derive Lambert parameters by rule (central lon = domain center;
standard parallels inset one-sixth from each latitude edge) -> snap outward to
whole `res_km` cells -> guard at 800x800 -> write `grids/<case>_grid.json`
plus the MET grid-spec string. Print which precedence rule fired.

Domain precedence:
1. `domain_override` in the case YAML, if present
2. track-within-window + 500 km pad, unioned with the WoFS box
3. WoFS box + pad alone (pure-remnant cases with no track points in window;
   these lose the track-relative mask, which is the correct degradation)

Domain clipping at long leads is checked by eye from the sanity map rather
than by a computed diagnostic.

Optionally union in the forecast tracks of the cycles in
`[init_start, init_end]` (the `atcfunix` files are tiny) for a more targeted
domain than a blanket pad.

*Check:* Nx x Ny, cell count, corner lat/lons, and the printed final track
time inside the window.

Writes `<out_dir>/<case>_grid.json` (Lambert parameters, corners, cell
counts, the MET grid-spec string, an east-positive PROJ.4 string, and the
spacing report) plus the A3 sanity map. Per-case YAMLs exist for all 18 storms
as `storms/build_grid_<year>_<storm>.yaml`.

**Settled on Orion (2026-09-30):** MET's grid-specification strings take
longitudes **east**-positive (negative for west), and MET negates them into
its internal west-positive form. `MET_LON_WEST_POSITIVE = False`.

The diagnosis, worth keeping because the failure is silent: a west-positive
spec placed the grid at 82.35 E and *every* output cell came back missing, for
every method and width -- there is no error, just an all-missing field. MET's
own debug output gives the check. It printed the HRAP input grid as
`Lon_orient: 105.000` (positive, for 105 W) and our spec as `Lon_orient:
-82.350`; same debug code, so positive means west, and opposite signs on two
grids in the same hemisphere means the spec is wrong. `read_grid_json`
rebuilds the spec from a JSON's numeric parameters when its stored convention
differs from the code's, so old grid JSONs correct themselves on read.

Also note MET silently clamps the BUDGET width: `RegridInfo::validate() ->
Resetting the regridding width from 5 to 2`. Tuning `width` upward for BUDGET
does nothing.

*MET:* none for construction; the output *is* the grid spec consumed
downstream (`regrid_data_plane -to_grid`, and `grid_stat`'s `regrid.to_grid`).

### A3. Grid sanity plot -- BUILT (part of `build-grid`)
One cartopy figure: grid outline, best track colored by status, 500 km swath,
WoFS box, US geography, obs coverage extent.

*Check:* by eye — track fully inside, WoFS box contained, no absurd extents.

*MET:* none. `plot_data_plane` renders a single field to PostScript; it cannot
overlay vector geometry.

### A4. Grid-spacing verification -- BUILT (`spacing_report`)
True ground spacing between adjacent grid points via `pyproj.Geod`, reported
as min/max/mean dx and dy plus the worst percentage deviation from `res_km`.
Printed by `build-grid` and stored in the JSON.

*Check:* under ~2%. Beryl on a 7/8--7/10 window came out at 463 x 513 cells
with spacing within **1.86%**. Note that the whole-track windows in the
generated YAMLs give much larger errors (Dorian 7.31%, Michael 6.40%) simply
because they span 40 degrees of latitude -- that number falling as the window
narrows is itself a useful signal that the window is right.

### A5. HAFS coverage check -- DROPPED
The HAFS parent spans ~100 x 82 degrees against verification grids of 4-10
degrees, and the measured per-cycle drift is 0.4-0.8 degrees, so containment
is never in doubt. Any failure would surface anyway as an anomalous B2
conservation percentage, on the same data, for free.

<details><summary>original step</summary>
For every cycle, confirm the parent domain contains the rectangle (the parent
domain slides northwest per cycle).

*Check:* a per-cycle PASS/FAIL table; any FAIL means a band of missing cells
and needs the window or pad adjusted.

*MET:* none (header read via `grid_spacing.py`'s eccodes path).
</details>

---

## Phase B — Re-regrid the observations

### B1. Repoint and clear the cache
Add the grid identity to the regridded cache path
(`<case>/<grid_name>_<res>km/`). Delete the old HAFS-parent regridded obs;
**keep the native downloads** — MRMS re-download is the expensive part and
those files are grid-independent.

### B2. Regrid Stage IV hourly to the case grid -- BUILT

    python analysis/run.py storms/build_grid_<year>_<storm>.yaml regrid-obs

Every command tees its console output to
`<out_dir>/logs/<case>_<command>_<stamp>.log` (override with `log_dir:`), so a
compute-node job keeps its own record independent of the scheduler's stdout
capture. The file is flushed on every write, so a killed job still leaves a
usable log.

`truth_source` defaults to `stage4`, and **only the truth product is
regridded** -- the other obs products stay in the obs-vs-obs scripts, which
read the native caches. `regrid.sources: [stage4, mrms]` overrides this for
the post-2020 sensitivity check.

*Check:* the per-hour conservation CSV — domain-mean percent difference within
a few tenths of a percent, as already validated at 6.7 km.

*MET:* `regrid_data_plane` (BUDGET, width 2, `vld_thresh 0.5`).

### B3. Visual QC of the new grid
Run `plot-regrid` and `stats-regrid` unchanged on the 6 km output.

*Check:* native-vs-regridded pairs look right; distributions have not shifted
in a way 6 km smoothing cannot explain.

*MET:* `plot_data_plane` as a fast one-off raster check on a single NetCDF
before committing to a full loop.

---

## Phase C — Masking regions

### C1. Coastal validity mask (150 km)
**BUILT** — `analysis/masks.py`, `run.py ... build-masks`. Writes
`<case>_mask_coastal.{nc,png}` into `figures/grids`.

Rasterize Natural Earth US-state polygons onto the case grid,
distance-transform, keep everything within 150 km of US land, then cut land
masses not connected to the US. The US land border is the only thing the
buffer measures from, and water and foreign soil follow the same rule: the
buffer reaches 150 km into Canada and Mexico and stops, since radar range does
not end at a political line but coverage does fade inland. The connectivity
test (not a country list) is what keeps Cuba, Hispaniola and the Bahamas out
while keeping Long Island, the Keys and the barrier islands in.

Milton: 67,427 of 168,921 cells (39.9%) — so `FULL` in Phase E must be
`FULL ∩ coastal_valid`, or domain-wide statistics are computed over a region
that is 60% no-truth.

*Caveat:* the NetCDF's MET Lambert global attributes are written from MET's
convention, with east-positive longitudes per the A2 lesson. Verified readable
with `plot_data_plane <mask>.nc /tmp/m.ps 'name="coastal_valid"; level="(*,*)";'`;
if MET ever rejects it, the fallback is writing the field as GRIB2 through
`regrid_data_plane` so MET writes the NetCDF itself.

*MET:* `gen_vx_mask` is available but its distance-thresholded types are
circle/track, not polygon-buffer — Python is simpler here, and the output
still feeds `grid_stat` natively.

### C2. Track swath mask (500 km)
**BUILT** — `masks.track_swath_mask`, written as `<case>_mask_swath.nc`.

The track is projected into the grid's own Lambert plane and buffered there,
so the corridor is a true constant radius. It buffers the track as a *line*,
not as discs at each fix, so a fast-moving storm with 6-hourly fixes gets a
continuous swath rather than a string of beads.

*MET:* `gen_vx_mask -type track -thresh le500` does this, but the region has
to be intersected with C1 and C3 on the same grid anyway, and one array pass
is simpler than three subprocess calls with intermediate files — the same
call made for C1. The NetCDF feeds `grid_stat` identically.

### C3. WoFS box masks
One mask per deployment (`WOFS_1`, `WOFS_2`, ...), each applied only to the
accumulation windows inside its own `valid_start`/`valid_end`; optionally a
`WOFS_ANY` union for a pooled view. `active_domains(domains, valid_time)`
already selects the live ones. Skipped for cases with no WoFS.

**BUILT** — `masks.wofs_box_mask`, written as `<case>_mask_wofs_<n>.nc` plus
`<case>_mask_wofs_any.nc`. The box is rasterised in lat/lon, since that is how
the deployments were estimated from the WoFS plots; its edges are not grid
lines, so the mask is a staircase at 6 km. A cell is in or out — nothing is
half-counted.

*MET:* `gen_vx_mask -type box`, not used, for the C2 reason.

### C4. Compose the masks
**BUILT** — `masks.build_regions` writes the intersection as `verify` in
`<case>_mask_verify.nc`: `coastal_valid AND track_swath AND (any WoFS box, if
the case has one)`. Cell counts for every region print once per case, and
`<case>_mask_regions.png` draws the three boundaries with the intersection
shaded.

For an individual accumulation window, `masks.active_verify_mask` rebuilds the
intersection from the deployments live at that valid time rather than the
all-hours union — scoring an hour against a box deployed on another day would
credit a forecast in a region WoFS never covered. With no deployment live the
WoFS term drops out instead of emptying the region, since HAFS-vs-obs is still
valid there and only the three-way comparison is not.

*Check:* `run.py <yaml> plot-masked` draws the regridded obs clipped to this
intersection, one map per hour, with the three boundaries over it — the field
`grid_stat` will actually accumulate counts from.

*MET:* `gen_vx_mask -intersection`, not used, for the C2 reason.

---

## Phase D — Forecast-side 3 h accumulations

### D1. Establish the accumulation convention
**ESTABLISHED for HAFS** (WoFS still open, pending data).

HAFS writes *both* forms in the same file, inspected with pygrib:

```
702: Total Precipitation:kg m**-2 (accum):surface:fcst time 21-24 hrs (accum):from 202308171800
703: Total Precipitation:kg m**-2 (accum):surface:fcst time  0-24 hrs (accum):from 202308171800
```

- `21-24` is a **per-interval 3 h bucket** — this is the one to use.
- `0-24` is the run total since init — the one to avoid.

Two consequences:

1. **No `pcp_combine` for HAFS 3 h windows.** The bucket already *is* the
   window. `pcp_combine` was planned to build 3 h sums; it is only needed if a
   longer window is ever wanted.
2. **The buckets land on the absolute clock for free.** HAFS initializes at
   00/06/12/18Z, all multiples of 3, so every bucket edge falls on the
   00/03/06...Z clock the obs windows use. In the example, init 2023081718 +
   21-24 h is exactly 15-18Z on 18 Aug. No realignment, no interpolation.

The two records are both APCP at surface and differ only in accumulation
interval, so **the selection must be by interval, not by name**:

```
name="APCP"; level="A3";      # the 3 h bucket
name="APCP"; level="A24";     # the run total -- not this
```

Getting this wrong is silent: both decode fine, and a run-total field just
looks like a very wet forecast.

*Still to confirm:* that the bucket is 3 h at **every** lead, not only at
multiples of 24. Some models emit `0-1`/`0-2` style buckets at the first
leads or widen the bucket beyond a certain lead, which would break the
one-bucket-per-window assumption.

```bash
for f in <case>/*.parent.atm.f*.grb2; do
  echo -n "$(basename $f)  "
  grib_get -p stepRange -w shortName=tp $f | tr '\n' ' '; echo
done
```

Every line should show one bucket of width 3 ending at that lead, plus the
`0-N` run total. If a lead shows a width other than 3, that lead needs
`pcp_combine` after all and D3 comes back.

*Check:* at f024, the eight 3 h buckets should sum to the `0-24` field. That
replaces the f003-minus-f000 test below, which assumed since-init output.

```bash
grib_get -p shortName,stepType,stepRange -w shortName=tp <parent.atm.f003.grb2>
```

<details><summary>Original plan text</summary>

`grib_get -p shortName,stepType,stepRange` on a HAFS parent file; record
whether APCP is since-init or per-interval. `stepRange` of `0-3` means
accumulated since init (`pcp_combine -subtract`); `2-3` means per-interval
(`-sum`). *Check:* an f003-minus-f000 field that goes negative anywhere means
the convention is wrong.

</details>

### D2. Absolute-clock window definition
**BUILT** — `analysis/accum_windows.py`, `run.py ... list-windows`. Reads no
data, so it runs before anything is staged.

Three decisions worth recording:

* **Whole minutes everywhere.** WoFS cycles every 30 min and writes every
  5, so f02:30 is routine; in float hours "is this lead on the model's
  accumulation granularity" becomes a tolerance question instead of an exact
  one.
* **`step_min` is accumulation granularity, not output frequency.** For HAFS
  that is the 3 h APCP bucket from D1 (180), not the hourly output — a 1 h
  slice of a 3 h bucket does not exist. This is what makes a 20-23Z window
  correctly unreachable from an 18Z HAFS init while 21-00Z is f003-f006. For
  WoFS it is the 5 min output step.
* **Lead is measured at the window *end*.** Defined once on `Coverage.lead_h`,
  so every lead-time plot in Phase F means the same thing: how far out the
  forecast had run by the time the accumulation was complete.

A small module producing the list of 3 h windows spanning
`valid_start`->`valid_end` on the 00/03/06...Z clock, and, for a given model
run, the lead offsets covering each window. Drop windows a run cannot fully
span.

Worked example for the 18-21Z window, WoFS 6 h runs at 30 min cycling:

| WoFS init | lead covering 18-21Z |
|---|---|
| 1500Z | f03-f06 |
| 1530Z | f02:30-f05:30 |
| 1600Z | f02-f05 |
| ... | ... |
| 1800Z | f00-f03 |

A 1430Z init ends at 2030Z (short of 21Z) and an 1830Z init starts after 18Z;
both are dropped. Every usable run carries its own `wofs_lead_h`, which gives
WoFS's internal lead-time dependence for free.

*Check:* unit tests — a 1730Z 6 h run yields exactly the 18-21Z window at
f00:30-f03:30; a 1430Z run yields none. Both are in
`tests/test_accum_windows.py`, with the HAFS bucket-granularity cases.

### D3+D4. HAFS 3 h windows on the case grid
**BUILT, as one step** — `analysis/hafs_windows.py`,
`run.py ... regrid-hafs`, configured by a `hafs:` block in each case YAML.

D3 and D4 merged because nothing sits between them. `regrid_data_plane` takes
`-field 'name="APCP"; level="A3";'` and selects the bucket out of the native
parent file, so the pass is: read the HAFS file, write the 6 km field. No
`pcp_combine`, and no intermediate APCP-only GRIB2 — each source file holds
exactly one bucket for exactly one window, so extracting first would be
read-big, write-small, read-small instead of read-big. (Extraction is still
worth doing for archiving or sharing; it just does not speed anything up. The
`wgrib2 -match ":APCP:surface:${start}-${lead} hour acc fcst:"` recipe is the
one to use, matching the literal interval so the run total is excluded.)

Which cycle covers which window comes from D2, including the rule that a HAFS
run can only serve windows made of whole 3 h buckets.

The file holding a window is the one named for the bucket's *end*: the
15-18Z accumulation from an 18Z-prior init is `21-24 hour acc fcst` in the
**f024** file, not f021.

`budget_check: true` (default) runs the same mass comparison as the obs side.
It costs a full scan of each native file — the I/O the no-staging design
otherwise avoids — so it is the one knob to turn off for a production re-run.

*Check:* `hafs_windows_<case>.csv`, one row per (cycle, window) with its
source file, status and mass columns. Rows marked `missing_source` are cycles
D2 expected but that are not on disk.

<details><summary>Original D3/D4 plan text</summary>

**Reduced by D1** to selecting the right record: the 3 h bucket is already the
window, so this step extracts `name="APCP"; level="A3";` and tags it with its
absolute window, with no accumulation arithmetic.

*Check:* one window's domain-total against the same window summed from the
run-total records (`0-24` minus `0-21`); they should agree to rounding.

*MET:* none needed, pending the per-lead bucket-width check in D1. If any lead
turns out not to be a clean 3 h bucket, `pcp_combine -sum` returns for those
leads only.

D4: Regrid HAFS windows to the case grid. *Check:* conservation CSV, same form
as obs. *MET:* `regrid_data_plane` (BUDGET).

</details>

### D5. Obs 3 h windows
Sum the already-regridded hourly Stage IV into the same clock windows; a missing
hour makes the window missing (no rescaling).

*Check:* the count of valid obs windows vs expected, and that missing ones
line up with known outages.

**Summation BUILT** inside `hafs_obs_plots.obs_window_field`, in numpy rather
than with `pcp_combine`: the hourly obs are already MET NetCDF on the case
grid, so the window is three arrays added, and D5b needs the field in memory
anyway. A missing hour returns no window at all — no rescaling, since a 2 h
total is not a 3 h total and pretending otherwise biases every threshold.

The hourly files are stamped at the END of their hour, so the window
(15Z, 18Z] is the 16Z, 17Z and 18Z files. Off by one here would shift the
whole verification an hour.

*MET:* `pcp_combine -sum`, not used, for the C2 reason.

### D5b. HAFS vs Stage IV eyeball panels
**BUILT** — `analysis/hafs_obs_plots.py`, `run.py ... plot-hafs`, configured
by a `hafs_plots:` block.

Before any score is computed, look at the fields. One figure per (cycle,
3 h window): Stage IV left, HAFS right, same grid and colour scale, so a
regridding mistake shows up as a displaced or empty panel rather than as a
strange FSS three steps later, and forecast quality can be judged by eye.

* **00Z cycles only** by default (`hafs_plots.init_hours`). Every cycle would
  be a few thousand figures per case, mostly repeating each other; the 00Z
  runs give one clean lead-time sequence across the case.
* **The map window follows WoFS**, since that is where the three-way
  comparison will happen. With several deployments the one nearest in time to
  the window wins — distance is zero while a deployment is live and grows with
  the gap either side — and the title says which, and how far away. With none
  configured it falls back to the whole verification grid.
* Panels are drawn **unclipped**, with the grid outline and WoFS boxes over
  them; the masked view is `plot-masked`. Judging the regridding means seeing
  the cells the mask would remove.

*Check:* subjective. A panel pair that disagrees in placement means the
regridding or the window pairing is wrong; one that disagrees in amount is
the forecast being wrong, which is the thing being measured.

### D6. Window manifest
A CSV inventory of every `(model, init, window_start, window_end, lead_h,
status)` before any statistics run.

*Check:* read it — this is where silent data gaps surface cheaply rather than
as strange scores later.

---

## Phase E — Statistics

### E1. `grid_stat` configuration
- thresholds: 0.5 / 1 / 2.5 / 5 / 10 / 25 / 50 mm per 3 h
- masks: `FULL`, `SWATH`, `WOFS` (each intersected with coastal validity)
- neighborhood widths: 1 / 3 / 5 / 9 / 21 cells = 6 / 18 / 30 / 54 / 126 km
- line types: `CTC CTS CNT SL1L2 NBRCTC NBRCNT`
- `interp` none (already on a common grid)

*Check:* run one window, read the ASCII output by hand, confirm POD/FAR/ETS
are sane at a low threshold.

*MET:* `grid_stat`.

### E2. Run across all windows, one model
HAFS vs Stage IV for every window and cycle of one case.

*Check:* runtime per window, total output size.

### E3. Parse into the long CSV schema
Columns: `case, model, init, valid_time, window_h, lead_h, mask,
threshold_mm, scale_km, stat_name, value`.

This is the table everything downstream queries. Carrying `lead_h` per model
means matched-lead comparison and the HAFS initialization-degradation test
(HAFS f24-f27 vs WoFS f00-f03, same valid window) are both just filters.

*Check:* row counts match windows x thresholds x masks x scales.

### E4. Pooling
Aggregate contingency counts and partial sums *before* deriving scores. Never
average per-cycle ratios: ETS = (H - H_rand)/(H + M + F - H_rand) with
H_rand depending on each cycle's base rate and N, so the mean of ratios is not
the ratio of means.

*MET:* `stat_analysis -job aggregate_stat` — CTC->CTS for ETS/CSI/POD/FAR,
SL1L2->CNT for RMSE/MAE/ME/bias, NBRCTC->NBRCTS and NBRCNT for FSS.

*Check:* a pooled ETS recomputed by hand from summed counts for one threshold.

### E5. Accumulation and track diagnostics
Keep the existing Python `distribution_stats` / `qq_percentiles` on the 3 h
fields, plus along/across track error.

*MET:* TC-Pairs is the proper tool for track error once HAFS ATCF files are in
hand; the existing custom `along_cross_km` is fine until then.

---

## Phase F — Figures

Colorblind-safe palettes throughout (Okabe-Ito; no red/green pairings).

- **F1.** FSS vs neighborhood scale, one curve per lead — the fairness plot
  for the HAFS/WoFS resolution difference.
- **F2.** ETS/CSI vs threshold, per lead and per mask.
- **F3.** Lead-time series of bias, RMSE, and FSS at a fixed scale.
- **F4.** Spatial 3 h panels: obs, model, difference, for selected windows.

*MET:* none — matplotlib reading the Phase E CSV. MET's plotting outputs are
PostScript diagnostics, not publication figures.

---

## Phase G — WoFS (when data access lands)

- **G1.** Confirm WoFS domains against the boxes estimated from example plots.
- **G2.** Member selection rule, documented so it cannot be mistaken for
  cherry-picking.
- **G3.** 5-min -> 3 h absolute-clock accumulation (`pcp_combine`, convention
  per D1).
- **G4.** Per-member `grid_stat` into the same CSV schema, then pool.

*MET:* `ensemble_stat` and `gen_ens_prod` become relevant if moving to
probabilistic verification; deliberately unused while verifying members
individually.

---

## MET programs

**Used**

| Program | Role |
|---|---|
| `regrid_data_plane` | budget regridding of obs and forecasts to the case grid |
| `pcp_combine` | 3 h absolute-clock accumulation windows |
| `gen_vx_mask` | track swath, WoFS box, mask composition |
| `grid_stat` | categorical, continuous, and neighborhood scores |
| `stat_analysis` | pooling counts and partial sums across cycles and cases |
| `plot_data_plane` | quick raster QC of MET NetCDF output |

**Available but not used, and why**

| Program | Why not |
|---|---|
| `MODE` / `MTD` | object-based verification; custom object code was removed and this will be designed deliberately later |
| `TC-RMW` | storm-relative radial composites; deferred, a natural later addition |
| `TC-Pairs` / `TC-Stat` | track error; will replace the custom code once HAFS ATCF output is wired in |
| `Grid-Diag` | joint histograms across cases; existing `obs_regrid_stats` accumulators already do this and are integrated with the plots |
| `Series-Analysis` | per-grid-point statistics across cases; genuinely useful for "where does HAFS fail" maps, but a separate question, best added once Phase E is stable |
| `Point-Stat` | needs point observations; everything here is gridded |
| `Ensemble-Stat` / `gen_ens_prod` | probabilistic WoFS verification, gated on the member-treatment decision in G2 |
