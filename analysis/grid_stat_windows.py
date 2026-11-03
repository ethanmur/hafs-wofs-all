"""grid_stat configuration and a one-window test run (plan step E1).

    python analysis/run.py storms/<case>.yaml grid-stat-test --res-km 3
    python analysis/run.py storms/<case>.yaml grid-stat --res-km 3      (E2)

Writes the grid_stat configuration for the case, builds the validity mask as a
MET NetCDF named FULL (gen_vx_mask), and scores one window: one WoFS member
and the shortest-lead cycle of each HAFS configuration, each against the truth
product, on the mask. The window is the wettest one with WoFS, HAFS and truth
files, or `grid_stat.test_end` in the YAML. It then reads the .stat files back
and prints the row counts per line type and the scores that can be checked by
hand, so the configuration is read before it is run over every window (E2).

What the configuration asks for (settings in `GridStatSettings`):

  CTC CTS            categorical counts and scores at each threshold (mm/3 h)
  CNT SL1L2          continuous scores and their partial sums, twice: over all
                     cells (cnt_thresh NA) and over wet cells only (either field
                     at or above `wet_thresh_mm`, union)
  NBRCTC NBRCNT      neighborhood counts and FSS at each threshold and width

The neighborhood widths are distances; they become cells for the grid in use,
rounded up to an odd count because grid_stat requires one (6 / 18 / 30 / 54 /
126 km is 1 / 3 / 5 / 9 / 21 cells at 6 km and 3 / 7 / 11 / 19 / 43 at 3 km, the
latter covering 9 / 21 / 33 / 57 / 129 km). The fields
are already on one grid, so there is no interpolation, and the only mask is the
validity mask.

`grid-stat` (E2) scores every window of the scored period with every run: all
members of every WoFS init and every HAFS cycle, each against the truth on the
same mask cells. It is resumable (a run with a current .stat file is skipped),
runs `GRID_STAT_WORKERS` (else SLURM_CPUS_PER_TASK, else 1) calls at a time, and
records every run, done, cached or failed, in grid_stat_runs_<case>.csv. A
WoFS-vs-HAFS comparison is a join of these per-run results on the window, with
no further grid_stat calls: both are scored against the same truth on the same
cells, so neither is treated as truth.

Files, under <grid_stat_dir> (data/grid-stat/<dep>_<storm>/):
    GridStat_<case>[_3km][_coast50km].conf
    <case>[_3km][_coast50km]_FULL_mask.nc
    <window end YYYYMMDDHH>/grid_stat_<run>[_3km]_<lead>L_<valid>V.stat (+ .log)
where <run> is wofs_m00_i<init> or <hfsa|hfsb>_c<cycle>.
"""

import csv
import os
import subprocess
import sys
import time
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import mask_check
import masks
import met_regrid
import wofs_windows as ww
from eyeball_plots import runs_for
from obs_regrid_plots import LABELS

OBS_NAME = met_regrid.OUTPUT_NAME
COMMON_COLUMNS = 24          # the .stat columns before the line-type values
OUTPUT_FLAGS = ("ctc", "cts", "cnt", "sl1l2", "nbrctc", "nbrcnt")


# =============================================================================
# Configuration
# =============================================================================

def neighbourhood_cells(widths_km, res_km):
    """Neighborhood widths in cells for this grid. A width that is not a whole
    number of cells is an error. grid_stat accepts only odd widths (a square
    centred on a cell), so an even count goes up to the next odd one: 6 km is
    2 cells at 3 km and becomes 3, an effective 9 km."""
    cells = []
    for km in widths_km:
        n = km / res_km
        if abs(n - round(n)) > 1e-6 or round(n) < 1:
            raise ValueError(f"a {km:g} km neighborhood is {n:g} cells at "
                             f"{res_km:g} km; widths must be whole cells")
        n = int(round(n))
        cells.append(n if n % 2 else n + 1)
    return cells


def effective_km(widths_km, res_km):
    """The distances the odd cell counts actually cover."""
    return [c * res_km for c in neighbourhood_cells(widths_km, res_km)]


def _thresholds(values):
    return ", ".join(f">={v:g}" for v in values)


def config_text(case):
    """The grid_stat configuration. The forecast field name, model label, mask
    file and output prefix change per run, so they are environment variables
    (MET substitutes ${NAME}); everything else is fixed for the case."""
    s = case.grid_stat
    cats = _thresholds(s.thresholds_mm)
    cnt = f"NA, >={s.wet_thresh_mm:g}"
    cells = neighbourhood_cells(s.widths_km, case.res_km)

    def field(name):
        return (f'{{ name = "{name}"; level = "(*,*)";\n'
                f"        cat_thresh = [ {cats} ];\n"
                f"        cnt_thresh = [ {cnt} ]; cnt_logic = UNION; }}")

    flags = "; ".join(f"{k} = STAT" for k in OUTPUT_FLAGS)
    return f'''// grid_stat configuration for {case.case_slug}
// Written by grid_stat_windows.py (plan E1); change the case YAML, not this file.
// Thresholds are mm per 3 h. cnt_thresh NA scores every cell; the second
// entry scores cells where either field is at least {s.wet_thresh_mm:g} mm.
// Neighborhood widths are {", ".join(f"{k:g}" for k in s.widths_km)} km,
// {", ".join(map(str, cells))} cells at {case.res_km:g} km (odd, as grid_stat
// requires), covering {", ".join(f"{k:g}" for k in effective_km(s.widths_km, case.res_km))} km.

model  = "${{MODEL}}";
obtype = "{LABELS[case.truth_source]}";

fcst = {{
   field = [ {field("${FCST_NAME}")} ];
}}
obs = {{
   field = [ {field(OBS_NAME)} ];
}}

nbrhd = {{
   shape      = SQUARE;
   width      = [ {", ".join(map(str, cells))} ];
   cov_thresh = [ >={s.cov_thresh:g} ];
   vld_thresh = 1.0;
}}

// The validity mask is the only region: the grid is the WoFS domain.
mask = {{
   grid = [];
   poly = [ "${{MASK_FILE}}" ];
}}

output_flag   = {{ {flags}; }}
nc_pairs_flag = FALSE;
output_prefix = "${{OUT_PREFIX}}";
'''


def write_config(case):
    """Write the configuration, but only when its text changed: its file time
    is what tells a finished run from one made under older settings."""
    path = case.grid_stat_conf_path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = config_text(case)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


class GridStatError(RuntimeError):
    """A grid_stat or gen_vx_mask call that did not produce its output."""


def prepare(case, grid_file):
    """The configuration and the MET mask for this case, written only when
    missing or out of date (the mask when older than the validity mask it
    comes from). `grid_file` is any truth window: the mask takes its grid."""
    conf = write_config(case)
    mask = case.grid_stat_mask_path
    if (not mask.exists()
            or mask.stat().st_mtime < case.mask_path.stat().st_mtime):
        ok, why = mask_check.make_met_mask(case, grid_file, mask, name="FULL")
        if not ok:
            raise GridStatError(f"cannot build the MET mask: {why}")
    return conf, mask


# =============================================================================
# Choosing the test window and runs
# =============================================================================

def choose_test(case, mask):
    """The window and runs to score: {"window", "truth", "wofs": (init, path),
    "hafs": [(config, init, path)]}. The window is `grid_stat.test_end`, else
    the one with the highest mean truth rain inside the mask among those with
    WoFS, truth and at least one HAFS file; the runs have the shortest lead."""
    source = case.truth_source
    wanted = case.grid_stat.test_end
    best = None
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        if wanted is not None and w.end != wanted:
            continue
        truth = case.obs_window_path(source, w.end)
        wofs = runs_for(case.wofs_regrid_3h_dir, w.end, "wofs", case.res_label)
        hafs = [(c, *runs_for(case.hafs_regrid_dir_for(c), w.end, c.lower(),
                              case.res_label)[-1])
                for c in case.hafs_models
                if runs_for(case.hafs_regrid_dir_for(c), w.end, c.lower(),
                            case.res_label)]
        if not (truth.exists() and wofs and hafs):
            continue
        field = met_regrid.read_regridded(truth)[2]
        if field.shape != mask.shape:
            raise SystemExit(f"the truth window is {field.shape} but the "
                             f"mask is {mask.shape}; rebuild the mask at the "
                             "same resolution")
        rain = float(np.nanmean(np.where(mask, field, np.nan)))
        if best is None or rain > best[0]:
            best = (rain, {"window": w, "truth": truth, "wofs": wofs[-1],
                           "hafs": hafs, "mean_mm": rain})
    if best is None:
        why = (f"window {wanted:%Y-%m-%d %HZ}" if wanted else "any window")
        raise SystemExit(f"grid-stat-test: no files for {why} with WoFS, HAFS "
                         "and truth together; run regrid-wofs, regrid-hafs and "
                         "window-obs first.")
    return best[1]


def runs_for_selection(case, pick):
    """[(tag, fcst path, fcst variable)] for the chosen window."""
    member = case.eyeball.members[0]
    init, path = pick["wofs"]
    runs = [(f"wofs_m{member:02d}_i{init:%Y%m%d%H%M}", path,
             f"{ww.VAR}_m{member:02d}")]
    for config, h_init, h_path in pick["hafs"]:
        runs.append((f"{config.lower()}_c{h_init:%Y%m%d%H}", h_path, OBS_NAME))
    return runs


# =============================================================================
# Running grid_stat and reading the result
# =============================================================================

def run_prefix(case, tag):
    return f"{tag}{case.grid_stat_tag}"


def existing_stat(case, tag, out_dir):
    """The finished .stat of a run, or None. One written before the
    configuration or the MET mask last changed is out of date, not finished."""
    found = sorted(Path(out_dir).glob(
        f"grid_stat_{run_prefix(case, tag)}_*.stat"))
    if not found or found[-1].stat().st_size == 0:
        return None
    newest = max(p.stat().st_mtime for p in (case.grid_stat_conf_path,
                                             case.grid_stat_mask_path)
                 if p.exists())
    return found[-1] if found[-1].stat().st_mtime >= newest else None


def run_grid_stat(case, fcst, fcst_name, obs, tag, out_dir):
    """One grid_stat call; returns the .stat path."""
    tool = met_regrid.met_tool("grid_stat", case.regrid.met_bin_dir)
    prefix = run_prefix(case, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MODEL=tag, FCST_NAME=fcst_name,
               MASK_FILE=str(case.grid_stat_mask_path), OUT_PREFIX=prefix)
    cmd = [str(tool), str(fcst), str(obs), str(case.grid_stat_conf_path),
           "-outdir", str(out_dir), "-v", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = out_dir / f"{prefix}.log"
    log.write_text(f"{' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}")
    found = sorted(out_dir.glob(f"grid_stat_{prefix}_*.stat"))
    if proc.returncode != 0 or not found:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
        raise GridStatError(f"grid_stat failed for {prefix} (log: {log}):\n"
                            f"{tail}")
    return found[-1]


def read_stat(path):
    """[{"line_type", "mask", "fcst_thresh", "points", "values": [str]}] from a
    .stat file. The columns after LINE_TYPE depend on the type, so they stay a
    list; the 24 common columns are fixed."""
    rows = []
    with open(path) as fh:
        next(fh, None)                                   # header line
        for line in fh:
            t = line.split()
            if len(t) <= COMMON_COLUMNS:
                continue
            rows.append({"mask": t[16], "points": int(float(t[18])),
                         "fcst_thresh": t[19], "cov_thresh": t[21],
                         "line_type": t[23], "values": t[COMMON_COLUMNS:]})
    return rows


def _num(text):
    try:
        return float(text)
    except ValueError:
        return float("nan")


def contingency_scores(total, hits, fa, miss):
    """POD, FAR, CSI and ETS from counts; NaN when undefined."""
    def ratio(a, b):
        return a / b if b else float("nan")
    h_rand = ratio((hits + miss) * (hits + fa), total)
    return {"pod": ratio(hits, hits + miss), "far": ratio(fa, hits + fa),
            "csi": ratio(hits, hits + miss + fa),
            "ets": ratio(hits - h_rand, hits + miss + fa - h_rand)}


def expected_rows(case):
    s = case.grid_stat
    n_t, n_w = len(s.thresholds_mm), len(s.widths_km)
    return {"CTC": n_t, "CTS": n_t, "CNT": 2, "SL1L2": 2,
            "NBRCTC": n_t * n_w, "NBRCNT": n_t * n_w}


def print_digest(case, rows, label):
    """Row counts against what the configuration asks for, then the scores a
    person can check by hand."""
    want = expected_rows(case)
    have = {}
    for r in rows:
        have[r["line_type"]] = have.get(r["line_type"], 0) + 1
    print(f"\n{label}")
    print("  rows    " + "  ".join(
        f"{k} {have.get(k, 0)}/{n}" for k, n in want.items()))
    for k, n in want.items():
        if have.get(k, 0) != n:
            print(f"  WARNING: {k} has {have.get(k, 0)} rows, expected {n}")
    masks_seen = sorted({r["mask"] for r in rows})
    print(f"  mask    {', '.join(masks_seen)}")
    nbr = sorted({r["points"] for r in rows if r["line_type"] == "NBRCTC"})
    cells = neighbourhood_cells(case.grid_stat.widths_km, case.res_km)
    print(f"  nbrhd   points {nbr}, expected {[c * c for c in cells]}")

    print("  thresh      hits       FA     miss          CN    POD    FAR"
          "    CSI    ETS")
    for r in rows:
        if r["line_type"] != "CTC":
            continue
        v = [_num(x) for x in r["values"][:5]]
        total, hits, fa, miss, cn = v
        sc = contingency_scores(total, hits, fa, miss)
        print(f"  {r['fcst_thresh']:>6} {hits:9.0f} {fa:8.0f} {miss:8.0f} "
              f"{cn:11.0f} {sc['pod']:6.3f} {sc['far']:6.3f} {sc['csi']:6.3f} "
              f"{sc['ets']:6.3f}")
    print("  cells         N     fbar     obar     bias     rmse      mae")
    for r in rows:
        if r["line_type"] != "SL1L2":
            continue
        total, fbar, obar, fobar, ffbar, oobar, mae = (
            _num(x) for x in r["values"][:7])
        mse = ffbar - 2 * fobar + oobar
        which = "all" if r["fcst_thresh"] == "NA" else f"wet{r['fcst_thresh']}"
        print(f"  {which:>6} {total:9.0f} {fbar:8.3f} {obar:8.3f} "
              f"{fbar - obar:8.3f} {np.sqrt(max(mse, 0.0)):8.3f} {mae:8.3f}")


# =============================================================================
# Driver
# =============================================================================

def grid_stat_test_case(case):
    """Driver for `grid-stat-test`."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"grid-stat-test: {case.case_slug} has no complete "
                         "WoFS window; nothing to score.")
    if not case.mask_path.exists():
        raise SystemExit(f"grid-stat-test: {case.mask_path} is missing; run "
                         "build-mask first (same --res-km and --coast-km).")
    mask = masks.read_mask(case.mask_path)
    pick = choose_test(case, mask)
    window = pick["window"]
    runs = runs_for_selection(case, pick)
    s = case.grid_stat
    print(f"Truth  : {LABELS[case.truth_source]}  |  mask "
          f"{case.mask_path.name} ({int(mask.sum()):,} of {mask.size:,} cells)")
    print(f"Window : 3 h ending {window.end:%Y-%m-%d %HZ} (mean truth "
          f"{pick['mean_mm']:.2f} mm inside the mask)")
    print(f"Config : {len(s.thresholds_mm)} thresholds "
          f"({s.thresholds_mm[0]:g}-{s.thresholds_mm[-1]:g} mm), "
          f"neighborhoods {list(map(float, s.widths_km))} km = "
          f"{neighbourhood_cells(s.widths_km, case.res_km)} cells "
          f"({effective_km(s.widths_km, case.res_km)} km), "
          f"wet cells >= {s.wet_thresh_mm:g} mm")
    try:
        conf, _ = prepare(case, pick["truth"])
    except GridStatError as err:
        raise SystemExit(f"grid-stat-test: {err}") from None
    out_dir = case.grid_stat_window_dir(window.end)
    print(f"Output : {out_dir}\nWrote  : {conf}\n         "
          f"{case.grid_stat_mask_path}", flush=True)

    written, started = [], time.monotonic()
    for tag, path, name in runs:
        try:
            stat = run_grid_stat(case, path, name, pick["truth"], tag, out_dir)
        except GridStatError as err:
            raise SystemExit(str(err)) from None
        written.append(stat)
        print_digest(case, read_stat(stat),
                     f"{tag}  ->  {stat.name}  (+{time.monotonic() - started:.0f}s)")
    print(f"\nScored {len(written)} runs for one window under {out_dir}")
    return written


# =============================================================================
# E2: every window, every run
# =============================================================================

Job = namedtuple("Job", "window tag fcst name truth out_dir")


def members_in(path):
    """The member variables (rain_mm_m00, ...) of a WoFS 3 h file."""
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        return sorted(n for n in nc.variables if n.startswith(f"{ww.VAR}_m"))


def work_list(case):
    """([Job], [windows skipped for want of truth]). Every clock window of the
    scored period that has a truth file gets every run on disk: each member of
    each WoFS init and each HAFS cycle of each configuration. A window with no
    truth is skipped; one with truth and no runs just has no jobs."""
    jobs, skipped = [], []
    for w in aw.clock_windows(case.valid_start, case.valid_end):
        truth = case.obs_window_path(case.truth_source, w.end)
        if not truth.exists():
            skipped.append(w)
            continue
        out_dir = case.grid_stat_window_dir(w.end)
        for init, path in runs_for(case.wofs_regrid_3h_dir, w.end, "wofs",
                                   case.res_label):
            for name in members_in(path):
                member = int(name.rsplit("_m", 1)[1])
                jobs.append(Job(w, f"wofs_m{member:02d}_i{init:%Y%m%d%H%M}",
                                path, name, truth, out_dir))
        for config in case.hafs_models:
            for init, path in runs_for(case.hafs_regrid_dir_for(config),
                                       w.end, config.lower(), case.res_label):
                jobs.append(Job(w, f"{config.lower()}_c{init:%Y%m%d%H}",
                                path, OBS_NAME, truth, out_dir))
    return jobs, skipped


def worker_count():
    """grid_stat calls to run at once: GRID_STAT_WORKERS, else the cores SLURM
    gave the task, else 1."""
    for key in ("GRID_STAT_WORKERS", "SLURM_CPUS_PER_TASK"):
        try:
            n = int(os.environ[key])
        except (KeyError, ValueError):
            continue
        if n >= 1:
            return n
    return 1


def run_job(case, job):
    """{"status": cached | done | failed, "stat", "seconds", "error"}."""
    cached = existing_stat(case, job.tag, job.out_dir)
    if cached:
        return {"status": "cached", "stat": cached, "seconds": 0.0,
                "error": ""}
    started = time.monotonic()
    try:
        stat = run_grid_stat(case, job.fcst, job.name, job.truth, job.tag,
                             job.out_dir)
    except GridStatError as err:
        return {"status": "failed", "stat": None, "error": str(err),
                "seconds": time.monotonic() - started}
    return {"status": "done", "stat": stat, "error": "",
            "seconds": time.monotonic() - started}


def write_runs_csv(case, rows):
    path = case.grid_stat_runs_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["window_end", "run", "status", "seconds", "stat_file"])
        for end, tag, r in rows:
            out.writerow([f"{end:%Y%m%d%H}", tag, r["status"],
                          f"{r['seconds']:.1f}",
                          r["stat"].name if r["stat"] else ""])
    return path


def grid_stat_case(case):
    """Driver for `grid-stat`: score every window with every run."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        raise SystemExit(f"grid-stat: {case.case_slug} has no complete WoFS "
                         "window; nothing to score.")
    if not case.mask_path.exists():
        raise SystemExit(f"grid-stat: {case.mask_path} is missing; run "
                         "build-mask first (same --res-km and --coast-km).")
    jobs, skipped = work_list(case)
    if not jobs:
        raise SystemExit("grid-stat: no runs found with truth; run "
                         "regrid-wofs, regrid-hafs and window-obs first.")
    try:
        conf, _ = prepare(case, jobs[0].truth)
    except GridStatError as err:
        raise SystemExit(f"grid-stat: {err}") from None
    workers = worker_count()
    families = Counter(j.tag.split("_")[0] for j in jobs)
    windows = sorted({j.window.end for j in jobs})
    print(f"Truth  : {LABELS[case.truth_source]}  |  mask {case.mask_path.name}")
    print(f"Work   : {len(jobs):,} runs over {len(windows)} windows ("
          + ", ".join(f"{k} {v:,}" for k, v in sorted(families.items()))
          + f"); {workers} at a time")
    if skipped:
        print(f"Skipped: {len(skipped)} windows with no "
              f"{LABELS[case.truth_source]} file (run window-obs): "
              + ", ".join(f"{w.end:%m-%d %HZ}" for w in skipped))
    print(f"Output : {case.grid_stat_dir}\nConfig : {conf}\n", flush=True)

    started = time.monotonic()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, case, j): j for j in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            job, r = futures[future], future.result()
            results[(job.window.end, job.tag)] = r
            if r["status"] != "cached" or n % 100 == 0:
                print(f"  [{n:>5}/{len(jobs)}] {job.window.end:%m-%d %HZ} "
                      f"{job.tag:<28} {r['status']:<6} {r['seconds']:5.1f}s",
                      flush=True)
    rows = [(j.window.end, j.tag, results[(j.window.end, j.tag)])
            for j in sorted(jobs, key=lambda j: (j.window.end, j.tag))]
    csv_path = write_runs_csv(case, rows)

    status = Counter(r["status"] for _, _, r in rows)
    print(f"\n{'window end':>14} {'runs':>6} {'done':>6} {'cached':>7} "
          f"{'failed':>7}")
    for end in windows:
        per = Counter(r["status"] for e, _, r in rows if e == end)
        print(f"{end:%Y-%m-%d %HZ} {sum(per.values()):>6} {per['done']:>6} "
              f"{per['cached']:>7} {per['failed']:>7}")
    size = sum(r["stat"].stat().st_size for _, _, r in rows if r["stat"])
    print(f"\n{status['done']} done, {status['cached']} already done, "
          f"{status['failed']} failed; {size / 1e6:.1f} MB of .stat files; "
          f"{(time.monotonic() - started) / 60:.1f} min")
    print(f"Runs   : {csv_path}")
    if status["failed"]:
        failed = [(e, t, r) for e, t, r in rows if r["status"] == "failed"]
        for end, tag, r in failed[:3]:
            print(f"\nFAILED {end:%Y-%m-%d %HZ} {tag}:\n{r['error']}")
        raise SystemExit(f"grid-stat: {status['failed']} of {len(rows)} runs "
                         f"failed; rerun to retry them (finished runs are "
                         f"kept). Details in {csv_path}")
    return [r["stat"] for _, _, r in rows]
