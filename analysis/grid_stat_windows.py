"""grid_stat configuration and a one-window test run (plan step E1).

    python analysis/run.py storms/<case>.yaml grid-stat-test --res-km 3

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

Files, under <grid_stat_dir> (data/grid-stat/<dep>_<storm>/):
    GridStat_<case>[_3km][_coast50km].conf
    <case>[_3km][_coast50km]_FULL_mask.nc
    <window end YYYYMMDDHH>/grid_stat_<run>[_3km]_<lead>L_<valid>V.stat (+ .log)
where <run> is wofs_m00_i<init> or <hfsa|hfsb>_c<cycle>.
"""

import os
import subprocess
import sys
import time
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
    path = case.grid_stat_conf_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_text(case))
    return path


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

def run_grid_stat(case, fcst, fcst_name, obs, tag, out_dir):
    """One grid_stat call; returns the .stat path."""
    tool = met_regrid.met_tool("grid_stat", case.regrid.met_bin_dir)
    prefix = f"{tag}{case.grid_stat_tag}"
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
        raise SystemExit(f"grid_stat failed for {prefix} (log: {log}):\n{tail}")
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
    conf = write_config(case)
    ok, why = mask_check.make_met_mask(case, pick["truth"],
                                       case.grid_stat_mask_path, name="FULL")
    if not ok:
        raise SystemExit(f"grid-stat-test: cannot build the MET mask: {why}")
    out_dir = case.grid_stat_window_dir(window.end)
    print(f"Output : {out_dir}\nWrote  : {conf}\n         "
          f"{case.grid_stat_mask_path}", flush=True)

    written, started = [], time.monotonic()
    for tag, path, name in runs:
        stat = run_grid_stat(case, path, name, pick["truth"], tag, out_dir)
        written.append(stat)
        print_digest(case, read_stat(stat),
                     f"{tag}  ->  {stat.name}  (+{time.monotonic() - started:.0f}s)")
    print(f"\nScored {len(written)} runs for one window under {out_dir}")
    return written
