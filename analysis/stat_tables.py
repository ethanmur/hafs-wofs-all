"""Parse the grid_stat .stat files of a case into tidy CSVs (plan step E3).

    python analysis/run.py storms/<case>.yaml parse-stat --res-km 3

Reads the .stat file of every run `grid-stat` scored (the checklist is
grid_stat_runs_<case>.csv, else the work list) and writes one CSV per line type
under data/grid-stat/csvs/, named stat_<linetype>_<case>[_3km][_coast50km].csv:
ctc, cts, cnt, sl1l2, nbrctc, nbrcnt. Line types have different value columns,
so they are separate files. Cases stay separate; pooling (E4) concatenates them.

Identity columns come from the MODEL tag (wofs_m00_i202410092100,
hfsa_c2024100918) and the window-end folder name, never from the .stat time
columns, which are wrong for WoFS (lead 0, valid 1970). `lead_h` is the lead at
the window end, so a forecast's first 3 h bucket has lead_h 3. `NA` in a .stat
file is an empty cell, never 0.

The command writes whatever it can parse, then exits non-zero listing runs with
no .stat file, files whose row counts differ from the configuration, files
whose MODEL tag is not the run they sit under, and duplicate keys.
"""

import csv
import math
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accum_windows as aw
import grid_stat_windows as gw

LINE_TYPES = ("CTC", "CTS", "CNT", "SL1L2", "NBRCTC", "NBRCNT")
NBR_TYPES = ("NBRCTC", "NBRCNT")
THRESH_TYPES = ("CTC", "CTS", "NBRCTC", "NBRCNT")
CELL_SET_TYPES = ("CNT", "SL1L2")

ID_COLUMNS = ("case", "storm", "deployment", "truth", "source", "member",
              "init_time", "valid_start", "valid_end", "lead_h")

# The columns after the 24 common ones, as (name, position). Confidence
# intervals are dropped (bootstrapping is off); positions follow MET 12.2's
# column order and are checked against real files in the tests.
_COUNTS = (("total", 0), ("fy_oy", 1), ("fy_on", 2), ("fn_oy", 3),
           ("fn_on", 4))
VALUES = {
    "CTC": _COUNTS,
    "NBRCTC": _COUNTS,
    "CTS": (("total", 0), ("baser", 1), ("fmean", 6), ("acc", 11),
            ("fbias", 16), ("pod", 19), ("podn", 24), ("pofd", 29),
            ("far", 34), ("csi", 39), ("ets", 44), ("hk", 47), ("hss", 52)),
    "NBRCNT": (("total", 0), ("fbs", 1), ("fss", 4), ("afss", 7),
               ("ufss", 10), ("f_rate", 13), ("o_rate", 16)),
    "CNT": (("total", 0), ("fbar", 1), ("fstdev", 6), ("obar", 11),
            ("ostdev", 16), ("pr_corr", 21), ("me", 31), ("estdev", 36),
            ("mbias", 41), ("mae", 44), ("mse", 47), ("bcmse", 50),
            ("rmse", 53)),
    "SL1L2": (("total", 0), ("fbar", 1), ("obar", 2), ("fobar", 3),
              ("ffbar", 4), ("oobar", 5), ("mae", 6)),
}
DERIVED = ("pod", "far", "csi", "ets", "fbias")     # CTC and NBRCTC only

MODEL_RE = re.compile(r"^(?P<source>[a-z0-9]+)_(?:m(?P<member>\d+)_i"
                      r"(?P<wofs>\d{12})|c(?P<cycle>\d{10}))$")


def extra_columns(line_type):
    """The columns between the identity and the values for this line type."""
    cols = []
    if line_type in THRESH_TYPES:
        cols.append("thresh_mm")
    if line_type in NBR_TYPES:
        cols += ["nbr_cells", "nbr_width_km"]
    if line_type == "NBRCTC":
        cols.append("cov_thresh")
    if line_type in CELL_SET_TYPES:
        cols += ["cell_set", "wet_thresh_mm"]
    return cols


def columns(line_type):
    cols = list(ID_COLUMNS) + extra_columns(line_type)
    cols += [name for name, _ in VALUES[line_type]]
    if line_type in ("CTC", "NBRCTC"):
        cols += list(DERIVED)
    return cols


def key_columns(line_type):
    return list(ID_COLUMNS) + extra_columns(line_type)


# =============================================================================
# Reading
# =============================================================================

def threshold_value(text):
    """The number in a MET threshold string (">=2.5", ">=0.5||"), or None."""
    if text in ("NA", ""):
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(m.group()) if m else None


def _fmt(value):
    """CSV text for a number; NaN and None are empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _cell(text):
    return "" if text == "NA" else text


def read_rows(path):
    """[dict] of a .stat file: model, interp_pnts, fcst_thresh, cov_thresh,
    line_type and the values after the common columns, all as text."""
    rows = []
    with open(path) as fh:
        next(fh, None)                                   # header line
        for line in fh:
            t = line.split()
            if len(t) <= gw.COMMON_COLUMNS:
                continue
            rows.append({"model": t[1], "interp_pnts": t[18],
                         "fcst_thresh": t[19], "cov_thresh": t[21],
                         "line_type": t[23], "values": t[gw.COMMON_COLUMNS:]})
    return rows


def parse_model(tag):
    """{"source", "member", "init"} from a MODEL tag, or None."""
    m = MODEL_RE.match(tag)
    if not m:
        return None
    if m.group("wofs"):
        init = datetime.strptime(m.group("wofs"), "%Y%m%d%H%M")
        member = f"m{int(m.group('member')):02d}"
    else:
        init = datetime.strptime(m.group("cycle"), "%Y%m%d%H")
        member = ""
    return {"source": m.group("source"), "member": member, "init": init}


def derived_scores(counts):
    """POD, FAR, CSI, ETS and frequency bias from CTC counts (hits, false
    alarms, misses, correct negatives); empty when undefined."""
    hits, fa, miss, cn = counts
    total = hits + fa + miss + cn
    scores = gw.contingency_scores(total, hits, fa, miss)
    scores["fbias"] = ((hits + fa) / (hits + miss)) if hits + miss else None
    return {k: scores[k] for k in DERIVED}


def row_dict(case, ident, r):
    """One output row for a .stat row, or None for an unknown line type or a
    row too short for the columns that line type needs."""
    lt = r["line_type"]
    if lt not in VALUES:
        return None
    vals = r["values"]
    if len(vals) <= max(i for _, i in VALUES[lt]):
        return None
    row = dict(ident)
    thr = threshold_value(r["fcst_thresh"])
    if lt in THRESH_TYPES:
        row["thresh_mm"] = _fmt(thr)
    if lt in NBR_TYPES:
        cells = int(round(math.sqrt(float(r["interp_pnts"]))))
        row["nbr_cells"] = cells
        row["nbr_width_km"] = _fmt(cells * case.res_km)
    if lt == "NBRCTC":
        row["cov_thresh"] = _fmt(threshold_value(r["cov_thresh"]))
    if lt in CELL_SET_TYPES:
        row["cell_set"] = "all" if r["fcst_thresh"] == "NA" else "wet"
        row["wet_thresh_mm"] = _fmt(thr)
    for name, i in VALUES[lt]:
        row[name] = _cell(vals[i])
    if lt in ("CTC", "NBRCTC"):
        counts = [int(float(row[k])) if row[k] != "" else 0
                  for k in ("fy_oy", "fy_on", "fn_oy", "fn_on")]
        row.update({k: _fmt(v) for k, v in derived_scores(counts).items()})
    return row


# =============================================================================
# Checklist and files
# =============================================================================

def checklist(case):
    """([(window end, run tag)] sorted, where it came from): the runs CSV that
    `grid-stat` wrote, else the work list."""
    path = case.grid_stat_runs_path
    if path.exists():
        with open(path, newline="") as fh:
            runs = [(datetime.strptime(r["window_end"], "%Y%m%d%H"), r["run"])
                    for r in csv.DictReader(fh)]
        return sorted(runs), path.name
    jobs, _ = gw.work_list(case)
    return sorted((j.window.end, j.tag) for j in jobs), "the work list"


def stat_file(case, end, tag):
    """The .stat file of a run (the last when there are several), or None."""
    prefix = gw.run_prefix(case, tag)
    name = re.compile(rf"^grid_stat_{re.escape(prefix)}_\d{{6}}L_\d{{8}}_"
                      rf"\d{{6}}V\.stat$")
    found = sorted(p for p in case.grid_stat_window_dir(end).glob(
        f"grid_stat_{prefix}_*.stat") if name.match(p.name))
    return found[-1] if found and found[-1].stat().st_size else None


def parse_run(case, end, tag, path):
    """({line type: [row]}, [problem]) for one .stat file."""
    problems = []
    rows = read_rows(path)
    models = {r["model"] for r in rows}
    if models != {tag}:
        return {}, [f"{end:%Y%m%d%H} {tag}: {path.name} holds MODEL tag(s) "
                    f"{sorted(models) or 'none'}, not {tag}"]
    parsed = parse_model(tag)
    if parsed is None:
        return {}, [f"{end:%Y%m%d%H} {tag}: not a wofs_m<NN>_i<init> or "
                    "<config>_c<cycle> tag"]
    lead = (end - parsed["init"]).total_seconds() / 3600
    start = end - timedelta(hours=aw.DEFAULT_WINDOW_H)
    ident = {"case": case.case_slug, "storm": case.storm_name,
             "deployment": case.deployment,
             "truth": case.truth_source, "source": parsed["source"],
             "member": parsed["member"],
             "init_time": f"{parsed['init']:%Y-%m-%d %H:%M}",
             "valid_start": f"{start:%Y-%m-%d %H:%M}",
             "valid_end": f"{end:%Y-%m-%d %H:%M}", "lead_h": _fmt(lead)}
    out = {lt: [] for lt in LINE_TYPES}
    for r in rows:
        row = row_dict(case, ident, r)
        if row is not None:
            out[r["line_type"]].append(row)
    have = {lt: len(v) for lt, v in out.items() if v}
    want = gw.expected_rows(case)
    if have != {lt: n for lt, n in want.items()}:
        diff = ", ".join(f"{lt} {have.get(lt, 0)}/{n}"
                         for lt, n in want.items() if have.get(lt, 0) != n)
        problems.append(f"{end:%Y%m%d%H} {tag}: rows differ from the "
                        f"configuration (have/want {diff})")
    return out, problems


def write_tables(case, tables):
    """Write each non-empty line type's rows; {line type: path}."""
    case.stat_csv_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lt in LINE_TYPES:
        if not tables[lt]:
            continue
        path = case.stat_csv_path(lt)
        with open(path, "w", newline="") as fh:
            out = csv.DictWriter(fh, fieldnames=columns(lt),
                                 extrasaction="ignore")
            out.writeheader()
            out.writerows(tables[lt])
        paths[lt] = path
    return paths


def duplicate_keys(tables):
    """['<line type>: <key>'] for each key that appears more than once."""
    problems = []
    for lt, rows in tables.items():
        keys = Counter(tuple(r.get(c, "") for c in key_columns(lt))
                       for r in rows)
        problems += [f"{lt}: duplicate key {k}" for k, n in keys.items()
                     if n > 1]
    return problems


def parse_stat_case(case):
    """Driver for `parse-stat`: {line type: csv path}."""
    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    runs, source = checklist(case)
    if not runs:
        raise SystemExit("parse-stat: no runs to parse; run grid-stat first.")
    print(f"Runs   : {len(runs):,} from {source}")
    tables = {lt: [] for lt in LINE_TYPES}
    problems, missing, used = [], [], set()
    for end, tag in runs:
        path = stat_file(case, end, tag)
        if path is None:
            missing.append(f"{end:%Y%m%d%H} {tag}")
            continue
        used.add(path)
        out, bad = parse_run(case, end, tag, path)
        problems += bad
        for lt, rows in out.items():
            tables[lt] += rows
    problems += duplicate_keys(tables)
    others = sum(1 for d in case.grid_stat_dir.glob("[0-9]" * 10)
                 for p in d.glob("grid_stat_*.stat") if p not in used)
    paths = write_tables(case, tables)

    parsed = len(runs) - len(missing)
    print(f"Parsed : {parsed:,} .stat files, {others} other .stat files "
          "ignored (another resolution or mask buffer)")
    print(f"\n{'line type':<8} {'rows':>9}  file")
    for lt in LINE_TYPES:
        print(f"{lt:<8} {len(tables[lt]):>9,}  "
              f"{paths[lt].name if lt in paths else '-'}")
    print(f"\nOutput : {case.stat_csv_dir}")
    if missing or problems:
        lines = ([f"{len(missing)} runs with no .stat file: "
                  + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")]
                 if missing else [])
        lines += problems[:10]
        if len(problems) > 10:
            lines.append(f"... and {len(problems) - 10} more problems")
        raise SystemExit("parse-stat: " + str(len(missing) + len(problems))
                         + " problems (tables written for what parsed):\n  "
                         + "\n  ".join(lines))
    return paths
