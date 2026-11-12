"""Pool the parsed grid_stat tables across runs, windows and cases (plan E4).

    python analysis/pool_stats.py --csv-dir data/grid-stat/csvs [--out-dir DIR]
        [--tag _3km] [--level case storm all] [--matched windows|leads]

Plain Python (pandas and numpy; no MET), so it runs the same on a laptop and on
Orion. It reads the stat_<linetype>_<case><tag>.csv files that `parse-stat`
wrote and writes pooled_<level>_<linetype><tag>[_matched].csv beside them (or
in --out-dir).

Scores are derived AFTER summing, never averaged from per-run scores: the
ratio of sums is not the mean of ratios (ETS depends on each run's base rate
and N). What is summed:

  ctc, nbrctc   the four counts, then POD FAR CSI ETS (MET's GSS) FBIAS HK HSS
                POFD PODN ACC BASER FMEAN from the pooled counts
  sl1l2         the partial means, weighted by `total`, then ME MSE RMSE MAE
                MBIAS the standard deviations and the correlation
  nbrcnt        FBS weighted by `total`; FSS = 1 - FBS / FBS_worst with
                FBS_worst recovered per run as FBS / (1 - FSS) and pooled the
                same way; f_rate and o_rate weighted; UFSS = 0.5 + o_rate / 2

Levels: `case` (one deployment), `storm` (all deployments of a storm) and
`all` (every case). Whatever the level, a row is one (source, lead_h,
threshold, scale / cell set) with every window, member and cycle of the level
pooled into it. A WoFS row therefore scores the typical member, not the
ensemble. `n_runs` and `n_windows` say how much went in, and `obs_events` (hits
+ misses) how many cells the threshold has to score: a threshold with few
events is noise whatever its score.

Cases in EXCLUDED_CASES (Beryl 20240710_d1: centered over Iowa, not
Beryl-related) are left out of every table; `--include-excluded` brings them
back (files end _withexcluded) and `--exclude-case` adds more. The per-case
stat_*.csv are never changed.

`--matched fixed` keeps the windows that have every lead of FIXED_LEADS (HAFS
6, 12 ... 48 h; WoFS 3-6 h) and only those leads, so a lead curve is drawn from
the same windows at every lead. `--pool-leads` sums all leads together
(lead_h = all; files end _allleads).

`--matched windows` keeps only the windows (per case) where every source has
runs, so WoFS and HAFS are pooled over the same windows. `--matched leads` is
stricter: only (window, lead_h) pairs every source has, the head-to-head sample.
WoFS runs are 6 h long and HAFS leads start at 6 here, so that is lead_h 6, and
only the windows ending 00/06/12/18Z (HAFS cycles are 6-hourly). The output
names gain `_matched_windows` or `_matched_leads`.

The pooled NBRCNT FSS is exact up to MET's 5-decimal rounding of FBS and FSS.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = {"case": ["case", "storm", "deployment"], "storm": ["storm"],
          "all": []}
BASE_KEYS = ["truth", "source", "lead_h"]
EXTRA_KEYS = {"ctc": ["thresh_mm"],
              "nbrctc": ["thresh_mm", "nbr_cells", "nbr_width_km",
                         "cov_thresh"],
              "sl1l2": ["cell_set", "wet_thresh_mm"],
              "nbrcnt": ["thresh_mm", "nbr_cells", "nbr_width_km"]}
LINE_TYPES = tuple(EXTRA_KEYS)
FIRST_BUCKET = ("exclude", "include")
HAFS_SOURCES = ("hfsa", "hfsb")
CASE_RE = r"\d{4}_\d{2}[a-z]_[a-z]+_\d{8}_d\d+"

# Cases left out of every pooled table unless asked back (--include-excluded).
# The raw per-case tables are untouched.
EXCLUDED_CASES = {
    "2024_02l_beryl_20240710_d1":
        "domain centered over Iowa; not Beryl-related by the best-track "
        "distances and tracks, and it has a single HAFS window",
}


def resolve_excluded(extra=(), include_default=False):
    """The set of case slugs to drop: the default list unless
    `include_default`, plus any `extra`."""
    out = set() if include_default else set(EXCLUDED_CASES)
    return out | set(extra)


def excluded_tag(excluded):
    """File-name ending for a non-default exclusion set ("" for the default)."""
    if set(excluded) == set(EXCLUDED_CASES):
        return ""
    return "_withexcluded" if not excluded else "_customexcl"


def _div(a, b):
    """a / b, NaN where b is not positive."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


# =============================================================================
# Reading
# =============================================================================

def csv_files(csv_dir, line_type, tag):
    """The per-case tables of one line type for this tag ("_3km"), sorted. A
    tagless request does not match tagged files, nor the reverse."""
    pat = re.compile(rf"^stat_{line_type}_{CASE_RE}{re.escape(tag)}\.csv$")
    return sorted(p for p in Path(csv_dir).glob(f"stat_{line_type}_*.csv")
                  if pat.match(p.name))


def load(csv_dir, line_type, tag, exclude=None, first_bucket="exclude"):
    """The per-case tables of a line type, minus the excluded cases (default:
    EXCLUDED_CASES; pass () to keep every case). `first_bucket` "exclude" drops
    the HAFS 0-3 h rows (a cycle initialized at the window start, lead_h 3: the
    spin-up); "include" keeps them. WoFS lead 3 is always kept."""
    files = csv_files(csv_dir, line_type, tag)
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, dtype={"member": str}) for f in files],
                   ignore_index=True)
    exclude = set(EXCLUDED_CASES if exclude is None else exclude)
    df = df[~df["case"].isin(exclude)]
    if first_bucket not in FIRST_BUCKET:
        raise ValueError(f"first_bucket must be one of {FIRST_BUCKET}")
    if first_bucket == "exclude":
        df = df[~(df["source"].isin(HAFS_SOURCES) & (df["lead_h"] == 3))]
    df = df.reset_index(drop=True)
    if df.empty:
        return None
    # blank member and wet_thresh_mm are legitimate (HAFS; all cells), so keep
    # them out of groupby's NaN dropping by giving the key columns a value
    for col in ("member",):
        if col in df:
            df[col] = df[col].fillna("")
    return df


# Mode "fixed": the leads each source must have in a window for the window to
# count, so a lead curve is drawn from the same windows at every lead. Windows
# ending 00/06/12/18Z have HAFS leads 6, 12, ... 48 (6-hourly cycles, 48 h
# lookback) and WoFS leads 3-6 (hourly inits of a 6 h run); the other four daily
# windows have HAFS leads 9 ... 51 and are not used.
FIXED_LEADS = {"wofs": (3, 4, 5, 6), "hfsa": tuple(range(6, 49, 6)),
               "hfsb": tuple(range(6, 49, 6))}


# Mode "fixed-spinup": the other window family (ending 03/09/15/21Z), whose HAFS
# leads are 3 (the 0-3 h spin-up bucket), 9 ... 51. Needs --first-bucket include.
FIXED_LEADS_SPINUP = {"wofs": (3, 4, 5, 6), "hfsa": tuple(range(3, 52, 6)),
                      "hfsb": tuple(range(3, 52, 6))}


def fixed_windows(df, leads=None):
    """Rows at FIXED_LEADS (or `leads`), only for the windows where every source
    in the table has every one of its fixed leads."""
    FIXED = leads or FIXED_LEADS
    present = list(df["source"].unique())
    unknown = [s for s in present if s not in FIXED]
    if unknown:
        raise ValueError(f"no fixed leads defined for source(s) {unknown}")
    mask = np.zeros(len(df), bool)
    for s in present:
        mask |= ((df["source"] == s) & df["lead_h"].isin(FIXED[s])
                 ).to_numpy()
    d = df[mask]
    have = (d.groupby(["case", "valid_end", "source"])["lead_h"].nunique()
            .unstack("source"))
    need = pd.Series({s: len(FIXED[s]) for s in present})
    ok = have.reindex(columns=present).eq(need).all(axis=1)
    idx = pd.MultiIndex.from_frame(d[["case", "valid_end"]])
    return d[idx.isin(ok[ok].index)]


def matched(df, mode):
    """Rows restricted, per case, to the windows (mode "windows") or the
    (window, lead_h) pairs (mode "leads") that every source in the table has
    runs for; mode "fixed" keeps the windows with every FIXED_LEADS lead."""
    if mode == "fixed":
        return fixed_windows(df)
    if mode == "fixed-spinup":
        return fixed_windows(df, FIXED_LEADS_SPINUP)
    keys = ["case", "valid_end"] + (["lead_h"] if mode == "leads" else [])
    have = df.groupby(keys)["source"].nunique()
    keep = have[have == df["source"].nunique()].index
    idx = pd.MultiIndex.from_frame(df[keys])
    return df[idx.isin(keep)]


def coverage(df):
    """Windows and runs per case and source, for the log."""
    runs = df.drop_duplicates(["case", "source", "member", "init_time",
                               "valid_end"])
    return (runs.groupby(["case", "source"])
            .agg(windows=("valid_end", "nunique"), runs=("valid_end", "size"))
            .unstack("source"))


# =============================================================================
# Scores from pooled sums
# =============================================================================

def contingency_scores(h, fa, miss, cn):
    """The scores MET's CTS reports, from pooled counts (arrays)."""
    h, fa, miss, cn = (np.asarray(x, float) for x in (h, fa, miss, cn))
    n = h + fa + miss + cn
    pod, pofd = _div(h, h + miss), _div(fa, fa + cn)
    h_rand = _div((h + miss) * (h + fa), n)
    return {
        "baser": _div(h + miss, n), "fmean": _div(h + fa, n),
        "acc": _div(h + cn, n), "fbias": _div(h + fa, h + miss),
        "pod": pod, "podn": _div(cn, fa + cn), "pofd": pofd,
        "far": _div(fa, h + fa), "csi": _div(h, h + miss + fa),
        "ets": _div(h - h_rand, h + miss + fa - h_rand), "hk": pod - pofd,
        "hss": _div(2 * (h * cn - fa * miss),
                    (h + miss) * (miss + cn) + (h + fa) * (fa + cn)),
    }


def _with_window_id(df):
    """A copy with `_wid`, one value per (case, window), to count windows."""
    df = df.copy()
    df["_wid"] = df["case"] + "|" + df["valid_end"]
    return df


def _group(df, keys):
    # dropna=False: blank wet_thresh_mm (all cells) and cov_thresh must group
    return df.groupby(keys, dropna=False, sort=True)


# What is added up when runs, windows or cases are pooled. For sl1l2 and nbrcnt
# these are the weighted sums (value x total), so every column is additive and a
# bootstrap can re-add any subset of windows.
_PARTS = ("fbar", "obar", "fobar", "ffbar", "oobar", "mae")
STAT_COLS = {
    "ctc": ["total", "fy_oy", "fy_on", "fn_oy", "fn_on"],
    "nbrctc": ["total", "fy_oy", "fy_on", "fn_oy", "fn_on"],
    "sl1l2": ["total"] + [f"w_{p}" for p in _PARTS],
    "nbrcnt": ["total", "w_fbs", "w_worst", "w_f", "w_o"],
}


def additive(df, line_type):
    """A copy of `df` with `_wid` and the additive columns of its line type."""
    df = _with_window_id(df)
    if line_type == "sl1l2":
        w = df["total"].astype(float)
        for p in _PARTS:
            df[f"w_{p}"] = (df[p].astype(float) * w).where(w > 0, 0.0
                                                          ).fillna(0.0)
    elif line_type == "nbrcnt":
        w = df["total"].astype(float)
        fss = df["fss"].astype(float)
        # FBS_worst of a run; FSS is empty or 1 only when there is nothing to
        # misplace (both fields dry), where the worst case is 0 too
        worst = (df["fbs"] / (1 - fss)).where((fss < 1) & fss.notna(), 0.0)
        df["w_fbs"] = df["fbs"].astype(float) * w
        df["w_worst"] = worst * w
        df["w_f"] = df["f_rate"].astype(float) * w
        df["w_o"] = df["o_rate"].astype(float) * w
    return df


def derive(line_type, s):
    """{score: array} from summed columns `s` ({STAT_COLS name: array}, any
    shape, so the same arithmetic serves pooled tables and bootstrap draws).
    Counts and means come back too, in table order."""
    if line_type in ("ctc", "nbrctc"):
        return {"obs_events": s["fy_oy"] + s["fn_oy"],
                **contingency_scores(s["fy_oy"], s["fy_on"], s["fn_oy"],
                                     s["fn_on"])}
    total = np.asarray(s["total"], float)
    if line_type == "sl1l2":
        m = {p: _div(s[f"w_{p}"], total) for p in _PARTS}
        mse = m["ffbar"] - 2 * m["fobar"] + m["oobar"]
        var_f = np.maximum(m["ffbar"] - m["fbar"] ** 2, 0)
        var_o = np.maximum(m["oobar"] - m["obar"] ** 2, 0)
        return {**m, "me": m["fbar"] - m["obar"], "mse": mse,
                "rmse": np.sqrt(np.maximum(mse, 0)),
                "mbias": _div(m["fbar"], np.where(m["obar"] > 0, m["obar"],
                                                  np.nan)),
                "fstdev": np.sqrt(var_f), "ostdev": np.sqrt(var_o),
                "pr_corr": _div(m["fobar"] - m["fbar"] * m["obar"],
                                np.sqrt(var_f * var_o))}
    o_rate = _div(s["w_o"], total)
    return {"fbs": _div(s["w_fbs"], total),
            "fss": 1 - _div(s["w_fbs"], s["w_worst"]),
            "f_rate": _div(s["w_f"], total), "o_rate": o_rate,
            "ufss": 0.5 + o_rate / 2}


def pool(df, line_type, level):
    """The pooled table of one line type at one level."""
    keys = LEVELS[level] + BASE_KEYS + EXTRA_KEYS[line_type]
    cols = STAT_COLS[line_type]
    g = _group(additive(df, line_type), keys)
    out = g[cols].sum()
    derived = derive(line_type, {c: out[c].to_numpy() for c in cols})
    out = out.drop(columns=[c for c in cols if c.startswith("w_")])
    for name, vals in derived.items():
        out[name] = vals
    out.insert(0, "n_windows", g["_wid"].nunique())
    out.insert(0, "n_runs", g.size())
    return out.reset_index()


# =============================================================================
# Command line
# =============================================================================

def name_suffix(mode=None, pool_leads=False, excluded=None,
                first_bucket="exclude"):
    """The file-name ending of a sample: _matched_<mode>, then _allleads, then
    _withexcluded / _customexcl when the excluded cases are not the default,
    then _withspinup when the HAFS 0-3 h bucket is included."""
    return ((f"_matched_{mode}" if mode else "")
            + ("_allleads" if pool_leads else "")
            + ("" if excluded is None else excluded_tag(excluded))
            + ("_withspinup" if first_bucket == "include" else ""))


def out_path(out_dir, level, line_type, tag, mode=None, prefix="pooled",
             pool_leads=False, excluded=None, first_bucket="exclude"):
    return Path(out_dir) / (f"{prefix}_{level}_{line_type}{tag}"
                            f"{name_suffix(mode, pool_leads, excluded, first_bucket)}.csv")


def pool_all(csv_dir, out_dir=None, tag="_3km", levels=tuple(LEVELS),
             mode=None, pool_leads=False, log=print, exclude=None,
             first_bucket="exclude"):
    """{(level, line type): path}. Missing line types are skipped with a note,
    and an empty directory is an error."""
    out_dir = Path(out_dir) if out_dir else Path(csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = set(EXCLUDED_CASES if exclude is None else exclude)
    check_first_bucket(mode, first_bucket)
    if exclude:
        log("Excluded : " + ", ".join(sorted(exclude)))
    log(f"HAFS 0-3 h bucket: {first_bucket}d")
    written = {}
    for lt in LINE_TYPES:
        df = load(csv_dir, lt, tag, exclude, first_bucket)
        if df is None:
            log(f"{lt:<7} no stat_{lt}_*{tag}.csv in {csv_dir}; skipped")
            continue
        if mode:
            df = matched(df, mode)
        if pool_leads:
            df = df.assign(lead_h="all")
        if lt == "ctc":
            log(f"\nCases   : {df['case'].nunique()}   (windows / runs per "
                "source)")
            log(coverage(df).to_string())
        for level in levels:
            table = pool(df, lt, level)
            path = out_path(out_dir, level, lt, tag, mode,
                            pool_leads=pool_leads, excluded=exclude,
                            first_bucket=first_bucket)
            table.to_csv(path, index=False, float_format="%.6g")
            written[(level, lt)] = path
            log(f"{lt:<7} {level:<5} {len(table):>7,} rows -> {path.name}")
    if not written:
        raise SystemExit(f"pool_stats: no stat_*{tag}.csv files in {csv_dir}; "
                         "run parse-stat first (or pass the right --tag).")
    return written


def check_first_bucket(mode, first_bucket):
    if first_bucket not in FIRST_BUCKET:
        raise SystemExit(f"--first-bucket must be one of {FIRST_BUCKET}")
    if mode == "fixed-spinup" and first_bucket != "include":
        raise SystemExit("--matched fixed-spinup needs --first-bucket include "
                         "(its HAFS lead 3 is the 0-3 h bucket)")


def add_exclude_args(ap):
    ap.add_argument("--first-bucket", choices=FIRST_BUCKET, default="exclude",
                    help="HAFS 0-3 h spin-up bucket (cycle initialized at the "
                         "window start): exclude (default) or include (files "
                         "gain _withspinup)")
    ap.add_argument("--exclude-case", nargs="+", default=[], metavar="CASE",
                    help="also leave these case slugs out (default list: "
                         + ", ".join(EXCLUDED_CASES) + ")")
    ap.add_argument("--include-excluded", action="store_true",
                    help="keep the default-excluded cases (files gain "
                         "_withexcluded)")


def excluded_from_args(a):
    return resolve_excluded(a.exclude_case, a.include_excluded)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv-dir", required=True,
                    help="folder holding the parse-stat tables")
    ap.add_argument("--out-dir", help="default: --csv-dir")
    ap.add_argument("--tag", default="_3km",
                    help="file-name suffix of the run: _3km (default), "
                         "_3km_coast50km, or '' for 6 km")
    ap.add_argument("--level", nargs="+", choices=list(LEVELS),
                    default=list(LEVELS))
    ap.add_argument("--matched", choices=("windows", "leads", "fixed", "fixed-spinup"),
                    help="pool only the windows, or the (window, lead) "
                         "pairs, every source has runs for; fixed: the "
                         "windows that have every lead of FIXED_LEADS")
    ap.add_argument("--pool-leads", action="store_true",
                    help="sum all leads together (lead_h becomes 'all')")
    add_exclude_args(ap)
    a = ap.parse_args(argv)
    pool_all(a.csv_dir, a.out_dir, a.tag, tuple(a.level), a.matched,
             a.pool_leads, exclude=excluded_from_args(a),
             first_bucket=a.first_bucket)


if __name__ == "__main__":
    main(sys.argv[1:])
