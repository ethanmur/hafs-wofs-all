"""Uncertainty for the pooled scores (plan E4b): a window bootstrap and a
leave-one-storm-out check.

    python analysis/bootstrap_stats.py --csv-dir data/grid-stat/csvs
        [--out-dir DIR] [--tag _3km] [--level all storm case]
        [--matched windows|leads|fixed] [--pool-leads] [--n-boot 1000]
        [--seed 0] [--no-loo]

Plain Python (pandas and numpy), like pool_stats.py, which it builds on.

WINDOW BOOTSTRAP. A window is one 3 h valid-time block of one deployment; every
run, threshold and scale scored for it travels with it, so the correlation
between the 18 WoFS members, the HAFS cycles and the thresholds of a window is
kept. One resample draws as many windows as there are, with replacement, adds up
the counts and partial sums of the drawn windows, and derives the scores from
those sums, exactly as pool_stats does. `--n-boot` resamples (default 1000)
give the 2.5th and 97.5th percentiles: `<score>_lo` and `<score>_hi` beside the
point estimate computed from all windows. An interval is left empty when fewer
than 90 % of the resamples have the score defined (a threshold few windows
reach). Rows are one (source, lead_h, threshold, scale or cell set), as in
pooled_*.csv, and `n_windows` is how many windows that row has.

With 12 windows there are C(23, 12) = 1,352,078 distinct resamples, so 1,000
draws are nearly all different. Windows of one storm are correlated, which the
resampling ignores: read the interval as a lower bound on the uncertainty. The
interval cannot cover weather the windows did not sample.

PAIRED DIFFERENCES (with --matched). WoFS minus HFSA, WoFS minus HFSB and HFSA
minus HFSB, scored on the same resampled windows, so a difference whose interval
contains zero is not distinguishable from none. Without --matched the sources
cover different windows and a difference would compare samples, so none is
written.

LEAVE ONE STORM OUT. The pooled table recomputed 4 (or however many) times, each
time without one storm, plus the full-sample row (`dropped_storm` = (none)). It
is a sensitivity check, not an interval: with this few storms there are too few
distinct storm resamples (C(7, 4) = 35 for 4) for a bootstrap.

Files (beside the stat_*.csv, or in --out-dir):
    boot_<level>_<linetype><tag>[_matched_<mode>].csv      point estimate + CI
    bootdiff_<level>_<linetype><tag>_matched_<mode>.csv    paired differences
    loo_storm_<linetype><tag>[_matched_<mode>].csv         leave one storm out
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool_stats as ps

# Scores that get an interval; counts and the other derived columns do not.
CI_SCORES = {
    "ctc": ["pod", "far", "csi", "ets", "fbias", "hk", "hss"],
    "nbrctc": ["pod", "far", "csi", "ets", "fbias", "hk", "hss"],
    "sl1l2": ["me", "rmse", "mae", "mbias", "pr_corr"],
    "nbrcnt": ["fss", "fbs"],
}
PAIRS = (("wofs", "hfsa"), ("wofs", "hfsb"), ("hfsa", "hfsb"))
ALPHA = 5.0                     # percent: 95 % intervals
MIN_VALID_FRAC = 0.9
CHUNK = 200                     # resamples scored at a time (memory)


def _interval(draws):
    """(lo, hi) per column of `draws` (resamples x columns); NaN where too few
    resamples define the score."""
    valid = np.isfinite(draws).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(draws, [ALPHA / 2, 100 - ALPHA / 2], axis=0)
    bad = valid < MIN_VALID_FRAC * draws.shape[0]
    lo, hi = np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)
    return lo, hi


def _dense(add, line_type, keys):
    """(A, present, group table, n windows): A[w, g, s] is the sum of stat `s`
    for group `g` over the runs of window `w`, zero where the group has none."""
    cols = ps.STAT_COLS[line_type]
    per = add.groupby(keys + ["_wid"], dropna=False, sort=True)[cols
                                                               ].sum()
    per = per.reset_index()
    gid = per.groupby(keys, dropna=False, sort=True).ngroup().to_numpy()
    widx, wids = pd.factorize(per["_wid"], sort=True)
    n_g, n_w = int(gid.max()) + 1, len(wids)
    A = np.zeros((n_w, n_g, len(cols)))
    A[widx, gid] = per[cols].to_numpy(float)
    present = np.zeros((n_w, n_g), bool)
    present[widx, gid] = True
    table = (per.assign(_gid=gid).drop_duplicates("_gid").sort_values("_gid")
             [keys].reset_index(drop=True))
    return A, present, table, n_w


def _scores(line_type, sums, names):
    s = {c: sums[..., i] for i, c in enumerate(ps.STAT_COLS[line_type])}
    d = ps.derive(line_type, s)
    return {n: d[n] for n in names}


def bootstrap_group(df, line_type, keys, n_boot, rng, diffs):
    """(estimates, differences or None) for one set of windows. `estimates` has
    one row per group; `differences` one per (pair, shared group)."""
    names = CI_SCORES[line_type]
    add = ps.additive(df, line_type)
    A, present, table, n_w = _dense(add, line_type, keys)
    point = _scores(line_type, A.sum(axis=0), names)
    p = np.full(n_w, 1.0 / n_w)
    draws = {n: [] for n in names}
    for start in range(0, n_boot, CHUNK):
        k = min(CHUNK, n_boot - start)
        counts = rng.multinomial(n_w, p, size=k).astype(float)
        sums = np.tensordot(counts, A, axes=(1, 0))          # (k, groups, s)
        for n, v in _scores(line_type, sums, names).items():
            draws[n].append(v)
    draws = {n: np.concatenate(v) for n, v in draws.items()}

    est = table.copy()
    est["n_windows"] = present.sum(axis=0)
    est["n_boot"] = n_boot
    totals = A.sum(axis=0)
    est["total"] = totals[:, 0]
    if line_type in ("ctc", "nbrctc"):
        est["obs_events"] = totals[:, 1] + totals[:, 3]
    for n in names:
        lo, hi = _interval(draws[n])
        est[n], est[f"{n}_lo"], est[f"{n}_hi"] = point[n], lo, hi
    if not diffs:
        return est, None
    return est, _differences(table, keys, names, point, draws, n_boot)


def _differences(table, keys, names, point, draws, n_boot):
    other = [k for k in keys if k != "source"]
    # one text key per group; built row by row because astype(str) keeps NaN
    # (the all-cells wet_thresh_mm) as a missing value in pandas 3
    tag = pd.Series(["|".join(map(str, row)) for row in
                     table[other].itertuples(index=False, name=None)],
                    index=table.index)
    by_source = {s: pd.Series(table.index[table["source"] == s].to_numpy(),
                              index=tag[table["source"] == s].to_numpy())
                 for s in table["source"].unique()}
    frames = []
    for a, b in PAIRS:
        if a not in by_source or b not in by_source:
            continue
        shared = by_source[a].index.intersection(by_source[b].index)
        if shared.empty:
            continue
        ia, ib = by_source[a][shared].to_numpy(), by_source[b][shared].to_numpy()
        rows = table.loc[ia, other].reset_index(drop=True)
        rows.insert(0, "source_b", b)
        rows.insert(0, "source_a", a)
        rows["n_boot"] = n_boot
        for n in names:
            lo, hi = _interval(draws[n][:, ia] - draws[n][:, ib])
            rows[f"{n}_diff"] = point[n][ia] - point[n][ib]
            rows[f"{n}_lo"], rows[f"{n}_hi"] = lo, hi
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else None


def bootstrap(df, line_type, level, n_boot=1000, seed=0, diffs=False):
    """(estimates, differences or None) at one level: windows are resampled
    within each level group (a case, a storm, or everything)."""
    level_cols = ps.LEVELS[level]
    keys = ps.BASE_KEYS + ps.EXTRA_KEYS[line_type]
    groups = ([((), df)] if not level_cols else
              [(k if isinstance(k, tuple) else (k,), sub)
               for k, sub in df.groupby(level_cols, sort=True)])
    ests, dfs = [], []
    for i, (label, sub) in enumerate(groups):
        rng = np.random.default_rng([seed, i])
        est, diff = bootstrap_group(sub, line_type, keys, n_boot, rng, diffs)
        for table in (est, diff):
            if table is not None:
                for col, val in reversed(list(zip(level_cols, label))):
                    table.insert(0, col, val)
        ests.append(est)
        if diff is not None:
            dfs.append(diff)
    return (pd.concat(ests, ignore_index=True),
            pd.concat(dfs, ignore_index=True) if dfs else None)


def leave_one_storm_out(df, line_type):
    """The pooled table with each storm dropped in turn, and the full sample as
    dropped_storm "(none)"."""
    parts = [("(none)", df)] + [(s, df[df["storm"] != s])
                                for s in sorted(df["storm"].unique())]
    out = []
    for dropped, sub in parts:
        if sub.empty:
            continue
        t = ps.pool(sub, line_type, "all")
        t.insert(0, "dropped_storm", dropped)
        out.append(t)
    return pd.concat(out, ignore_index=True)


# =============================================================================
# Command line
# =============================================================================

def bootstrap_all(csv_dir, out_dir=None, tag="_3km", levels=("all",),
                  mode=None, n_boot=1000, seed=0, loo=True, pool_leads=False,
                  log=print, exclude=None):
    """{name: path} of the files written."""
    out_dir = Path(out_dir) if out_dir else Path(csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = set(ps.EXCLUDED_CASES if exclude is None else exclude)
    if exclude:
        log("Excluded : " + ", ".join(sorted(exclude)))
    written = {}
    for lt in ps.LINE_TYPES:
        df = ps.load(csv_dir, lt, tag, exclude)
        if df is None:
            log(f"{lt:<7} no stat_{lt}_*{tag}.csv in {csv_dir}; skipped")
            continue
        if mode:
            df = ps.matched(df, mode)
        if df.empty:
            log(f"{lt:<7} no windows left after --matched {mode}; skipped")
            continue
        if pool_leads:
            df = df.assign(lead_h="all")
        n_w = df[["case", "valid_end"]].drop_duplicates().shape[0]
        log(f"{lt:<7} {n_w} windows, {df['storm'].nunique()} storms, "
            f"{n_boot} resamples")
        for level in levels:
            est, diff = bootstrap(df, lt, level, n_boot, seed,
                                  diffs=bool(mode))
            path = ps.out_path(out_dir, level, lt, tag, mode, "boot",
                               pool_leads, exclude)
            est.to_csv(path, index=False, float_format="%.6g")
            written[("boot", level, lt)] = path
            log(f"        {level:<5} {len(est):>7,} rows -> {path.name}")
            if diff is not None:
                path = ps.out_path(out_dir, level, lt, tag, mode, "bootdiff",
                                   pool_leads, exclude)
                diff.to_csv(path, index=False, float_format="%.6g")
                written[("bootdiff", level, lt)] = path
                log(f"        {level:<5} {len(diff):>7,} rows -> {path.name}")
        if loo:
            table = leave_one_storm_out(df, lt)
            path = (Path(out_dir) / f"loo_storm_{lt}{tag}"
                    f"{ps.name_suffix(mode, pool_leads, exclude)}.csv")
            table.to_csv(path, index=False, float_format="%.6g")
            written[("loo", "storm", lt)] = path
            log(f"        loo   {len(table):>7,} rows -> {path.name}")
    if not written and mode:
        unit = "window" if mode == "windows" else "(window, lead) pair"
        raise SystemExit(f"bootstrap_stats: nothing is left after --matched "
                         f"{mode}: no {unit} has runs from every source.")
    if not written:
        raise SystemExit(f"bootstrap_stats: no stat_*{tag}.csv files in "
                         f"{csv_dir}; run parse-stat first (or pass --tag).")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out-dir", help="default: --csv-dir")
    ap.add_argument("--tag", default="_3km")
    ap.add_argument("--level", nargs="+", choices=list(ps.LEVELS),
                    default=["all"])
    ap.add_argument("--matched", choices=("windows", "leads", "fixed"))
    ap.add_argument("--pool-leads", action="store_true",
                    help="sum all leads together (lead_h becomes 'all')")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-loo", action="store_true",
                    help="skip the leave-one-storm-out tables")
    ps.add_exclude_args(ap)
    a = ap.parse_args(argv)
    if a.n_boot < 100:
        ap.error("--n-boot below 100 is too few for 95 % percentiles")
    bootstrap_all(a.csv_dir, a.out_dir, a.tag, tuple(a.level), a.matched,
                  a.n_boot, a.seed, not a.no_loo, a.pool_leads,
                  exclude=ps.excluded_from_args(a))


if __name__ == "__main__":
    main(sys.argv[1:])
