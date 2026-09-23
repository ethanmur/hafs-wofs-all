"""Compare multi-cycle verification tables across HAFS configurations.

The comparison is intentionally table-driven: run each available model with
the ``cycles`` command first, then use ``cycles-compare`` to combine its CSVs.
Models without output yet remain visible as "awaiting data" placeholders.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import yaml

from hafs_case import cycles_from_yaml
from plot_units import inches, miles


MODEL_COLORS = {
    "HAFS-A": "#2563a6",
    "HAFS-B": "#dd6b4d",
    "HAFS-M": "#2a9d78",
}


def _resolve_path(value, config_path):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return config_path.parent / path


def load_cycles_comparison(path):
    """Load model-cycle comparison metadata without requiring model data."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    models = raw.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError(f"'models' must contain at least two entries in {path}")
    parsed = []
    for item in models:
        if not isinstance(item, dict) or "name" not in item or "cycles_yaml" not in item:
            raise ValueError("Each model needs 'name' and 'cycles_yaml'.")
        parsed.append({"name": str(item["name"]),
                       "cycles_yaml": _resolve_path(item["cycles_yaml"], path)})
    out_dir = Path(raw.get("out_dir", "analysis/output/helene_cycles_compare"))
    thresholds = [float(v) for v in raw.get(
        "ets_thresholds_in", list(range(2, 25, 2)))]
    fss_thresholds = [float(v) for v in raw.get("fss_thresholds_in", [1, 2])]
    return {"label": raw.get("label", path.stem), "models": parsed,
            "out_dir": out_dir, "ets_thresholds_in": thresholds,
            "fss_thresholds_in": fss_thresholds}


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _read_summary_csv(path):
    """Read a cycle summary, converting metric columns to floats."""
    text_fields = {"init", "init_dt", "valid_start", "valid_end"}
    output = []
    for row in _read_csv(path):
        parsed = {}
        for key, value in row.items():
            if key in text_fields:
                parsed[key] = value
            else:
                try:
                    parsed[key] = float(value) if value != "" else np.nan
                except (TypeError, ValueError):
                    parsed[key] = np.nan
        output.append(parsed)
    return output


def load_model_tables(model):
    """Return categorical/FSS rows and paths for one configured model."""
    try:
        case = cycles_from_yaml(model["cycles_yaml"])
    except (OSError, KeyError, ValueError) as exc:
        return {**model, "case": None, "categorical": [], "fss": [],
                "summary": [],
                "status": f"configuration unavailable: {exc}"}
    slug = case.output_slug
    cat_path = case.out_dir / f"cycles_{slug}.csv"
    fss_path = case.out_dir / f"cycles_fss_{slug}.csv"
    summary_path = case.out_dir / f"cycles_summary_{slug}.csv"
    categorical = _read_csv(cat_path)
    fss = _read_csv(fss_path)
    summary = _read_summary_csv(summary_path)
    status = "available" if categorical and fss else "awaiting cycle output"
    return {**model, "case": case, "categorical": categorical, "fss": fss,
            "summary": summary, "cat_path": cat_path, "fss_path": fss_path,
            "summary_path": summary_path, "status": status}


def pooled_ets(rows, thresholds_in):
    """Pool MRMS parent contingency counts across cycles at inch thresholds."""
    selected = [r for r in rows if r.get("forecast") == "parent"
                and r.get("observation") == "MRMS"]
    output = []
    for threshold_in in thresholds_in:
        threshold_mm = threshold_in * 25.4
        matched = [r for r in selected
                   if np.isclose(float(r["threshold"]), threshold_mm)]
        counts = {key: sum(int(float(r[key])) for r in matched)
                  for key in ("a", "b", "c", "d")}
        n = sum(counts.values())
        random_hits = (((counts["a"] + counts["b"])
                        * (counts["a"] + counts["c"]) / n) if n else 0.0)
        denominator = counts["a"] + counts["b"] + counts["c"] - random_hits
        score = ((counts["a"] - random_hits) / denominator
                 if denominator else np.nan)
        output.append({"threshold_in": threshold_in, "ets": score,
                       "n_cycles": len({r["init"] for r in matched})})
    return output


def trim_trailing_empty_ets_thresholds(models, thresholds_in):
    """Drop only trailing thresholds with zero or unavailable ETS everywhere."""
    thresholds = list(thresholds_in)
    last_active = None
    for index, threshold in enumerate(thresholds):
        scores = [pooled_ets(model["categorical"], [threshold])[0]["ets"]
                  for model in models]
        if any(np.isfinite(score) and not np.isclose(score, 0.0)
               for score in scores):
            last_active = index
    return thresholds if last_active is None else thresholds[:last_active + 1]


def plot_ets_comparison(models, thresholds_in, label, out_path):
    """Tasteful grouped ETS bars for all configured models."""
    thresholds_in = trim_trailing_empty_ets_thresholds(models, thresholds_in)
    x = np.arange(len(thresholds_in))
    width = min(0.24, 0.78 / max(len(models), 1))
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("#f7f9fb")
    missing = []
    for index, model in enumerate(models):
        summary = pooled_ets(model["categorical"], thresholds_in)
        values = np.asarray([row["ets"] for row in summary], dtype=float)
        offset = (index - (len(models) - 1) / 2) * width
        color = MODEL_COLORS.get(model["name"], plt.cm.Set2(index))
        if np.isfinite(values).any():
            ax.bar(x + offset, np.nan_to_num(values), width=width * 0.90,
                   color=color, edgecolor="white", linewidth=0.8,
                   label=model["name"], zorder=3)
        else:
            missing.append(model["name"])
    ax.axhline(0, color="#607080", linewidth=0.8)
    ax.set_xticks(x, [f"{value:g}" for value in thresholds_in])
    ax.set_xlabel("Rainfall accumulation threshold (inches)", labelpad=10)
    ax.set_ylabel("Equitable Threat Score (ETS)")
    ax.grid(axis="y", color="#d8e0e8", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=MODEL_COLORS.get(model["name"], "gray"),
                     alpha=1.0 if model["name"] not in missing else 0.35,
                     label=(model["name"] if model["name"] not in missing
                            else f"{model['name']} · awaiting data"))
               for model in models]
    ax.legend(handles=handles, frameon=False, ncols=len(models),
              loc="upper right")
    ax.set_title(f"{label}\nPooled rainfall skill across forecast cycles",
                 loc="left", fontsize=16, fontweight="semibold", pad=16)
    ax.text(1, 1.015, "Higher is better", transform=ax.transAxes,
            ha="right", color="#607080", fontsize=9)
    if missing:
        ax.text(0.01, 0.02, "Awaiting output: " + ", ".join(missing),
                transform=ax.transAxes, fontsize=9, color="#607080")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _fss_series(rows, threshold_in):
    selected = []
    for row in rows:
        if row.get("forecast") != "parent" or row.get("observation") != "MRMS":
            continue
        value_in = float(row.get("threshold_in") or float(row["threshold"]) / 25.4)
        if np.isclose(value_in, threshold_in):
            selected.append(row)
    scales = sorted({float(r["scale_km"]) for r in selected})
    output = []
    for scale in scales:
        values = np.asarray([float(r["fss"]) for r in selected
                             if np.isclose(float(r["scale_km"]), scale)])
        values = values[np.isfinite(values)]
        if values.size:
            output.append((scale, float(values.mean()),
                           float(np.percentile(values, 25)),
                           float(np.percentile(values, 75))))
    return output


def plot_fss_comparison(models, thresholds_in, label, out_path):
    """Mean cycle FSS by scale with an interquartile band per model."""
    fig, axes = plt.subplots(1, len(thresholds_in),
                             figsize=(6.5 * len(thresholds_in), 5.8),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor("#f7f9fb")
    for ax, threshold in zip(axes[0], thresholds_in):
        ax.set_facecolor("#f7f9fb")
        have_data = False
        for index, model in enumerate(models):
            series = _fss_series(model["fss"], threshold)
            if not series:
                continue
            have_data = True
            scale, mean, q25, q75 = map(np.asarray, zip(*series))
            scale = miles(scale)
            color = MODEL_COLORS.get(model["name"], plt.cm.Set2(index))
            ax.fill_between(scale, q25, q75, color=color, alpha=0.14,
                            linewidth=0)
            ax.plot(scale, mean, color=color, linewidth=2.5, marker="o",
                    markersize=5.5, markeredgecolor="white",
                    label=model["name"])
        unit = "inch" if np.isclose(threshold, 1.0) else "inches"
        ax.set_title(f"Rainfall ≥ {threshold:g} {unit}",
                     fontweight="semibold", fontsize=13)
        ax.set_xlabel("Neighborhood scale (miles)")
        ax.set_ylim(0, 1)
        ax.grid(color="#d8e0e8", linewidth=0.8, alpha=0.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if not have_data:
            ax.text(0.5, 0.5, "Awaiting cycle output", transform=ax.transAxes,
                    ha="center", color="#607080")
    axes[0, 0].set_ylabel("Fractions Skill Score (FSS)")
    handles = [plt.Line2D([], [], color=MODEL_COLORS.get(m["name"], "gray"),
                          marker="o", lw=2.5,
                          label=(m["name"] if m["fss"] else
                                 f"{m['name']} · awaiting data"))
               for m in models]
    fig.legend(handles=handles, frameon=False, ncols=len(models),
               loc="upper center", bbox_to_anchor=(0.64, 0.85))
    fig.suptitle(f"{label}\nMean FSS across cycles · shading shows interquartile range",
                 x=0.06, y=0.985, ha="left", fontsize=16,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.74))
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _finite_summary_pairs(rows, x_key, y_key):
    pairs = [(row.get(x_key, np.nan), row.get(y_key, np.nan))
             for row in rows]
    pairs = [(float(x), float(y)) for x, y in pairs
             if np.isfinite(x) and np.isfinite(y)]
    if not pairs:
        return np.asarray([]), np.asarray([])
    return map(np.asarray, zip(*pairs))


def _spearman_text(x, y):
    if len(x) < 3:
        return "n<3"
    try:
        from scipy.stats import spearmanr
        with np.errstate(all="ignore"):
            rho = float(spearmanr(x, y).statistic)
    except (ValueError, FloatingPointError):
        rho = np.nan
    return f"ρ={rho:.2f}"


def plot_track_error_comparison(models, label, out_path):
    """Mean track error versus landfall lead for each model."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for index, model in enumerate(models):
        x, y = _finite_summary_pairs(
            model["summary"], "mean_track_err_km", "lead_hours_to_landfall")
        if len(x) == 0:
            continue
        order = np.argsort(y)
        ax.plot(miles(x[order]), y[order], marker="o", lw=2,
                color=MODEL_COLORS.get(model["name"], plt.cm.Set2(index)),
                label=model["name"])
    ax.set_xlabel("Mean track error (miles)")
    ax.set_ylabel("Hours before landfall")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(frameon=False)
    ax.set_title(f"{label}\nTrack error by forecast-cycle lead", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_rmse_comparison(models, label, out_path):
    """Storm-total RMSE versus initialization lead for each model."""
    fig, ax = plt.subplots(figsize=(9, 6))
    have_data = False
    for index, model in enumerate(models):
        lead, rmse = _finite_summary_pairs(
            model["summary"], "lead_hours_to_landfall", "rmse")
        if len(lead) == 0:
            continue
        have_data = True
        order = np.argsort(lead)
        rmse_in = inches(rmse[order])
        mean_in = float(np.mean(rmse_in))
        ax.plot(lead[order], rmse_in, marker="o", lw=2.2,
                color=MODEL_COLORS.get(model["name"], plt.cm.Set2(index)),
                label=f"{model['name']} · mean {mean_in:.2f} in")
    if not have_data:
        plt.close(fig)
        return False
    ax.set_xlabel("Hours before landfall (forecast initialization)")
    ax.set_ylabel("Storm-total RMSE (inches)")
    ax.invert_xaxis()
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(frameon=False)
    ax.set_title(
        f"{label}\nRainfall RMSE by forecast-cycle lead · "
        "matching init-clipped windows", loc="left")
    ax.text(1.0, 1.015, "Lower is better", transform=ax.transAxes,
            ha="right", color="#607080", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_track_precip_comparison(models, label, out_path):
    """Pooled model-colored track-error versus headline ETS scatter."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    pooled_x = []
    pooled_y = []
    for index, model in enumerate(models):
        x, y = _finite_summary_pairs(
            model["summary"], "mean_track_err_km", "ets_headline")
        if len(x) == 0:
            continue
        pooled_x.extend(miles(x))
        pooled_y.extend(y)
        ax.scatter(miles(x), y, s=52,
                   color=MODEL_COLORS.get(model["name"], plt.cm.Set2(index)),
                   edgecolor="white",
                   label=f"{model['name']} · {_spearman_text(x, y)}")
    pooled_x = np.asarray(pooled_x)
    pooled_y = np.asarray(pooled_y)
    ax.set_xlabel("Mean track error (miles)")
    ax.set_ylabel("Headline ETS")
    ax.grid(True, ls=":", alpha=0.45)
    ax.legend(frameon=False, title=f"Pooled {_spearman_text(pooled_x, pooled_y)}")
    ax.set_title(f"{label}\nTrack error and precipitation skill", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_shifted_ets_comparison(models, label, out_path):
    """Compare model-mean unshifted and track-shifted headline ETS."""
    x = np.arange(len(models), dtype=float)
    width = 0.34
    raw_means = []
    shifted_means = []
    counts = []
    for model in models:
        raw, shifted = _finite_summary_pairs(
            model["summary"], "ets_headline", "ets_shifted")
        raw_means.append(float(np.mean(raw)) if len(raw) else np.nan)
        shifted_means.append(float(np.mean(shifted)) if len(shifted) else np.nan)
        counts.append(len(raw))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(x - width / 2, raw_means, width, color="#607d9b",
           label="Unshifted")
    bars = ax.bar(x + width / 2, shifted_means, width, color="#d97941",
                  label="Track-shifted")
    for bar, n in zip(bars, counts):
        height = bar.get_height()
        if np.isfinite(height):
            ax.annotate(f"n={n}", (bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9)
    ax.set_xticks(x, [model["name"] for model in models])
    ax.set_ylabel("Mean headline ETS")
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.legend(frameon=False)
    ax.set_title(f"{label}\nEffect of track-shifting on ETS", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_cycles_comparison(config):
    """Load available cycle tables and write the comparison figures."""
    models = [load_model_tables(model) for model in config["models"]]
    config["out_dir"].mkdir(parents=True, exist_ok=True)
    ets_path = config["out_dir"] / "cycles_compare_ets_bars.png"
    fss_path = config["out_dir"] / "cycles_compare_fss.png"
    plot_ets_comparison(models, config["ets_thresholds_in"],
                        config["label"], ets_path)
    plot_fss_comparison(models, config["fss_thresholds_in"],
                        config["label"], fss_path)
    for model in models:
        print(f"{model['name']}: {model['status']}")
    print(f"Saved plot : {ets_path}")
    print(f"Saved plot : {fss_path}")
    if any(model["summary"] for model in models):
        summary_plots = [
            ("cycles_compare_rmse.png", plot_rmse_comparison),
            ("cycles_compare_track_error.png", plot_track_error_comparison),
            ("cycles_compare_track_precip.png", plot_track_precip_comparison),
            ("cycles_compare_shifted_ets.png", plot_shifted_ets_comparison),
        ]
        for filename, plotter in summary_plots:
            path = config["out_dir"] / filename
            plotter(models, config["label"], path)
            print(f"Saved plot : {path}")
    else:
        print("Summary comparison plots skipped — no model has summary rows.")
    return models
