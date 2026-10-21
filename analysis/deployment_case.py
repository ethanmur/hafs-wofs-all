"""Case config for one WoFS deployment: the unit the verification works on.

Loaded from the per-deployment YAML written by make_case_yamls.py (plan A1):
which storm and deployment, where the WoFS and HAFS data live, the scored
period, and which truth product to use. A deployment with no complete WoFS
window (``wofs.available: false``) loads fine and carries no scoring window, so
later steps can list it as unavailable instead of tripping over it.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_RES_KM = 6.0
TRUTH_SOURCES = ("stage4", "mrms")
TOP_KEYS = {"storm_name", "storm_id", "deployment", "best_track",
            "truth_source", "wofs", "valid_start", "valid_end", "hafs",
            "grid_json_dir", "grid_plot_dir", "regrid_dir", "regrid_plot_dir",
            "log_dir", "mrms_cache_dir", "stage4_cache_dir", "grid",
            "regrid", "regrid_plots"}
# Where each kind of file goes. Similar files sit together, flat, under
# data/ or figures/; there is no per-case output folder.
PATH_KEYS = ("grid_json_dir", "grid_plot_dir", "regrid_dir",
             "regrid_plot_dir", "log_dir")
WOFS_KEYS = {"root", "available", "complete_windows"}
HAFS_KEYS = {"lookback_h", "min_members", "init_start", "init_end", "models"}
GRID_KEYS = {"res_km"}
REGRID_KEYS = {"sources", "method", "width", "vld_thresh", "tolerance_pct",
               "met_bin_dir", "fields"}
PLOT_KEYS = {"panels"}
PLOT_PANELS = ("compare-regrid", "compare-products", "compare-anomaly")


def parse_stamp(value, key):
    """datetime from YYYYMMDDHH (int or string)."""
    try:
        return datetime.strptime(str(value), "%Y%m%d%H")
    except ValueError:
        raise ValueError(f"{key} must be YYYYMMDDHH, got {value!r}") from None


def _unknown(block, allowed, where):
    extra = set(block) - allowed
    if extra:
        raise ValueError(f"unknown {where} keys: {', '.join(sorted(extra))}")


@dataclass
class RegridSettings:
    """How regrid-obs runs MET. `sources` empty means just the truth product."""
    sources: tuple = ()
    method: str = "BUDGET"
    width: int = 2
    vld_thresh: float = 0.5
    tolerance_pct: float = 2.0
    met_bin_dir: Optional[Path] = None
    fields: dict = field(default_factory=dict)


@dataclass
class DeploymentCase:
    case_slug: str
    storm_name: str
    storm_id: str
    deployment: str
    best_track: Path
    truth_source: str
    wofs_root: Path
    wofs_available: bool
    complete_windows: int
    grid_json_dir: Path        # data/json-grids/        <case>_grid.json
    grid_plot_dir: Path        # figures/regrid-wofs/    <case>_grid.png
    regrid_dir: Path           # data/stage-iv-regrid/<dep>_<storm>/  regridded obs + CSV
    regrid_plot_dir: Path      # figures/regrid-obs-tests/<dep>_<storm>/
    log_dir: Path              # data/logs/
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    valid_start: Optional[datetime] = None
    valid_end: Optional[datetime] = None
    hafs_lookback_h: int = 48
    hafs_min_members: int = 8
    hafs_init_start: Optional[datetime] = None
    hafs_init_end: Optional[datetime] = None
    hafs_models: dict = field(default_factory=dict)
    res_km: float = DEFAULT_RES_KM
    regrid: RegridSettings = field(default_factory=RegridSettings)
    regrid_panels: tuple = ()               # empty = every panel

    @property
    def output_slug(self):
        """Names the per-window outputs, so two windows never overwrite."""
        return (f"{self.case_slug}_{self.valid_start:%Y%m%d%H}_"
                f"{self.valid_end:%Y%m%d%H}")

    @property
    def grid_json_path(self):
        """Where build-grid writes this deployment's grid."""
        return self.grid_json_dir / f"{self.case_slug}_grid.json"

    @property
    def grid_plot_path(self):
        return self.grid_plot_dir / f"{self.case_slug}_grid.png"

    @property
    def folder_name(self):
        """<deployment>_<storm>, e.g. 20240927_d1_helene: names this
        deployment's folder under both data/ and figures/. The deployment
        carries its date because d1 alone repeats (Helene has a d1 on two days,
        overlapping in time)."""
        return f"{self.deployment}_{self.storm_name.lower()}"

    @property
    def res_label(self):
        return f"{self.res_km:g}km"

    @property
    def budget_csv_path(self):
        """The conservation check, beside the regridded data it describes."""
        return self.regrid_dir / (f"regrid_budget_{self.valid_start:%Y%m%d%H}_"
                                  f"{self.valid_end:%Y%m%d%H}.csv")

    @property
    def regrid_sources(self):
        """Obs products regrid-obs handles: the truth product unless
        regrid.sources names more (e.g. both, for the MRMS sensitivity)."""
        return self.regrid.sources or (self.truth_source,)

    @property
    def wofs_storm_folder(self):
        """The archive's storm folder (e.g. 2024_BERYL), the parent of root."""
        return self.wofs_root.parent.name


def from_yaml(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    _unknown(cfg, TOP_KEYS, "case")
    for key in ("storm_name", "storm_id", "deployment", "best_track", "wofs",
                *PATH_KEYS, "mrms_cache_dir", "stage4_cache_dir"):
        if key not in cfg:
            raise ValueError(f"{path.name}: {key} is required")

    truth = str(cfg.get("truth_source", "stage4")).lower()
    if truth not in TRUTH_SOURCES:
        raise ValueError(f"truth_source must be one of {TRUTH_SOURCES}, "
                         f"got {truth!r}")

    wofs = cfg["wofs"] or {}
    _unknown(wofs, WOFS_KEYS, "wofs")
    if "root" not in wofs:
        raise ValueError(f"{path.name}: wofs.root is required")
    available = bool(wofs.get("available", True))

    hafs = cfg.get("hafs") or {}
    _unknown(hafs, HAFS_KEYS, "hafs")
    grid = cfg.get("grid") or {}
    _unknown(grid, GRID_KEYS, "grid")
    regrid = cfg.get("regrid") or {}
    _unknown(regrid, REGRID_KEYS, "regrid")
    plots = cfg.get("regrid_plots") or {}
    _unknown(plots, PLOT_KEYS, "regrid_plots")
    panels = tuple(str(v) for v in (plots.get("panels") or ()))
    bad = [k for k in panels if k not in PLOT_PANELS]
    if bad:
        raise ValueError(f"regrid_plots.panels has unknown panel(s) {bad}; "
                         f"expected any of {PLOT_PANELS}")
    sources = tuple(str(v).lower() for v in (regrid.get("sources") or ()))
    bad = [v for v in sources if v not in TRUTH_SOURCES]
    if bad:
        raise ValueError(f"regrid.sources has unknown source(s) {bad}; "
                         f"expected any of {TRUTH_SOURCES}")

    case = DeploymentCase(
        case_slug=path.stem,
        storm_name=str(cfg["storm_name"]),
        storm_id=str(cfg["storm_id"]),
        deployment=str(cfg["deployment"]),
        best_track=Path(cfg["best_track"]),
        truth_source=truth,
        wofs_root=Path(wofs["root"]),
        wofs_available=available,
        complete_windows=int(wofs.get("complete_windows", 0)),
        grid_json_dir=Path(cfg["grid_json_dir"]),
        grid_plot_dir=Path(cfg["grid_plot_dir"]),
        regrid_dir=Path(cfg["regrid_dir"]),
        regrid_plot_dir=Path(cfg["regrid_plot_dir"]),
        log_dir=Path(cfg["log_dir"]),
        mrms_cache_dir=Path(cfg["mrms_cache_dir"]),
        stage4_cache_dir=Path(cfg["stage4_cache_dir"]),
        res_km=float(grid.get("res_km", DEFAULT_RES_KM)),
        regrid=RegridSettings(
            sources=sources,
            method=str(regrid.get("method", "BUDGET")).upper(),
            width=int(regrid.get("width", 2)),
            vld_thresh=float(regrid.get("vld_thresh", 0.5)),
            tolerance_pct=float(regrid.get("tolerance_pct", 2.0)),
            met_bin_dir=(Path(regrid["met_bin_dir"])
                         if regrid.get("met_bin_dir") else None),
            fields=dict(regrid.get("fields") or {})),
        regrid_panels=panels,
    )
    unknown_fields = set(case.regrid.fields) - set(TRUTH_SOURCES)
    if unknown_fields:
        raise ValueError(f"regrid.fields has unknown source(s) "
                         f"{sorted(unknown_fields)}")
    if case.res_km <= 0:
        raise ValueError("grid.res_km must be positive")

    if available:
        for key in ("valid_start", "valid_end"):
            if key not in cfg:
                raise ValueError(f"{path.name}: {key} is required when "
                                 "wofs.available is true")
        case.valid_start = parse_stamp(cfg["valid_start"], "valid_start")
        case.valid_end = parse_stamp(cfg["valid_end"], "valid_end")
        if case.valid_end <= case.valid_start:
            raise ValueError("valid_end must be after valid_start")
        case.hafs_lookback_h = int(hafs.get("lookback_h", 48))
        case.hafs_min_members = int(hafs.get("min_members", 8))
        if hafs.get("init_start") is not None:
            case.hafs_init_start = parse_stamp(hafs["init_start"],
                                               "hafs.init_start")
        if hafs.get("init_end") is not None:
            case.hafs_init_end = parse_stamp(hafs["init_end"], "hafs.init_end")
        case.hafs_models = {k: Path(v)
                            for k, v in (hafs.get("models") or {}).items()}
    return case
