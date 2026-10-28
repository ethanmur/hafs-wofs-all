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
            "grid_json_dir", "grid_plot_dir", "stage4_regrid_dir",
            "mrms_regrid_dir", "regrid_plot_dir",
            "mask_dir", "mask_plot_dir", "manifest_dir", "wofs_regrid_1h_dir",
            "wofs_regrid_3h_dir", "log_dir", "mrms_cache_dir",
            "stage4_cache_dir", "grid", "regrid", "regrid_plots", "masks"}
# Where each kind of file goes. Similar files sit together, flat, under
# data/ or figures/; there is no per-case output folder.
PATH_KEYS = ("grid_json_dir", "grid_plot_dir", "stage4_regrid_dir",
             "mrms_regrid_dir",
             "regrid_plot_dir", "mask_dir", "mask_plot_dir", "manifest_dir",
             "wofs_regrid_1h_dir", "wofs_regrid_3h_dir", "log_dir")
MASK_KEYS = {"coast_buffer_km", "ne_scale"}
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


def _resolution_dir(path, res_km):
    """A plot folder for the default resolution is used as written; another
    resolution gets its own sibling (20240927_d1_helene_3km), so its maps never
    overwrite the default run's."""
    if res_km == DEFAULT_RES_KM:
        return path
    return path.with_name(f"{path.name}_{res_km:g}km")


@dataclass
class MaskSettings:
    """The observation-validity rule: how far past US land the truth product
    is trusted, and which Natural Earth scale draws the coast."""
    coast_buffer_km: float = 150.0
    ne_scale: str = "50m"


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
    stage4_regrid_dir: Path    # data/stage-iv-regrid/<dep>_<storm>/  Stage IV + its CSV
    mrms_regrid_dir: Path      # data/mrms-regrid/<dep>_<storm>/      MRMS + its CSV
    regrid_plot_dir: Path      # figures/regrid-obs-tests/<dep>_<storm>/
    mask_dir: Path             # data/obs-valid-masks/   <case>_obs_valid.nc
    mask_plot_dir: Path        # figures/obs-valid-masks/ <case>_obs_valid.png
    manifest_dir: Path         # data/window-manifests/ <case>_manifest.csv, _windows.csv
    wofs_regrid_1h_dir: Path   # data/wofs-regrid-1h/<dep>_<storm>/  WoFS 1 h accumulations
    wofs_regrid_3h_dir: Path   # data/wofs-regrid-3h/<dep>_<storm>/  WoFS 3 h accumulations
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
    masks: MaskSettings = field(default_factory=MaskSettings)
    regrid_panels: tuple = ()               # empty = every panel

    @property
    def output_slug(self):
        """Names the per-window outputs, so two windows never overwrite."""
        return (f"{self.case_slug}_{self.valid_start:%Y%m%d%H}_"
                f"{self.valid_end:%Y%m%d%H}")

    @property
    def res_suffix(self):
        """"" at the default 6 km, so every default path is unchanged; any
        other resolution adds `_<res>km` to the names that would otherwise
        collide with the default run's (grid JSON, grid map, budget CSV)."""
        return "" if self.res_km == DEFAULT_RES_KM else f"_{self.res_label}"

    @property
    def grid_json_path(self):
        """Where build-grid writes this deployment's grid."""
        return self.grid_json_dir / f"{self.case_slug}{self.res_suffix}_grid.json"

    @property
    def grid_plot_path(self):
        return self.grid_plot_dir / f"{self.case_slug}{self.res_suffix}_grid.png"

    @property
    def mask_path(self):
        """The observation-validity mask, on this deployment's grid."""
        return self.mask_dir / f"{self.case_slug}{self.res_suffix}_obs_valid.nc"

    @property
    def mask_plot_path(self):
        return self.mask_plot_dir / (
            f"{self.case_slug}{self.res_suffix}_obs_valid.png")

    @property
    def manifest_path(self):
        """Every (window, model run) and what is on disk for it. Windows do
        not depend on the resolution, so neither does the name."""
        return self.manifest_dir / f"{self.case_slug}_manifest.csv"

    @property
    def window_summary_path(self):
        return self.manifest_dir / f"{self.case_slug}_windows.csv"

    def wofs_regrid_dir(self, hours):
        """The folder for 1 h or 3 h WoFS accumulations."""
        if hours not in (1, 3):
            raise ValueError("WoFS accumulations are kept for 1 h and 3 h")
        return self.wofs_regrid_1h_dir if hours == 1 else self.wofs_regrid_3h_dir

    def wofs_window_path(self, end, init, hours=3):
        """One WoFS (init, accumulation ending at `end`): all members, summed
        and on the grid. The folder says 1 h or 3 h; the name says which hour
        or window ends, which run it came from, and the resolution."""
        return self.wofs_regrid_dir(hours) / (
            f"{end:%Y%m%d%H}_i{init:%Y%m%d%H%M}_wofs_{self.res_label}.nc")

    def wofs_budget_csv_path(self, hours=3):
        """The WoFS regrid check, beside the files it describes."""
        return self.wofs_regrid_dir(hours) / (
            f"regrid_budget_{self.valid_start:%Y%m%d%H}_"
            f"{self.valid_end:%Y%m%d%H}{self.res_suffix}.csv")

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

    def regrid_dir_for(self, source):
        """The folder holding one obs product's regridded files."""
        return {"stage4": self.stage4_regrid_dir,
                "mrms": self.mrms_regrid_dir}[source]

    def budget_csv_path(self, source=None):
        """One product's conservation check, beside the files it describes
        (default: the truth product)."""
        source = source or self.truth_source
        return self.regrid_dir_for(source) / (
            f"regrid_budget_{self.valid_start:%Y%m%d%H}_"
            f"{self.valid_end:%Y%m%d%H}{self.res_suffix}.csv")

    @property
    def regrid_sources(self):
        """Obs products regrid-obs handles: the truth product unless
        regrid.sources names more (e.g. both, for the MRMS sensitivity)."""
        return self.regrid.sources or (self.truth_source,)

    @property
    def wofs_storm_folder(self):
        """The archive's storm folder (e.g. 2024_BERYL), the parent of root."""
        return self.wofs_root.parent.name


def from_yaml(path, res_km=None, sources=None):
    """The case in `path`. `res_km` overrides grid.res_km (run.py --res-km) and
    `sources` overrides regrid.sources (run.py --sources), so another
    resolution or obs product needs no edit to the YAML."""
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
    masks = cfg.get("masks") or {}
    _unknown(masks, MASK_KEYS, "masks")
    if float(masks.get("coast_buffer_km", 150.0)) < 0:
        raise ValueError("masks.coast_buffer_km cannot be negative")
    panels = tuple(str(v) for v in (plots.get("panels") or ()))
    bad = [k for k in panels if k not in PLOT_PANELS]
    if bad:
        raise ValueError(f"regrid_plots.panels has unknown panel(s) {bad}; "
                         f"expected any of {PLOT_PANELS}")
    res = float(res_km if res_km is not None
                else grid.get("res_km", DEFAULT_RES_KM))
    sources = tuple(str(v).lower()
                    for v in (sources or regrid.get("sources") or ()))
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
        stage4_regrid_dir=Path(cfg["stage4_regrid_dir"]),
        mrms_regrid_dir=Path(cfg["mrms_regrid_dir"]),
        regrid_plot_dir=_resolution_dir(Path(cfg["regrid_plot_dir"]), res),
        mask_dir=Path(cfg["mask_dir"]),
        mask_plot_dir=Path(cfg["mask_plot_dir"]),
        manifest_dir=Path(cfg["manifest_dir"]),
        wofs_regrid_1h_dir=Path(cfg["wofs_regrid_1h_dir"]),
        wofs_regrid_3h_dir=Path(cfg["wofs_regrid_3h_dir"]),
        masks=MaskSettings(
            coast_buffer_km=float(masks.get("coast_buffer_km", 150.0)),
            ne_scale=str(masks.get("ne_scale", "50m"))),
        log_dir=Path(cfg["log_dir"]),
        mrms_cache_dir=Path(cfg["mrms_cache_dir"]),
        stage4_cache_dir=Path(cfg["stage4_cache_dir"]),
        res_km=res,
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
