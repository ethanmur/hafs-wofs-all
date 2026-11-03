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
            "mrms_regrid_dir", "stage4_regrid_3h_dir", "mrms_regrid_3h_dir",
            "regrid_plot_dir", "eyeball_plot_dir", "hafs_check_plot_dir",
            "mask_dir", "mask_plot_dir", "manifest_dir", "wofs_regrid_1h_dir",
            "wofs_regrid_3h_dir", "hafs_regrid_dir", "grid_stat_dir", "log_dir",
            "mrms_cache_dir", "stage4_cache_dir", "grid", "regrid",
            "regrid_plots", "masks", "eyeball", "grid_stat"}
# Where each kind of file goes. Similar files sit together, flat, under
# data/ or figures/; there is no per-case output folder.
PATH_KEYS = ("grid_json_dir", "grid_plot_dir", "stage4_regrid_dir",
             "mrms_regrid_dir", "stage4_regrid_3h_dir", "mrms_regrid_3h_dir",
             "regrid_plot_dir", "eyeball_plot_dir", "hafs_check_plot_dir",
             "mask_dir", "mask_plot_dir", "manifest_dir",
             "wofs_regrid_1h_dir", "wofs_regrid_3h_dir", "hafs_regrid_dir",
             "grid_stat_dir", "log_dir")
MASK_KEYS = {"coast_buffer_km", "ne_scale"}
WOFS_KEYS = {"root", "available", "complete_windows"}
HAFS_KEYS = {"lookback_h", "min_members", "init_start", "init_end", "models"}
GRID_KEYS = {"res_km"}
DEFAULT_COAST_KM = 150.0
REGRID_KEYS = {"sources", "method", "width", "vld_thresh", "tolerance_pct",
               "met_bin_dir", "fields", "budget_check"}
PLOT_KEYS = {"panels"}
GRID_STAT_KEYS = {"thresholds_mm", "widths_km", "wet_thresh_mm", "cov_thresh",
                  "test_end"}
EYEBALL_KEYS = {"members", "windows", "cycles", "colorbar_style"}
COLORBAR_STYLES = ("true_scale", "standard")
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
    coast_buffer_km: float = DEFAULT_COAST_KM
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
    budget_check: bool = True     # regrid-hafs: compare each file with its source


@dataclass
class EyeballSettings:
    """How much plot-eyeball and plot-hafs-check draw: every pair would be
    thousands of figures, so only a few windows, HAFS cycles and WoFS members.
    `windows` and `cycles` are counts, spread evenly (first, last, between);
    `members` are WoFS member numbers."""
    members: tuple = (0, 9)
    windows: int = 3
    cycles: int = 2
    colorbar_style: str = "true_scale"


@dataclass
class GridStatSettings:
    """The grid_stat configuration (plan E1). Thresholds are mm per 3 h.
    Neighborhood widths are distances, converted to cells for the grid in use,
    so a scale means the same distance at any resolution. `wet_thresh_mm`
    defines the wet-cells-only continuous scores (either field at or above it);
    `cov_thresh` is the event fraction a neighborhood needs to count as a hit.
    `test_end` pins the window grid-stat-test scores (None: the wettest)."""
    thresholds_mm: tuple = (0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0,
                            100.0, 125.0, 150.0)
    widths_km: tuple = (6.0, 18.0, 30.0, 54.0, 126.0)
    wet_thresh_mm: float = 0.5
    cov_thresh: float = 0.5
    test_end: Optional[datetime] = None


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
    stage4_regrid_3h_dir: Path  # data/stage-iv-regrid-3h/<dep>_<storm>/  Stage IV 3 h windows
    mrms_regrid_3h_dir: Path   # data/mrms-regrid-3h/<dep>_<storm>/      MRMS 3 h windows
    regrid_plot_dir: Path      # figures/regrid-obs-tests/<dep>_<storm>/
    eyeball_plot_dir: Path     # figures/eyeball-panels/<dep>_<storm>/  WoFS | HAFS | truth
    hafs_check_plot_dir: Path  # figures/hafs-regrid-checks/<dep>_<storm>/  HAFS native vs regridded
    mask_dir: Path             # data/obs-valid-masks/   <case>_obs_valid.nc
    mask_plot_dir: Path        # figures/obs-valid-masks/ <case>_obs_valid.png
    manifest_dir: Path         # data/window-manifests/ <case>_manifest.csv, _windows.csv
    wofs_regrid_1h_dir: Path   # data/wofs-regrid-1h/<dep>_<storm>/  WoFS 1 h accumulations
    wofs_regrid_3h_dir: Path   # data/wofs-regrid-3h/<dep>_<storm>/  WoFS 3 h accumulations
    hafs_regrid_dir: Path      # data/hafs-regrid/<dep>_<storm>/  + _<CONFIG> per HAFS configuration
    grid_stat_dir: Path        # data/grid-stat/<dep>_<storm>/  config, mask, <window end>/ outputs
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
    eyeball: EyeballSettings = field(default_factory=EyeballSettings)
    grid_stat: GridStatSettings = field(default_factory=GridStatSettings)

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
    def mask_suffix(self):
        """"" at the default 150 km buffer; any other adds `_coast<N>km` to
        the mask file and its figures, so a sensitivity run never overwrites
        the default mask."""
        km = self.masks.coast_buffer_km
        return "" if km == DEFAULT_COAST_KM else f"_coast{km:g}km"

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
        return self.mask_dir / (
            f"{self.case_slug}{self.res_suffix}{self.mask_suffix}_obs_valid.nc")

    @property
    def mask_plot_path(self):
        return self.mask_plot_dir / (
            f"{self.case_slug}{self.res_suffix}{self.mask_suffix}"
            "_obs_valid.png")

    def mask_check_path(self, end=None):
        """plot-mask-check figures, beside the mask map: one per window
        (named for its end), or the all-windows coverage summary."""
        tag = "coverage" if end is None else f"{end:%Y%m%d%H}"
        return self.mask_plot_dir / (
            f"{self.case_slug}{self.res_suffix}{self.mask_suffix}"
            f"_mask_check_{tag}.png")

    @property
    def grid_stat_tag(self):
        """Resolution and mask-buffer suffixes: every grid_stat file carries
        them, so a 6 km run or a 50 km coastal buffer never overwrites the
        default one."""
        return f"{self.res_suffix}{self.mask_suffix}"

    @property
    def grid_stat_conf_path(self):
        return self.grid_stat_dir / (
            f"GridStat_{self.case_slug}{self.grid_stat_tag}.conf")

    @property
    def grid_stat_mask_path(self):
        """The validity mask as MET's own NetCDF, named FULL for grid_stat."""
        return self.grid_stat_dir / (
            f"{self.case_slug}{self.grid_stat_tag}_FULL_mask.nc")

    def grid_stat_window_dir(self, end):
        """One 3 h window's grid_stat output, named by its end."""
        return self.grid_stat_dir / f"{end:%Y%m%d%H}"

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

    def hafs_regrid_dir_for(self, config):
        """One HAFS configuration's folder: the YAML's hafs_regrid_dir with the
        configuration appended (20241009_d1_milton_HFSA), so HFSA and HFSB
        never share a folder, a settings record or a budget CSV."""
        base = self.hafs_regrid_dir
        return base.with_name(f"{base.name}_{config.upper()}")

    def hafs_window_path(self, config, end, init):
        """One HAFS (cycle, 3 h window ending at `end`) on the grid. Named like
        the WoFS files: the valid end, the cycle, the model and the
        resolution."""
        return self.hafs_regrid_dir_for(config) / (
            f"{end:%Y%m%d%H}_i{init:%Y%m%d%H%M}_{config.lower()}_"
            f"{self.res_label}.nc")

    def hafs_budget_csv_path(self, config):
        """The HAFS regrid check, beside the files it describes."""
        return self.hafs_regrid_dir_for(config) / (
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

    def regrid_3h_dir_for(self, source):
        """The folder holding one obs product's 3 h window sums."""
        return {"stage4": self.stage4_regrid_3h_dir,
                "mrms": self.mrms_regrid_3h_dir}[source]

    def obs_window_path(self, source, end):
        """One obs 3 h window (the three hourly files ending at `end`) on the
        grid. Named like the hourly files; the folder says it is 3 h."""
        return self.regrid_3h_dir_for(source) / (
            f"{end:%Y%m%d%H}_{source}_{self.res_label}.nc")

    def obs_window_csv_path(self, source):
        """The window sums and what was missing, beside the files."""
        return self.regrid_3h_dir_for(source) / (
            f"window_sums_{self.valid_start:%Y%m%d%H}_"
            f"{self.valid_end:%Y%m%d%H}{self.res_suffix}.csv")

    def budget_csv_path(self, source=None):
        """One product's conservation check, beside the files it describes
        (default: the truth product)."""
        source = source or self.truth_source
        return self.regrid_dir_for(source) / (
            f"regrid_budget_{self.valid_start:%Y%m%d%H}_"
            f"{self.valid_end:%Y%m%d%H}{self.res_suffix}.csv")

    @property
    def plot_truth(self):
        """The product plot-eyeball draws as truth: the one `--sources` (or
        regrid.sources) names when it names exactly one, so a run can switch
        products without editing the YAML; otherwise truth_source."""
        return (self.regrid.sources[0] if len(self.regrid.sources) == 1
                else self.truth_source)

    @property
    def regrid_sources(self):
        """Obs products regrid-obs handles: the truth product unless
        regrid.sources names more (e.g. both, for the MRMS sensitivity)."""
        return self.regrid.sources or (self.truth_source,)

    @property
    def wofs_storm_folder(self):
        """The archive's storm folder (e.g. 2024_BERYL), the parent of root."""
        return self.wofs_root.parent.name


def from_yaml(path, res_km=None, sources=None, coast_km=None):
    """The case in `path`. `res_km` overrides grid.res_km (run.py --res-km),
    `sources` overrides regrid.sources (run.py --sources) and `coast_km`
    overrides masks.coast_buffer_km (run.py --coast-km), so another
    resolution, obs product or mask buffer needs no edit to the YAML."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    _unknown(cfg, TOP_KEYS, "case")
    for key in ("storm_name", "storm_id", "deployment", "best_track", "wofs",
                *PATH_KEYS, "mrms_cache_dir", "stage4_cache_dir"):
        if key not in cfg:
            raise ValueError(f"{path.name}: {key} is required")

    truth = str(cfg.get("truth_source", "mrms")).lower()
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
    gs = cfg.get("grid_stat") or {}
    _unknown(gs, GRID_STAT_KEYS, "grid_stat")
    defaults = GridStatSettings()
    gs_settings = GridStatSettings(
        thresholds_mm=tuple(float(v) for v in
                            gs.get("thresholds_mm", defaults.thresholds_mm)),
        widths_km=tuple(float(v) for v in
                        gs.get("widths_km", defaults.widths_km)),
        wet_thresh_mm=float(gs.get("wet_thresh_mm", defaults.wet_thresh_mm)),
        cov_thresh=float(gs.get("cov_thresh", defaults.cov_thresh)),
        test_end=(parse_stamp(gs["test_end"], "grid_stat.test_end")
                  if gs.get("test_end") is not None else None))
    if (not gs_settings.thresholds_mm or min(gs_settings.thresholds_mm) < 0
            or not gs_settings.widths_km or min(gs_settings.widths_km) <= 0
            or not 0 < gs_settings.cov_thresh <= 1):
        raise ValueError("grid_stat needs thresholds >= 0, widths > 0 and "
                         "0 < cov_thresh <= 1")
    eyeball = cfg.get("eyeball") or {}
    _unknown(eyeball, EYEBALL_KEYS, "eyeball")
    style = str(eyeball.get("colorbar_style", "true_scale"))
    if style not in COLORBAR_STYLES:
        raise ValueError(f"eyeball.colorbar_style must be one of "
                         f"{COLORBAR_STYLES}, got {style!r}")
    members = tuple(int(m) for m in eyeball.get("members", (0, 9)))
    if (not members or min(members) < 0
            or int(eyeball.get("windows", 3)) < 1
            or int(eyeball.get("cycles", 2)) < 1):
        raise ValueError("eyeball needs members >= 0 and windows, cycles >= 1")
    plots = cfg.get("regrid_plots") or {}
    _unknown(plots, PLOT_KEYS, "regrid_plots")
    masks = cfg.get("masks") or {}
    _unknown(masks, MASK_KEYS, "masks")
    coast = float(coast_km if coast_km is not None
                  else masks.get("coast_buffer_km", DEFAULT_COAST_KM))
    if coast < 0:
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
        stage4_regrid_3h_dir=Path(cfg["stage4_regrid_3h_dir"]),
        mrms_regrid_3h_dir=Path(cfg["mrms_regrid_3h_dir"]),
        regrid_plot_dir=_resolution_dir(Path(cfg["regrid_plot_dir"]), res),
        eyeball_plot_dir=_resolution_dir(Path(cfg["eyeball_plot_dir"]), res),
        hafs_check_plot_dir=_resolution_dir(Path(cfg["hafs_check_plot_dir"]),
                                            res),
        mask_dir=Path(cfg["mask_dir"]),
        mask_plot_dir=Path(cfg["mask_plot_dir"]),
        manifest_dir=Path(cfg["manifest_dir"]),
        wofs_regrid_1h_dir=Path(cfg["wofs_regrid_1h_dir"]),
        wofs_regrid_3h_dir=Path(cfg["wofs_regrid_3h_dir"]),
        hafs_regrid_dir=Path(cfg["hafs_regrid_dir"]),
        grid_stat_dir=Path(cfg["grid_stat_dir"]),
        grid_stat=gs_settings,
        masks=MaskSettings(
            coast_buffer_km=coast,
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
            budget_check=bool(regrid.get("budget_check", True)),
            met_bin_dir=(Path(regrid["met_bin_dir"])
                         if regrid.get("met_bin_dir") else None),
            fields=dict(regrid.get("fields") or {})),
        regrid_panels=panels,
        eyeball=EyeballSettings(
            members=members, windows=int(eyeball.get("windows", 3)),
            cycles=int(eyeball.get("cycles", 2)), colorbar_style=style),
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
