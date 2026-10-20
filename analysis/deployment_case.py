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
            "out_dir", "mrms_cache_dir", "stage4_cache_dir", "grid",
            "log_dir"}
WOFS_KEYS = {"root", "available", "complete_windows"}
HAFS_KEYS = {"lookback_h", "min_members", "init_start", "init_end", "models"}
GRID_KEYS = {"res_km"}


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
    out_dir: Path
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
    log_dir: Optional[Path] = None

    @property
    def wofs_storm_folder(self):
        """The archive's storm folder (e.g. 2024_BERYL), the parent of root."""
        return self.wofs_root.parent.name


def from_yaml(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    _unknown(cfg, TOP_KEYS, "case")
    for key in ("storm_name", "storm_id", "deployment", "best_track", "wofs",
                "out_dir", "mrms_cache_dir", "stage4_cache_dir"):
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
        out_dir=Path(cfg["out_dir"]),
        mrms_cache_dir=Path(cfg["mrms_cache_dir"]),
        stage4_cache_dir=Path(cfg["stage4_cache_dir"]),
        res_km=float(grid.get("res_km", DEFAULT_RES_KM)),
        log_dir=Path(cfg["log_dir"]) if cfg.get("log_dir") else None,
    )
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
