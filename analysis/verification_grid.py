"""Configuration for the common verification grid (one per storm case).

Both observations and forecasts are regridded onto this grid with MET's budget
interpolation so that every comparison happens cell-for-cell at a constant
physical spacing. The grid itself is built in a later step; this module holds
the configuration surface and its validation.

Two radii matter here and they are deliberately separate:

  pad_km          how far the *grid* extends beyond the trimmed best track.
  mask_radius_km  how far the *statistics* swath extends from the track.

The grid pad is the larger of the two. A forecast initialized days before
landfall can be several hundred km off in track, and any of its rain that
falls outside the grid is silently uncounted -- an asymmetric bias, since
misses inside the domain still count while false alarms beyond the boundary
disappear. Padding the grid past the scoring mask leaves room for that error.
"""

from dataclasses import dataclass
from typing import Optional


DEFAULT_RES_KM = 6.0
DEFAULT_PAD_KM = 750.0
DEFAULT_MASK_RADIUS_KM = 500.0
DEFAULT_MARGIN_H = 6
DEFAULT_MAX_CELLS = 800


@dataclass
class GridConfig:
    """Geometry of one case's verification grid."""
    res_km: float = DEFAULT_RES_KM
    pad_km: float = DEFAULT_PAD_KM
    mask_radius_km: float = DEFAULT_MASK_RADIUS_KM
    margin_h: int = DEFAULT_MARGIN_H     # track kept either side of the window
    max_cells: int = DEFAULT_MAX_CELLS   # guard on Nx and Ny
    std_parallels: Optional[tuple] = None    # override the derived parallels
    domain_override: Optional[tuple] = None  # (lat_min, lat_max, lon_*, lon_*)
    name: Optional[str] = None               # grid id used in cache paths

    @property
    def grid_name(self):
        """Identity for cache paths; keeps resolutions from mixing on disk."""
        return self.name or f"lambert{self.res_km:g}km"


def _domain_tuple(value, key):
    """Validate a [lat_min, lat_max, lon_min, lon_max] box."""
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{key} must be "
                         "[lat_min, lat_max, lon_min, lon_max]")
    lat0, lat1, lon0, lon1 = (float(v) for v in value)
    if lat0 >= lat1 or lon0 >= lon1:
        raise ValueError(f"{key} must be ordered "
                         "[lat_min, lat_max, lon_min, lon_max]")
    return (lat0, lat1, lon0, lon1)


def grid_config_from_dict(cfg):
    """Build a GridConfig from a YAML `verification_grid:` block.

    Returns the defaults when the block is absent, so a case YAML needs the
    block only to deviate from them.
    """
    cfg = cfg or {}
    unknown = set(cfg) - {"res_km", "pad_km", "mask_radius_km", "margin_h",
                          "max_cells", "std_parallels", "domain_override",
                          "name"}
    if unknown:
        raise ValueError("unknown verification_grid keys: "
                         + ", ".join(sorted(unknown)))
    res_km = float(cfg.get("res_km", DEFAULT_RES_KM))
    pad_km = float(cfg.get("pad_km", DEFAULT_PAD_KM))
    mask_radius_km = float(cfg.get("mask_radius_km",
                                   DEFAULT_MASK_RADIUS_KM))
    margin_h = int(cfg.get("margin_h", DEFAULT_MARGIN_H))
    max_cells = int(cfg.get("max_cells", DEFAULT_MAX_CELLS))
    if res_km <= 0:
        raise ValueError("verification_grid.res_km must be positive")
    if mask_radius_km <= 0:
        raise ValueError("verification_grid.mask_radius_km must be positive")
    if pad_km < mask_radius_km:
        raise ValueError(
            f"verification_grid.pad_km ({pad_km:g} km) must be at least "
            f"mask_radius_km ({mask_radius_km:g} km), or the statistics swath "
            "would extend past the edge of the grid")
    if margin_h < 0:
        raise ValueError("verification_grid.margin_h cannot be negative")
    if max_cells <= 0:
        raise ValueError("verification_grid.max_cells must be positive")
    parallels = cfg.get("std_parallels")
    if parallels is not None:
        if len(parallels) != 2:
            raise ValueError("verification_grid.std_parallels must be "
                             "[lat1, lat2]")
        parallels = tuple(float(v) for v in parallels)
        if parallels[0] >= parallels[1]:
            raise ValueError("verification_grid.std_parallels must be "
                             "increasing")
    return GridConfig(
        res_km=res_km,
        pad_km=pad_km,
        mask_radius_km=mask_radius_km,
        margin_h=margin_h,
        max_cells=max_cells,
        std_parallels=parallels,
        domain_override=_domain_tuple(cfg.get("domain_override"),
                                      "verification_grid.domain_override"),
        name=cfg.get("name"),
    )
