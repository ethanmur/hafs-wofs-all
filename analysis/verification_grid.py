"""The common verification grid: each WoFS deployment's own domain at 6 km.

Observations and forecasts are regridded onto this grid with MET's budget
interpolation, so every comparison happens cell-for-cell at one physical
spacing. The grid is not chosen: it is read from the WoFS files, which carry
their native Lambert conformal grid (300 x 300 cells at 3 km, 900 km square),
and coarsened by whole 2 x 2 blocks to 6 km, so each 6 km cell is exactly four
WoFS cells. Each deployment has its own grid; a storm can have several.

The "legacy case configuration" block below (GridConfig, WofsDomain, ...) is
what the old best-track grid builder read. It stays only while obs_cases.py
and masks.py still import it, and goes with them in plan steps B0 and C0.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =============================================================================
# Legacy case configuration (removed with obs_cases.py / masks.py, steps B0/C0)
# =============================================================================

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


def parse_stamp(value, key, where=""):
    """Optional YYYYMMDDHH (or YYYYMMDDHHMM) timestamp from a YAML value."""
    from datetime import datetime
    if value in (None, ""):
        return None
    text = str(value)
    fmt = "%Y%m%d%H%M" if len(text) == 12 else "%Y%m%d%H"
    try:
        return datetime.strptime(text, fmt)
    except ValueError:
        suffix = f" in {where}" if where else ""
        raise ValueError(f"'{key}' must be YYYYMMDDHH{suffix}, "
                         f"got {value!r}") from None


@dataclass
class WofsDomain:
    """One WoFS deployment: a box, and the period it was actually running.

    WoFS is re-sited between events, so a storm can have several deployments
    in different places on different days. The grid takes the union of them,
    but each stays a separate statistics mask applied only inside its own time
    window -- masking day-3 statistics with a box that was live on day 1 would
    score HAFS in a region WoFS never covered.
    """
    domain: tuple                  # (lat_min, lat_max, lon_min, lon_max)
    name: str = "wofs"
    valid_start: Optional[object] = None    # datetime, or None for always-on
    valid_end: Optional[object] = None

    def covers(self, when):
        """Is this deployment live at `when`? Untimed boxes always are."""
        if self.valid_start is not None and when < self.valid_start:
            return False
        if self.valid_end is not None and when > self.valid_end:
            return False
        return True

    @property
    def label(self):
        if self.valid_start is None and self.valid_end is None:
            return self.name
        lo = f"{self.valid_start:%m-%d %HZ}" if self.valid_start else "..."
        hi = f"{self.valid_end:%m-%d %HZ}" if self.valid_end else "..."
        return f"{self.name}  {lo}-{hi}"


def wofs_domains_from_cfg(cfg, where=""):
    """Read `wofs_domains:` (a list) or the `wofs_domain:` shorthand.

    A list entry is either a bare [lat_min, lat_max, lon_min, lon_max] box or
    a mapping with `domain` plus optional `name`, `valid_start`, `valid_end`.
    """
    cfg = cfg or {}
    if "wofs_domains" in cfg and "wofs_domain" in cfg:
        raise ValueError("give either wofs_domain or wofs_domains, not both")
    if cfg.get("wofs_domain") is not None:
        return [WofsDomain(domain=_domain_tuple(cfg["wofs_domain"],
                                                "wofs_domain"))]
    entries = cfg.get("wofs_domains") or []
    if isinstance(entries, dict):
        entries = [entries]
    out = []
    for i, entry in enumerate(entries, start=1):
        default_name = f"wofs_{i}"
        if isinstance(entry, dict):
            unknown = set(entry) - {"domain", "name", "valid_start",
                                    "valid_end"}
            if unknown:
                raise ValueError("unknown wofs_domains keys: "
                                 + ", ".join(sorted(unknown)))
            if "domain" not in entry:
                raise ValueError(f"wofs_domains[{i}] needs a 'domain'")
            box = _domain_tuple(entry["domain"], f"wofs_domains[{i}].domain")
            start = parse_stamp(entry.get("valid_start"),
                                f"wofs_domains[{i}].valid_start", where)
            end = parse_stamp(entry.get("valid_end"),
                              f"wofs_domains[{i}].valid_end", where)
            if start is not None and end is not None and end <= start:
                raise ValueError(f"wofs_domains[{i}] valid_end must be after "
                                 "valid_start")
            out.append(WofsDomain(domain=box,
                                  name=str(entry.get("name", default_name)),
                                  valid_start=start, valid_end=end))
        else:
            out.append(WofsDomain(
                domain=_domain_tuple(entry, f"wofs_domains[{i}]"),
                name=default_name))
    names = [d.name for d in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate wofs_domains names: {names}")
    return out


def active_domains(domains, when):
    """The deployments live at a given valid time."""
    return [d for d in domains if d.covers(when)]


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


# =============================================================================
# Grid construction
# =============================================================================

R_EARTH_KM = 6371.2      # MET's default spherical earth radius for grid specs

# MET's grid-specification strings take longitudes EAST-positive (negative
# for west), the same sign as everywhere else here, and MET negates them into
# its own internal west-positive form.
#
# Established from MET's own debug output: regridding Stage IV, it printed the
# HRAP input grid's centre meridian as "Lon_orient: 105.000" -- positive for
# 105 W -- while printing a west-positive spec of ours as "Lon_orient:
# -82.350". Same debug code, so positive means west internally, and our grid
# had been placed at 82.35 E, over the Indian Ocean. Every output cell came
# back missing for every method and width, which is the signature to watch
# for: compare the sign of the two printed Lon_orient values.
MET_LON_WEST_POSITIVE = False

# Okabe-Ito, distinguishable under red-green colour blindness.
STATUS_COLORS = {"DB": "#56b4e9", "TD": "#0072b2", "TS": "#e69f00",
                 "HU": "#d55e00", "SD": "#009e73", "SS": "#009e73",
                 "LO": "#cc79a7", "EX": "#000000"}
STATUS_LABELS = {"DB": "disturbance", "TD": "tropical depression",
                 "TS": "tropical storm", "HU": "hurricane",
                 "SD": "subtropical dep.", "SS": "subtropical storm",
                 "LO": "remnant low", "EX": "extratropical"}


@dataclass
class GridSpec:
    """One deployment's Lambert conformal verification grid."""
    name: str
    nx: int
    ny: int
    res_km: float
    lat_0: float
    lon_0: float
    lat_1: float
    lat_2: float
    lat_ll: float       # lower-left grid point, degrees north
    lon_ll: float       # lower-left grid point, degrees EAST (negative = west)
    x_ll_km: float      # lower-left corner in projected coordinates
    y_ll_km: float
    source: str = ""    # the WoFS file the grid was read from

    @property
    def proj4(self):
        """East-positive PROJ.4 string, for pyproj/cartopy."""
        return (f"+proj=lcc +lat_1={self.lat_1} +lat_2={self.lat_2} "
                f"+lat_0={self.lat_0} +lon_0={self.lon_0} "
                f"+R={R_EARTH_KM * 1000.0:.1f} +units=m +no_defs")

    @property
    def met_spec(self):
        """MET grid-specification string for -to_grid / regrid.to_grid."""
        sign = -1.0 if MET_LON_WEST_POSITIVE else 1.0
        # The trailing hemisphere is required, and MET wants the single
        # letter: parse_lambert_grid() rejects "north" with "bad hemisphere
        # in grid spec" just as it rejects the token being absent.
        hemisphere = "N" if self.lat_0 >= 0 else "S"
        return (f"lambert {self.nx} {self.ny} "
                f"{self.lat_ll:.6f} {sign * self.lon_ll:.6f} "
                f"{sign * self.lon_0:.6f} {self.res_km:.6f} "
                f"{R_EARTH_KM:.3f} {self.lat_1:.6f} {self.lat_2:.6f} "
                f"{hemisphere}")

    @property
    def n_cells(self):
        return self.nx * self.ny


def _crs(spec_or_proj):
    """pyproj CRS for a GridSpec or a projection-parameter dict."""
    import pyproj
    if isinstance(spec_or_proj, GridSpec):
        return pyproj.CRS.from_proj4(spec_or_proj.proj4)
    p = spec_or_proj
    return pyproj.CRS.from_proj4(
        f"+proj=lcc +lat_1={p['lat_1']} +lat_2={p['lat_2']} "
        f"+lat_0={p['lat_0']} +lon_0={p['lon_0']} "
        f"+R={R_EARTH_KM * 1000.0:.1f} +units=m +no_defs")


def _transformer(proj, inverse=False):
    import pyproj
    lonlat = pyproj.CRS.from_epsg(4326)
    crs = _crs(proj)
    a, b = (crs, lonlat) if inverse else (lonlat, crs)
    return pyproj.Transformer.from_crs(a, b, always_xy=True)


def trim_track(track, valid_start, valid_end, margin_h):
    """Track points inside the valid window, widened by margin_h either side.

    The grid follows the *scoring* window, not the cycle range: a forecast
    initialized days earlier is still only verified inside this window, so its
    earlier positions must not inflate the domain.
    """
    from datetime import timedelta
    lo = valid_start - timedelta(hours=margin_h)
    hi = valid_end + timedelta(hours=margin_h)
    return [p for p in track if lo <= p[0] <= hi]


def grid_latlon(spec):
    """(lat, lon) 2-D arrays of the grid points, shape (ny, nx)."""
    import numpy as np
    x = (spec.x_ll_km + np.arange(spec.nx) * spec.res_km) * 1000.0
    y = (spec.y_ll_km + np.arange(spec.ny) * spec.res_km) * 1000.0
    xx, yy = np.meshgrid(x, y)
    lon, lat = _transformer(spec, inverse=True).transform(xx, yy)
    return np.asarray(lat), np.asarray(lon)


def spacing_report(spec, lat=None, lon=None):
    """True ground spacing of the grid, for the A4 fairness check.

    Returns min/max/mean dx and dy in km plus the worst percentage deviation
    from the nominal resolution. A conformal projection keeps dx == dy at each
    point; what varies is the scale factor away from the standard parallels.
    """
    import numpy as np
    import pyproj
    if lat is None or lon is None:
        lat, lon = grid_latlon(spec)
    geod = pyproj.Geod(a=R_EARTH_KM * 1000.0, b=R_EARTH_KM * 1000.0)
    _, _, dx = geod.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])
    _, _, dy = geod.inv(lon[:-1, :], lat[:-1, :], lon[1:, :], lat[1:, :])
    dx = np.asarray(dx) / 1000.0
    dy = np.asarray(dy) / 1000.0
    worst = max(abs(dx.min() - spec.res_km), abs(dx.max() - spec.res_km),
                abs(dy.min() - spec.res_km), abs(dy.max() - spec.res_km))
    return {"dx_min": float(dx.min()), "dx_max": float(dx.max()),
            "dx_mean": float(dx.mean()), "dy_min": float(dy.min()),
            "dy_max": float(dy.max()), "dy_mean": float(dy.mean()),
            "worst_pct": float(100.0 * worst / spec.res_km)}


def grid_to_dict(spec, report=None, extra=None):
    """Serialisable record of a grid, written beside the sanity plot."""
    out = {
        "name": spec.name,
        "source": spec.source,
        "nx": spec.nx, "ny": spec.ny, "n_cells": spec.n_cells,
        "res_km": spec.res_km,
        "projection": "lambert_conformal",
        "lat_0": spec.lat_0, "lon_0": spec.lon_0,
        "lat_1": spec.lat_1, "lat_2": spec.lat_2,
        "lat_ll": spec.lat_ll, "lon_ll": spec.lon_ll,
        "x_ll_km": spec.x_ll_km, "y_ll_km": spec.y_ll_km,
        "earth_radius_km": R_EARTH_KM,
        "proj4": spec.proj4,
        "met_spec": spec.met_spec,
        "met_lon_west_positive": MET_LON_WEST_POSITIVE,
    }
    if report is not None:
        out["spacing_km"] = report
    if extra:
        out.update(extra)
    return out


def write_grid_json(path, payload):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def _grid_outline(lat, lon):
    """Perimeter of the grid, walked so curved edges plot correctly."""
    import numpy as np
    edges = [(lat[0, :], lon[0, :]), (lat[:, -1], lon[:, -1]),
             (lat[-1, ::-1], lon[-1, ::-1]), (lat[::-1, 0], lon[::-1, 0])]
    return (np.concatenate([e[0] for e in edges]),
            np.concatenate([e[1] for e in edges]))




# =============================================================================
# The WoFS-domain grid
# =============================================================================

WOFS_REQUIRED_ATTRS = ("start_lat", "start_lon", "lon_orient", "scale_lat_1",
                       "scale_lat_2", "d_km", "nx", "ny")
# Worst tolerated disagreement between the grid built from a file's attributes
# and the file's own xlat/xlon arrays, as a fraction of a verification cell.
MAX_OFFSET_FRACTION = 0.1


def aggregation_factor(d_km, res_km):
    """Whole number of native cells per verification cell along each axis."""
    ratio = res_km / d_km
    n = round(ratio)
    if n < 1 or abs(ratio - n) > 1e-6:
        raise ValueError(
            f"res_km={res_km:g} is not a whole multiple of the native "
            f"{d_km:g} km spacing, so verification cells would split WoFS "
            "cells")
    return n


def wofs_grid_spec(attrs, res_km=DEFAULT_RES_KM, name=None, source=""):
    """The verification grid for a WoFS file's global attributes.

    The file's attributes describe its native grid the way MET does: the
    lower-left grid point (start_lat/start_lon), the Lambert standard
    longitude and parallels, and the spacing. The verification grid is that
    grid coarsened by n x n blocks, so its lower-left point is the centre of
    the first block, (n - 1) * d / 2 km up and to the right of the native one.
    Nothing here is chosen: change the file and the grid changes with it.
    """
    missing = [k for k in WOFS_REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise KeyError("WoFS file lacks grid attributes: "
                       + ", ".join(missing))
    d = float(attrs["d_km"])
    nx_n, ny_n = int(float(attrs["nx"])), int(float(attrs["ny"]))
    n = aggregation_factor(d, res_km)
    if nx_n % n or ny_n % n:
        raise ValueError(f"the {nx_n} x {ny_n} native grid does not divide "
                         f"into {n} x {n} blocks")
    lat_1, lat_2 = float(attrs["scale_lat_1"]), float(attrs["scale_lat_2"])
    proj = {"lat_0": float(attrs.get("CEN_LAT", 0.5 * (lat_1 + lat_2))),
            "lon_0": float(attrs["lon_orient"]),
            "lat_1": lat_1, "lat_2": lat_2}
    x0, y0 = _transformer(proj).transform(float(attrs["start_lon"]),
                                          float(attrs["start_lat"]))
    shift = (n - 1) * d / 2.0
    x_ll, y_ll = x0 / 1000.0 + shift, y0 / 1000.0 + shift
    lon_ll, lat_ll = _transformer(proj, inverse=True).transform(
        x_ll * 1000.0, y_ll * 1000.0)
    return GridSpec(
        name=name or f"wofs{res_km:g}km", nx=nx_n // n, ny=ny_n // n,
        res_km=res_km, lat_0=proj["lat_0"], lon_0=proj["lon_0"],
        lat_1=lat_1, lat_2=lat_2, lat_ll=round(lat_ll, 6),
        lon_ll=round(lon_ll, 6), x_ll_km=x_ll, y_ll_km=y_ll, source=source)


def _haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def offset_from_file_km(spec, xlat, xlon, n):
    """(max, mean) km between the grid's cell centres and the file's own.

    `xlat`/`xlon` are the file's native 2-D arrays; for n > 1 they are averaged
    over the n x n blocks first, which is what each verification cell is. This
    checks the projection assumptions against the data itself: a misread
    standard parallel or longitude shows up as an offset growing away from the
    lower-left corner.
    """
    import numpy as np
    lat, lon = grid_latlon(spec)
    ny, nx = lat.shape
    xlat, xlon = np.asarray(xlat, float), np.asarray(xlon, float)
    if xlat.shape != (ny * n, nx * n):
        raise ValueError(f"file arrays are {xlat.shape}, expected "
                         f"{(ny * n, nx * n)} for this grid")
    blat = xlat.reshape(ny, n, nx, n).mean(axis=(1, 3))
    blon = xlon.reshape(ny, n, nx, n).mean(axis=(1, 3))
    d = _haversine_km(lat, lon, blat, blon)
    return float(d.max()), float(d.mean())


def read_wofs_latlon(path):
    """The native xlat/xlon arrays of one WoFS file."""
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        return ds["xlat"].values, ds["xlon"].values


def plot_grid(spec, track, out_path, window=None, title=None, report=None,
              offset=None):
    """A3 sanity map: the domain outline and the best track around it.

    The whole b-deck is drawn faintly for context and the fixes inside the
    scored window in colour by ATCF status. The track is context only; it
    plays no part in the grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    from plot_common import add_us_geography

    lat, lon = grid_latlon(spec)
    globe = ccrs.Globe(ellipse=None, semimajor_axis=R_EARTH_KM * 1000.0,
                       semiminor_axis=R_EARTH_KM * 1000.0)
    proj = ccrs.LambertConformal(
        central_longitude=spec.lon_0, central_latitude=spec.lat_0,
        standard_parallels=(spec.lat_1, spec.lat_2), globe=globe)
    plain = ccrs.PlateCarree()

    inside = [p for p in track if window and window[0] <= p[0] <= window[1]]
    olat, olon = _grid_outline(lat, lon)
    lats = list(olat) + [p[1] for p in inside]
    lons = list(olon) + [p[2] for p in inside]
    margin = 2.0

    fig = plt.figure(figsize=(10.0, 8.5))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([min(lons) - margin, max(lons) + margin,
                   max(min(lats) - margin, -85.0),
                   min(max(lats) + margin, 85.0)], crs=plain)
    add_us_geography(ax)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb",
                 alpha=0.7, linestyle=":")

    if track:
        ax.plot([p[2] for p in track], [p[1] for p in track], transform=plain,
                color="#aaaaaa", linewidth=1.0, zorder=3)
    ax.plot(olon, olat, transform=plain, color="#444444", linewidth=2.4,
            zorder=5)

    seen = []
    for (t0, lat0, lon0, st0), (t1, lat1, lon1, _) in zip(inside, inside[1:]):
        ax.plot([lon0, lon1], [lat0, lat1], transform=plain,
                color=STATUS_COLORS.get(st0, "#888888"), linewidth=2.2,
                solid_capstyle="round", zorder=6)
    for t0, lat0, lon0, st0 in inside:
        ax.plot(lon0, lat0, transform=plain, marker="o", markersize=3.4,
                color=STATUS_COLORS.get(st0, "#888888"), zorder=7)
        if st0 not in seen:
            seen.append(st0)

    handles = [Line2D([], [], color="#444444", linewidth=2.4,
                      label=f"grid {spec.nx}x{spec.ny} @ {spec.res_km:g} km")]
    handles += [Line2D([], [], color=STATUS_COLORS.get(s, "#888888"),
                       linewidth=2.2, label=STATUS_LABELS.get(s, s))
                for s in seen]
    if track:
        handles.append(Line2D([], [], color="#aaaaaa", linewidth=1.0,
                              label="best track outside the window"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    lines = [title or "verification grid"]
    if window:
        lines.append(f"scored {window[0]:%Y-%m-%d %HZ} to "
                     f"{window[1]:%Y-%m-%d %HZ}   "
                     f"({len(inside)} best-track fixes in the window)")
    sub = f"{spec.n_cells:,} cells"
    if report:
        sub += f"   ground spacing within {report['worst_pct']:.2f}% of nominal"
    if offset:
        sub += f"   centres within {offset[0]:.2f} km of the WoFS file"
    lines.append(sub)
    ax.set_title("\n".join(lines), fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_grid_case(case):
    """Driver for the `build-grid` command: read, build, check, write, plot."""
    import wofs_inventory
    from best_track import parse_bdeck_status

    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    if not case.wofs_available:
        print("WoFS   : no complete 3 h window in the archive; no grid built "
              "for this deployment")
        return None
    print(f"Window : {case.valid_start:%Y-%m-%d %HZ} -> "
          f"{case.valid_end:%Y-%m-%d %HZ}")

    dep = wofs_inventory.scan_deployment(case.wofs_storm_folder,
                                         case.wofs_root)
    if not dep.inits:
        raise SystemExit(f"ERROR: no WoFS files under {case.wofs_root}")
    if len(dep.grids) > 1:
        raise SystemExit(
            f"ERROR: {len(dep.grids)} different grids under {case.wofs_root}: "
            "one deployment must have one grid")
    first_file = dep.inits[0].first_file
    n = aggregation_factor(float(dep.attrs["d_km"]), case.res_km)
    print(f"WoFS   : {len(dep.inits)} init folders, one grid; read from "
          f"{first_file.name}")

    spec = wofs_grid_spec(dep.attrs, case.res_km, name=case.case_slug,
                          source=str(first_file))
    lat, lon = grid_latlon(spec)
    report = spacing_report(spec, lat, lon)
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km "
          f"({spec.n_cells:,} cells), {n} x {n} WoFS cells each")
    print(f"Lambert: lon_0={spec.lon_0} parallels=({spec.lat_1}, "
          f"{spec.lat_2})  centre {dep.attrs.get('CEN_LAT')}, "
          f"{dep.attrs.get('CEN_LON')}")
    print(f"Extent : {lat.min():.2f} to {lat.max():.2f} N, "
          f"{lon.min():.2f} to {lon.max():.2f} E")
    print(f"Spacing: dx {report['dx_min']:.3f}-{report['dx_max']:.3f} km, "
          f"dy {report['dy_min']:.3f}-{report['dy_max']:.3f} km  "
          f"(worst {report['worst_pct']:.2f}% from {spec.res_km:g} km; the "
          "WoFS map scale, not an error)")

    xlat, xlon = read_wofs_latlon(first_file)
    native = wofs_grid_spec(dep.attrs, float(dep.attrs["d_km"]),
                            name="native")
    off_native = offset_from_file_km(native, xlat, xlon, 1)
    off_grid = offset_from_file_km(spec, xlat, xlon, n)
    print(f"Check  : native cell centres within {off_native[0]:.3f} km "
          f"(mean {off_native[1]:.3f}) of the file's xlat/xlon; verification "
          f"cells within {off_grid[0]:.3f} km (mean {off_grid[1]:.3f})")
    if off_grid[0] > MAX_OFFSET_FRACTION * spec.res_km:
        raise SystemExit(
            f"ERROR: the grid built from the attributes disagrees with the "
            f"file's own coordinates by {off_grid[0]:.2f} km "
            f"(> {MAX_OFFSET_FRACTION * spec.res_km:.2f} km); the projection "
            "assumptions need a look before anything is regridded")
    print(f"MET    : {spec.met_spec}")

    track = parse_bdeck_status(case.best_track)
    window = (case.valid_start, case.valid_end)
    payload = grid_to_dict(spec, report, extra={
        "case": case.case_slug,
        "storm_name": case.storm_name,
        "deployment": case.deployment,
        "wofs_root": str(case.wofs_root),
        "best_track": str(case.best_track),
        "valid_start": f"{case.valid_start:%Y%m%d%H}",
        "valid_end": f"{case.valid_end:%Y%m%d%H}",
        "wofs_native": {"nx": int(float(dep.attrs["nx"])),
                        "ny": int(float(dep.attrs["ny"])),
                        "d_km": float(dep.attrs["d_km"]),
                        "start_lat": float(dep.attrs["start_lat"]),
                        "start_lon": float(dep.attrs["start_lon"]),
                        "aggregation": f"{n}x{n}"},
        "offset_from_file_km": {"native_max": off_native[0],
                                "native_mean": off_native[1],
                                "grid_max": off_grid[0],
                                "grid_mean": off_grid[1]},
        "init_folders_checked": len(dep.inits),
    })
    out_dir = Path(case.out_dir)
    json_path = write_grid_json(out_dir / f"{case.case_slug}_grid.json",
                                payload)
    png_path = plot_grid(spec, track, out_dir / f"{case.case_slug}_grid.png",
                         window=window,
                         title=f"{case.storm_name} {case.deployment}: "
                               "verification grid",
                         report=report, offset=off_grid)
    print(f"Wrote  : {json_path}")
    print(f"Wrote  : {png_path}")
    return spec
