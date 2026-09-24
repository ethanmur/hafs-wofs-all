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

import math
from dataclasses import dataclass
from pathlib import Path
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

# MET's grid-specification strings take longitudes as degrees WEST-positive,
# the opposite sign to every other longitude in this codebase. Flip this to
# False if a regridded field lands in the wrong hemisphere -- the JSON output
# also carries an unambiguous east-positive PROJ.4 string for Python use, so
# only the MET string depends on this constant.
MET_LON_WEST_POSITIVE = True

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
    """One case's Lambert conformal verification grid."""
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
    rule: str           # which domain-precedence rule produced this grid

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
        return (f"lambert {self.nx} {self.ny} "
                f"{self.lat_ll:.6f} {sign * self.lon_ll:.6f} "
                f"{sign * self.lon_0:.6f} {self.res_km:.6f} "
                f"{R_EARTH_KM:.3f} {self.lat_1:.6f} {self.lat_2:.6f}")

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


def _boxes(wofs_domains):
    """Normalise WofsDomain objects or bare boxes to a list of box tuples."""
    if not wofs_domains:
        return []
    if isinstance(wofs_domains, WofsDomain):
        wofs_domains = [wofs_domains]
    elif (len(wofs_domains) == 4
          and all(isinstance(v, (int, float)) for v in wofs_domains)):
        wofs_domains = [WofsDomain(domain=tuple(float(v)
                                                for v in wofs_domains))]
    return [d.domain if isinstance(d, WofsDomain) else tuple(float(v)
            for v in d) for d in wofs_domains]


def _corners(boxes):
    """Corner pseudo-fixes (_, lat, lon) for a list of boxes."""
    return [(None, lat, lon) for box in boxes
            for lat in box[:2] for lon in box[2:]]


def _bbox(points, boxes=()):
    """(lat_min, lat_max, lon_min, lon_max) over track points and WoFS boxes."""
    lats = [p[1] for p in points]
    lons = [p[2] for p in points]
    for box in boxes:
        lats.extend(box[:2])
        lons.extend(box[2:])
    if not lats:
        raise ValueError("no positions to bound")
    return (min(lats), max(lats), min(lons), max(lons))


def derive_projection(bbox):
    """Lambert parameters from a lat/lon box.

    The standard parallels are inset one sixth of the latitude span from each
    end, which brackets the domain and holds the scale error to order 1%. A
    single parallel is exact only on that line, and a lat/lon or Mercator grid
    varies with cos(lat) in the zonal direction -- either would make a fixed
    neighbourhood width mean different physical distances at different
    latitudes, which is what the FSS scales depend on.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    if lon_max - lon_min > 180.0:
        raise ValueError("domain spans more than 180 degrees of longitude; "
                         "a Lambert conformal grid is not appropriate")
    span = lat_max - lat_min
    mid_lat = 0.5 * (lat_min + lat_max)
    if span < 6.0:                      # too narrow to inset meaningfully
        lat_1, lat_2 = mid_lat - 2.0, mid_lat + 2.0
    else:
        lat_1, lat_2 = lat_min + span / 6.0, lat_max - span / 6.0
    return {"lat_0": round(mid_lat, 4), "lon_0": round(0.5 * (lon_min + lon_max), 4),
            "lat_1": round(lat_1, 4), "lat_2": round(lat_2, 4)}


def build_grid(track, cfg, wofs_domains=None, name=None):
    """Build a GridSpec from already-trimmed track points.

    Every WoFS deployment is unioned into the extent, so one grid per storm
    covers all of them; their time windows matter only for masking later.

    Domain precedence, reported in GridSpec.rule:
      1. cfg.domain_override
      2. trimmed track + cfg.pad_km, unioned with the WoFS boxes
      3. WoFS boxes + cfg.pad_km alone, when no track point survives the window
    """
    import numpy as np
    boxes = _boxes(wofs_domains)
    if cfg.domain_override is not None:
        rule = "domain_override"
        box = cfg.domain_override
        corners = _corners([box])
        pad_km = 0.0
    elif track:
        rule = "track+wofs" if boxes else "track"
        corners = list(track) + _corners(boxes)
        box = _bbox(track, boxes)
        pad_km = cfg.pad_km
    elif boxes:
        rule = "wofs-only"
        box = _bbox([], boxes)
        corners = _corners(boxes)
        pad_km = cfg.pad_km
    else:
        raise ValueError(
            "no track points inside the valid window and no wofs_domains or "
            "domain_override given; nothing to build a grid from")

    # Pick the projection from the padded geographic extent, so the standard
    # parallels bracket the grid rather than just the track.
    deg_pad = pad_km / 111.0
    lat_min, lat_max, lon_min, lon_max = box
    proj = derive_projection((max(lat_min - deg_pad, -89.0),
                              min(lat_max + deg_pad, 89.0),
                              lon_min - deg_pad, lon_max + deg_pad))

    fwd = _transformer(proj)
    xs, ys = fwd.transform([p[2] for p in corners], [p[1] for p in corners])
    xs = np.asarray(xs) / 1000.0
    ys = np.asarray(ys) / 1000.0
    res = cfg.res_km
    # snap outward to whole cells so the grid is reproducible from the spec
    x_ll = math.floor((xs.min() - pad_km) / res) * res
    y_ll = math.floor((ys.min() - pad_km) / res) * res
    x_ur = math.ceil((xs.max() + pad_km) / res) * res
    y_ur = math.ceil((ys.max() + pad_km) / res) * res
    nx = int(round((x_ur - x_ll) / res)) + 1
    ny = int(round((y_ur - y_ll) / res)) + 1
    if nx > cfg.max_cells or ny > cfg.max_cells:
        raise ValueError(
            f"grid is {nx} x {ny} cells at {res:g} km, above max_cells="
            f"{cfg.max_cells}. Narrow valid_start/valid_end, reduce pad_km, "
            f"or raise max_cells deliberately.")

    inv = _transformer(proj, inverse=True)
    lon_ll, lat_ll = inv.transform(x_ll * 1000.0, y_ll * 1000.0)
    return GridSpec(
        name=name or cfg.grid_name,
        nx=nx, ny=ny, res_km=res,
        lat_0=proj["lat_0"], lon_0=proj["lon_0"],
        lat_1=proj["lat_1"], lat_2=proj["lat_2"],
        lat_ll=round(lat_ll, 6), lon_ll=round(lon_ll, 6),
        x_ll_km=x_ll, y_ll_km=y_ll, rule=rule)


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
        "rule": spec.rule,
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


def plot_grid(spec, track, out_path, cfg, wofs_domains=None, title=None,
              report=None):
    """A3 sanity map: grid outline, best track by status, swath, WoFS box."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from compare import _add_us_geography

    lat, lon = grid_latlon(spec)
    globe = ccrs.Globe(ellipse=None, semimajor_axis=R_EARTH_KM * 1000.0,
                       semiminor_axis=R_EARTH_KM * 1000.0)
    proj = ccrs.LambertConformal(
        central_longitude=spec.lon_0, central_latitude=spec.lat_0,
        standard_parallels=(spec.lat_1, spec.lat_2), globe=globe)
    plain = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11.0, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    margin = max(3.0, 0.08 * (lat.max() - lat.min()))
    ax.set_extent([lon.min() - margin, lon.max() + margin,
                   max(lat.min() - margin, -85.0),
                   min(lat.max() + margin, 85.0)], crs=plain)
    _add_us_geography(ax)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb",
                 alpha=0.7, linestyle=":")

    # statistics swath: everything within mask_radius_km of the track
    if track:
        try:
            from shapely.geometry import LineString, Point
            fwd = _transformer(
                {"lat_0": spec.lat_0, "lon_0": spec.lon_0,
                 "lat_1": spec.lat_1, "lat_2": spec.lat_2})
            xs, ys = fwd.transform([p[2] for p in track],
                                   [p[1] for p in track])
            pts = list(zip(np.asarray(xs) / 1000.0, np.asarray(ys) / 1000.0))
            geom = (LineString(pts) if len(pts) > 1 else Point(pts[0]))
            swath = geom.buffer(cfg.mask_radius_km)
            bx, by = swath.exterior.xy
            ax.fill(np.asarray(bx) * 1000.0, np.asarray(by) * 1000.0,
                    transform=proj, facecolor="#0072b2", alpha=0.10,
                    edgecolor="#0072b2", linewidth=0.8, linestyle="--",
                    zorder=2)
        except Exception:
            pass          # swath is illustrative; never fail the figure for it

    # grid outline
    olat, olon = _grid_outline(lat, lon)
    ax.plot(olon, olat, transform=plain, color="#444444", linewidth=2.4,
            zorder=5)

    # best track, coloured by ATCF status
    seen = []
    for (t0, lat0, lon0, st0), (t1, lat1, lon1, _) in zip(track, track[1:]):
        colour = STATUS_COLORS.get(st0, "#888888")
        ax.plot([lon0, lon1], [lat0, lat1], transform=plain, color=colour,
                linewidth=2.2, solid_capstyle="round", zorder=6)
        if st0 not in seen:
            seen.append(st0)
    for t0, lat0, lon0, st0 in track:
        ax.plot(lon0, lat0, transform=plain, marker="o", markersize=3.4,
                color=STATUS_COLORS.get(st0, "#888888"), zorder=7)
        if st0 not in seen:
            seen.append(st0)

    domains = (wofs_domains if isinstance(wofs_domains, list)
               else ([wofs_domains] if wofs_domains else []))
    domains = [d if isinstance(d, WofsDomain) else WofsDomain(domain=tuple(d))
               for d in domains]
    for dom in domains:
        wlat0, wlat1, wlon0, wlon1 = dom.domain
        ax.add_patch(Rectangle(
            (wlon0, wlat0), wlon1 - wlon0, wlat1 - wlat0, transform=plain,
            facecolor="none", edgecolor="#cc79a7", linewidth=1.6,
            linestyle="--", zorder=8))
        ax.text(wlon0, wlat1, f" {dom.label}", transform=plain, fontsize=7.5,
                color="#cc79a7", va="bottom", ha="left", zorder=9,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75))

    handles = [Line2D([], [], color="#444444", linewidth=2.4,
                      label=f"grid {spec.nx}x{spec.ny} @ {spec.res_km:g} km"),
               Patch(facecolor="#0072b2", alpha=0.10, edgecolor="#0072b2",
                     linestyle="--",
                     label=f"swath {cfg.mask_radius_km:g} km")]
    if domains:
        label = ("WoFS domain" if len(domains) == 1
                 else f"WoFS domains ({len(domains)})")
        handles.append(Line2D([], [], color="#cc79a7", linewidth=1.6,
                              linestyle="--", label=label))
    handles += [Line2D([], [], color=STATUS_COLORS.get(s, "#888888"),
                       linewidth=2.2,
                       label=STATUS_LABELS.get(s, s))
                for s in seen]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    lines = [title or "verification grid"]
    if track:
        lines.append(f"track {track[0][0]:%Y-%m-%d %HZ} to "
                     f"{track[-1][0]:%Y-%m-%d %HZ}  ({len(track)} fixes)")
    sub = (f"rule: {spec.rule}   pad {cfg.pad_km:g} km   "
           f"{spec.n_cells:,} cells")
    if report:
        sub += f"   spacing within {report['worst_pct']:.2f}%"
    lines.append(sub)
    ax.set_title("\n".join(lines), fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_grid_case(case):
    """Driver for the `build-grid` command: build, report, write, plot."""
    from best_track import parse_bdeck_status

    cfg = case.grid or GridConfig()
    full = parse_bdeck_status(case.best_track)
    track = trim_track(full, case.valid_start, case.valid_end, cfg.margin_h)

    print(f"Case   : {case.storm_name}  ({case.case_slug})")
    print(f"Window : {case.valid_start:%Y-%m-%d %HZ} -> "
          f"{case.valid_end:%Y-%m-%d %HZ}  (+/-{cfg.margin_h}h margin)")
    if case.init_start is not None:
        print(f"Cycles : {case.init_start:%Y-%m-%d %HZ} -> "
              f"{case.init_end:%Y-%m-%d %HZ}  (pulled, not scored)")
    print(f"B-deck : {len(full)} fixes, {full[0][0]:%Y-%m-%d %HZ} -> "
          f"{full[-1][0]:%Y-%m-%d %HZ}")
    if not track:
        print("  WARNING: no track points inside the window; falling back to "
              "the WoFS box if one is configured")
    else:
        print(f"Trimmed: {len(track)} fixes, {track[0][0]:%Y-%m-%d %HZ} -> "
              f"{track[-1][0]:%Y-%m-%d %HZ}  "
              f"statuses {','.join(dict.fromkeys(p[3] for p in track))}")

    domains = case.wofs_domains
    if domains:
        print(f"WoFS   : {len(domains)} deployment(s): "
              + "; ".join(d.label for d in domains))
    spec = build_grid(track, cfg, wofs_domains=domains)
    lat, lon = grid_latlon(spec)
    report = spacing_report(spec, lat, lon)

    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km "
          f"({spec.n_cells:,} cells)  rule={spec.rule}")
    print(f"Lambert: lat_0={spec.lat_0} lon_0={spec.lon_0} "
          f"parallels=({spec.lat_1}, {spec.lat_2})")
    print(f"Extent : {lat.min():.2f} to {lat.max():.2f} N, "
          f"{lon.min():.2f} to {lon.max():.2f} E")
    print(f"Spacing: dx {report['dx_min']:.3f}-{report['dx_max']:.3f} km, "
          f"dy {report['dy_min']:.3f}-{report['dy_max']:.3f} km  "
          f"(worst {report['worst_pct']:.2f}% from {spec.res_km:g} km)")
    print(f"MET    : {spec.met_spec}")

    out_dir = Path(case.out_dir)
    payload = grid_to_dict(spec, report, extra={
        "case": case.case_slug,
        "storm_name": case.storm_name,
        "best_track": str(case.best_track),
        "valid_start": f"{case.valid_start:%Y%m%d%H}",
        "valid_end": f"{case.valid_end:%Y%m%d%H}",
        "init_start": (f"{case.init_start:%Y%m%d%H}"
                       if case.init_start else None),
        "init_end": f"{case.init_end:%Y%m%d%H}" if case.init_end else None,
        "margin_h": cfg.margin_h,
        "pad_km": cfg.pad_km,
        "mask_radius_km": cfg.mask_radius_km,
        "wofs_domains": [
            {"name": d.name, "domain": list(d.domain),
             "valid_start": (f"{d.valid_start:%Y%m%d%H}"
                             if d.valid_start else None),
             "valid_end": f"{d.valid_end:%Y%m%d%H}" if d.valid_end else None}
            for d in domains],
        "track_fixes_in_window": len(track),
    })
    json_path = write_grid_json(out_dir / f"{case.case_slug}_grid.json",
                               payload)
    png_path = plot_grid(spec, track, out_dir / f"{case.case_slug}_grid.png",
                         cfg, wofs_domains=domains,
                         title=f"{case.storm_name} — verification grid",
                         report=report)
    print(f"Wrote  : {json_path}")
    print(f"Wrote  : {png_path}")
    return spec
