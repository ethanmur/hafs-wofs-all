"""Masking regions for grid_stat, built on a case's verification grid.

Phase C. Four regions, all rasterised onto the case grid and written as
MET-readable NetCDF so grid_stat can take them as `mask.poly`:

  C1  coastal_valid   where Stage IV has truth (below)
  C2  track_swath     within mask_radius_km of the best track
  C3  WOFS_<n>        one box per WoFS deployment, plus their union WOFS_ANY
  C4  verify          the intersection of the three, which is what statistics
                      are actually accumulated over

C2-C4 are computed here rather than with `gen_vx_mask` for the reason C1
already established: the regions have to live on the same grid and be
intersected with each other, and doing that in one array pass is simpler than
three subprocess calls plus intermediate files. The NetCDF output feeds
grid_stat identically either way.

C1, the observation-validity mask. Stage IV is a CONUS product: it covers US
land and thins out offshore, so a verification region that reaches past its
coverage counts cells where there is no truth. Those cells are missing rather
than zero, which quietly removes them from some statistics and not others.
This module draws the region where Stage IV can be trusted -- US land, plus a
fixed distance beyond it in every direction -- so every other masking region
can be intersected with it.

The construction:

  1. Rasterise US land onto the grid from Natural Earth state polygons
     (the `_lakes` variant, so the Great Lakes are not land).
  2. Rasterise all land, and keep only the land masses connected to US land.
     A land mass that touches the US (Canada, Mexico) is treated exactly like
     open water: the buffer reaches into it and stops. One that does not
     (Cuba, Hispaniola, the Bahamas) is cut, however close it lies.
  3. Distance-transform the US-land raster, in cells, and scale by the grid
     spacing. This is only valid because the verification grid is Lambert at
     a near-constant km spacing -- the same property the FSS neighbourhoods
     rely on. On a lat/lon grid the distance would be wrong away from the
     reference latitude.
  4. mask = (within buffer_km of US land) AND NOT on a detached land mass.

So the US land border is the only thing the buffer measures from, and there is
one rule for water and for foreign soil alike. Radar range does not stop at a
political line, and cutting the region there splits contiguous rainfall in the
Texas and Pacific Northwest cases -- but coverage does fade inland, so the
buffer penetrates Canada and Mexico rather than swallowing them whole.

Step 2 is a connectivity test, not a country list. That is deliberate: it
keeps Cuba and the Bahamas out for Gulf and Florida cases, where parts of them
sit inside a 150 km buffer yet lie beyond the US radar network, without
needing a maintained list of which nations to name. It also keeps offshore US
land -- Long Island, the Keys, the barrier islands -- since those land masses
contain US cells of their own.

Point 4 is why nothing here is coast-specific: the mask is distance-to-land,
not distance-to-a-named-coastline, so the Atlantic, Gulf and Pacific coasts
are all handled by the same transform, and an East Pacific case needs no
different treatment.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import verification_grid as vg

DEFAULT_BUFFER_KM = 150.0
DEFAULT_NE_SCALE = "50m"
MASK_NAME = "coastal_valid"
SWATH_NAME = "track_swath"
WOFS_ANY_NAME = "WOFS_ANY"
VERIFY_NAME = "verify"
# Okabe-Ito, one per region, used by every mask figure.
REGION_COLORS = {MASK_NAME: "#0072b2", SWATH_NAME: "#009e73",
                 WOFS_ANY_NAME: "#cc79a7", VERIFY_NAME: "#e69f00"}
MET_MISSING = -9999.0


@dataclass
class MaskConfig:
    """Configuration for the observation-validity mask."""
    coast_buffer_km: float = DEFAULT_BUFFER_KM
    ne_scale: str = DEFAULT_NE_SCALE
    out_dir: Optional[Path] = None      # default: the case's out_dir


def mask_config_from_dict(cfg):
    cfg = cfg or {}
    unknown = set(cfg) - {"coast_buffer_km", "ne_scale", "out_dir"}
    if unknown:
        raise ValueError("unknown masks keys: " + ", ".join(sorted(unknown)))
    buffer_km = float(cfg.get("coast_buffer_km", DEFAULT_BUFFER_KM))
    if buffer_km < 0:
        raise ValueError("masks.coast_buffer_km cannot be negative")
    return MaskConfig(
        coast_buffer_km=buffer_km,
        ne_scale=str(cfg.get("ne_scale", DEFAULT_NE_SCALE)),
        out_dir=Path(cfg["out_dir"]) if cfg.get("out_dir") else None)


US_ADMINS = ("United States of America", "USA")


def _land_geometry(scale, us_only):
    """Unioned land polygons: US states only, or every land mass."""
    import shapely
    from cartopy.io import shapereader
    if us_only:
        path = shapereader.natural_earth(
            resolution=scale, category="cultural",
            name="admin_1_states_provinces_lakes")
        geoms = [rec.geometry for rec in shapereader.Reader(path).records()
                 if (rec.attributes.get("admin")
                     or rec.attributes.get("adm0_a3", "")) in US_ADMINS]
    else:
        path = shapereader.natural_earth(resolution=scale,
                                         category="physical", name="land")
        geoms = list(shapereader.Reader(path).geometries())
    if not geoms:
        raise RuntimeError(
            f"no {'US state' if us_only else 'land'} polygons found in the "
            f"Natural Earth {scale} data")
    return shapely.union_all(shapely.GeometryCollection(geoms))


def _rasterise(geom, lat, lon):
    """Boolean raster of `geom` on the grid's lat/lon points."""
    import shapely
    return np.asarray(
        shapely.contains_xy(geom, lon.ravel(), lat.ravel())
    ).reshape(lat.shape)


def _attached_land(all_land, us_land):
    """Land masses that touch US land, as a boolean raster.

    scipy labels 8-connected components of the land raster; a component is
    kept when it holds at least one US-land cell. Canada and Mexico share a
    component with the lower 48, so they are kept (and then buffered); Cuba
    and the Bahamas form their own and are dropped.
    """
    from scipy.ndimage import label
    structure = np.ones((3, 3), dtype=bool)
    labels, n = label(all_land, structure=structure)
    if n == 0:
        return np.zeros_like(all_land)
    touching = np.unique(labels[us_land & (labels > 0)])
    return np.isin(labels, touching)


def coastal_valid_mask(spec, buffer_km=DEFAULT_BUFFER_KM,
                       scale=DEFAULT_NE_SCALE, lat=None, lon=None):
    """(mask, detail) for the observation-validity region on `spec`.

    `mask` is boolean, True where observations can be trusted. `detail`
    carries the intermediate rasters and the cell counts, for the plot and
    the log.
    """
    from scipy.ndimage import distance_transform_edt
    if lat is None or lon is None:
        lat, lon = vg.grid_latlon(spec)
    us_land = _rasterise(_land_geometry(scale, us_only=True), lat, lon)
    all_land = _rasterise(_land_geometry(scale, us_only=False), lat, lon)
    all_land |= us_land        # the two layers can disagree by a cell at the coast
    attached = _attached_land(all_land, us_land)
    detached = all_land & ~attached
    # Distance in cells to the nearest US-land cell, scaled to km. Valid
    # because the grid spacing is near-constant by construction.
    dist_km = distance_transform_edt(~us_land) * spec.res_km
    within = dist_km <= buffer_km
    # One rule for water and for foreign soil: within range of US land, and
    # not on a land mass the US radar network cannot reach across.
    mask = within & ~detached
    detail = {
        "us_land": us_land,
        "all_land": all_land,
        "dist_km": dist_km,
        "n_total": int(mask.size),
        "n_mask": int(mask.sum()),
        "detached": detached,
        "n_us_land": int(us_land.sum()),
        "n_offshore": int((mask & ~all_land).sum()),
        "n_foreign_land": int((mask & attached & ~us_land).sum()),
        "n_detached_excluded": int((within & detached).sum()),
    }
    return mask, detail


# =============================================================================
# C2, C3, C4
# =============================================================================

def grid_xy_km(spec):
    """(x, y) 2-D projected coordinates of the grid points, in km.

    Read straight off the grid definition rather than transformed back from
    lat/lon, so distances in this plane are exactly the ones the grid was
    built on.
    """
    x = spec.x_ll_km + np.arange(spec.nx) * spec.res_km
    y = spec.y_ll_km + np.arange(spec.ny) * spec.res_km
    return np.meshgrid(x, y)


def track_swath_mask(spec, track, radius_km):
    """C2: cells within `radius_km` of the best-track line.

    The track is projected into the grid's own Lambert plane and buffered
    there, so the swath is a true constant-radius corridor -- the same reason
    the grid is Lambert in the first place. Buffering in degrees would be
    narrower in longitude the further north the storm goes.

    A swath is drawn through the track as a line, not as a series of discs,
    so a fast-moving storm with 6-hourly fixes gets a continuous corridor
    rather than a string of beads.
    """
    import shapely
    from shapely.geometry import LineString, Point
    if not track:
        return np.zeros((spec.ny, spec.nx), dtype=bool)
    fwd = vg._transformer({"lat_0": spec.lat_0, "lon_0": spec.lon_0,
                           "lat_1": spec.lat_1, "lat_2": spec.lat_2})
    xs, ys = fwd.transform([p[2] for p in track], [p[1] for p in track])
    pts = list(zip(np.asarray(xs) / 1000.0, np.asarray(ys) / 1000.0))
    geom = LineString(pts) if len(pts) > 1 else Point(pts[0])
    swath = geom.buffer(radius_km)
    gx, gy = grid_xy_km(spec)
    return np.asarray(
        shapely.contains_xy(swath, gx.ravel(), gy.ravel())
    ).reshape(gx.shape)


def wofs_box_mask(domain, lat, lon):
    """C3: cells inside one WoFS deployment's lat/lon box.

    The box is applied in lat/lon because that is how the deployment was
    estimated from the WoFS plots; its edges are not grid lines, so the
    rasterised box is a staircase at the 6 km scale. That is the right
    behaviour -- a cell is in or out, and no cell is half-counted.
    """
    lat0, lat1, lon0, lon1 = domain
    return ((lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1))


def build_regions(spec, track, cfg, grid_cfg, wofs_domains, lat, lon):
    """Every Phase C region on this grid: {name: bool array}, plus detail.

    The returned dict is ordered coastal, swath, each WoFS box, WOFS_ANY,
    verify. `verify` is C4: the intersection that statistics are accumulated
    over. A region that does not apply to this case (no WoFS deployments) is
    simply absent, and the intersection skips it rather than emptying.
    """
    regions = {}
    coastal, detail = coastal_valid_mask(spec, cfg.coast_buffer_km,
                                         cfg.ne_scale, lat, lon)
    regions[MASK_NAME] = coastal
    regions[SWATH_NAME] = track_swath_mask(spec, track,
                                           grid_cfg.mask_radius_km)
    boxes = {}
    for i, dom in enumerate(wofs_domains or [], 1):
        name = f"WOFS_{i}"
        boxes[name] = wofs_box_mask(dom.domain, lat, lon)
        regions[name] = boxes[name]
    if boxes:
        regions[WOFS_ANY_NAME] = np.logical_or.reduce(list(boxes.values()))
    # C4: what statistics are actually accumulated over. WOFS_ANY is the union
    # of the deployments, so this is the all-hours region; a per-hour mask uses
    # only the deployments live at that valid time (see active_regions).
    verify = regions[MASK_NAME] & regions[SWATH_NAME]
    if WOFS_ANY_NAME in regions:
        verify = verify & regions[WOFS_ANY_NAME]
    regions[VERIFY_NAME] = verify
    return regions, detail


def active_verify_mask(regions, wofs_domains, when):
    """The C4 intersection using only the WoFS boxes live at `when`.

    Scoring an hour against a box that was deployed on another day would
    credit or penalise a forecast in a region WoFS never covered, so the
    intersection is rebuilt per accumulation window. With no deployment live,
    the WoFS term drops out rather than emptying the region: the HAFS-vs-obs
    comparison is still valid there, it is only the three-way one that is not.
    """
    mask = regions[MASK_NAME] & regions[SWATH_NAME]
    live = [f"WOFS_{i}" for i, dom in enumerate(wofs_domains or [], 1)
            if dom.covers(when)]
    if live:
        mask = mask & np.logical_or.reduce([regions[n] for n in live])
    return mask, live


def write_mask_netcdf(path, spec, mask, lat, lon, name=MASK_NAME,
                      buffer_km=DEFAULT_BUFFER_KM, description=None):
    """Write the mask as a MET-style gridded NetCDF.

    The variable layout matches regrid_data_plane's output, which
    met_regrid.read_regridded already reads. The global attributes describe
    the Lambert grid in MET's NetCDF convention so that grid_stat can take
    the file as a masking region.
    """
    import netCDF4
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.where(mask, 1.0, 0.0)
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("lat", spec.ny)
        nc.createDimension("lon", spec.nx)
        # 2-D lat/lon named after their own dimensions, as MET writes them
        nc.createVariable("lat", "f4", ("lat", "lon"))[:] = lat
        nc.createVariable("lon", "f4", ("lat", "lon"))[:] = lon
        var = nc.createVariable(name, "f4", ("lat", "lon"),
                                fill_value=MET_MISSING)
        var[:] = values
        var.long_name = "observation validity mask"
        var.units = "none"
        var.description = description or (
            f"1 where Stage IV coverage is trusted: within {buffer_km:g} km "
            f"of US land, excluding land masses not connected to the US")
        # MET NetCDF grid definition (Lambert conformal).
        nc.Projection = "Lambert Conformal"
        nc.scale_lat_1 = f"{spec.lat_1:.6f} degrees_north"
        nc.scale_lat_2 = f"{spec.lat_2:.6f} degrees_north"
        nc.lat_pin = f"{spec.lat_ll:.6f} degrees_north"
        nc.lon_pin = f"{spec.lon_ll:.6f} degrees_east"
        nc.x_pin = "0.000000"
        nc.y_pin = "0.000000"
        nc.lon_orient = f"{spec.lon_0:.6f} degrees_east"
        nc.d_km = f"{spec.res_km:.6f} km"
        nc.r_km = f"{vg.R_EARTH_KM:.6f} km"
        nc.nx = f"{spec.nx}"
        nc.ny = f"{spec.ny}"
        nc.mask_name = name
        nc.coast_buffer_km = f"{buffer_km:g}"
    return path


def plot_mask(spec, mask, detail, lat, lon, out_path, track=None,
              buffer_km=DEFAULT_BUFFER_KM, title=None, wofs_domains=None):
    """Map of the validity mask: kept cells, excluded land, grid, track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from compare import _add_us_geography

    # 0 outside, 1 buffer over water, 2 US land, 3 buffer over foreign land,
    # 4 detached land mass cut out of the buffer
    code = np.zeros(mask.shape)
    code[mask] = 1.0
    code[mask & detail["all_land"]] = 3.0
    code[detail["us_land"]] = 2.0
    code[detail["detached"] & (detail["dist_km"] <= buffer_km)] = 4.0
    # Okabe-Ito; no red/green pairing
    cmap = ListedColormap(["#f2f2f2", "#56b4e9", "#0072b2", "#009e73",
                           "#e69f00"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    globe = ccrs.Globe(ellipse=None, semimajor_axis=vg.R_EARTH_KM * 1000.0,
                       semiminor_axis=vg.R_EARTH_KM * 1000.0)
    proj = ccrs.LambertConformal(
        central_longitude=spec.lon_0, central_latitude=spec.lat_0,
        standard_parallels=(spec.lat_1, spec.lat_2), globe=globe)
    plain = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11.0, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    margin = max(1.5, 0.04 * (lat.max() - lat.min()))
    ax.set_extent([lon.min() - margin, lon.max() + margin,
                   max(lat.min() - margin, -85.0),
                   min(lat.max() + margin, 85.0)], crs=plain)
    ax.pcolormesh(lon, lat, code, transform=plain, cmap=cmap, norm=norm,
                  shading="nearest", zorder=1)
    _add_us_geography(ax)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb", alpha=0.7,
                 linestyle=":")

    olat, olon = vg._grid_outline(lat, lon)
    ax.plot(olon, olat, transform=plain, color="#444444", linewidth=2.0,
            zorder=5)
    if track:
        for (t0, la0, lo0, st0), (t1, la1, lo1, _) in zip(track, track[1:]):
            ax.plot([lo0, lo1], [la0, la1], transform=plain, linewidth=2.0,
                    color=vg.STATUS_COLORS.get(st0, "#888888"), zorder=6)
    for dom in (wofs_domains or []):
        wlat0, wlat1, wlon0, wlon1 = dom.domain
        ax.add_patch(Rectangle(
            (wlon0, wlat0), wlon1 - wlon0, wlat1 - wlat0, transform=plain,
            facecolor="none", edgecolor="#cc79a7", linewidth=1.5,
            linestyle="--", zorder=8))

    pct = 100.0 * detail["n_mask"] / detail["n_total"]
    handles = [
        Patch(facecolor="#0072b2", label=f"US land ({detail['n_us_land']:,})"),
        Patch(facecolor="#56b4e9",
              label=f"water within {buffer_km:g} km "
                    f"({detail['n_offshore']:,})"),
        Patch(facecolor="#009e73",
              label=f"foreign land within {buffer_km:g} km "
                    f"({detail['n_foreign_land']:,})"),
        Patch(facecolor="#e69f00",
              label=f"detached land mass cut "
                    f"({detail['n_detached_excluded']:,})"),
        Patch(facecolor="#f2f2f2", edgecolor="#999999",
              label="outside the mask"),
        Line2D([], [], color="#444444", linewidth=2.0,
               label=f"grid {spec.nx}x{spec.ny} @ {spec.res_km:g} km")]
    if wofs_domains:
        handles.append(Line2D([], [], color="#cc79a7", linewidth=1.5,
                              linestyle="--", label="WoFS domain"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{title or 'observation validity mask'}\n"
                 f"{detail['n_mask']:,} of {detail['n_total']:,} cells "
                 f"({pct:.1f}%) inside the mask", fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_regions(spec, regions, lat, lon, out_path, track=None,
                 wofs_domains=None, title=None):
    """Phase C summary map: each region's boundary, the intersection filled.

    Boundaries rather than filled patches, because three filled regions over
    each other hide exactly the thing being checked -- where they disagree.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from compare import _add_us_geography

    globe = ccrs.Globe(ellipse=None, semimajor_axis=vg.R_EARTH_KM * 1000.0,
                       semiminor_axis=vg.R_EARTH_KM * 1000.0)
    proj = ccrs.LambertConformal(
        central_longitude=spec.lon_0, central_latitude=spec.lat_0,
        standard_parallels=(spec.lat_1, spec.lat_2), globe=globe)
    plain = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11.0, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    margin = max(1.5, 0.04 * (lat.max() - lat.min()))
    ax.set_extent([lon.min() - margin, lon.max() + margin,
                   max(lat.min() - margin, -85.0),
                   min(lat.max() + margin, 85.0)], crs=plain)

    verify = regions[VERIFY_NAME]
    ax.pcolormesh(lon, lat, np.where(verify, 1.0, np.nan), transform=plain,
                  cmap=matplotlib.colors.ListedColormap(["#e69f00"]),
                  shading="nearest", alpha=0.35, zorder=1)
    _add_us_geography(ax)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb", alpha=0.7,
                 linestyle=":")

    handles = []
    for name in (MASK_NAME, SWATH_NAME, WOFS_ANY_NAME):
        if name not in regions:
            continue
        colour = REGION_COLORS[name]
        ax.contour(lon, lat, regions[name].astype(float), levels=[0.5],
                   colors=[colour], linewidths=1.8, transform=plain, zorder=4)
        handles.append(Line2D([], [], color=colour, linewidth=1.8,
                              label=f"{name} ({int(regions[name].sum()):,})"))
    handles.append(Patch(facecolor="#e69f00", alpha=0.35,
                         label=f"{VERIFY_NAME} = intersection "
                               f"({int(verify.sum()):,})"))

    olat, olon = vg._grid_outline(lat, lon)
    ax.plot(olon, olat, transform=plain, color="#444444", linewidth=2.0,
            zorder=5)
    handles.append(Line2D([], [], color="#444444", linewidth=2.0,
                          label=f"grid {spec.nx}x{spec.ny} @ "
                                f"{spec.res_km:g} km"))
    for (t0, la0, lo0, st0), (t1, la1, lo1, _) in zip(track or [],
                                                      (track or [])[1:]):
        ax.plot([lo0, lo1], [la0, la1], transform=plain, linewidth=2.0,
                color=vg.STATUS_COLORS.get(st0, "#888888"), zorder=6)
    for dom in (wofs_domains or []):
        ax.text(dom.domain[2], dom.domain[1], f" {dom.label}", transform=plain,
                fontsize=7.5, color=REGION_COLORS[WOFS_ANY_NAME], va="bottom",
                ha="left", zorder=9,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.75))

    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    pct = 100.0 * verify.sum() / verify.size
    ax.set_title(f"{title or 'verification regions'}\n"
                 f"{int(verify.sum()):,} of {verify.size:,} cells ({pct:.1f}%) "
                 f"scored", fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_grid_spec(case):
    """The GridSpec written by build-grid for this case."""
    import json
    spec_path = Path(case.out_dir) / f"{case.case_slug}_grid.json"
    if not spec_path.exists():
        raise SystemExit(
            f"ERROR: no grid for this case ({spec_path}). Run build-grid "
            f"first:\n  python analysis/run.py <yaml> build-grid")
    payload = json.loads(spec_path.read_text())
    spec = vg.GridSpec(
        name=payload.get("name", "grid"), nx=int(payload["nx"]),
        ny=int(payload["ny"]), res_km=float(payload["res_km"]),
        lat_0=float(payload["lat_0"]), lon_0=float(payload["lon_0"]),
        lat_1=float(payload["lat_1"]), lat_2=float(payload["lat_2"]),
        lat_ll=float(payload["lat_ll"]), lon_ll=float(payload["lon_ll"]),
        x_ll_km=float(payload["x_ll_km"]), y_ll_km=float(payload["y_ll_km"]),
        rule=payload.get("rule", ""))
    return spec, spec_path


def case_regions(case):
    """(spec, regions, lat, lon, track) for a case -- the shared entry point.

    Rebuilt from the grid JSON and the b-deck rather than read back from the
    NetCDFs, so a plot can never be drawn against a stale mask file.
    """
    from best_track import parse_bdeck_status
    grid_cfg = case.grid or vg.GridConfig()
    cfg = case.masks or MaskConfig()
    spec, _ = load_grid_spec(case)
    lat, lon = vg.grid_latlon(spec)
    track = []
    if case.best_track and Path(case.best_track).exists():
        track = vg.trim_track(parse_bdeck_status(case.best_track),
                              case.valid_start, case.valid_end,
                              grid_cfg.margin_h)
    regions, _ = build_regions(spec, track, cfg, grid_cfg, case.wofs_domains,
                               lat, lon)
    return spec, regions, lat, lon, track


_REGION_DESC = {
    SWATH_NAME: "1 within the track-swath radius of the best track",
    WOFS_ANY_NAME: "1 inside any WoFS deployment box for this case",
    VERIFY_NAME: ("1 where statistics are accumulated: coastal_valid AND "
                  "track_swath AND (any WoFS box, if the case has one)"),
}


def build_masks_case(case):
    """Driver for `build-masks`: every Phase C region for this case."""
    grid_cfg = case.grid or vg.GridConfig()
    cfg = case.masks or MaskConfig()
    spec, spec_path = load_grid_spec(case)

    print(f"Case   : {case.storm_name}  ({case.case_slug})")
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km  "
          f"({spec.n_cells:,} cells)  from {spec_path.name}")
    print(f"Buffer : {cfg.coast_buffer_km:g} km beyond US land  "
          f"(Natural Earth {cfg.ne_scale})")
    print(f"Swath  : {grid_cfg.mask_radius_km:g} km of the best track",
          flush=True)

    from best_track import parse_bdeck_status
    lat, lon = vg.grid_latlon(spec)
    track = []
    if case.best_track and Path(case.best_track).exists():
        track = vg.trim_track(parse_bdeck_status(case.best_track),
                              case.valid_start, case.valid_end,
                              grid_cfg.margin_h)
    if not track:
        print("  WARNING: no track fixes in the window; the swath is empty "
              "and the intersection with it would be too")
    domains = case.wofs_domains
    if domains:
        print(f"WoFS   : {len(domains)} deployment(s): "
              + "; ".join(d.label for d in domains))
    else:
        print("WoFS   : none configured; the intersection is coastal x swath")

    regions, detail = build_regions(spec, track, cfg, grid_cfg, domains,
                                    lat, lon)
    print(f"\nC1 {MASK_NAME}")
    print(f"  US land            {detail['n_us_land']:,}")
    print(f"  water buffer       {detail['n_offshore']:,}")
    print(f"  foreign land       {detail['n_foreign_land']:,}")
    print(f"  detached land cut  {detail['n_detached_excluded']:,}")
    print("\nRegions (cells, % of grid)")
    for name, arr in regions.items():
        n = int(arr.sum())
        print(f"  {name:<14} {n:>10,}  {100.0 * n / arr.size:5.1f}%")
    if regions[VERIFY_NAME].sum() == 0:
        print("  WARNING: the intersection is empty -- nothing would be "
              "scored. Check the WoFS boxes and the valid window.")

    out_dir = Path(cfg.out_dir or case.out_dir)
    written = []
    for name, arr in regions.items():
        suffix = {MASK_NAME: "coastal", SWATH_NAME: "swath",
                  VERIFY_NAME: "verify"}.get(name, name.lower())
        written.append(write_mask_netcdf(
            out_dir / f"{case.case_slug}_mask_{suffix}.nc", spec, arr, lat,
            lon, name=name, buffer_km=cfg.coast_buffer_km,
            description=_REGION_DESC.get(
                name, f"1 inside the {name} verification region")))
    written.append(plot_mask(
        spec, regions[MASK_NAME], detail, lat, lon,
        out_dir / f"{case.case_slug}_mask_coastal.png", track=track,
        buffer_km=cfg.coast_buffer_km, wofs_domains=domains,
        title=f"{case.storm_name} — Stage IV validity mask"))
    written.append(plot_regions(
        spec, regions, lat, lon,
        out_dir / f"{case.case_slug}_mask_regions.png", track=track,
        wofs_domains=domains,
        title=f"{case.storm_name} — verification regions"))
    print()
    for path in written:
        print(f"Wrote  : {path}")
    return regions
