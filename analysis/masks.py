"""The observation-validity mask for a deployment's verification grid.

Stage IV is a CONUS product: it covers US land and thins out offshore, so a
verification region that reaches past its coverage counts cells where there is
no truth. Those cells are missing rather than zero, which quietly removes them
from some statistics and not others. This module draws the region where the
truth product can be trusted -- US land, plus a fixed distance beyond it in
every direction -- and writes it as a MET-readable NetCDF on the deployment
grid. The verification region is the whole WoFS domain intersected with this
mask; there is no track swath and no WoFS-box mask, since the grid *is* the
WoFS domain.

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

The rule is for Stage IV, and MRMS has the same CONUS footprint, so one mask
serves both truth products for now. If MRMS needs a different region, the rule
becomes a function of `truth_source` here without touching any caller.

Usage:
    python analysis/run.py storms/<case>.yaml build-mask
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import verification_grid as vg

DEFAULT_BUFFER_KM = 150.0
DEFAULT_NE_SCALE = "50m"
MASK_NAME = "obs_valid"
MET_MISSING = -9999.0

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


def obs_valid_mask(spec, buffer_km=DEFAULT_BUFFER_KM, scale=DEFAULT_NE_SCALE,
                   lat=None, lon=None, us_land=None, all_land=None):
    """(mask, detail) for the observation-validity region on `spec`.

    `mask` is boolean, True where observations can be trusted. `detail`
    carries the intermediate rasters and the cell counts, for the plot and
    the log. `us_land` and `all_land` are boolean rasters on the grid; they
    are read from Natural Earth when not given, and exist as arguments so the
    construction can be tested without the shapefiles.
    """
    from scipy.ndimage import distance_transform_edt
    if lat is None or lon is None:
        lat, lon = vg.grid_latlon(spec)
    if us_land is None:
        us_land = _rasterise(_land_geometry(scale, us_only=True), lat, lon)
    if all_land is None:
        all_land = _rasterise(_land_geometry(scale, us_only=False), lat, lon)
    all_land = all_land | us_land   # the two layers can disagree by a cell
    attached = _attached_land(all_land, us_land)
    detached = all_land & ~attached
    # Distance in cells to the nearest US-land cell, scaled to km. Valid
    # because the grid spacing is near-constant by construction. With no US
    # land on the grid at all there is nothing to measure from, and the mask
    # is empty rather than everywhere.
    if us_land.any():
        dist_km = distance_transform_edt(~us_land) * spec.res_km
    else:
        dist_km = np.full(us_land.shape, np.inf)
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
            f"1 where Stage IV / MRMS coverage is trusted: within "
            f"{buffer_km:g} km of US land, excluding land masses not "
            f"connected to the US")
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


def read_mask(path, name=MASK_NAME):
    """Boolean (ny, nx) array from a mask NetCDF written by this module."""
    import met_regrid
    _, _, values = met_regrid.read_regridded(path, name=name)
    return np.asarray(values) > 0.5


def plot_mask(spec, mask, detail, lat, lon, out_path, track=None,
              buffer_km=DEFAULT_BUFFER_KM, title=None):
    """Map of the validity mask: kept cells, excluded land, grid, track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from plot_common import add_us_geography

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

    olat, olon = vg._grid_outline(lat, lon)
    fig = plt.figure(figsize=(11.0, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    margin = max(1.5, 0.04 * (lat.max() - lat.min()))
    ax.set_extent([olon.min() - margin, olon.max() + margin,
                   max(olat.min() - margin, -85.0),
                   min(olat.max() + margin, 85.0)], crs=plain)
    ax.pcolormesh(lon, lat, code, transform=plain, cmap=cmap, norm=norm,
                  shading="nearest", zorder=1)
    add_us_geography(ax)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb", alpha=0.7,
                 linestyle=":")
    ax.plot(olon, olat, transform=plain, color="#444444", linewidth=2.0,
            zorder=5)
    for (t0, la0, lo0, st0), (t1, la1, lo1, _) in zip(track or [],
                                                      (track or [])[1:]):
        ax.plot([lo0, lo1], [la0, la1], transform=plain, linewidth=2.0,
                color=vg.STATUS_COLORS.get(st0, "#888888"), zorder=6)

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
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{title or 'observation validity mask'}\n"
                 f"{detail['n_mask']:,} of {detail['n_total']:,} cells "
                 f"({pct:.1f}%) inside the mask", fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_mask_case(case):
    """Driver for `build-mask`: the validity mask for this deployment."""
    from best_track import parse_bdeck_status
    if not case.wofs_available:
        print(f"Case   : {case.storm_name} {case.deployment}  "
              f"({case.case_slug})\nNo complete WoFS window; no grid, so no "
              "mask for this deployment")
        return None
    try:
        spec = vg.grid_spec_from_json(case.grid_json_path)
    except FileNotFoundError as err:
        raise SystemExit(f"ERROR: {err}") from None
    settings = case.masks
    lat, lon = vg.grid_latlon(spec)

    print(f"Case   : {case.storm_name} {case.deployment}  ({case.case_slug})")
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km  "
          f"({spec.n_cells:,} cells)  from {case.grid_json_path.name}")
    print(f"Rule   : within {settings.coast_buffer_km:g} km of US land, "
          f"detached land cut  (Natural Earth {settings.ne_scale})",
          flush=True)

    mask, detail = obs_valid_mask(spec, settings.coast_buffer_km,
                                  settings.ne_scale, lat, lon)
    pct = 100.0 * detail["n_mask"] / detail["n_total"]
    print(f"\n{MASK_NAME}: {detail['n_mask']:,} of {detail['n_total']:,} "
          f"cells ({pct:.1f}%)")
    print(f"  US land            {detail['n_us_land']:,}")
    print(f"  water buffer       {detail['n_offshore']:,}")
    print(f"  foreign land       {detail['n_foreign_land']:,}")
    print(f"  detached land cut  {detail['n_detached_excluded']:,}")
    if detail["n_mask"] == 0:
        print("  WARNING: the mask is empty -- no US land on this grid, so "
              "every statistic restricted to it would have no cells.")

    track = []
    if case.best_track and Path(case.best_track).exists():
        track = [p for p in parse_bdeck_status(case.best_track)
                 if case.valid_start <= p[0] <= case.valid_end]
    nc_path = write_mask_netcdf(case.mask_path, spec, mask, lat, lon,
                                buffer_km=settings.coast_buffer_km)
    png_path = plot_mask(
        spec, mask, detail, lat, lon, case.mask_plot_path, track=track,
        buffer_km=settings.coast_buffer_km,
        title=f"{case.storm_name} {case.deployment}: observation validity")
    print(f"\nWrote  : {nc_path}")
    print(f"Wrote  : {png_path}")
    return mask
