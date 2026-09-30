"""Masking regions for grid_stat, built on a case's verification grid.

C1, the observation-validity mask. Stage IV is a CONUS product: it covers US
land and thins out offshore, so a verification region that reaches past its
coverage counts cells where there is no truth. Those cells are missing rather
than zero, which quietly removes them from some statistics and not others.
This module draws the region where Stage IV can be trusted -- US land, plus a
fixed distance out over water -- so every other masking region can be
intersected with it.

The construction:

  1. Rasterise US land onto the grid from Natural Earth state polygons
     (the `_lakes` variant, so the Great Lakes are not land).
  2. Rasterise all land, to tell water from foreign land.
  3. Distance-transform the US-land raster, in cells, and scale by the grid
     spacing. This is only valid because the verification grid is Lambert at
     a near-constant km spacing -- the same property the FSS neighbourhoods
     rely on. On a lat/lon grid the distance would be wrong away from the
     reference latitude.
  4. mask = US land, OR (within buffer_km of US land AND over water).

Point 4 is why nothing here is coast-specific: the mask is distance-to-land,
not distance-to-a-named-coastline, so the Atlantic, Gulf and Pacific coasts
are all handled by the same transform, and an East Pacific case needs no
different treatment. Excluding foreign land matters for the Gulf and southwest
cases, where a 150 km buffer around US land would otherwise sweep in parts of
Mexico, Cuba and the Bahamas that Stage IV never observes.
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
                     or rec.attributes.get("adm0_a3", "")) in
                 ("United States of America", "USA")]
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
    water = ~all_land
    # Distance in cells to the nearest US-land cell, scaled to km. Valid
    # because the grid spacing is near-constant by construction.
    dist_km = distance_transform_edt(~us_land) * spec.res_km
    within = dist_km <= buffer_km
    mask = us_land | (within & water)
    detail = {
        "us_land": us_land,
        "all_land": all_land,
        "dist_km": dist_km,
        "n_total": int(mask.size),
        "n_mask": int(mask.sum()),
        "n_us_land": int(us_land.sum()),
        "n_offshore": int((within & water).sum()),
        "n_foreign_land_excluded": int((within & all_land & ~us_land).sum()),
    }
    return mask, detail


def write_mask_netcdf(path, spec, mask, lat, lon, name=MASK_NAME,
                      buffer_km=DEFAULT_BUFFER_KM):
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
        var.description = (f"1 where Stage IV coverage is trusted: US land, "
                           f"or water within {buffer_km:g} km of US land")
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

    # 0 outside, 1 offshore buffer, 2 US land, 3 foreign land excluded
    code = np.zeros(mask.shape)
    code[mask] = 1.0
    code[detail["us_land"]] = 2.0
    excluded = detail["all_land"] & ~detail["us_land"]
    code[excluded & (detail["dist_km"] <= buffer_km)] = 3.0
    # Okabe-Ito; no red/green pairing
    cmap = ListedColormap(["#f2f2f2", "#56b4e9", "#0072b2", "#e69f00"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

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
        Patch(facecolor="#e69f00",
              label=f"foreign land excluded "
                    f"({detail['n_foreign_land_excluded']:,})"),
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


def build_masks_case(case):
    """Driver for `build-masks`: the C1 observation-validity mask."""
    from best_track import parse_bdeck_status

    grid_cfg = case.grid or vg.GridConfig()
    cfg = case.masks or MaskConfig()
    spec_path = Path(case.out_dir) / f"{case.case_slug}_grid.json"
    if not spec_path.exists():
        raise SystemExit(
            f"ERROR: no grid for this case ({spec_path}). Run build-grid "
            f"first:\n  python analysis/run.py <yaml> build-grid")

    import json
    payload = json.loads(spec_path.read_text())
    spec = vg.GridSpec(
        name=payload.get("name", "grid"), nx=int(payload["nx"]),
        ny=int(payload["ny"]), res_km=float(payload["res_km"]),
        lat_0=float(payload["lat_0"]), lon_0=float(payload["lon_0"]),
        lat_1=float(payload["lat_1"]), lat_2=float(payload["lat_2"]),
        lat_ll=float(payload["lat_ll"]), lon_ll=float(payload["lon_ll"]),
        x_ll_km=float(payload["x_ll_km"]), y_ll_km=float(payload["y_ll_km"]),
        rule=payload.get("rule", ""))

    print(f"Case   : {case.storm_name}  ({case.case_slug})")
    print(f"Grid   : {spec.nx} x {spec.ny} @ {spec.res_km:g} km  "
          f"({spec.n_cells:,} cells)  from {spec_path.name}")
    print(f"Buffer : {cfg.coast_buffer_km:g} km offshore of US land  "
          f"(Natural Earth {cfg.ne_scale})", flush=True)

    lat, lon = vg.grid_latlon(spec)
    mask, detail = coastal_valid_mask(spec, cfg.coast_buffer_km,
                                      cfg.ne_scale, lat, lon)
    pct = 100.0 * detail["n_mask"] / detail["n_total"]
    print(f"Mask   : {detail['n_mask']:,} of {detail['n_total']:,} cells "
          f"({pct:.1f}%)")
    print(f"  US land            {detail['n_us_land']:,}")
    print(f"  offshore buffer    {detail['n_offshore']:,}")
    print(f"  foreign land cut   {detail['n_foreign_land_excluded']:,}")
    if detail["n_mask"] == 0:
        print("  WARNING: the mask is empty -- no US land within the grid?")

    out_dir = Path(cfg.out_dir or case.out_dir)
    nc_path = write_mask_netcdf(
        out_dir / f"{case.case_slug}_mask_coastal.nc", spec, mask, lat, lon,
        buffer_km=cfg.coast_buffer_km)
    track = []
    if case.best_track and Path(case.best_track).exists():
        track = vg.trim_track(parse_bdeck_status(case.best_track),
                              case.valid_start, case.valid_end,
                              grid_cfg.margin_h)
    png_path = plot_mask(
        spec, mask, detail, lat, lon,
        out_dir / f"{case.case_slug}_mask_coastal.png", track=track,
        buffer_km=cfg.coast_buffer_km, wofs_domains=case.wofs_domains,
        title=f"{case.storm_name} — Stage IV validity mask")
    print(f"Wrote  : {nc_path}")
    print(f"Wrote  : {png_path}")
    return mask
