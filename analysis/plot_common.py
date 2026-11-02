"""Plotting pieces shared by the map-producing modules.

The QPF colour ladder and the US-geography overlay were previously buried in
the HAFS-only product modules; they live here so the grid, mask and obs
plotting code does not depend on any product module.
"""

import numpy as np
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plot_units import inches

QPF_LEVELS = [0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
QPF_COLORS = [
    "#ffffff", "#c8f0f0", "#64d2ff", "#3296ff",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
]


def qpf_cmap():
    cmap = mcolors.ListedColormap(QPF_COLORS)
    norm = mcolors.BoundaryNorm(QPF_LEVELS, cmap.N)
    return cmap, norm


# --- precipitation scales for the maps that compare fields -----------------
# "true_scale" is a log scale in inches with the ladder's colours blended into
# a continuous ramp, so light and heavy rain are both legible on one colour
# bar. "standard" is the banded ladder above. Data come in mm and are drawn in
# inches, like the rest of the figures.
COLORBAR_STYLES = ("true_scale", "standard")
DEFAULT_COLORBAR_STYLE = "true_scale"
LOG_FLOOR_IN = 0.01      # the log scale cannot include 0: this is the plot floor
LOG_VMAX_IN = 5.0
LOG_N_LEVELS = 20
LOG_TICK_CANDIDATES = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50]
NODATA_COLOR = "#d0d0d0"


def qpf_continuous_cmap():
    return mcolors.LinearSegmentedColormap.from_list("qpf_continuous",
                                                     QPF_COLORS)


def precip_scale(style=DEFAULT_COLORBAR_STYLE):
    """Levels, colour handling and colour-bar settings for one style:
    {"levels", "kwargs" (for contourf), "ticks", "format", "extend"}."""
    if style == "true_scale":
        levels = np.logspace(np.log10(LOG_FLOOR_IN), np.log10(LOG_VMAX_IN),
                             LOG_N_LEVELS)
        return {"levels": levels,
                "kwargs": dict(cmap=qpf_continuous_cmap(),
                               norm=mcolors.LogNorm(vmin=LOG_FLOOR_IN,
                                                    vmax=LOG_VMAX_IN)),
                "ticks": [t for t in LOG_TICK_CANDIDATES
                          if LOG_FLOOR_IN <= t <= LOG_VMAX_IN],
                "format": "%g", "extend": "max"}
    if style == "standard":
        levels = np.asarray(inches(np.asarray(QPF_LEVELS, dtype=float)))
        return {"levels": levels, "kwargs": dict(colors=QPF_COLORS),
                "ticks": levels[::2], "format": "%.2g", "extend": "max"}
    raise ValueError(f"Unknown colorbar_style: {style!r} "
                     f"(use one of {COLORBAR_STYLES})")


def draw_precip(ax, lon, lat, data_mm, style=DEFAULT_COLORBAR_STYLE):
    """Filled contours of precipitation (mm in, inches drawn) on a cartopy
    axis. Returns (mappable, scale) for the colour bar. Dry cells take the
    lowest colour (white); cells with no data are drawn grey."""
    scale = precip_scale(style)
    data = np.ma.masked_invalid(np.asarray(inches(np.asarray(data_mm,
                                                             dtype=float))))
    bad = np.ma.getmaskarray(data)
    if bad.any():
        ax.pcolormesh(lon, lat, np.where(bad, 1.0, np.nan),
                      cmap=mcolors.ListedColormap([NODATA_COLOR]),
                      shading="nearest", transform=ccrs.PlateCarree(),
                      zorder=1)
    floor = scale["levels"][0]
    # Contours exclude values on the lowest level, so lift dry cells just
    # above it rather than leave holes.
    data = np.ma.clip(data, floor * 1.0001 if floor > 0 else 1e-6, None)
    cf = ax.contourf(lon, lat, data, levels=scale["levels"],
                     extend=scale["extend"], transform=ccrs.PlateCarree(),
                     zorder=2, **scale["kwargs"])
    return cf, scale


def add_us_geography(ax):
    """Overlay US coastline, state, and national borders on a cartopy axis.
    Best-effort: a no-op if the Natural Earth data is unavailable (e.g. an
    offline test host), so the map still renders."""
    try:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
    except Exception:
        pass
