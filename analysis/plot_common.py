"""Plotting pieces shared by the map-producing modules.

The QPF colour ladder and the US-geography overlay were previously buried in
the HAFS-only product modules; they live here so the grid, mask and obs
plotting code does not depend on any product module.
"""

import matplotlib.colors as mcolors
import cartopy.feature as cfeature

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
