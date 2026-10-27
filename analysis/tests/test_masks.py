"""Tests for the observation-validity mask. The land rasters are injected, so
nothing here needs the Natural Earth shapefiles or the network."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import masks
import verification_grid as vg
from test_verification_grid import HELENE_ATTRS


def _spec():
    return vg.wofs_grid_spec(HELENE_ATTRS, 6.0, name="t")


def _blank(spec):
    return np.zeros((spec.ny, spec.nx), dtype=bool)


def test_buffer_is_measured_in_km_from_us_land():
    spec = _spec()
    us = _blank(spec)
    us[:, :10] = True                       # a US coast along the left edge
    mask, detail = masks.obs_valid_mask(spec, 150.0, us_land=us, all_land=us)
    # 150 km at 6 km is 25 cells: land (10) plus 25 cells of water
    assert mask[:, :35].all() and not mask[:, 36:].any()
    assert detail["n_us_land"] == us.sum()
    assert detail["n_offshore"] == mask.sum() - us.sum()


def test_buffer_scales_with_the_grid_spacing():
    # the same 150 km is 50 cells on a 3 km grid, 25 on a 6 km grid
    six = _spec()
    three = vg.wofs_grid_spec(HELENE_ATTRS, 3.0)
    for spec, cells in ((six, 25), (three, 50)):
        us = _blank(spec)
        us[:, :4] = True
        mask, _ = masks.obs_valid_mask(spec, 150.0, us_land=us, all_land=us)
        assert mask[0, 4 + cells - 1] and not mask[0, 4 + cells + 1]


def test_foreign_land_touching_the_us_is_buffered_not_dropped():
    spec = _spec()
    us = _blank(spec)
    us[:, :10] = True
    allland = us.copy()
    allland[:, 10:20] = True                # Mexico/Canada: touches US land
    mask, detail = masks.obs_valid_mask(spec, 150.0, us_land=us,
                                        all_land=allland)
    assert mask[:, 10:20].all()
    assert detail["n_foreign_land"] == 10 * spec.ny
    assert detail["n_detached_excluded"] == 0


def test_a_detached_island_inside_the_buffer_is_cut():
    spec = _spec()
    us = _blank(spec)
    us[:, :10] = True
    allland = us.copy()
    allland[70:74, 20:24] = True            # Cuba-like: close, but not attached
    mask, detail = masks.obs_valid_mask(spec, 150.0, us_land=us,
                                        all_land=allland)
    assert not mask[70:74, 20:24].any()
    assert mask[69, 20] or mask[75, 20]     # the water around it is kept
    assert detail["n_detached_excluded"] == 16


def test_offshore_us_land_is_kept():
    spec = _spec()
    us = _blank(spec)
    us[:, :10] = True
    us[50:52, 30:32] = True                 # Long Island-like: US cells of its own
    mask, _ = masks.obs_valid_mask(spec, 150.0, us_land=us, all_land=us)
    assert mask[50:52, 30:32].all()


def test_no_us_land_on_the_grid_gives_an_empty_mask_not_a_full_one():
    spec = _spec()
    none = _blank(spec)
    mask, detail = masks.obs_valid_mask(spec, 150.0, us_land=none,
                                        all_land=none)
    assert detail["n_mask"] == 0 and not mask.any()


def test_zero_buffer_keeps_only_us_land():
    spec = _spec()
    us = _blank(spec)
    us[:, :10] = True
    mask, _ = masks.obs_valid_mask(spec, 0.0, us_land=us, all_land=us)
    assert (mask == us).all()


def test_mask_netcdf_round_trips_and_carries_the_met_grid(tmp_path):
    import netCDF4
    spec = _spec()
    lat, lon = vg.grid_latlon(spec)
    mask = _blank(spec)
    mask[10:20, 30:60] = True
    path = masks.write_mask_netcdf(tmp_path / "m" / "x_obs_valid.nc", spec,
                                   mask, lat, lon)
    again = masks.read_mask(path)
    assert again.shape == (spec.ny, spec.nx) and (again == mask).all()
    with netCDF4.Dataset(path) as nc:
        assert nc.Projection == "Lambert Conformal"
        assert nc.nx == str(spec.nx) and nc.ny == str(spec.ny)
        assert float(nc.d_km.split()[0]) == spec.res_km
        assert masks.MASK_NAME in nc.variables


def test_only_the_validity_mask_remains_in_this_module():
    for gone in ("track_swath_mask", "wofs_box_mask", "build_regions",
                 "active_verify_mask", "plot_regions", "case_regions",
                 "MaskConfig"):
        assert not hasattr(masks, gone), gone
