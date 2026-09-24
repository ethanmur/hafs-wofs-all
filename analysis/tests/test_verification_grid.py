"""Unit tests for the verification-grid configuration block.

Run directly:   python3 analysis/tests/test_verification_grid.py
Or via pytest:  pytest analysis/tests/test_verification_grid.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification_grid import (GridConfig, grid_config_from_dict,
                               trim_track, derive_projection,
                               build_grid, grid_latlon,
                               spacing_report, grid_to_dict,
                               R_EARTH_KM)


def test_defaults_when_block_absent():
    for block in (None, {}):
        cfg = grid_config_from_dict(block)
        assert cfg.res_km == 6.0
        assert cfg.pad_km == 750.0
        assert cfg.mask_radius_km == 500.0
        assert cfg.margin_h == 6
        assert cfg.max_cells == 800
        assert cfg.std_parallels is None
        assert cfg.domain_override is None


def test_grid_pad_is_larger_than_the_scoring_mask_by_default():
    # long-lead forecasts drift off the best track; the grid must outrun the
    # swath so false alarms are not clipped away at the boundary
    cfg = grid_config_from_dict({})
    assert cfg.pad_km - cfg.mask_radius_km == 250.0


def test_overrides_are_read():
    cfg = grid_config_from_dict({
        "res_km": 5, "pad_km": 1000, "mask_radius_km": 400, "margin_h": 12,
        "max_cells": 1200, "std_parallels": [15, 35], "name": "atlantic5km"})
    assert (cfg.res_km, cfg.pad_km, cfg.mask_radius_km) == (5.0, 1000.0, 400.0)
    assert cfg.margin_h == 12 and cfg.max_cells == 1200
    assert cfg.std_parallels == (15.0, 35.0)
    assert cfg.grid_name == "atlantic5km"


def test_grid_name_defaults_to_resolution():
    assert grid_config_from_dict({}).grid_name == "lambert6km"
    assert grid_config_from_dict({"res_km": 5}).grid_name == "lambert5km"


def test_pad_smaller_than_mask_is_rejected():
    try:
        grid_config_from_dict({"pad_km": 300, "mask_radius_km": 500})
    except ValueError as err:
        assert "pad_km" in str(err) and "mask_radius_km" in str(err)
        return
    raise AssertionError("expected ValueError")


def test_bad_numbers_rejected():
    for block in ({"res_km": 0}, {"mask_radius_km": -1}, {"margin_h": -1},
                  {"max_cells": 0}, {"std_parallels": [20]},
                  {"std_parallels": [45, 20]}):
        try:
            grid_config_from_dict(block)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {block}")


def test_domain_override_must_be_an_ordered_box():
    cfg = grid_config_from_dict(
        {"domain_override": [30, 40, -90, -80]})
    assert cfg.domain_override == (30.0, 40.0, -90.0, -80.0)
    for bad in ([30, 40, -90], [40, 30, -90, -80], [30, 40, -80, -90]):
        try:
            grid_config_from_dict({"domain_override": bad})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_unknown_keys_are_rejected():
    try:
        grid_config_from_dict({"resolution_km": 6})
    except ValueError as err:
        assert "resolution_km" in str(err)
        return
    raise AssertionError("expected ValueError")


# --- track trimming -------------------------------------------------------

def _track(hours, lat0=25.0, lon0=-90.0):
    """Synthetic track, one fix every 6 h, moving north-east."""
    return [(datetime(2024, 7, 5) + timedelta(hours=h), lat0 + 0.4 * i,
             lon0 + 0.3 * i, "HU")
            for i, h in enumerate(hours)]


def test_trim_track_follows_the_valid_window_not_the_cycle_range():
    track = _track(range(0, 145, 6))          # 7/5 00Z through 7/11 00Z
    kept = trim_track(track, datetime(2024, 7, 8), datetime(2024, 7, 10),
                      margin_h=6)
    assert kept[0][0] == datetime(2024, 7, 7, 18)
    assert kept[-1][0] == datetime(2024, 7, 10, 6)
    # a forecast initialized on 7/5 does not drag the grid back to 7/5
    assert all(t >= datetime(2024, 7, 7, 18) for t, _, _, _ in kept)


def test_trim_track_margin_of_zero():
    track = _track(range(0, 49, 6))
    kept = trim_track(track, datetime(2024, 7, 5, 12),
                      datetime(2024, 7, 6, 0), margin_h=0)
    assert [t.hour for t, _, _, _ in kept] == [12, 18, 0]


def test_trim_track_can_be_empty():
    track = _track(range(0, 25, 6))
    assert trim_track(track, datetime(2025, 1, 1), datetime(2025, 1, 2), 6) == []


# --- projection -----------------------------------------------------------

def test_derive_projection_insets_standard_parallels():
    proj = derive_projection((20.0, 44.0, -100.0, -70.0))
    assert proj["lat_0"] == 32.0 and proj["lon_0"] == -85.0
    # one sixth of the 24-degree span inset from each end
    assert proj["lat_1"] == 24.0 and proj["lat_2"] == 40.0


def test_derive_projection_narrow_domain_falls_back():
    proj = derive_projection((30.0, 33.0, -90.0, -85.0))
    assert proj["lat_1"] < proj["lat_2"]
    assert proj["lat_1"] == 29.5 and proj["lat_2"] == 33.5


def test_derive_projection_rejects_hemispheric_span():
    try:
        derive_projection((10.0, 50.0, -170.0, 40.0))
    except ValueError as err:
        assert "180 degrees" in str(err)
        return
    raise AssertionError("expected ValueError")


# --- grid construction ----------------------------------------------------

def test_build_grid_from_track_and_spacing_is_even():
    cfg = grid_config_from_dict({})
    spec = build_grid(_track(range(0, 49, 6)), cfg)
    assert spec.rule == "track"
    assert spec.res_km == 6.0 and spec.nx > 0 and spec.ny > 0
    report = spacing_report(spec)
    # the whole point of Lambert with two standard parallels
    assert report["worst_pct"] < 2.0
    lat, lon = grid_latlon(spec)
    assert lat.shape == (spec.ny, spec.nx) == lon.shape


def test_build_grid_pad_sets_the_extent():
    cfg = grid_config_from_dict({})
    track = _track([0])                      # a single fix
    spec = build_grid(track, cfg)
    # a lone point padded by 750 km either side spans ~1500 km = 250 cells
    assert 248 <= spec.nx <= 254 and 248 <= spec.ny <= 254


def test_build_grid_unions_the_wofs_box():
    cfg = grid_config_from_dict({})
    track = _track(range(0, 25, 6))
    plain = build_grid(track, cfg)
    with_wofs = build_grid(track, cfg, wofs_domain=(40.0, 46.0, -80.0, -74.0))
    assert with_wofs.rule == "track+wofs"
    assert with_wofs.n_cells > plain.n_cells
    lat, lon = grid_latlon(with_wofs)
    assert lat.max() >= 46.0 and lon.max() >= -74.0


def test_build_grid_domain_override_wins():
    cfg = grid_config_from_dict({"domain_override": [30.0, 36.0, -92.0, -86.0]})
    spec = build_grid(_track(range(0, 49, 6)), cfg,
                      wofs_domain=(10.0, 12.0, -60.0, -58.0))
    assert spec.rule == "domain_override"
    lat, lon = grid_latlon(spec)
    # no pad and neither the track nor the WoFS box widen it
    assert lat.min() > 28.0 and lat.max() < 38.0
    assert lon.min() > -94.0 and lon.max() < -84.0


def test_build_grid_wofs_only_when_no_track_survives():
    cfg = grid_config_from_dict({})
    spec = build_grid([], cfg, wofs_domain=(31.0, 39.0, -87.0, -79.0))
    assert spec.rule == "wofs-only"
    assert spec.nx > 0 and spec.ny > 0


def test_build_grid_needs_something_to_bound():
    try:
        build_grid([], grid_config_from_dict({}))
    except ValueError as err:
        assert "nothing to build a grid from" in str(err)
        return
    raise AssertionError("expected ValueError")


def test_build_grid_max_cells_guard():
    cfg = grid_config_from_dict({"max_cells": 100})
    try:
        build_grid(_track(range(0, 145, 6)), cfg)
    except ValueError as err:
        assert "max_cells" in str(err)
        return
    raise AssertionError("expected ValueError")


def test_build_grid_is_reproducible():
    cfg = grid_config_from_dict({})
    track = _track(range(0, 49, 6))
    a, b = build_grid(track, cfg), build_grid(track, cfg)
    assert a == b


# --- output ---------------------------------------------------------------

def test_met_spec_is_west_positive():
    cfg = grid_config_from_dict({})
    spec = build_grid(_track(range(0, 25, 6)), cfg)
    fields = spec.met_spec.split()
    assert fields[0] == "lambert"
    assert int(fields[1]) == spec.nx and int(fields[2]) == spec.ny
    # MET grid specs take longitude as degrees WEST-positive, the opposite
    # sign to lon_ll/lon_0 on the GridSpec itself
    assert float(fields[4]) == -spec.lon_ll
    assert float(fields[5]) == -spec.lon_0
    assert float(fields[6]) == spec.res_km
    assert float(fields[7]) == R_EARTH_KM
    assert (float(fields[8]), float(fields[9])) == (spec.lat_1, spec.lat_2)


def test_proj4_is_east_positive():
    spec = build_grid(_track(range(0, 25, 6)), grid_config_from_dict({}))
    assert f"+lon_0={spec.lon_0}" in spec.proj4
    assert "+proj=lcc" in spec.proj4


def test_grid_to_dict_carries_the_spec_and_report():
    spec = build_grid(_track(range(0, 25, 6)), grid_config_from_dict({}))
    payload = grid_to_dict(spec, spacing_report(spec), extra={"case": "x"})
    for key in ("nx", "ny", "met_spec", "proj4", "rule", "spacing_km",
                "case", "met_lon_west_positive"):
        assert key in payload, key


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
