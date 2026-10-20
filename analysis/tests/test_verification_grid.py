"""Unit tests for the verification grid: the WoFS-domain builder and the
legacy case-configuration block.

Run directly:   python3 analysis/tests/test_verification_grid.py
Or via pytest:  pytest analysis/tests/test_verification_grid.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from verification_grid import (GridConfig, grid_config_from_dict,
                               trim_track, grid_latlon,
                               spacing_report, grid_to_dict,
                               wofs_domains_from_cfg,
                               active_domains, WofsDomain,
                               aggregation_factor, wofs_grid_spec,
                               offset_from_file_km, R_EARTH_KM)


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


# --- grid construction ----------------------------------------------------


# --- output ---------------------------------------------------------------


# --- multiple WoFS deployments -------------------------------------------

_WOFS_CFG = {"wofs_domains": [
    {"name": "wofs_tx", "domain": [26.0, 32.0, -99.0, -93.0],
     "valid_start": 2024070800, "valid_end": 2024070812},
    {"name": "wofs_oh", "domain": [36.0, 42.0, -89.0, -83.0],
     "valid_start": 2024070918, "valid_end": 2024071006},
]}


def test_wofs_domains_list_with_windows():
    doms = wofs_domains_from_cfg(_WOFS_CFG)
    assert [d.name for d in doms] == ["wofs_tx", "wofs_oh"]
    assert doms[0].domain == (26.0, 32.0, -99.0, -93.0)
    assert doms[0].valid_start == datetime(2024, 7, 8, 0)
    assert doms[1].valid_end == datetime(2024, 7, 10, 6)


def test_wofs_domains_singular_shorthand_still_works():
    doms = wofs_domains_from_cfg({"wofs_domain": [31.5, 39.5, -87.0, -78.0]})
    assert len(doms) == 1 and doms[0].name == "wofs"
    assert doms[0].valid_start is None and doms[0].covers(datetime(1999, 1, 1))


def test_wofs_domains_bare_boxes_are_auto_named():
    doms = wofs_domains_from_cfg({"wofs_domains": [[26.0, 32.0, -99.0, -93.0],
                                                   [36.0, 42.0, -89.0, -83.0]]})
    assert [d.name for d in doms] == ["wofs_1", "wofs_2"]


def test_wofs_domains_absent_is_empty():
    assert wofs_domains_from_cfg({}) == []
    assert wofs_domains_from_cfg(None) == []


def test_wofs_domains_rejects_bad_input():
    for bad in ({"wofs_domain": [1, 2], "wofs_domains": []},
                {"wofs_domains": [{"name": "a"}]},
                {"wofs_domains": [{"domain": [1, 2, 3]}]},
                {"wofs_domains": [{"domain": [26, 32, -99, -93],
                                   "valid_start": 2024070812,
                                   "valid_end": 2024070800}]},
                {"wofs_domains": [{"domain": [26, 32, -99, -93],
                                   "bogus": 1}]},
                {"wofs_domains": [{"domain": [26, 32, -99, -93], "name": "a"},
                                  {"domain": [36, 42, -89, -83], "name": "a"}]}):
        try:
            wofs_domains_from_cfg(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_wofs_domain_covers_only_its_own_window():
    dom = wofs_domains_from_cfg(_WOFS_CFG)[0]
    assert dom.covers(datetime(2024, 7, 8, 6))
    # the day-1 Texas box must not mask day-3 statistics in Ohio
    assert not dom.covers(datetime(2024, 7, 10, 0))
    assert not dom.covers(datetime(2024, 7, 7, 0))


def test_active_domains_selects_by_valid_time():
    doms = wofs_domains_from_cfg(_WOFS_CFG)
    assert [d.name for d in active_domains(doms, datetime(2024, 7, 8, 6))] \
        == ["wofs_tx"]
    assert [d.name for d in active_domains(doms, datetime(2024, 7, 10, 0))] \
        == ["wofs_oh"]
    assert active_domains(doms, datetime(2024, 7, 9, 0)) == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()


# --- the WoFS-domain grid --------------------------------------------------

# Global attributes of a real WoFS file (Helene, deployment 20240927_d1).
HELENE_ATTRS = {"start_lat": 29.02168, "start_lon": -86.84998,
                "lon_orient": -82.25424, "scale_lat_1": 30.0,
                "scale_lat_2": 60.0, "d_km": 3.0, "nx": "300", "ny": "300",
                "CEN_LAT": 33.19005}


def test_aggregation_factor():
    assert aggregation_factor(3.0, 6.0) == 2
    assert aggregation_factor(3.0, 3.0) == 1
    with pytest.raises(ValueError):
        aggregation_factor(3.0, 7.0)


def test_wofs_grid_is_150_by_150_at_6_km():
    spec = wofs_grid_spec(HELENE_ATTRS, 6.0, name="helene", source="f.nc")
    assert (spec.nx, spec.ny, spec.res_km) == (150, 150, 6.0)
    assert spec.name == "helene" and spec.source == "f.nc"
    assert spec.lon_0 == -82.25424 and (spec.lat_1, spec.lat_2) == (30.0, 60.0)


def test_native_grid_starts_exactly_at_the_files_lower_left_point():
    native = wofs_grid_spec(HELENE_ATTRS, 3.0)
    lat, lon = grid_latlon(native)
    assert (native.nx, native.ny) == (300, 300)
    assert abs(lat[0, 0] - 29.02168) < 1e-4
    assert abs(lon[0, 0] - -86.84998) < 1e-4


def test_each_6_km_cell_is_the_mean_of_its_four_3_km_cells():
    native = wofs_grid_spec(HELENE_ATTRS, 3.0)
    coarse = wofs_grid_spec(HELENE_ATTRS, 6.0)
    nlat, nlon = grid_latlon(native)
    clat, clon = grid_latlon(coarse)
    blat = nlat.reshape(150, 2, 150, 2).mean(axis=(1, 3))
    blon = nlon.reshape(150, 2, 150, 2).mean(axis=(1, 3))
    # projected positions average exactly; lat/lon of the mean differ by
    # curvature only, far below a metre at this scale
    assert abs(clat - blat).max() < 1e-5
    assert abs(clon - blon).max() < 1e-5


def test_domain_is_900_km_square():
    spec = wofs_grid_spec(HELENE_ATTRS, 6.0)
    assert spec.nx * spec.res_km == 900.0 and spec.ny * spec.res_km == 900.0


def test_met_spec_is_east_positive_and_has_the_wofs_projection():
    spec = wofs_grid_spec(HELENE_ATTRS, 6.0)
    fields = spec.met_spec.split()
    assert fields[0] == "lambert"
    assert (int(fields[1]), int(fields[2])) == (150, 150)
    assert float(fields[4]) == spec.lon_ll and float(fields[4]) < 0
    assert float(fields[5]) == -82.25424
    assert float(fields[6]) == 6.0 and float(fields[7]) == R_EARTH_KM
    assert (float(fields[8]), float(fields[9])) == (30.0, 60.0)
    assert fields[10] == "N"


def test_ground_spacing_is_close_to_nominal_over_the_domain():
    spec = wofs_grid_spec(HELENE_ATTRS, 6.0)
    assert spacing_report(spec)["worst_pct"] < 3.0


def test_grid_rejects_a_native_grid_that_does_not_divide():
    attrs = dict(HELENE_ATTRS, nx="301")
    with pytest.raises(ValueError):
        wofs_grid_spec(attrs, 6.0)


def test_grid_needs_the_projection_attributes():
    attrs = {k: v for k, v in HELENE_ATTRS.items() if k != "scale_lat_1"}
    with pytest.raises(KeyError):
        wofs_grid_spec(attrs, 6.0)


def test_offset_from_file_is_zero_for_a_consistent_file_and_flags_a_shift():
    native = wofs_grid_spec(HELENE_ATTRS, 3.0)
    coarse = wofs_grid_spec(HELENE_ATTRS, 6.0)
    xlat, xlon = grid_latlon(native)
    assert offset_from_file_km(native, xlat, xlon, 1)[0] < 1e-6
    assert offset_from_file_km(coarse, xlat, xlon, 2)[0] < 1e-3
    # a file whose grid sits 0.05 degrees (~5 km) east is not this grid
    assert offset_from_file_km(coarse, xlat, xlon + 0.05, 2)[0] > 3.0


def test_offset_from_file_rejects_wrong_array_shape():
    coarse = wofs_grid_spec(HELENE_ATTRS, 6.0)
    import numpy as np
    with pytest.raises(ValueError):
        offset_from_file_km(coarse, np.zeros((300, 299)), np.zeros((300, 299)), 2)


def test_grid_to_dict_carries_the_spec_and_report():
    spec = wofs_grid_spec(HELENE_ATTRS, 6.0, source="f.nc")
    payload = grid_to_dict(spec, spacing_report(spec), extra={"case": "x"})
    for key in ("nx", "ny", "met_spec", "proj4", "source", "spacing_km",
                "case", "met_lon_west_positive"):
        assert key in payload, key
    assert payload["source"] == "f.nc"
