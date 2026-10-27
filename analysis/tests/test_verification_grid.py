"""Unit tests for the verification grid: the WoFS-domain builder.

Run directly:   python3 analysis/tests/test_verification_grid.py
Or via pytest:  pytest analysis/tests/test_verification_grid.py -v
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from verification_grid import (grid_latlon, spacing_report, grid_to_dict,
                               aggregation_factor, wofs_grid_spec,
                               offset_from_file_km, grid_spec_from_json,
                               write_grid_json, MAX_OFFSET_KM, R_EARTH_KM)


# --- track trimming -------------------------------------------------------


# --- projection -----------------------------------------------------------


# --- grid construction ----------------------------------------------------


# --- output ---------------------------------------------------------------


# --- multiple WoFS deployments -------------------------------------------


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


def test_grid_spec_round_trips_through_its_json(tmp_path):
    spec = wofs_grid_spec(HELENE_ATTRS, 6.0, name="helene", source="f.nc")
    path = write_grid_json(tmp_path / "g.json", grid_to_dict(spec))
    again = grid_spec_from_json(path)
    assert again == spec
    lat, lon = grid_latlon(spec)
    lat2, lon2 = grid_latlon(again)
    assert abs(lat - lat2).max() == 0 and abs(lon - lon2).max() == 0


def test_grid_spec_from_missing_json_says_to_run_build_grid(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        grid_spec_from_json(tmp_path / "nope.json")
    assert "build-grid" in str(err.value)


def test_offset_limit_is_a_fixed_distance_that_fits_every_real_file():
    # real 2023-2024 deployments disagree with their own xlat/xlon by up to
    # 0.56 km; the limit must clear that and not scale down with resolution
    assert MAX_OFFSET_KM >= 0.6
    assert MAX_OFFSET_KM <= 0.25 * 3.0
