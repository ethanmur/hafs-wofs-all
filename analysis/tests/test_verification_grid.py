"""Unit tests for the verification-grid configuration block.

Run directly:   python3 analysis/tests/test_verification_grid.py
Or via pytest:  pytest analysis/tests/test_verification_grid.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification_grid import GridConfig, grid_config_from_dict


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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
