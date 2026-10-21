import sys
from datetime import datetime
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deployment_case as dc
import make_case_yamls as mc


def _write(tmp_path, cfg, name="2024_02l_beryl_20240708_d1.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    return path


def _generated(windows):
    inv = {"storm": "2024_BERYL", "deployment": "20240708_d1"}
    return yaml.safe_load(mc.render_case(mc.build_case(inv, windows)))


GOOD = [{"window_start": "2024-07-08 1800Z",
         "window_end": "2024-07-08 2100Z", "status": "complete"}]


def test_loads_a_generated_available_case(tmp_path):
    case = dc.from_yaml(_write(tmp_path, _generated(GOOD)))
    assert case.case_slug == "2024_02l_beryl_20240708_d1"
    assert case.storm_id == "2024-02l" and case.truth_source == "stage4"
    assert case.wofs_available and case.complete_windows == 1
    assert case.valid_start == datetime(2024, 7, 8, 18)
    assert case.valid_end == datetime(2024, 7, 8, 21)
    assert case.hafs_lookback_h == 48 and case.hafs_min_members == 8
    assert set(case.hafs_models) == {"HFSA", "HFSB"}
    assert case.res_km == 6.0
    assert case.wofs_storm_folder == "2024_BERYL"


def test_loads_an_unavailable_case_without_a_window(tmp_path):
    cfg = _generated([])
    case = dc.from_yaml(_write(tmp_path, cfg))
    assert not case.wofs_available
    assert case.valid_start is None and case.hafs_models == {}


def test_truth_source_can_switch_to_mrms(tmp_path):
    cfg = _generated(GOOD)
    cfg["truth_source"] = "mrms"
    assert dc.from_yaml(_write(tmp_path, cfg)).truth_source == "mrms"


def test_rejects_unknown_truth_source_and_keys(tmp_path):
    cfg = _generated(GOOD)
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, dict(cfg, truth_source="aorc")))
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, dict(cfg, domain=[1, 2, 3, 4])))
    bad = dict(cfg, wofs=dict(cfg["wofs"], members=18))
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, bad))


def test_available_case_needs_a_window_and_it_must_be_ordered(tmp_path):
    cfg = _generated(GOOD)
    no_window = {k: v for k, v in cfg.items() if k != "valid_end"}
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, no_window))
    backwards = dict(cfg, valid_start=2024070821, valid_end=2024070818)
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, backwards))


def test_grid_resolution_can_be_set(tmp_path):
    cfg = dict(_generated(GOOD), grid={"res_km": 3.0})
    assert dc.from_yaml(_write(tmp_path, cfg)).res_km == 3.0


def test_regrid_defaults_to_the_truth_product_and_derives_paths(tmp_path):
    case = dc.from_yaml(_write(tmp_path, _generated(GOOD)))
    assert case.regrid_sources == ("stage4",)
    assert case.regrid.method == "BUDGET" and case.regrid.width == 2
    assert case.regrid_cache_dir.name == "20240708_d1"
    assert case.regrid_plot_dir == case.out_dir / "regrid"
    assert case.grid_json_path.name == "2024_02l_beryl_20240708_d1_grid.json"
    assert case.output_slug == "2024_02l_beryl_20240708_d1_2024070818_2024070821"


def test_regrid_block_can_ask_for_both_products(tmp_path):
    cfg = dict(_generated(GOOD),
               regrid={"sources": ["stage4", "mrms"], "width": 3})
    case = dc.from_yaml(_write(tmp_path, cfg))
    assert case.regrid_sources == ("stage4", "mrms") and case.regrid.width == 3


def test_regrid_block_rejects_unknown_keys_and_sources(tmp_path):
    cfg = _generated(GOOD)
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, dict(cfg, regrid={"sources": ["aorc"]})))
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, dict(cfg, regrid={"grid_template": "x"})))
    with pytest.raises(ValueError):
        dc.from_yaml(_write(tmp_path, dict(cfg, regrid_plots={"panels": ["x"]})))
