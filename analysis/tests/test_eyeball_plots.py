"""Tests for the eyeball plots: choosing windows, cycles and members, the
precipitation colour scales, and both commands on synthetic regridded files
(figures are really rendered)."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deployment_case as dc
import eyeball_plots as ep
import hafs_windows as hw
import plot_common as pc
from case_helpers import make_case
from test_window_manifest import _write_hafs, _write_wofs

T = datetime
NY = NX = 8


def _latlon():
    lon, lat = np.meshgrid(np.linspace(-86, -83, NX), np.linspace(28, 31, NY))
    return lat, lon


def _write_nc(path, variables):
    import netCDF4
    path.parent.mkdir(parents=True, exist_ok=True)
    lat, lon = _latlon()
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("lat", NY)
        nc.createDimension("lon", NX)
        nc.createVariable("lat", "f4", ("lat", "lon"))[:] = lat
        nc.createVariable("lon", "f4", ("lat", "lon"))[:] = lon
        for name, arr in variables.items():
            v = nc.createVariable(name, "f4", ("lat", "lon"),
                                  fill_value=-9999.0)
            v[:] = arr


def _field(seed, scale=40.0):
    rng = np.random.default_rng(seed)
    base = rng.random((NY, NX)) * scale
    base[0, 0] = 0.0
    return base


def _bdeck(path):
    # one BEST fix per 6 h, ATCF format: basin, num, date, tech-num, tech, tau,
    # lat, lon, wind, pres, status
    lines = []
    for h in range(0, 30, 6):
        t = T(2024, 7, 8, 0) + timedelta(hours=h)
        lines.append(f"AL, 02, {t:%Y%m%d%H}, 03, BEST, 0, 295N, 0850W, 60, "
                     "990, TS, 34, NEQ, 0, 0, 0, 0")
    path.write_text("\n".join(lines) + "\n")


def _case(tmp_path, with_wofs=True, with_truth=True, truth_files="stage4",
          **kw):
    wofs_root = tmp_path / "wofs" / "2024_BERYL" / "20240708_d1"
    _write_wofs(wofs_root, T(2024, 7, 8, 18), range(0, 73))
    _bdeck(tmp_path / "bt.dat")
    case = make_case(
        tmp_path, wofs_root=wofs_root, valid_start=T(2024, 7, 8, 18),
        valid_end=T(2024, 7, 9, 0),
        hafs_models={"HFSA": tmp_path / "A", "HFSB": tmp_path / "B"},
        hafs_regrid_dir=tmp_path / "hafs-regrid" / "20240708_d1_beryl",
        eyeball_plot_dir=tmp_path / "figs" / "eyeball",
        hafs_check_plot_dir=tmp_path / "figs" / "check", **kw)
    for w_end in (T(2024, 7, 8, 21), T(2024, 7, 9, 0)):
        for config in ("HFSA", "HFSB"):
            for k, init in enumerate((T(2024, 7, 7, 0), T(2024, 7, 7, 12),
                                      T(2024, 7, 8, 0), T(2024, 7, 8, 12))):
                _write_nc(case.hafs_window_path(config, w_end, init),
                          {"precip": _field(k + w_end.hour)})
        if with_wofs:
            for init in (T(2024, 7, 8, 17), T(2024, 7, 8, 18)):
                _write_nc(case.wofs_window_path(w_end, init, 3),
                          {f"rain_mm_m{m:02d}": _field(10 + m)
                           for m in range(12)})
        if with_truth:
            _write_nc(case.obs_window_path(truth_files, w_end),
                      {"precip": _field(99)})
    return case


# --- choosing ---------------------------------------------------------------

def test_pick_evenly_spreads_and_keeps_the_ends():
    xs = list(range(10))
    assert ep.pick_evenly(xs, 1) == [9]
    assert ep.pick_evenly(xs, 2) == [0, 9]
    assert ep.pick_evenly(xs, 3) == [0, 4, 9] or ep.pick_evenly(xs, 3) == [0, 5, 9]
    assert ep.pick_evenly(xs, 20) == xs
    assert ep.pick_evenly([], 2) == []


def test_runs_for_parses_cycles_and_inits_of_both_name_lengths(tmp_path):
    for name in ("2024070821_i202407081200_hfsa_6km.nc",
                 "2024070821_i202407080600_hfsa_6km.nc",
                 "2024070821_i202407081200_hfsb_6km.nc",
                 "2024070821_i202407081200_hfsa_3km.nc",
                 "2024070900_i202407081200_hfsa_6km.nc"):
        (tmp_path / name).write_text("")
    runs = ep.runs_for(tmp_path, T(2024, 7, 8, 21), "hfsa", "6km")
    assert [r[0] for r in runs] == [T(2024, 7, 8, 6), T(2024, 7, 8, 12)]


def test_lead_text():
    w = type("W", (), {"start": T(2024, 7, 8, 18), "end": T(2024, 7, 8, 21)})
    assert ep.lead_text(T(2024, 7, 8, 12), w) == "6-9 h"


def test_selection_respects_the_limits_and_needs_the_inputs(tmp_path):
    case = _case(tmp_path)
    chosen = ep.selection(case, need_wofs=True, need_truth=True)
    assert [c["window"].end for c in chosen] == [T(2024, 7, 8, 21),
                                                 T(2024, 7, 9, 0)]
    first = chosen[0]
    assert first["wofs"][0] == T(2024, 7, 8, 18)        # shortest lead
    # 2 cycles per configuration: the earliest and the latest
    assert sorted((c, i.day, i.hour) for c, i, _ in first["hafs"]) == [
        ("HFSA", 7, 0), ("HFSA", 8, 12), ("HFSB", 7, 0), ("HFSB", 8, 12)]
    one = make_case(tmp_path, wofs_root=case.wofs_root,
                    valid_start=case.valid_start, valid_end=case.valid_end,
                    hafs_models=case.hafs_models,
                    hafs_regrid_dir=case.hafs_regrid_dir,
                    eyeball=dc.EyeballSettings(windows=1, cycles=1))
    small = ep.selection(one, need_wofs=True, need_truth=True)
    assert len(small) == 1 and small[0]["window"].end == T(2024, 7, 9, 0)
    assert len(small[0]["hafs"]) == 2                    # 1 cycle x 2 configs
    nothing = _case(tmp_path / "x", with_truth=False)
    assert ep.selection(nothing, need_wofs=True, need_truth=True) == []
    assert ep.selection(nothing, need_wofs=True, need_truth=False)


# --- colour scales ----------------------------------------------------------

def test_true_scale_is_a_log_scale_in_inches_from_the_preset():
    s = pc.precip_scale("true_scale")
    assert s["levels"][0] == pytest.approx(0.01)
    assert s["levels"][-1] == pytest.approx(5.0)
    assert len(s["levels"]) == 20 and s["extend"] == "max"
    assert s["ticks"] == [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
    ratios = s["levels"][1:] / s["levels"][:-1]
    assert np.allclose(ratios, ratios[0])               # log-spaced
    assert s["kwargs"]["cmap"](0.0)[:3] == (1.0, 1.0, 1.0)   # dry is white


def test_standard_scale_is_the_banded_ladder_in_inches():
    s = pc.precip_scale("standard")
    assert s["levels"][1] == pytest.approx(5 / 25.4)
    assert s["kwargs"]["colors"] == pc.QPF_COLORS
    with pytest.raises(ValueError):
        pc.precip_scale("rainbow")


def test_dry_and_trace_cells_are_not_drawn_so_the_map_shows_through():
    mm = np.array([[0.0, 0.1, 0.3, 5.0], [np.nan, 30.0, 200.0, 0.0]])
    drawn = pc.precip_drawn(mm, "true_scale")            # floor 0.01 in = 0.254 mm
    assert drawn.tolist() == [[False, False, True, True],
                              [False, True, True, False]]
    std = pc.precip_drawn(np.array([0.0, 2.0, 5.0, 50.0]), "standard")
    assert std.tolist() == [False, False, True, True]    # first band is "no rain"


def test_an_entirely_dry_field_still_draws_with_its_colour_bar(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    lat, lon = _latlon()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    for style in pc.COLORBAR_STYLES:
        cf, scale = pc.draw_precip(ax, lon, lat, np.zeros_like(lat), style)
        fig.colorbar(cf, ticks=scale["ticks"], extend="max")
    fig.savefig(tmp_path / "dry.png")
    plt.close(fig)


# --- the commands -----------------------------------------------------------

def test_plot_eyeball_writes_one_figure_per_window_cycle_and_member(tmp_path):
    case = _case(tmp_path)
    written = ep.plot_eyeball(case)
    # 2 windows x (2 configs x 2 cycles) x 2 members
    assert len(written) == 16
    assert all(p.exists() and p.stat().st_size > 5000 for p in written)
    names = {p.name for p in written}
    assert "eyeball_2024070821_hfsa_c2024070812_m00.png" in names
    assert "eyeball_2024070900_hfsb_c2024070700_m09.png" in names
    assert {p.parent for p in written} == {case.eyeball_plot_dir}


def test_the_truth_panel_follows_truth_source_or_a_single_sources_choice(
        tmp_path, capsys):
    # MRMS windows on disk and truth_source mrms: MRMS is drawn
    mrms = _case(tmp_path / "a", truth_files="mrms", truth_source="mrms",
                 eyeball=dc.EyeballSettings(members=(0,), windows=1, cycles=1))
    assert len(ep.plot_eyeball(mrms)) == 2
    assert "Truth  : MRMS" in capsys.readouterr().out
    # truth_source mrms in the YAML, but --sources stage4 for this run
    both = _case(tmp_path / "b", truth_files="stage4", truth_source="mrms",
                 regrid=dc.RegridSettings(sources=("stage4",)),
                 eyeball=dc.EyeballSettings(members=(0,), windows=1, cycles=1))
    assert both.plot_truth == "stage4"
    assert len(ep.plot_eyeball(both)) == 2
    assert "Truth  : Stage IV" in capsys.readouterr().out
    # the named product has no windows: nothing is drawn, and it says so
    wrong = _case(tmp_path / "c", truth_files="stage4", truth_source="mrms")
    assert ep.plot_eyeball(wrong) == []


def test_plot_eyeball_skips_a_member_the_run_does_not_have(tmp_path):
    case = _case(tmp_path, eyeball=dc.EyeballSettings(members=(0, 40),
                                                      windows=1, cycles=1))
    written = ep.plot_eyeball(case)
    assert written and all(p.name.endswith("_m00.png") for p in written)


def test_plot_eyeball_standard_scale_renders_too(tmp_path):
    case = _case(tmp_path, eyeball=dc.EyeballSettings(
        members=(0,), windows=1, cycles=1, colorbar_style="standard"))
    assert len(ep.plot_eyeball(case)) == 2


def test_plot_eyeball_refuses_fields_on_different_grids(tmp_path):
    case = _case(tmp_path)
    bad = case.obs_window_path("stage4", T(2024, 7, 8, 21))
    import netCDF4
    with netCDF4.Dataset(bad, "w") as nc:
        nc.createDimension("lat", 3)
        nc.createDimension("lon", 3)
        nc.createVariable("lat", "f4", ("lat", "lon"))[:] = 30.0
        nc.createVariable("lon", "f4", ("lat", "lon"))[:] = -85.0
        nc.createVariable("precip", "f4", ("lat", "lon"))[:] = 1.0
    with pytest.raises(ValueError) as err:
        ep.plot_eyeball(case)
    assert "not on the same grid" in str(err.value)


def test_plot_eyeball_says_what_is_missing(tmp_path, capsys):
    case = _case(tmp_path, with_truth=False)
    assert ep.plot_eyeball(case) == []
    assert "regrid-wofs, regrid-hafs and window-obs" in capsys.readouterr().out


def _native_reader(path, valid_end, hours):
    lon, lat = np.meshgrid(np.linspace(-90, -80, 60), np.linspace(24, 34, 60))
    return lat, lon, np.random.default_rng(1).random(lat.shape) * 30


def test_plot_hafs_check_pairs_native_and_regridded(tmp_path):
    case = _case(tmp_path, eyeball=dc.EyeballSettings(windows=1, cycles=1))
    # the manifest wants HAFS archives to find the native files
    cycles = [T(2024, 7, 6, 0) + timedelta(hours=6 * k) for k in range(12)]
    _write_hafs(tmp_path / "A", "02l", "HFSA", cycles)
    _write_hafs(tmp_path / "B", "02l", "HFSB", cycles)
    written = ep.plot_hafs_check(case, reader=_native_reader)
    assert len(written) == 2
    assert {p.name for p in written} == {
        "hafs_check_2024070900_hfsa_c2024070812.png",
        "hafs_check_2024070900_hfsb_c2024070812.png"}
    assert all(p.stat().st_size > 5000 for p in written)
    assert {p.parent for p in written} == {case.hafs_check_plot_dir}


def test_plot_hafs_check_headline_carries_the_budget_change():
    row = {"mean_pct_diff": "0.42", "native_max_mm": "50.8",
           "regrid_max_mm": "25.4"}
    text = ep._budget_text(row)
    assert "+0.42%" in text and "2.00 -> 1.00 in" in text
    assert ep._budget_text(None) == "" and ep._budget_text(
        {"mean_pct_diff": ""}) == ""


def test_plot_commands_skip_an_unavailable_deployment(tmp_path):
    case = make_case(tmp_path, wofs_available=False, valid_start=None,
                     valid_end=None)
    for fn in (ep.plot_eyeball, ep.plot_hafs_check):
        with pytest.raises(SystemExit) as err:
            fn(case)
        assert "no complete WoFS window" in str(err.value)
