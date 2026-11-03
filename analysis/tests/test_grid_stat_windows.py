"""Tests for the E1 grid_stat configuration and one-window test: the
neighborhood widths, the config text, choosing the window and runs, and the
whole command against fake gen_vx_mask and grid_stat tools."""
import json
import stat
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import deployment_case as dc
import grid_stat_windows as gw
from case_helpers import make_case
from test_deployment_case import GOOD, _generated, _write
from test_eyeball_plots import NX, NY, _case as _eyeball_case, _field, _write_nc

T = datetime
COUNTS = (100, 20, 10, 10, 60)        # total, hits, false alarms, misses, CN

FAKE_VX = '''#!{python}
import sys
import netCDF4
args = sys.argv
name = args[args.index("-name") + 1]
with netCDF4.Dataset(args[2]) as nc:
    vals = nc.variables["obs_valid"][:].astype(float)
with netCDF4.Dataset(args[3], "w") as nc:
    nc.createDimension("lat", vals.shape[0])
    nc.createDimension("lon", vals.shape[1])
    nc.createVariable(name, "f4", ("lat", "lon"))[:] = (vals > 0.5)
'''

FAKE_GS = '''#!{python}
import json, os, re, sys
args = sys.argv
if "{mode}" == "fail":
    sys.stderr.write("ERROR: bad config\\n")
    sys.exit(1)
conf = open(args[3]).read()
outdir = args[args.index("-outdir") + 1]
prefix = os.environ["OUT_PREFIX"]
with open(os.path.join(os.path.dirname(args[0]), "calls.jsonl"), "a") as fh:
    fh.write(json.dumps({{"fcst": args[1], "obs": args[2],
                         "name": os.environ["FCST_NAME"],
                         "mask": os.environ["MASK_FILE"],
                         "model": os.environ["MODEL"],
                         "prefix": prefix}}) + "\\n")
cats = re.search(r"cat_thresh = \\[ (.*?) \\]", conf).group(1).split(", ")
widths = [int(w) for w in re.search(r"width\\s+= \\[ (.*?) \\]", conf)
          .group(1).split(", ")]
if "{mode}" == "short":
    cats = cats[:3]
head = ("VERSION MODEL DESC FCST_LEAD FCST_VALID_BEG FCST_VALID_END OBS_LEAD "
        "OBS_VALID_BEG OBS_VALID_END FCST_VAR FCST_UNITS FCST_LEV OBS_VAR "
        "OBS_UNITS OBS_LEV OBTYPE VX_MASK INTERP_MTHD INTERP_PNTS FCST_THRESH "
        "OBS_THRESH COV_THRESH ALPHA LINE_TYPE")
def row(line, thresh, pts, values):
    common = ["V12.2", os.environ["MODEL"], "NA", "030000", "20240708_210000",
              "20240708_210000", "000000", "20240708_210000", "20240708_210000",
              "x", "mm", "(*,*)", "precip", "mm", "(*,*)", "MRMS", "FULL",
              "NEAREST", str(pts), thresh, thresh, "NA", "NA", line]
    return " ".join(common + [str(v) for v in values])
lines = [head]
ctc = {counts}
for c in cats:
    lines.append(row("CTC", c, 1, ctc))
    lines.append(row("CTS", c, 1, [100, 0.3]))
for cnt in ("NA", ">=0.5"):
    lines.append(row("CNT", cnt, 1, [100, 1.0]))
    lines.append(row("SL1L2", cnt, 1, [100, 2.0, 1.0, 3.0, 8.0, 5.0, 1.5]))
for w in widths:
    for c in cats:
        lines.append(row("NBRCTC", c, w * w, ctc))
        lines.append(row("NBRCNT", c, w * w, [100, 0.2]))
name = "grid_stat_" + prefix + "_030000L_20240708_210000V.stat"
open(os.path.join(outdir, name), "w").write("\\n".join(lines) + "\\n")
'''


def fake_tools(tmp_path, mode="ok"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in (("gen_vx_mask", FAKE_VX), ("grid_stat", FAKE_GS)):
        tool = bin_dir / name
        tool.write_text(text.format(python=sys.executable, mode=mode,
                                    counts=list(COUNTS)))
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _case(tmp_path, mode="ok", **kw):
    case = _eyeball_case(tmp_path, truth_files="mrms", truth_source="mrms",
                         regrid=dc.RegridSettings(
                             sources=("mrms",),
                             met_bin_dir=fake_tools(tmp_path, mode)),
                         **kw)
    mask = np.zeros((NY, NX))
    mask[:, :6] = 1.0
    _write_nc(case.mask_path, {"obs_valid": mask})
    return case


# --- configuration ----------------------------------------------------------

def test_neighbourhood_widths_are_distances_in_whole_cells():
    km = (6, 18, 30, 54, 126)
    # grid_stat takes only odd widths: the even counts at 3 km go up one
    assert gw.neighbourhood_cells(km, 3.0) == [3, 7, 11, 19, 43]
    assert gw.effective_km(km, 3.0) == [9, 21, 33, 57, 129]
    assert gw.neighbourhood_cells(km, 6.0) == [1, 3, 5, 9, 21]
    assert all(c % 2 for c in gw.neighbourhood_cells(km, 1.0))
    with pytest.raises(ValueError, match="whole cells"):
        gw.neighbourhood_cells(km, 4.0)


def test_config_asks_for_the_planned_scores_on_the_validity_mask(tmp_path):
    case = make_case(tmp_path, res_km=3.0, truth_source="mrms")
    text = gw.config_text(case)
    assert (">=0.5, >=1, >=2.5, >=5, >=10, >=25, >=50, >=75, >=100, "
            ">=125, >=150") in text
    assert "cnt_thresh = [ NA, >=0.5 ]; cnt_logic = UNION;" in text
    assert "width      = [ 3, 7, 11, 19, 43 ];" in text
    assert "cov_thresh = [ >=0.5 ];" in text
    assert 'poly = [ "${MASK_FILE}" ];' in text and "grid = [];" in text
    assert 'name = "${FCST_NAME}"' in text and 'name = "precip"' in text
    assert 'obtype = "MRMS";' in text
    for line_type in ("ctc", "cts", "cnt", "sl1l2", "nbrctc", "nbrcnt"):
        assert f"{line_type} = STAT" in text
    assert "nc_pairs_flag = FALSE" in text


def test_config_follows_the_settings_and_the_resolution(tmp_path):
    settings = dc.GridStatSettings(thresholds_mm=(1.0, 5.0), widths_km=(6.0,),
                                   wet_thresh_mm=2.0, cov_thresh=0.25)
    case = make_case(tmp_path, res_km=6.0, grid_stat=settings)
    text = gw.config_text(case)
    assert "cat_thresh = [ >=1, >=5 ]" in text
    assert "cnt_thresh = [ NA, >=2 ]" in text
    assert "width      = [ 1 ];" in text and "cov_thresh = [ >=0.25 ]" in text


def test_settings_load_from_the_yaml_and_are_validated(tmp_path):
    cfg = dict(_generated(GOOD), grid_stat={
        "thresholds_mm": [1, 10], "widths_km": [9], "test_end": 2024070821})
    case = dc.from_yaml(_write(tmp_path, cfg))
    assert case.grid_stat.thresholds_mm == (1.0, 10.0)
    assert case.grid_stat.test_end == T(2024, 7, 8, 21)
    default = dc.from_yaml(_write(tmp_path, _generated(GOOD)))
    assert default.grid_stat.thresholds_mm[-1] == 150.0
    for bad in ({"thresholds_mm": []}, {"widths_km": [0]},
                {"cov_thresh": 1.5}, {"nbrhd": 3}):
        with pytest.raises(ValueError):
            dc.from_yaml(_write(tmp_path, dict(_generated(GOOD),
                                               grid_stat=bad)))


def test_names_carry_resolution_and_buffer_only_off_default(tmp_path):
    path = _write(tmp_path, _generated(GOOD))
    base = dc.from_yaml(path)
    assert base.grid_stat_tag == ""
    assert base.grid_stat_conf_path.name == (
        "GridStat_2024_02l_beryl_20240708_d1.conf")
    both = dc.from_yaml(path, res_km=3.0, coast_km=50.0)
    assert both.grid_stat_tag == "_3km_coast50km"
    assert both.grid_stat_mask_path.name == (
        "2024_02l_beryl_20240708_d1_3km_coast50km_FULL_mask.nc")
    assert both.grid_stat_conf_path != base.grid_stat_conf_path
    assert base.grid_stat_window_dir(T(2024, 7, 9, 0)).name == "2024070900"


# --- choosing ---------------------------------------------------------------

def test_the_wettest_window_with_all_inputs_is_chosen(tmp_path):
    case = _case(tmp_path)
    mask = np.zeros((NY, NX), dtype=bool)
    mask[:, :6] = True
    _write_nc(case.obs_window_path("mrms", T(2024, 7, 9, 0)),
              {"precip": _field(99) + 50.0})
    pick = gw.choose_test(case, mask)
    assert pick["window"].end == T(2024, 7, 9, 0)
    assert pick["wofs"][0] == T(2024, 7, 8, 18)            # shortest lead
    assert {c: i for c, i, _ in pick["hafs"]} == {
        "HFSA": T(2024, 7, 8, 12), "HFSB": T(2024, 7, 8, 12)}


def test_test_end_pins_the_window_and_missing_inputs_stop(tmp_path):
    mask = np.ones((NY, NX), dtype=bool)
    case = _case(tmp_path)
    _write_nc(case.obs_window_path("mrms", T(2024, 7, 9, 0)),
              {"precip": _field(99) + 50.0})
    case.grid_stat = dc.GridStatSettings(test_end=T(2024, 7, 8, 21))
    assert gw.choose_test(case, mask)["window"].end == T(2024, 7, 8, 21)
    case.grid_stat = dc.GridStatSettings(test_end=T(2024, 7, 8, 23))
    with pytest.raises(SystemExit, match="regrid-wofs"):
        gw.choose_test(case, mask)


def test_runs_are_one_wofs_member_and_one_cycle_per_configuration(tmp_path):
    case = _case(tmp_path)
    pick = gw.choose_test(case, np.ones((NY, NX), dtype=bool))
    runs = gw.runs_for_selection(case, pick)
    assert [(t, v) for t, _, v in runs] == [
        ("wofs_m00_i202407081800", "rain_mm_m00"),
        ("hfsa_c2024070812", "precip"), ("hfsb_c2024070812", "precip")]


# --- the stat file ----------------------------------------------------------

def test_contingency_scores_match_a_hand_calculation():
    sc = gw.contingency_scores(100, 20, 10, 10)
    assert sc["pod"] == pytest.approx(20 / 30)
    assert sc["far"] == pytest.approx(10 / 30)
    assert sc["csi"] == pytest.approx(20 / 40)
    # H_rand = (30 * 30) / 100 = 9; ETS = (20 - 9) / (40 - 9)
    assert sc["ets"] == pytest.approx(11 / 31)
    assert np.isnan(gw.contingency_scores(100, 0, 0, 0)["pod"])


# --- the command ------------------------------------------------------------

def test_the_test_scores_one_window_with_three_runs(tmp_path, capsys):
    case = _case(tmp_path)
    written = gw.grid_stat_test_case(case)
    out = capsys.readouterr().out
    assert len(written) == 3
    window_dir = case.grid_stat_window_dir(T(2024, 7, 8, 21))
    assert all(p.parent == window_dir for p in written)
    assert {p.name.split("_")[2] for p in written} == {"wofs", "hfsa", "hfsb"}
    assert case.grid_stat_conf_path.exists()
    assert case.grid_stat_mask_path.exists()

    calls = [json.loads(line) for line in
             (tmp_path / "bin" / "calls.jsonl").read_text().splitlines()]
    assert [c["name"] for c in calls] == ["rain_mm_m00", "precip", "precip"]
    assert {c["mask"] for c in calls} == {str(case.grid_stat_mask_path)}
    assert {Path(c["obs"]).name for c in calls} == {
        case.obs_window_path("mrms", T(2024, 7, 8, 21)).name}
    assert (window_dir / f"{calls[0]['prefix']}.log").exists()

    # the fake's mask is the validity mask under the name grid_stat will see
    import netCDF4
    with netCDF4.Dataset(case.grid_stat_mask_path) as nc:
        assert nc.variables["FULL"][:].sum() == NY * 6
    assert "CTC 11/11" in out and "NBRCNT 55/55" in out
    assert "WARNING" not in out and "FULL" in out
    assert "0.355" in out                                  # ETS = 11/31
    assert "wet>=0.5" in out and "   all" in out


def test_a_short_file_is_flagged_in_the_digest(tmp_path, capsys):
    case = _case(tmp_path, mode="short")
    gw.grid_stat_test_case(case)
    out = capsys.readouterr().out
    assert "WARNING: CTC has 3 rows, expected 11" in out


def test_a_failing_grid_stat_stops_with_the_log(tmp_path):
    with pytest.raises(SystemExit, match="bad config"):
        gw.grid_stat_test_case(_case(tmp_path, mode="fail"))


def test_the_buffer_tag_reaches_the_output_names(tmp_path):
    case = _case(tmp_path)
    case.masks = dc.MaskSettings(coast_buffer_km=50.0)
    _write_nc(case.mask_path, {"obs_valid": np.ones((NY, NX))})
    written = gw.grid_stat_test_case(case)
    assert all("_coast50km_" in p.name for p in written)
    assert "coast50km" in case.grid_stat_conf_path.name


def test_missing_mask_or_an_unavailable_case_stops(tmp_path):
    case = _case(tmp_path)
    case.mask_path.unlink()
    with pytest.raises(SystemExit, match="build-mask"):
        gw.grid_stat_test_case(case)
    off = make_case(tmp_path / "off", wofs_available=False, valid_start=None,
                    valid_end=None)
    with pytest.raises(SystemExit, match="no complete WoFS window"):
        gw.grid_stat_test_case(off)


def test_it_is_a_command():
    import run
    assert run.parse_args(["c.yaml", "grid-stat-test"])[1] == "grid-stat-test"
