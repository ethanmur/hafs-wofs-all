"""Tests for the E3 .stat parser, against two real grid_stat outputs from the
Milton 20241009_d1 test (a WoFS member, whose time columns are wrong, and an
HFSA cycle) kept in tests/fixtures/stat."""
import csv
import math
import shutil
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stat_tables as st
from case_helpers import make_case

FIX = Path(__file__).resolve().parent / "fixtures" / "stat"
WOFS = "grid_stat_wofs_m00_i202410092100_3km_000000L_19700101_000000V.stat"
HFSA = "grid_stat_hfsa_c2024100918_3km_060000L_20241010_000000V.stat"
END = datetime(2024, 10, 10, 0)
T_WOFS, T_HFSA = "wofs_m00_i202410092100", "hfsa_c2024100918"


def _case(tmp_path):
    return make_case(tmp_path, res_km=3.0)


def _install(case, end=END, files=((T_WOFS, WOFS), (T_HFSA, HFSA))):
    """Put fixture .stat files in the window folder; write the runs CSV."""
    out = case.grid_stat_window_dir(end)
    out.mkdir(parents=True, exist_ok=True)
    for _, name in files:
        shutil.copy(FIX / name, out / name)
    runs = [(end, tag) for tag, _ in files]
    _write_runs(case, runs)


def _write_runs(case, runs):
    case.grid_stat_runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(case.grid_stat_runs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_end", "run", "status", "seconds", "stat_file"])
        for end, tag in runs:
            w.writerow([f"{end:%Y%m%d%H}", tag, "done", "1.0", ""])


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(text):
    return float(text) if text != "" else math.nan


# --- small pieces ------------------------------------------------------------

def test_threshold_value_reads_the_number_and_ignores_the_union_marks():
    assert st.threshold_value(">=2.5") == 2.5
    assert st.threshold_value(">=0.5||") == 0.5
    assert st.threshold_value(">=150") == 150.0
    assert st.threshold_value("NA") is None


def test_model_tags_give_source_member_and_init():
    w = st.parse_model("wofs_m07_i202410092100")
    assert (w["source"], w["member"], w["init"]) == (
        "wofs", "m07", datetime(2024, 10, 9, 21, 0))
    h = st.parse_model("hfsb_c2024100918")
    assert (h["source"], h["member"], h["init"]) == (
        "hfsb", "", datetime(2024, 10, 9, 18))
    assert st.parse_model("something_else") is None


def test_derived_scores_are_empty_when_undefined():
    s = st.derived_scores([0, 0, 0, 50])
    assert all(math.isnan(v) if v is not None else True for v in s.values())
    s = st.derived_scores([20, 10, 10, 60])
    assert s["pod"] == pytest.approx(20 / 30)
    assert s["far"] == pytest.approx(10 / 30)
    assert s["fbias"] == pytest.approx(1.0)


def test_the_csv_paths_sit_in_one_folder_and_carry_the_tags(tmp_path):
    case = _case(tmp_path)
    assert case.stat_csv_dir == tmp_path / "csvs"
    assert case.stat_csv_path("CTC").name == (
        "stat_ctc_2024_02l_beryl_20240708_d1_3km.csv")


# --- a real file --------------------------------------------------------------

def test_time_fields_come_from_the_model_tag_not_the_stat_columns(tmp_path):
    case = _case(tmp_path)
    _install(case)
    out, problems = st.parse_run(case, END, T_WOFS,
                                 case.grid_stat_window_dir(END) / WOFS)
    assert problems == []
    row = out["CTC"][0]
    assert (row["source"], row["member"]) == ("wofs", "m00")
    assert row["init_time"] == "2024-10-09 21:00"
    assert row["valid_start"] == "2024-10-09 21:00"
    assert row["valid_end"] == "2024-10-10 00:00"
    assert row["lead_h"] == "3"             # the file says lead 0, valid 1970
    out, _ = st.parse_run(case, END, T_HFSA,
                          case.grid_stat_window_dir(END) / HFSA)
    assert out["CTC"][0]["lead_h"] == "6"
    assert out["CTC"][0]["member"] == ""


def test_row_counts_and_columns_per_line_type(tmp_path):
    case = _case(tmp_path)
    _install(case)
    out, _ = st.parse_run(case, END, T_WOFS,
                          case.grid_stat_window_dir(END) / WOFS)
    assert {k: len(v) for k, v in out.items()} == {
        "CTC": 11, "CTS": 11, "CNT": 2, "SL1L2": 2, "NBRCTC": 55,
        "NBRCNT": 55}
    for lt, rows in out.items():
        assert set(st.columns(lt)) <= set(rows[0]) | set(), lt


def test_ctc_counts_match_the_stat_file_and_empty_means_na(tmp_path):
    case = _case(tmp_path)
    _install(case)
    out, _ = st.parse_run(case, END, T_WOFS,
                          case.grid_stat_window_dir(END) / WOFS)
    first = out["CTC"][0]
    assert first["thresh_mm"] == "0.5"
    assert [first[k] for k in ("total", "fy_oy", "fy_on", "fn_oy", "fn_on")] \
        == ["60793", "24364", "4032", "5790", "26607"]
    cts = out["CTS"][0]
    assert cts["pod"] == "0.80799" and cts["far"] == "0.14199"
    # HFSA at 150 mm: no forecast events, 38 misses. POD is 0, FAR undefined
    out, _ = st.parse_run(case, END, T_HFSA,
                          case.grid_stat_window_dir(END) / HFSA)
    top = out["CTC"][-1]
    assert top["thresh_mm"] == "150"
    assert top["pod"] == "0" and top["far"] == ""
    assert out["CTS"][-1]["far"] == ""


def test_derived_ctc_scores_equal_mets_cts(tmp_path):
    case = _case(tmp_path)
    _install(case)
    for tag, name in ((T_WOFS, WOFS), (T_HFSA, HFSA)):
        out, _ = st.parse_run(case, END, tag,
                              case.grid_stat_window_dir(END) / name)
        for ctc, cts in zip(out["CTC"], out["CTS"]):
            assert ctc["thresh_mm"] == cts["thresh_mm"]
            for mine, mets in (("pod", "pod"), ("far", "far"),
                               ("csi", "csi"), ("ets", "ets"),
                               ("fbias", "fbias")):
                a, b = _f(ctc[mine]), _f(cts[mets])
                if math.isnan(b):
                    assert math.isnan(a), (ctc["thresh_mm"], mine)
                else:
                    assert a == pytest.approx(b, abs=2e-5), (
                        ctc["thresh_mm"], mine)


def test_sl1l2_sums_agree_with_cnt_for_all_and_wet_cells(tmp_path):
    case = _case(tmp_path)
    _install(case)
    out, _ = st.parse_run(case, END, T_WOFS,
                          case.grid_stat_window_dir(END) / WOFS)
    assert [r["cell_set"] for r in out["CNT"]] == ["all", "wet"]
    assert [r["wet_thresh_mm"] for r in out["SL1L2"]] == ["", "0.5"]
    for cnt, sl in zip(out["CNT"], out["SL1L2"]):
        assert cnt["cell_set"] == sl["cell_set"]
        for k in ("total", "fbar", "obar", "mae"):
            assert _f(cnt[k]) == pytest.approx(_f(sl[k]), rel=1e-5), k
        # mse (CNT) is E[(f-o)^2] = ffbar - 2 fobar + oobar (SL1L2)
        mse = _f(sl["ffbar"]) - 2 * _f(sl["fobar"]) + _f(sl["oobar"])
        assert _f(cnt["mse"]) == pytest.approx(mse, rel=1e-4)
        assert _f(cnt["me"]) == pytest.approx(
            _f(sl["fbar"]) - _f(sl["obar"]), rel=1e-3, abs=1e-4)


def test_neighborhood_rows_carry_width_in_cells_and_km(tmp_path):
    case = _case(tmp_path)
    _install(case)
    out, _ = st.parse_run(case, END, T_WOFS,
                          case.grid_stat_window_dir(END) / WOFS)
    widths = sorted({(int(r["nbr_cells"]), float(r["nbr_width_km"]))
                     for r in out["NBRCTC"]})
    assert widths == [(3, 9.0), (7, 21.0), (11, 33.0), (19, 57.0),
                      (43, 129.0)]
    assert {r["cov_thresh"] for r in out["NBRCTC"]} == {"0.5"}
    fss = [r for r in out["NBRCNT"] if r["nbr_cells"] == 3
           and r["thresh_mm"] == "0.5"][0]
    assert fss["fss"] == "0.85628" and fss["total"] == "60296"


# --- the command --------------------------------------------------------------

def test_the_command_writes_one_csv_per_line_type(tmp_path, capsys):
    case = _case(tmp_path)
    _install(case)
    paths = st.parse_stat_case(case)
    assert sorted(paths) == sorted(st.LINE_TYPES)
    for lt, path in paths.items():
        assert path.parent == case.stat_csv_dir
        assert path.name == f"stat_{lt.lower()}_{case.case_slug}_3km.csv"
        rows = _read(path)
        assert list(rows[0]) == st.columns(lt)
        assert {r["source"] for r in rows} == {"wofs", "hfsa"}
    assert len(_read(paths["CTC"])) == 22
    assert len(_read(paths["NBRCNT"])) == 110
    out = capsys.readouterr().out
    assert "2 .stat files" in out and "NBRCTC" in out


def test_without_the_runs_csv_the_work_list_is_the_checklist(
        tmp_path, monkeypatch):
    case = _case(tmp_path)
    _install(case)
    case.grid_stat_runs_path.unlink()
    Job = namedtuple("Job", "window tag")
    Win = namedtuple("Win", "end")
    monkeypatch.setattr(st.gw, "work_list", lambda c: (
        [Job(Win(END), T_WOFS), Job(Win(END), T_HFSA)], []))
    runs, source = st.checklist(case)
    assert runs == sorted([(END, T_WOFS), (END, T_HFSA)])
    assert source == "the work list"
    assert len(st.parse_stat_case(case)) == 6


def test_a_missing_stat_file_is_listed_and_the_rest_still_written(
        tmp_path):
    case = _case(tmp_path)
    _install(case)
    _write_runs(case, [(END, T_WOFS), (END, T_HFSA),
                       (END, "wofs_m01_i202410092100")])
    with pytest.raises(SystemExit) as exc:
        st.parse_stat_case(case)
    assert "1 runs with no .stat file" in str(exc.value)
    assert "wofs_m01_i202410092100" in str(exc.value)
    assert len(_read(case.stat_csv_path("CTC"))) == 22


def test_a_truncated_file_is_a_row_count_problem(tmp_path):
    case = _case(tmp_path)
    _install(case)
    path = case.grid_stat_window_dir(END) / HFSA
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-3]))
    with pytest.raises(SystemExit) as exc:
        st.parse_stat_case(case)
    assert "rows differ from the configuration" in str(exc.value)
    assert "NBRCNT" in str(exc.value)


def test_a_file_under_the_wrong_run_name_is_refused(tmp_path):
    case = _case(tmp_path)
    _install(case)
    out = case.grid_stat_window_dir(END)
    shutil.copy(FIX / WOFS, out / WOFS.replace("m00", "m01"))
    _write_runs(case, [(END, "wofs_m01_i202410092100")])
    with pytest.raises(SystemExit) as exc:
        st.parse_stat_case(case)
    assert "MODEL tag" in str(exc.value)


def test_a_run_listed_twice_is_a_duplicate_key(tmp_path):
    case = _case(tmp_path)
    _install(case)
    _write_runs(case, [(END, T_WOFS), (END, T_WOFS)])
    with pytest.raises(SystemExit) as exc:
        st.parse_stat_case(case)
    assert "duplicate key" in str(exc.value)


def test_other_resolution_files_are_not_picked_up(tmp_path):
    """A default-resolution case (no suffix) must not read the _3km .stat of
    the same run, and the 3 km case must find it."""
    six = make_case(tmp_path)
    out = six.grid_stat_window_dir(END)
    out.mkdir(parents=True)
    shutil.copy(FIX / WOFS, out / WOFS)            # a _3km file
    assert st.stat_file(six, END, T_WOFS) is None
    assert st.stat_file(_case(tmp_path), END, T_WOFS).name == WOFS


def test_it_is_a_command():
    import run
    assert run.parse_args(["c.yaml", "parse-stat"])[1] == "parse-stat"
