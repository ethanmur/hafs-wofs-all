"""Unit tests for fetch_bdecks' storm-identifier parsing and storm lists.

Run directly:   python3 analysis/tests/test_fetch_bdecks.py
Or via pytest:  pytest analysis/tests/test_fetch_bdecks.py -v
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fetch_bdecks import bdeck_name, bdeck_year, read_storm_list


def test_bdeck_name_accepts_every_spelling():
    for token in ("2024-02l", "2024-2l", "2024-02L",
                  "AL022024", "al022024", "bal022024.dat"):
        assert bdeck_name(token) == "bal022024.dat", token


def test_bdeck_name_maps_basins():
    assert bdeck_name("2023-10l") == "bal102023.dat"
    assert bdeck_name("2023-10e") == "bep102023.dat"
    assert bdeck_name("2023-10c") == "bcp102023.dat"
    assert bdeck_name("2023-10w") == "bwp102023.dat"


def test_bdeck_name_rejects_junk():
    for bad in ("beryl", "2024-02x", "24-02l", "", "AL0220"):
        try:
            bdeck_name(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_bdeck_year_from_filename():
    # the archive subdirectory must come from the b-deck, not the token, so
    # that ATCF-style ids like AL022024 resolve to the right year
    assert bdeck_year("bal022024.dat") == "2024"
    assert bdeck_year("bep102023.dat") == "2023"


def test_read_storm_list_skips_comments_and_blanks():
    tmp = Path(tempfile.mkdtemp()) / "storms.txt"
    tmp.write_text("# Atlantic\n2024-02l\n\nAL092024  # Helene\n\n")
    assert read_storm_list(tmp) == ["2024-02l", "AL092024"]


def test_read_storm_list_empty_raises():
    tmp = Path(tempfile.mkdtemp()) / "storms.txt"
    tmp.write_text("# nothing here\n\n")
    try:
        read_storm_list(tmp)
    except ValueError:
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
