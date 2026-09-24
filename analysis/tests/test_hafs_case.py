"""Local unit tests for hafs_case ATCF parsing (no Hercules data needed).

Run directly:   python3 analysis/tests/test_hafs_case.py
Or via pytest:  pytest analysis/tests/test_hafs_case.py -v
"""
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Make analysis/ importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIX = Path(__file__).resolve().parent / "fixtures"

import numpy as np
from hafs_case import (decode_latlon, parse_atcfunix,
                       parse_atcfunix_fixes, normalize_storm_id,
                       detect_model, auto_domain, StormCase,
                       from_yaml, find_atcfunix,
                       position_on_track, filter_inits,
                       cycles_from_yaml)


def test_decode_latlon():
    assert decode_latlon("168N") == 16.8
    assert decode_latlon("832W") == -83.2
    assert decode_latlon("105S") == -10.5
    assert decode_latlon("50E") == 5.0


def test_parse_atcfunix_reproduces_helene_track():
    name, init_dt, track = parse_atcfunix(FIX / "helene.atcfunix")
    assert name == "Helene"
    assert init_dt == datetime(2024, 9, 24, 0)
    # First three known 6-hourly fixes from the current hardcoded TC_TRACK_6H.
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    # Sorted ascending by valid time, no duplicate lead hours.
    times = [t for t, _, _ in track]
    assert times == sorted(times)
    assert len(times) == len(set(times))


def test_detect_model():
    assert detect_model("/work2/.../helene/HFSA") == "HAFS-A"
    assert detect_model("/work2/.../helene/HFSB") == "HAFS-B"
    assert detect_model("/data/hfsa_run/lower") == "HAFS-A"   # case-insensitive
    assert detect_model("/work2/.../helene/other") == "HAFS"
    # HAFS-M (experimental multistorm): the 'hfsb_multistorm' file tag contains
    # 'HFSB', so it must resolve to HAFS-M, not HAFS-B.
    assert detect_model("/work2/.../helene/HFSM") == "HAFS-M"
    assert detect_model("/data/hfsb_multistorm/parent") == "HAFS-M"
    assert detect_model(
        "/runs/00l.2024092412.hfsb_multistorm.parent.atm.f000.grb2"
    ) == "HAFS-M"


def test_auto_domain_pads_track_bbox():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 26, 0), 28.8, -84.1),
        (datetime(2024, 9, 29, 6), 44.3, -61.5),
    ]
    lat_min, lat_max, lon_min, lon_max = auto_domain(track, pad_deg=2.0)
    assert lat_min == 14.8 and lat_max == 46.3
    assert lon_min == -86.1 and lon_max == -59.5


def _toy_case():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 24, 6), 17.8, -83.5),
    ]
    return StormCase(
        run_dir=Path("/tmp/HFSA"), init_dt=datetime(2024, 9, 24, 0),
        storm_name="Helene", model_label="HAFS-A",
        domain=(15.0, 20.0, -90.0, -80.0), grid_res=1.0,
        mask_radius_km=500.0, display_radius_km=750.0,
        thresholds_mm=[1, 5], out_dir=Path("/tmp/out"),
        mrms_cache_dir=Path("/tmp/mrms"), stage4_cache_dir=Path("/tmp/s4"),
        fhours_filter=None, track=track, case_slug="helene_hfsa",
        init_str="2024092400",
    )


def test_position_on_track_interpolates_and_clamps():
    track = [
        (datetime(2024, 9, 24, 0), 16.8, -83.2),
        (datetime(2024, 9, 24, 6), 17.8, -83.5),
    ]
    lat, lon = position_on_track(track, datetime(2024, 9, 24, 3))
    assert abs(lat - 17.3) < 1e-9 and abs(lon - (-83.35)) < 1e-9
    assert position_on_track(track, datetime(2024, 9, 23, 0)) == (16.8, -83.2)
    assert position_on_track(track, datetime(2024, 9, 25, 0)) == (17.8, -83.5)


def test_position_at_interpolates_and_clamps():
    c = _toy_case()
    # Midpoint between the two 6-hourly fixes.
    lat, lon = c.position_at(datetime(2024, 9, 24, 3))
    assert abs(lat - 17.3) < 1e-9 and abs(lon - (-83.35)) < 1e-9
    # Before/after the track clamps to the endpoints.
    assert c.position_at(datetime(2024, 9, 23, 0)) == (16.8, -83.2)
    assert c.position_at(datetime(2024, 9, 25, 0)) == (17.8, -83.5)


def test_fixed_grid_shape():
    c = _toy_case()
    grid_lat, grid_lon = c.fixed_grid()
    # lon -90..-80 step 1 -> 11 cols; lat 15..20 step 1 -> 6 rows.
    assert grid_lat.shape == (6, 11)
    assert grid_lon.shape == (6, 11)


def test_globs_use_init_str():
    c = _toy_case()
    assert c.parent_glob() == "**/*2024092400*parent.atm.f*.grb2"
    assert c.storm_glob() == "**/*2024092400*storm.atm.f*.grb2"


def test_output_slug_appends_init():
    c = _toy_case()
    assert c.case_slug == "helene_hfsa"
    assert c.output_slug == "helene_hfsa_2024092400"


def test_output_slug_dedups_when_init_already_in_slug():
    import dataclasses
    c = dataclasses.replace(_toy_case(), case_slug="helene_hfsa_2024092400")
    assert c.output_slug == "helene_hfsa_2024092400"


def test_from_yaml_minimal_autoderives():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "helene" / "HFSA"
        run_dir.mkdir(parents=True)
        shutil.copy(FIX / "helene.atcfunix", run_dir / "helene.atcfunix")
        yaml_path = tmpdir / "helene_hfsa.yaml"
        yaml_path.write_text(f"run_dir: {run_dir}\n")

        case = from_yaml(yaml_path)

        assert case.case_slug == "helene_hfsa"
        assert case.model_label == "HAFS-A"          # auto from path
        assert case.init_dt == datetime(2024, 9, 24, 0)  # from atcfunix
        assert case.storm_name == "Helene"            # from atcfunix name field
        assert case.init_str == "2024092400"
        assert case.grid_res == 0.05                  # default
        assert case.mask_radius_km == 500.0           # default
        assert case.thresholds_mm == [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]
        assert case.out_dir == Path("analysis/output/helene_hfsa")
        # Domain auto-derived from the track bbox (non-empty, sane ordering).
        lat_min, lat_max, lon_min, lon_max = case.domain
        assert lat_min < lat_max and lon_min < lon_max
    finally:
        shutil.rmtree(tmpdir)


def test_from_yaml_overrides_win():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "HFSA"
        run_dir.mkdir(parents=True)
        shutil.copy(FIX / "helene.atcfunix", run_dir / "x.atcfunix")
        yaml_path = tmpdir / "case.yaml"
        yaml_path.write_text(
            f"run_dir: {run_dir}\n"
            "storm_name: Test Storm\n"
            "model_label: HAFS-X\n"
            "domain: [15.0, 42.0, -100.0, -60.0]\n"
            "mask_radius_km: 300\n"
            "out_dir: /tmp/custom_out\n"
        )
        case = from_yaml(yaml_path)
        assert case.storm_name == "Test Storm"
        assert case.model_label == "HAFS-X"
        assert case.domain == (15.0, 42.0, -100.0, -60.0)
        assert case.mask_radius_km == 300.0
        assert case.out_dir == Path("/tmp/custom_out")
    finally:
        shutil.rmtree(tmpdir)


def test_find_atcfunix_missing_raises():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        try:
            find_atcfunix(tmpdir)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert str(tmpdir) in str(e)
    finally:
        shutil.rmtree(tmpdir)


def test_parse_atcfunix_realistic_wind_radii():
    """Fix 2: storm name found even when wind-radii columns push it past index 27."""
    name, init_dt, track = parse_atcfunix(FIX / "helene_realistic.atcfunix")
    assert name == "Helene"
    assert init_dt == datetime(2024, 9, 24, 0)
    # Same first three track points as the simple fixture (deduped by TAU).
    assert track[0] == (datetime(2024, 9, 24, 0), 16.8, -83.2)
    assert track[1] == (datetime(2024, 9, 24, 6), 17.8, -83.5)
    assert track[2] == (datetime(2024, 9, 24, 12), 19.0, -83.8)
    assert len(track) == 5


def test_find_atcfunix_skips_aggregate():
    """Fix 1: with aggregate (00L) + real (09L), find_atcfunix picks the real one."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Aggregate track (cyclone 00)
        agg = tmpdir / "hafs.2024092400.00l.atcfunix"
        agg.write_text(
            "AL, 00, 2024092400, 03, HFSA, 000, 168N, 832W, 65, 985\n"
        )
        # Real track (cyclone 09)
        real = tmpdir / "hafs.2024092400.09l.atcfunix"
        real.write_text(
            "AL, 09, 2024092400, 03, HFSA, 000, 168N, 832W, 65, 985\n"
        )
        chosen = find_atcfunix(tmpdir)
        assert chosen == real, f"Expected {real}, got {chosen}"
    finally:
        shutil.rmtree(tmpdir)


def test_find_atcfunix_selects_configured_init():
    """A storm-level run root may contain tracks from many cycles."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        for init in ("2024092400", "2024092412"):
            cycle = tmpdir / init
            cycle.mkdir()
            (cycle / f"hafs.{init}.09l.atcfunix").write_text(
                f"AL, 09, {init}, 03, HFSA, 000, 168N, 832W, 65, 985\n"
            )
        chosen = find_atcfunix(tmpdir, "2024092412")
        assert "2024092412" in str(chosen)
    finally:
        shutil.rmtree(tmpdir)


def test_from_yaml_explicit_atcfunix():
    """Fix 1: YAML 'atcfunix' key overrides auto-discovery."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "helene" / "HFSA"
        run_dir.mkdir(parents=True)
        # Put the fixture in a non-standard location.
        target = run_dir / "subdir"
        target.mkdir()
        shutil.copy(FIX / "helene.atcfunix", target / "my_track.atcfunix")
        yaml_path = tmpdir / "explicit.yaml"
        yaml_path.write_text(
            f"run_dir: {run_dir}\n"
            f"atcfunix: subdir/my_track.atcfunix\n"
        )
        case = from_yaml(yaml_path)
        assert case.storm_name == "Helene"
        assert case.init_dt == datetime(2024, 9, 24, 0)
        assert len(case.track) == 5
    finally:
        shutil.rmtree(tmpdir)


def test_from_yaml_empty_track_raises():
    """Fix 4: atcfunix with no valid data lines raises ValueError."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        run_dir = tmpdir / "HFSA"
        run_dir.mkdir(parents=True)
        bad = run_dir / "bad.atcfunix"
        bad.write_text("# no valid data lines here\ngarbage,line\n")
        yaml_path = tmpdir / "bad_case.yaml"
        yaml_path.write_text(f"run_dir: {run_dir}\n")
        try:
            from_yaml(yaml_path)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "0 track fixes" in str(e)
    finally:
        shutil.rmtree(tmpdir)


# A multistorm parent.trak.atcfunix.all: Helene (AL 09) interleaved with a
# concurrent Pacific storm (EP 10), the way the real hfsb_multistorm file is.
_MULTISTORM_ALL = (
    "AL, 09, 2024092400, 03, HFSB, 000, 179N,  818W,  31, 1002\n"
    "EP, 10, 2024092400, 03, HFSB, 000, 161N,  990W,  52,  989\n"
    "AL, 09, 2024092400, 03, HFSB, 003, 185N,  823W,  36, 1003\n"
    "EP, 10, 2024092400, 03, HFSB, 003, 165N,  995W,  55,  985\n"
)


def test_normalize_storm_id_forms():
    assert normalize_storm_id("AL09") == ("AL", 9)
    assert normalize_storm_id("al9") == ("AL", 9)
    assert normalize_storm_id("09L") == ("AL", 9)
    assert normalize_storm_id("AL, 09") == ("AL", 9)
    assert normalize_storm_id("EP10") == ("EP", 10)
    assert normalize_storm_id(None) is None
    assert normalize_storm_id("garbage") is None


def test_parse_atcfunix_filters_multistorm_to_target():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        trk = tmpdir / "00l.2024092400.hfsb_multistorm.parent.trak.atcfunix.all"
        trk.write_text(_MULTISTORM_ALL)
        # Helene only: two fixes, both Atlantic (negative lon near 82W).
        _, _, hel = parse_atcfunix(trk, "AL09")
        assert len(hel) == 2
        assert hel[0][1] == np.float64(17.9) or abs(hel[0][1] - 17.9) < 1e-9
        assert abs(hel[0][2] - (-81.8)) < 1e-9   # 818W
        assert abs(hel[1][1] - 18.5) < 1e-9      # f003 lat
        # The Pacific storm must be excluded entirely.
        assert all(lon < -50 for _, _, lon in hel)
        # Targeting the other storm isolates it instead.
        _, _, john = parse_atcfunix(trk, "EP10")
        assert len(john) == 2
        assert abs(john[0][2] - (-99.0)) < 1e-9  # 990W
    finally:
        shutil.rmtree(tmpdir)


def test_find_atcfunix_falls_back_to_all_for_multistorm():
    """No bare *.atcfunix (multistorm): use the parent *.atcfunix.all, not the
    .orig backup or the combined track, even though it is a 00l aggregate."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        base = "00l.2024092400.hfsb_multistorm"
        for suffix in (
            "parent.trak.atcfunix.all",
            "parent.trak.atcfunix.all.orig",
            "trak.atcfunix.all",
            "parent.trak.atcfunix.f000",
        ):
            (tmpdir / f"{base}.{suffix}").write_text(_MULTISTORM_ALL)
        chosen = find_atcfunix(tmpdir)
        assert chosen.name == f"{base}.parent.trak.atcfunix.all", chosen
    finally:
        shutil.rmtree(tmpdir)


_INITS = ["2024070500", "2024070512", "2024070600", "2024070812",
          "2024071000", "2024071100"]


def test_filter_inits_unbounded_is_a_passthrough():
    assert filter_inits(_INITS) == _INITS


def test_filter_inits_bounds_are_inclusive():
    kept = filter_inits(_INITS, datetime(2024, 7, 5, 12),
                        datetime(2024, 7, 10, 0))
    assert kept == ["2024070512", "2024070600", "2024070812", "2024071000"]


def test_filter_inits_drops_unparseable_names():
    assert filter_inits(["2024070500", "latest", "scratch"],
                        datetime(2024, 7, 1), None) == ["2024070500"]


def test_cycles_from_yaml_init_range():
    import tempfile
    import yaml as _yaml
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "HFSA"
        root.mkdir()
        base = {"run_root": str(root), "valid_start": 2024070800,
                "valid_end": 2024071000, "domain": [20.0, 40.0, -95.0, -80.0],
                "model_label": "HFSA"}
        path = Path(tmp) / "cycles.yaml"
        path.write_text(_yaml.safe_dump(base))
        ccase = cycles_from_yaml(path)
        assert ccase.init_start is None and ccase.init_end is None

        path.write_text(_yaml.safe_dump({**base, "init_start": 2024070500}))
        ccase = cycles_from_yaml(path)
        assert ccase.init_start == datetime(2024, 7, 5, 0)
        assert ccase.init_end == datetime(2024, 7, 10, 0)  # defaults to window

        path.write_text(_yaml.safe_dump({**base, "init_start": 2024070500,
                                         "init_end": 2024070400}))
        try:
            cycles_from_yaml(path)
        except ValueError:
            return
        raise AssertionError("expected ValueError for init_end < init_start")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
