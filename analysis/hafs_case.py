"""StormCase config + ATCF track parsing for the HAFS QPF/ETS framework.

Dependency-light on purpose (stdlib + numpy + yaml only) so it imports and
tests off-Hercules, away from cfgrib/boto3/eccodes/cartopy.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml


def decode_latlon(token):
    """ATCF tenths-of-degree + hemisphere token -> signed float degrees.

    '168N' -> 16.8, '832W' -> -83.2, '105S' -> -10.5, '50E' -> 5.0.
    """
    token = token.strip()
    hemi = token[-1].upper()
    value = int(token[:-1]) / 10.0
    if hemi in ("S", "W"):
        value = -value
    return value


_ATCF_STOPLIST = frozenset({
    "NEQ", "SEQ", "AAA", "XX", "L", "M", "D", "S", "N", "E", "W",
    "TS", "HU", "TD", "DB", "SD", "SS", "EX", "LO", "WV", "TY", "ST",
})


def _extract_storm_name(cols):
    """Scan columns from the end for the storm name.

    Returns the first (from end) all-alpha token of length >= 3 that is
    NOT a known ATCF code word, title-cased.  Returns None if not found.
    """
    for token in reversed(cols):
        if len(token) >= 3 and re.fullmatch(r"[A-Za-z]+", token):
            if token.upper() not in _ATCF_STOPLIST:
                return token.title()
    return None


def _atcf_value(cols, index):
    """Optional positive ATCF numeric value, with sentinels removed."""
    try:
        value = float(cols[index])
    except (IndexError, TypeError, ValueError):
        return None
    return value if value not in (0.0, -99.0) else None


# ATCF single-letter basin suffix (as used in filenames like '09l') -> the
# two-letter basin code that appears in the file's first column ('AL').
_BASIN_BY_LETTER = {"L": "AL", "E": "EP", "C": "CP", "W": "WP"}


def normalize_storm_id(storm_id):
    """('AL', 9) from 'AL09'/'al9'/'09L'/'9l'/'AL, 09'; None if unset/bad.

    Used to isolate one cyclone in a multistorm .atcfunix.all that carries
    several storms (e.g. Helene 'AL, 09' alongside 'EP, 10'), where the file
    lines never spell the storm name.
    """
    if not storm_id:
        return None
    s = str(storm_id).replace(",", "").replace(" ", "").upper()
    m = re.fullmatch(r"([A-Z]{2})(\d{1,2})", s)          # AL09
    if m:
        return m.group(1), int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})([A-Z])", s)             # 09L
    if m:
        return _BASIN_BY_LETTER.get(m.group(2), m.group(2)), int(m.group(1))
    return None


def parse_atcfunix_fixes(path, storm_id=None):
    """Parse full position and intensity fixes from a HAFS .atcfunix file.

    Returns (storm_name_or_None, init_dt, fixes) where each fix is
    (valid_dt, lat, lon, vmax_kt_or_None, mslp_hpa_or_None), deduped by lead
    hour (TAU) with the first line winning and sorted ascending.

    When ``storm_id`` is given (e.g. 'AL09'), only lines for that basin +
    cyclone number are kept — required for multistorm tracks that interleave
    several storms in one file.
    """
    target = normalize_storm_id(storm_id)
    init_dt = None
    name = None
    by_tau = {}
    with open(path) as fh:
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 8:
                continue
            if target is not None:
                try:
                    if (cols[0], int(cols[1])) != target:
                        continue
                except (ValueError, IndexError):
                    continue
            try:
                warn = datetime.strptime(cols[2], "%Y%m%d%H")
                tau = int(cols[5])
                lat = decode_latlon(cols[6])
                lon = decode_latlon(cols[7])
            except (ValueError, IndexError):
                continue
            if init_dt is None:
                init_dt = warn
            if tau not in by_tau:
                by_tau[tau] = (warn + timedelta(hours=tau), lat, lon,
                               _atcf_value(cols, 8), _atcf_value(cols, 9))
            # Storm name: robust reverse scan for last all-alpha token.
            if name is None:
                name = _extract_storm_name(cols)
    fixes = [by_tau[t] for t in sorted(by_tau)]
    return name, init_dt, fixes


def parse_atcfunix(path, storm_id=None):
    """Parse position fixes from a HAFS .atcfunix track file."""
    name, init_dt, fixes = parse_atcfunix_fixes(path, storm_id)
    return name, init_dt, [(t, lat, lon) for t, lat, lon, _, _ in fixes]


def detect_model(run_dir):
    """'HAFS-A'/'HAFS-B'/'HAFS-M'/'HAFS' from the model tag in the run-dir path.

    HAFS-M is the experimental multistorm run. Its files carry an
    'hfsb_multistorm' tag (e.g. 00l.2024092412.hfsb_multistorm.parent.atm.
    f000.grb2), so 'multistorm'/'hfsm' must be tested before 'hfsb' — the
    multistorm tag contains the 'HFSB' substring and would otherwise be
    misread as HAFS-B.
    """
    s = str(run_dir).upper()
    if "MULTISTORM" in s or "HFSM" in s:
        return "HAFS-M"
    if "HFSA" in s:
        return "HAFS-A"
    if "HFSB" in s:
        return "HAFS-B"
    return "HAFS"


def auto_domain(track, pad_deg=2.0):
    """(lat_min, lat_max, lon_min, lon_max) = padded bbox of the track."""
    lats = [la for _, la, _ in track]
    lons = [lo for _, _, lo in track]
    return (min(lats) - pad_deg, max(lats) + pad_deg,
            min(lons) - pad_deg, max(lons) + pad_deg)


def position_on_track(track, valid_dt):
    """Linear interpolation of a [(dt, lat, lon), ...] track to any time.

    Clamps to the endpoints outside the track's time span.
    """
    times = [t for t, _, _ in track]
    lats = [la for _, la, _ in track]
    lons = [lo for _, _, lo in track]
    if valid_dt <= times[0]:
        return lats[0], lons[0]
    if valid_dt >= times[-1]:
        return lats[-1], lons[-1]
    for i in range(len(times) - 1):
        if times[i] <= valid_dt <= times[i + 1]:
            frac = ((valid_dt - times[i]).total_seconds()
                    / (times[i + 1] - times[i]).total_seconds())
            return (lats[i] + frac * (lats[i + 1] - lats[i]),
                    lons[i] + frac * (lons[i + 1] - lons[i]))
    return lats[-1], lons[-1]


def make_fixed_grid(domain, grid_res):
    """Fixed lat/lon verification/plot mesh from a domain + resolution."""
    lat_min, lat_max, lon_min, lon_max = domain
    fixed_lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    fixed_lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    return np.meshgrid(fixed_lons, fixed_lats)[::-1]  # (grid_lat, grid_lon)


@dataclass
class StormCase:
    run_dir: Path
    init_dt: datetime
    storm_name: str
    model_label: str
    domain: tuple          # (lat_min, lat_max, lon_min, lon_max)
    grid_res: float
    mask_radius_km: float
    display_radius_km: float
    thresholds_mm: list
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    fhours_filter: list
    track: list            # [(valid_dt, lat, lon), ...]
    case_slug: str
    init_str: str
    track_fixes: list = None
    storm_id: str = None   # e.g. 'AL09'; isolates one storm in a multistorm track

    def position_at(self, valid_dt):
        """Linear interpolation of the track to any time; clamps to endpoints."""
        return position_on_track(self.track, valid_dt)

    def fixed_grid(self):
        """Fixed lat/lon verification/plot mesh from domain + grid_res."""
        return make_fixed_grid(self.domain, self.grid_res)

    def parent_glob(self):
        return f"**/*{self.init_str}*parent.atm.f*.grb2"

    def storm_glob(self):
        return f"**/*{self.init_str}*storm.atm.f*.grb2"

    @property
    def output_slug(self):
        """case_slug with the init appended for output filenames, de-duplicated.

        'helene_hfsa' -> 'helene_hfsa_2024092400'; returned unchanged if the
        slug already contains the init string (YAML was named with it).
        """
        if self.init_str in self.case_slug:
            return self.case_slug
        return f"{self.case_slug}_{self.init_str}"


_DEFAULT_THRESHOLDS = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]


def _is_aggregate_track(path):
    """True if *path* looks like a cyclone-00 aggregate track.

    Checks the filename for '00l'/'00e'/'00c'/'00w' (case-insensitive),
    then falls back to reading the first data line's cyclone-number field.
    """
    fname = path.name.lower()
    if re.search(r"00[lecw]", fname):
        return True
    try:
        with open(path) as fh:
            for line in fh:
                cols = [c.strip() for c in line.split(",")]
                if len(cols) >= 2:
                    return cols[1] == "00"
    except OSError:
        pass
    return False


def find_atcfunix(run_dir, init_str=None):
    """Best *.atcfunix under run_dir for an optional initialization.

    When ``init_str`` is supplied, candidates are restricted to paths or track
    contents matching that cycle before aggregate (cyclone 00) tracks are
    deprioritized. Raises FileNotFoundError naming the directory and cycle when
    no file matches.
    """
    hits = sorted(Path(run_dir).glob("**/*.atcfunix"))
    if not hits:
        # Multistorm/experimental runs emit no bare *.atcfunix — only the
        # concatenated *.atcfunix.all (plus per-lead .fNNN and .all.orig
        # backups). Fall back to the complete .all track; the .orig backups
        # end in '.orig' and are excluded by the glob.
        hits = sorted(Path(run_dir).glob("**/*.atcfunix.all"))
    if not hits:
        raise FileNotFoundError(
            f"No .atcfunix or .atcfunix.all track file under {run_dir} "
            f"(globs '**/*.atcfunix', '**/*.atcfunix.all'). Add an explicit "
            f"'track', 'init', and 'domain' to the YAML to run without one."
        )
    if init_str is not None:
        init_str = str(init_str)
        matching = [p for p in hits if init_str in str(p)]
        if not matching:
            for path in hits:
                try:
                    _, track_init, _ = parse_atcfunix(path)
                except OSError:
                    continue
                if track_init and track_init.strftime("%Y%m%d%H") == init_str:
                    matching.append(path)
        if not matching:
            raise FileNotFoundError(
                f"No .atcfunix track for init {init_str} under {run_dir}."
            )
        hits = matching
    if len(hits) == 1:
        print(f"Using track file: {hits[0]}")
        return hits[0]
    # Multiple files: deprioritize aggregate (cyclone 00) tracks.
    real = [p for p in hits if not _is_aggregate_track(p)]
    if real:
        skipped = len(hits) - len(real)
        print(f"Found {len(hits)} atcfunix files; skipped {skipped} "
              f"aggregate track(s).")
        choice = real[0]
    else:
        print(f"Found {len(hits)} atcfunix files (all appear aggregate); "
              f"using first sorted.")
        choice = hits[0]
    print(f"Using track file: {choice}")
    return choice


def from_yaml(yaml_path):
    """Load a StormCase from a YAML case file (run_dir required)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if "run_dir" not in cfg:
        if "run_root" in cfg:
            raise KeyError(
                f"{yaml_path} looks like a cycles YAML (has 'run_root') — "
                f"run it with the 'cycles' command."
            )
        raise KeyError(f"'run_dir' is required in {yaml_path}")
    run_dir = Path(cfg["run_dir"])
    configured_init = (datetime.strptime(str(cfg["init"]), "%Y%m%d%H")
                       if cfg.get("init") else None)
    configured_init_str = (configured_init.strftime("%Y%m%d%H")
                           if configured_init else None)

    # Explicit atcfunix path from YAML, or auto-discover.
    if cfg.get("atcfunix"):
        atcf_path = Path(cfg["atcfunix"])
        if not atcf_path.is_absolute():
            atcf_path = run_dir / atcf_path
        if not atcf_path.exists():
            raise FileNotFoundError(
                f"Explicit atcfunix path does not exist: {atcf_path}"
            )
        print(f"Using track file: {atcf_path}")
    else:
        atcf_path = find_atcfunix(run_dir, configured_init_str)
    storm_id = cfg.get("storm_id")
    name, init_from_atcf, track = parse_atcfunix(atcf_path, storm_id)

    if not track:
        hint = (f" for storm_id {storm_id!r}" if storm_id else "")
        raise ValueError(
            f"Parsed 0 track fixes from {atcf_path}{hint}; "
            f"cannot derive domain/position."
        )

    # init: YAML override (YYYYMMDDHH) else from atcfunix.
    init_dt = configured_init or init_from_atcf
    init_str = init_dt.strftime("%Y%m%d%H")

    domain = tuple(cfg["domain"]) if cfg.get("domain") else auto_domain(track)
    out_dir = (Path(cfg["out_dir"]) if cfg.get("out_dir")
               else Path("analysis/output") / yaml_path.stem)

    return StormCase(
        run_dir=run_dir,
        init_dt=init_dt,
        storm_name=(cfg["storm_name"] if "storm_name" in cfg
                    else (name or "Storm")),
        model_label=(cfg["model_label"] if "model_label" in cfg
                     else detect_model(run_dir)),
        domain=domain,
        grid_res=float(cfg.get("grid_res", 0.05)),
        mask_radius_km=float(cfg.get("mask_radius_km", 500.0)),
        display_radius_km=float(cfg.get("display_radius_km", 750.0)),
        thresholds_mm=cfg.get("thresholds_mm", list(_DEFAULT_THRESHOLDS)),
        out_dir=out_dir,
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        stage4_cache_dir=Path(cfg.get("stage4_cache_dir", "/tmp/stage4_cache")),
        fhours_filter=cfg.get("fhours"),
        track=track,
        case_slug=yaml_path.stem,
        init_str=init_str,
        storm_id=storm_id,
    )


# =============================================================================
# Cycle comparison: config + window math
# =============================================================================

_INIT_DIR_RE = re.compile(r"^\d{10}$")


@dataclass
class CyclesCase:
    """One storm x model across initializations, scored on a common window."""
    run_root: Path
    valid_start: datetime
    valid_end: datetime
    storm_name: str
    model_label: str
    domain: tuple          # (lat_min, lat_max, lon_min, lon_max)
    grid_res: float
    mask_radius_km: float
    display_radius_km: float
    thresholds_mm: list
    ets_threshold_mm: float
    out_dir: Path
    mrms_cache_dir: Path
    stage4_cache_dir: Path
    inits: list            # explicit YYYYMMDDHH strings, or None to discover
    case_slug: str
    landfall_time: datetime = None
    ets_bar_thresholds_in: list = field(
        default_factory=lambda: list(range(2, 25, 2)))
    fss_thresholds_in: list = field(default_factory=lambda: [1.0, 2.0])
    fss_scales_cells: list = field(default_factory=lambda: [1, 3, 5, 11, 21, 41])
    make_animation: bool = True
    best_track: Path = None
    track_step_hours: int = 6
    headline_fss_threshold_in: float = None
    headline_fss_scale_cells: int = None
    storm_id: str = None   # e.g. 'AL09'; isolates one storm in a multistorm track

    def fixed_grid(self):
        """Fixed lat/lon verification/plot mesh from domain + grid_res."""
        return make_fixed_grid(self.domain, self.grid_res)

    @property
    def output_slug(self):
        """YAML stem + window, so different windows never overwrite output."""
        return (f"{self.case_slug}_{self.valid_start:%Y%m%d%H}_"
                f"{self.valid_end:%Y%m%d%H}")


def discover_inits(run_root):
    """Sorted YYYYMMDDHH-named subdirectories of run_root."""
    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError(f"run_root does not exist: {root}")
    inits = []
    for p in root.iterdir():
        if not (p.is_dir() and _INIT_DIR_RE.fullmatch(p.name)):
            continue
        try:
            datetime.strptime(p.name, "%Y%m%d%H")
        except ValueError:
            continue
        inits.append(p.name)
    return sorted(inits)


def window_hours(init_dt, valid_start, valid_end):
    """Express an absolute valid window as forecast hours of one cycle."""
    f1 = (valid_start - init_dt).total_seconds() / 3600.0
    f2 = (valid_end - init_dt).total_seconds() / 3600.0
    return int(round(f1)), int(round(f2))


def cycle_eligibility(init_dt, max_fhour, valid_start, valid_end):
    """Eligibility for an init-clipped window ending at ``valid_end``.

    Cycles initialized after ``valid_start`` begin verification at their own
    initialization time. Every retained run must still reach the common end.
    """
    if init_dt >= valid_end:
        return False, (f"init {init_dt:%Y-%m-%d %HZ} is at or after the window "
                       f"end {valid_end:%Y-%m-%d %HZ}")
    run_end = init_dt + timedelta(hours=max_fhour)
    if run_end < valid_end:
        return False, (f"run ends {run_end:%Y-%m-%d %HZ}, before the "
                       f"window end {valid_end:%Y-%m-%d %HZ}")
    return True, ""


def cycles_from_yaml(yaml_path):
    """Load a CyclesCase from a cycles YAML (run_root/valid_start/valid_end
    and domain required)."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if "run_root" not in cfg:
        hint = (" This looks like a per-init case YAML (has 'run_dir'); "
                "the 'cycles' command needs a cycles YAML with 'run_root'."
                if "run_dir" in cfg else "")
        raise KeyError(
            f"'run_root' is required in a cycles YAML ({yaml_path}).{hint}")
    for key in ("valid_start", "valid_end", "domain"):
        if key not in cfg:
            raise KeyError(f"'{key}' is required in a cycles YAML "
                           f"({yaml_path})")
    valid_start = datetime.strptime(str(cfg["valid_start"]), "%Y%m%d%H")
    valid_end = datetime.strptime(str(cfg["valid_end"]), "%Y%m%d%H")
    if valid_end <= valid_start:
        raise ValueError(
            f"valid_end must be after valid_start in {yaml_path}")
    run_root = Path(cfg["run_root"])
    out_dir = (Path(cfg["out_dir"]) if cfg.get("out_dir")
               else Path("analysis/output") / yaml_path.stem)
    landfall_raw = cfg.get("landfall_time")
    landfall_time = None
    if landfall_raw:
        value = str(landfall_raw)
        fmt = "%Y%m%d%H%M" if len(value) == 12 else "%Y%m%d%H"
        landfall_time = datetime.strptime(value, fmt)
    return CyclesCase(
        run_root=run_root,
        valid_start=valid_start,
        valid_end=valid_end,
        storm_name=cfg.get("storm_name", "Storm"),
        model_label=(cfg["model_label"] if "model_label" in cfg
                     else detect_model(run_root)),
        domain=tuple(cfg["domain"]),
        grid_res=float(cfg.get("grid_res", 0.05)),
        mask_radius_km=float(cfg.get("mask_radius_km", 500.0)),
        display_radius_km=float(cfg.get("display_radius_km", 750.0)),
        thresholds_mm=cfg.get("thresholds_mm", list(_DEFAULT_THRESHOLDS)),
        ets_threshold_mm=float(cfg.get("ets_threshold_mm", 25.0)),
        out_dir=out_dir,
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        stage4_cache_dir=Path(cfg.get("stage4_cache_dir",
                                      "/tmp/stage4_cache")),
        inits=[str(i) for i in cfg["inits"]] if cfg.get("inits") else None,
        case_slug=yaml_path.stem,
        storm_id=cfg.get("storm_id"),
        landfall_time=landfall_time,
        ets_bar_thresholds_in=[float(v) for v in cfg.get(
            "ets_bar_thresholds_in", list(range(2, 25, 2)))],
        fss_thresholds_in=[float(v) for v in cfg.get(
            "fss_thresholds_in",
            [float(v) / 25.4 for v in cfg["fss_thresholds_mm"]]
            if "fss_thresholds_mm" in cfg else [1.0, 2.0])],
        fss_scales_cells=[int(v) for v in cfg.get(
            "fss_scales_cells", [1, 3, 5, 11, 21, 41])],
        make_animation=bool(cfg.get("make_animation", True)),
        best_track=Path(cfg["best_track"]) if cfg.get("best_track") else None,
        track_step_hours=int(cfg.get("track_step_hours", 6)),
        headline_fss_threshold_in=(
            float(cfg["headline_fss_threshold_in"])
            if cfg.get("headline_fss_threshold_in") is not None else None),
        headline_fss_scale_cells=(
            int(cfg["headline_fss_scale_cells"])
            if cfg.get("headline_fss_scale_cells") is not None else None),
    )


def cycle_storm_case(ccase, init_str):
    """StormCase for one cycle of a CyclesCase (run_dir = run_root/<init>).

    Track comes from the cycle's own .atcfunix; grid/mask/output settings
    are inherited from the CyclesCase so every cycle verifies identically.
    """
    run_dir = ccase.run_root / init_str
    atcf_path = find_atcfunix(run_dir)
    _, _, fixes = parse_atcfunix_fixes(atcf_path, ccase.storm_id)
    track = [(t, lat, lon) for t, lat, lon, _, _ in fixes]
    if not track:
        hint = (f" for storm_id {ccase.storm_id!r}" if ccase.storm_id else "")
        raise ValueError(f"Parsed 0 track fixes from {atcf_path}{hint}")
    return StormCase(
        run_dir=run_dir,
        init_dt=datetime.strptime(init_str, "%Y%m%d%H"),
        storm_name=ccase.storm_name,
        model_label=ccase.model_label,
        domain=ccase.domain,
        grid_res=ccase.grid_res,
        mask_radius_km=ccase.mask_radius_km,
        display_radius_km=ccase.display_radius_km,
        thresholds_mm=ccase.thresholds_mm,
        out_dir=ccase.out_dir,
        mrms_cache_dir=ccase.mrms_cache_dir,
        stage4_cache_dir=ccase.stage4_cache_dir,
        fhours_filter=None,
        track=track,
        case_slug=ccase.case_slug,
        init_str=init_str,
        track_fixes=fixes,
        storm_id=ccase.storm_id,
    )
