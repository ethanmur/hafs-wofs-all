"""One place that builds a DeploymentCase for tests, so adding a path to the
case changes this file and not every test module."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deployment_case as dc

_DIRS = ("grid_json_dir", "grid_plot_dir", "stage4_regrid_dir",
         "mrms_regrid_dir", "regrid_plot_dir",
         "mask_dir", "mask_plot_dir", "manifest_dir", "wofs_regrid_1h_dir",
         "wofs_regrid_3h_dir", "log_dir", "mrms_cache_dir", "stage4_cache_dir")


def make_case(tmp_path, **overrides):
    """A Beryl 20240708_d1 case with every directory under `tmp_path`."""
    tmp_path = Path(tmp_path)
    kwargs = dict(
        case_slug="2024_02l_beryl_20240708_d1", storm_name="Beryl",
        storm_id="2024-02l", deployment="20240708_d1",
        best_track=tmp_path / "bt.dat", truth_source="stage4",
        wofs_root=tmp_path / "wofs" / "2024_BERYL" / "20240708_d1",
        wofs_available=True, complete_windows=3,
        valid_start=datetime(2024, 9, 26, 0),
        valid_end=datetime(2024, 9, 26, 2))
    kwargs.update({d: tmp_path / d for d in _DIRS})
    kwargs.update(overrides)
    return dc.DeploymentCase(**kwargs)
