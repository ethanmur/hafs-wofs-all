"""Unit tests for precipitation distribution and pattern helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precip_structure import DIST_FIELDS, distribution_stats, qq_percentiles


def test_distribution_stats_uniform_field_exact_volume_and_percentiles():
    grid_res = 0.1
    lat = np.zeros((2, 3))
    field = np.full((2, 3), 10.0)
    swath = np.array([[True, True, False], [True, False, False]])
    stats = distribution_stats(field, swath, lat, grid_res)
    cell_area = (grid_res * 111.0) ** 2
    assert stats["p50"] == stats["p90"] == stats["p95"] == 10.0
    assert stats["p99"] == stats["max_mm"] == 10.0
    assert stats["volume_km3"] == 3 * 10.0 * cell_area * 1e-6
    assert stats["wet_frac"] == 1.0


def test_qq_identical_fields_is_identity():
    field = np.arange(25, dtype=float).reshape(5, 5)
    fcst_q, obs_q = qq_percentiles(
        field, field.copy(), np.ones_like(field, dtype=bool))
    np.testing.assert_allclose(fcst_q, obs_q)


def test_distribution_stats_all_nan_returns_nan_dict():
    field = np.full((4, 4), np.nan)
    stats = distribution_stats(field, np.ones_like(field, dtype=bool),
                               np.zeros_like(field), 0.1)
    assert set(stats) == set(DIST_FIELDS)
    assert all(np.isnan(value) for value in stats.values())
