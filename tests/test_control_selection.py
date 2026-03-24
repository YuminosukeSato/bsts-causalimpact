"""Tests for DTW distance, LB_Keogh, and select_controls."""

import numpy as np
import pandas as pd
import pytest
from causal_impact._core import (
    py_dtw_distance,
    py_lb_keogh_distance,
    py_lb_keogh_envelope,
)
from causal_impact.selection import select_controls

# ── DTW distance tests ──────────────────────────────────────────────


class TestDtwDistance:
    def test_identical_series_is_zero(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert py_dtw_distance(x, x) == 0.0

    def test_different_series_positive(self):
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        assert py_dtw_distance(x, y) > 0.0

    def test_single_element_known_value(self):
        assert py_dtw_distance([1.0], [3.0]) == pytest.approx(2.0)

    def test_empty_x_raises(self):
        with pytest.raises(ValueError, match="x must not be empty"):
            py_dtw_distance([], [1.0])

    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="y must not be empty"):
            py_dtw_distance([1.0], [])

    def test_window_none_unconstrained(self):
        x = [1.0, 2.0, 3.0]
        assert py_dtw_distance(x, x, window=None) == 0.0

    def test_window_zero_equals_l1(self):
        x = [1.0, 2.0, 3.0]
        y = [2.0, 4.0, 6.0]
        d = py_dtw_distance(x, y, window=0)
        l1 = sum(abs(a - b) for a, b in zip(x, y))
        assert d == pytest.approx(l1)

    def test_different_length_valid(self):
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = py_dtw_distance(x, y)
        assert d >= 0.0
        assert np.isfinite(d)

    def test_symmetry(self):
        x = [1.0, 3.0, 5.0, 2.0]
        y = [2.0, 4.0, 1.0, 3.0]
        assert py_dtw_distance(x, y) == pytest.approx(py_dtw_distance(y, x))

    def test_early_abandon_fires(self):
        x = [0.0, 0.0, 0.0]
        y = [100.0, 100.0, 100.0]
        d = py_dtw_distance(x, y, best_so_far=1.0)
        assert d == float("inf")


# ── LB_Keogh tests ──────────────────────────────────────────────────


class TestLbKeogh:
    def test_distance_le_dtw(self):
        """LB_Keogh is a lower bound: LB <= DTW."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50).tolist()
        y = rng.normal(size=50).tolist()
        w = 5
        lo, hi = py_lb_keogh_envelope(y, w)
        lb = py_lb_keogh_distance(x, lo, hi)
        dtw = py_dtw_distance(x, y, window=w)
        assert lb <= dtw + 1e-10

    def test_identical_series_zero(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = py_lb_keogh_envelope(y, 2)
        assert py_lb_keogh_distance(y, lo, hi) == 0.0

    def test_window_zero_equals_l1(self):
        x = [1.0, 5.0, 3.0]
        y = [2.0, 3.0, 6.0]
        lo, hi = py_lb_keogh_envelope(y, 0)
        lb = py_lb_keogh_distance(x, lo, hi)
        l1 = sum(abs(a - b) for a, b in zip(x, y))
        assert lb == pytest.approx(l1)

    def test_avx2_matches_scalar(self):
        """AVX2 and scalar implementations must agree (n=100)."""
        rng = np.random.default_rng(123)
        xi = rng.normal(size=100).tolist()
        y = rng.normal(size=100).tolist()
        lo, hi = py_lb_keogh_envelope(y, 5)
        lb = py_lb_keogh_distance(xi, lo, hi)
        # Re-compute manually (scalar reference)
        ref = sum(max(0, lo_i - x) + max(0, x - h) for x, lo_i, h in zip(xi, lo, hi))
        assert lb == pytest.approx(ref, abs=1e-10)

    def test_non_multiple_of_4(self):
        """n % 4 != 0 triggers tail handling in AVX2."""
        for n in [101, 102, 103]:
            xi = list(range(n))
            y = [v + 10.0 for v in xi]
            lo, hi = py_lb_keogh_envelope(y, 2)
            lb = py_lb_keogh_distance([float(v) for v in xi], lo, hi)
            ref = sum(
                max(0, lo_i - x) + max(0, x - h) for x, lo_i, h in zip(xi, lo, hi)
            )
            assert lb == pytest.approx(ref, abs=1e-8), f"Failed for n={n}"


# ── select_controls tests ───────────────────────────────────────────


class TestSelectControls:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(42)
        t = 100
        y = np.cumsum(rng.normal(size=t))
        # x1: correlated with y (similar shape)
        x1 = y + rng.normal(0, 0.5, size=t)
        # x2: moderately correlated
        x2 = y * 0.5 + np.cumsum(rng.normal(size=t)) * 0.5
        # x3: uncorrelated
        x3 = np.cumsum(rng.normal(size=t)) * 5
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

    def test_returns_dataframe(self, sample_df):
        result = select_controls(sample_df, "y", [0, 69], top_k=2)
        assert isinstance(result, pd.DataFrame)

    def test_target_col_preserved(self, sample_df):
        result = select_controls(sample_df, "y", [0, 69], top_k=2)
        assert "y" in result.columns

    def test_correct_column_count(self, sample_df):
        result = select_controls(sample_df, "y", [0, 69], top_k=2)
        assert result.shape[1] == 3  # target + 2 controls

    def test_selects_most_similar(self, sample_df):
        """x1 (most correlated) should be selected first."""
        result = select_controls(sample_df, "y", [0, 69], top_k=1)
        selected = [c for c in result.columns if c != "y"]
        assert selected[0] == "x1"

    def test_k_exceeds_candidates(self, sample_df):
        result = select_controls(sample_df, "y", [0, 69], top_k=10)
        # Should return all 3 candidates + target
        assert result.shape[1] == 4

    def test_k_zero_raises(self, sample_df):
        with pytest.raises(ValueError, match="top_k must be > 0"):
            select_controls(sample_df, "y", [0, 69], top_k=0)

    def test_k_negative_raises(self, sample_df):
        with pytest.raises(ValueError, match="top_k must be > 0"):
            select_controls(sample_df, "y", [0, 69], top_k=-1)

    def test_target_missing_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            select_controls(sample_df, "nonexistent", [0, 69])

    def test_with_datetime_index(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "y": np.cumsum(rng.normal(size=50)),
                "x1": np.cumsum(rng.normal(size=50)),
            },
            index=dates,
        )
        result = select_controls(df, "y", ["2020-01-01", "2020-02-01"], top_k=1)
        assert "y" in result.columns
        assert result.shape[1] == 2

    def test_scale_invariant(self):
        """DTW on z-scored data should select by shape, not scale."""
        t = 80
        y = np.sin(np.linspace(0, 4 * np.pi, t))
        # x1: same shape, 1000x scale
        x1 = y * 1000 + 5000
        # x2: different shape
        x2 = np.cos(np.linspace(0, 8 * np.pi, t)) * 0.01
        df = pd.DataFrame({"y": y, "x1_scaled": x1, "x2_noise": x2})
        result = select_controls(df, "y", [0, 59], top_k=1)
        selected = [c for c in result.columns if c != "y"]
        assert selected[0] == "x1_scaled"

    def test_no_candidates(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        result = select_controls(df, "y", [0, 1], top_k=1)
        assert list(result.columns) == ["y"]
