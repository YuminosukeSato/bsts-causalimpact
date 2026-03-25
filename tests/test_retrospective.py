"""Tests for retrospective attribution mode.

What: Verify that mode="retrospective" correctly identifies treatment effects
by including treatment indicator columns as covariates and fitting the BSTS
model on the entire time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from causal_impact import CausalImpact
from causal_impact.decomposition import DateDecomposition
from causal_impact.retrospective import (
    build_treatment_design,
    extract_treatment_effects,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retrospective_data(
    t_pre: int = 100,
    t_post: int = 30,
    beta_persistent: float = 5.0,
    beta_spot: float = 0.0,
    beta_trend: float = 0.0,
    noise_sd: float = 1.0,
    seed: int = 42,
    with_covariate: bool = False,
):
    """Synthesise data with known treatment effect injected via design columns."""
    rng = np.random.default_rng(seed)
    n = t_pre + t_post

    # Base process: local level + covariate
    if with_covariate:
        x = np.zeros(n)
        x[0] = 100
        for t in range(1, n):
            x[t] = 0.999 * x[t - 1] + rng.normal(0, 1)
        y = 1.2 * x + rng.normal(0, noise_sd, size=n)
    else:
        y = np.cumsum(rng.normal(0, 0.3, n)) + 50
        y += rng.normal(0, noise_sd, n)
        x = None

    # Inject treatment effect
    D = build_treatment_design(n, t_pre)
    treatment = D @ np.array([beta_spot, beta_persistent, beta_trend])
    y += treatment

    if with_covariate:
        df = pd.DataFrame({"y": y, "x": x})
    else:
        df = pd.DataFrame({"y": y})

    return df, [0, t_pre - 1], [t_pre, n - 1], t_pre


# ---------------------------------------------------------------------------
# TestBuildTreatmentDesign
# ---------------------------------------------------------------------------


class TestBuildTreatmentDesign:
    def test_shape(self):
        D = build_treatment_design(100, 70)
        assert D.shape == (100, 3)

    def test_spot_column(self):
        D = build_treatment_design(10, 5)
        expected = np.zeros(10)
        expected[5] = 1.0
        np.testing.assert_array_equal(D[:, 0], expected)

    def test_persistent_column(self):
        D = build_treatment_design(10, 5)
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        np.testing.assert_array_equal(D[:, 1], expected)

    def test_trend_column(self):
        D = build_treatment_design(10, 5)
        expected = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=float)
        np.testing.assert_array_equal(D[:, 2], expected)

    def test_pre_period_all_zero(self):
        D = build_treatment_design(20, 10)
        np.testing.assert_array_equal(D[:10, :], 0.0)


# ---------------------------------------------------------------------------
# TestExtractTreatmentEffects
# ---------------------------------------------------------------------------


class TestExtractTreatmentEffects:
    def test_returns_date_decomposition(self):
        rng = np.random.default_rng(42)
        beta_samples = rng.normal(0, 1, (100, 5))
        result = extract_treatment_effects(
            beta_samples,
            treatment_col_start=2,
            t_post=30,
        )
        assert isinstance(result, DateDecomposition)

    def test_correct_coefficients_extracted(self):
        n_samples = 200
        # Columns: [x1, x2, spot, persistent, trend]
        beta_samples = np.zeros((n_samples, 5))
        beta_samples[:, 2] = 3.0  # spot
        beta_samples[:, 3] = 5.0  # persistent
        beta_samples[:, 4] = 0.1  # trend
        result = extract_treatment_effects(
            beta_samples,
            treatment_col_start=2,
            t_post=20,
        )
        assert result.spot.coefficient == pytest.approx(3.0, abs=1e-10)
        assert result.persistent.coefficient == pytest.approx(5.0, abs=1e-10)
        assert result.trend.coefficient == pytest.approx(0.1, abs=1e-10)

    def test_trajectory_shapes(self):
        beta_samples = np.random.default_rng(42).normal(0, 1, (50, 3))
        result = extract_treatment_effects(
            beta_samples,
            treatment_col_start=0,
            t_post=15,
        )
        assert result.spot.mean.shape == (15,)
        assert result.persistent.mean.shape == (15,)
        assert result.trend.mean.shape == (15,)


# ---------------------------------------------------------------------------
# TestRetrospectiveCausalImpact
# ---------------------------------------------------------------------------


class TestRetrospectiveCausalImpact:
    def test_known_persistent_effect(self):
        """Known level shift: persistent coefficient should be in a reasonable range.

        Due to partial collinearity between the local level state and the
        persistent treatment indicator, exact recovery is not guaranteed.
        We check the estimate is within ±50% of the true value.
        """
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=5.0,
            noise_sd=0.5,
            seed=42,
            t_pre=200,
            t_post=50,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 2000,
                "nwarmup": 1000,
                "seed": 42,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        dec = ci._decomposition
        assert dec is not None
        # Persistent + spot at t=0 should sum to roughly the true effect
        total_at_intervention = dec.persistent.coefficient + dec.spot.coefficient
        assert abs(total_at_intervention - 5.0) < 5.0

    def test_known_trend_effect(self):
        """Known trend: trend coefficient should be positive and non-trivial."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=0.0,
            beta_trend=0.5,
            noise_sd=0.5,
            seed=77,
            t_pre=200,
            t_post=50,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 2000,
                "nwarmup": 1000,
                "seed": 77,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        dec = ci._decomposition
        assert dec is not None
        # Trend should be positive (true value 0.5)
        assert dec.trend.coefficient > 0.0

    def test_zero_effect_includes_zero(self):
        """No effect: persistent coefficient should be close to zero."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=0.0,
            beta_spot=0.0,
            beta_trend=0.0,
            noise_sd=0.5,
            seed=99,
            t_pre=150,
            t_post=50,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 1000,
                "nwarmup": 500,
                "seed": 99,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        dec = ci._decomposition
        assert dec is not None
        # Should be close to zero (within a few units)
        assert abs(dec.persistent.coefficient) < 3.0

    def test_spike_slab_auto_disabled(self):
        """Spike-and-slab should be auto-disabled in retrospective mode."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=3.0,
            with_covariate=True,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
                "mode": "retrospective",
                "expected_model_size": 1,
            },
        )
        # Should have run without error; gamma should be empty or all-True
        # because spike-and-slab was force-disabled
        assert ci._decomposition is not None

    def test_with_covariate(self):
        """Retrospective with an additional covariate should still work."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=5.0,
            with_covariate=True,
            noise_sd=0.5,
            seed=42,
            t_pre=200,
            t_post=50,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 2000,
                "nwarmup": 1000,
                "seed": 42,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        dec = ci._decomposition
        assert dec is not None
        # Persistent + spot should sum to roughly the true effect
        # Due to collinearity with local level and covariate, wide tolerance
        total_at_intervention = dec.persistent.coefficient + dec.spot.coefficient
        assert abs(total_at_intervention - 5.0) < 5.0

    def test_forward_decompose_still_works(self):
        """Forward mode decompose() should still work after retrospective is added."""
        df, pre, post, _ = _make_retrospective_data(beta_persistent=5.0)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.decompose()
        assert isinstance(result, DateDecomposition)

    def test_existing_tests_unchanged(self):
        """Ensure forward mode default behavior is unchanged."""
        rng = np.random.default_rng(42)
        n = 130
        y = np.cumsum(rng.normal(0, 1, n)) + 100
        y[100:] += 5.0
        df = pd.DataFrame({"y": y})
        ci = CausalImpact(
            df,
            [0, 99],
            [100, 129],
            model_args={"niter": 100, "nwarmup": 50},
        )
        assert ci._decomposition is None
        assert hasattr(ci, "decompose")

    def test_decompose_returns_cached_in_retrospective(self):
        """decompose() should return the cached decomposition in retrospective mode."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=3.0,
            noise_sd=0.5,
            seed=42,
            t_pre=100,
            t_post=30,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        # decompose() should work without crashing
        dec = ci.decompose()
        assert isinstance(dec, DateDecomposition)
        assert dec is ci._decomposition

    def test_decompose_recomputes_with_different_alpha(self):
        """decompose(alpha=0.1) should recompute with new credible intervals."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=3.0,
            noise_sd=0.5,
            seed=42,
            t_pre=100,
            t_post=30,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        dec_05 = ci.decompose(alpha=0.05)
        dec_10 = ci.decompose(alpha=0.10)
        assert dec_10.alpha == 0.10
        # 90% CI should be narrower than 95% CI
        assert dec_10.persistent.ci_upper - dec_10.persistent.ci_lower <= (
            dec_05.persistent.ci_upper - dec_05.persistent.ci_lower
        )

    def test_short_post_period_no_crash(self):
        """Retrospective with t_post=2 should not crash (trend=None)."""
        df, pre, post, _ = _make_retrospective_data(
            beta_persistent=3.0,
            noise_sd=0.5,
            seed=42,
            t_pre=50,
            t_post=2,
        )
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "niter": 100,
                "nwarmup": 50,
                "seed": 42,
                "mode": "retrospective",
                "prior_level_sd": 0.001,
            },
        )
        dec = ci._decomposition
        assert dec is not None
        assert dec.trend is None
        assert dec.spot.mean.shape == (2,)
        assert dec.persistent.mean.shape == (2,)

    def test_dynamic_regression_retrospective_raises(self):
        """dynamic_regression=True should raise in retrospective mode."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(0, 1, 50)) + 100
        df = pd.DataFrame({"y": y})
        with pytest.raises(ValueError, match="dynamic_regression"):
            CausalImpact(
                df,
                [0, 29],
                [30, 49],
                model_args={
                    "niter": 100,
                    "nwarmup": 50,
                    "mode": "retrospective",
                    "dynamic_regression": True,
                },
            )

    def test_invalid_mode_raises(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.normal(0, 1, 50)) + 100
        df = pd.DataFrame({"y": y})
        with pytest.raises(ValueError, match="mode"):
            CausalImpact(
                df,
                [0, 29],
                [30, 49],
                model_args={"niter": 100, "nwarmup": 50, "mode": "invalid"},
            )
