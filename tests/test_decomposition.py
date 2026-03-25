"""Tests for DATE decomposition of pointwise causal effects.

What: Verify that decompose_effects correctly separates pointwise effects
into spot, persistent, and trend components via OLS projection.

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836, Eq. (11e).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_impact.decomposition import (
    DateDecomposition,
    EffectComponent,
    build_design_matrix,
    decompose_effects,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_effects(
    n_samples: int,
    t_post: int,
    beta_spot: float,
    beta_persistent: float,
    beta_trend: float = 0.0,
    noise_sd: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthesise effect trajectories from known coefficients."""
    rng = np.random.default_rng(seed)
    D = build_design_matrix(t_post, include_trend=(t_post >= 3))
    beta = np.array([beta_spot, beta_persistent, beta_trend][: D.shape[1]])
    signal = D @ beta
    noise = rng.normal(0, noise_sd, (n_samples, t_post)) if noise_sd > 0 else 0.0
    return np.tile(signal, (n_samples, 1)) + noise


# ---------------------------------------------------------------------------
# TestBuildDesignMatrix
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    @pytest.mark.parametrize(
        "t_post, expected_cols",
        [(1, 2), (2, 2), (3, 3), (10, 3), (100, 3)],
    )
    def test_shape_parametrized(self, t_post, expected_cols):
        D = build_design_matrix(t_post)
        assert D.shape == (t_post, expected_cols)

    def test_spot_column_impulse(self):
        D = build_design_matrix(5)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(D[:, 0], expected)

    def test_persistent_column_ones(self):
        D = build_design_matrix(5)
        np.testing.assert_array_equal(D[:, 1], np.ones(5))

    def test_trend_column_ramp(self):
        D = build_design_matrix(5)
        np.testing.assert_array_equal(D[:, 2], np.arange(5, dtype=np.float64))

    def test_tpost1_shape_2cols(self):
        D = build_design_matrix(1)
        assert D.shape == (1, 2)

    def test_tpost0_raises(self):
        with pytest.raises(ValueError, match="t_post must be >= 1"):
            build_design_matrix(0)

    def test_include_trend_false(self):
        D = build_design_matrix(10, include_trend=False)
        assert D.shape == (10, 2)

    def test_dtype_float64(self):
        D = build_design_matrix(5)
        assert D.dtype == np.float64


# ---------------------------------------------------------------------------
# TestDecomposeRecovery
# ---------------------------------------------------------------------------


class TestDecomposeRecovery:
    @pytest.mark.parametrize(
        "beta_spot, beta_persistent, beta_trend",
        [
            (5.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 3.0, 0.5),
            (-1.0, 4.0, -0.2),
        ],
    )
    def test_noiseless_parametrized(self, beta_spot, beta_persistent, beta_trend):
        t_post = 10
        effects = _make_effects(100, t_post, beta_spot, beta_persistent, beta_trend)
        result = decompose_effects(effects)
        assert result.spot.coefficient == pytest.approx(beta_spot, abs=1e-12)
        assert result.persistent.coefficient == pytest.approx(beta_persistent, abs=1e-12)
        assert result.trend is not None
        assert result.trend.coefficient == pytest.approx(beta_trend, abs=1e-12)

    def test_noisy_mean_close(self):
        effects = _make_effects(5000, 20, 3.0, 2.0, 0.5, noise_sd=0.5)
        result = decompose_effects(effects)
        assert result.spot.coefficient == pytest.approx(3.0, abs=0.1)
        assert result.persistent.coefficient == pytest.approx(2.0, abs=0.1)
        assert result.trend.coefficient == pytest.approx(0.5, abs=0.1)

    def test_residual_zero_noiseless(self):
        effects = _make_effects(100, 10, 2.0, 1.0, 0.3)
        result = decompose_effects(effects)
        np.testing.assert_allclose(result.residual_mean, 0.0, atol=1e-12)

    def test_components_sum_to_original(self):
        effects = _make_effects(200, 15, 1.5, 2.5, 0.4, noise_sd=0.3, seed=7)
        result = decompose_effects(effects)
        reconstructed = result.spot.mean + result.persistent.mean
        if result.trend is not None:
            reconstructed = reconstructed + result.trend.mean
        reconstructed = reconstructed + result.residual_mean
        np.testing.assert_allclose(reconstructed, effects.mean(axis=0), atol=1e-10)

    def test_spot_trajectory_is_impulse_scaled(self):
        effects = _make_effects(50, 8, 5.0, 0.0, 0.0)
        result = decompose_effects(effects)
        expected = np.zeros(8)
        expected[0] = 5.0
        np.testing.assert_allclose(result.spot.mean, expected, atol=1e-10)

    def test_persistent_trajectory_is_step(self):
        effects = _make_effects(50, 8, 0.0, 3.0, 0.0)
        result = decompose_effects(effects)
        np.testing.assert_allclose(result.persistent.mean, np.full(8, 3.0), atol=1e-10)


# ---------------------------------------------------------------------------
# TestDecomposeShortPost
# ---------------------------------------------------------------------------


class TestDecomposeShortPost:
    def test_tpost2_trend_is_none(self):
        effects = _make_effects(100, 2, 1.0, 2.0)
        result = decompose_effects(effects)
        assert result.trend is None

    def test_tpost2_spot_persistent_recovered(self):
        effects = _make_effects(100, 2, 1.0, 2.0)
        with pytest.warns(UserWarning, match="T_post=2 < 3"):
            result = decompose_effects(effects)
        assert result.spot.coefficient == pytest.approx(1.0, abs=1e-10)
        assert result.persistent.coefficient == pytest.approx(2.0, abs=1e-10)

    def test_tpost2_emits_warning(self):
        effects = _make_effects(100, 2, 1.0, 2.0)
        with pytest.warns(UserWarning, match="T_post=2 < 3"):
            decompose_effects(effects)

    def test_tpost1_trend_is_none(self):
        effects = np.full((100, 1), 4.0)
        with pytest.warns(UserWarning):
            result = decompose_effects(effects)
        assert result.trend is None

    def test_tpost1_emits_collinearity_warning(self):
        effects = np.full((100, 1), 4.0)
        with pytest.warns(UserWarning, match="collinear"):
            decompose_effects(effects)

    def test_tpost1_equal_split(self):
        effects = np.full((100, 1), 4.0)
        with pytest.warns(UserWarning):
            result = decompose_effects(effects)
        assert result.spot.coefficient == pytest.approx(2.0, abs=1e-10)
        assert result.persistent.coefficient == pytest.approx(2.0, abs=1e-10)

    def test_tpost1_coefficients_sum_to_effect(self):
        effects = np.full((100, 1), 6.0)
        with pytest.warns(UserWarning):
            result = decompose_effects(effects)
        total = result.spot.coefficient + result.persistent.coefficient
        assert total == pytest.approx(6.0, abs=1e-10)


# ---------------------------------------------------------------------------
# TestDecomposeEdgeCases
# ---------------------------------------------------------------------------


class TestDecomposeEdgeCases:
    def test_nsamples0_raises(self):
        with pytest.raises(ValueError, match="0 samples"):
            decompose_effects(np.empty((0, 5)))

    def test_tpost0_raises(self):
        with pytest.raises(ValueError, match="0 time points"):
            decompose_effects(np.empty((10, 0)))

    def test_single_sample_ci_collapses(self):
        effects = _make_effects(1, 10, 2.0, 1.0, 0.5)
        result = decompose_effects(effects)
        assert result.spot.ci_lower == result.spot.coefficient
        assert result.spot.ci_upper == result.spot.coefficient

    def test_1d_input_promoted(self):
        effects_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = decompose_effects(effects_1d)
        assert result.spot.samples.shape[0] == 1

    def test_frozen_dataclass_immutable(self):
        effects = _make_effects(50, 5, 1.0, 1.0, 0.1)
        result = decompose_effects(effects)
        with pytest.raises(AttributeError):
            result.spot = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestDecomposeCI
# ---------------------------------------------------------------------------


class TestDecomposeCI:
    def test_lower_le_mean_le_upper(self):
        effects = _make_effects(500, 10, 2.0, 1.0, 0.3, noise_sd=0.5)
        result = decompose_effects(effects)
        for comp in [result.spot, result.persistent, result.trend]:
            assert comp is not None
            assert comp.ci_lower <= comp.coefficient <= comp.ci_upper

    def test_alpha01_wider_than_alpha05(self):
        effects = _make_effects(500, 10, 2.0, 1.0, 0.3, noise_sd=0.5)
        r01 = decompose_effects(effects, alpha=0.01)
        r05 = decompose_effects(effects, alpha=0.05)
        width_01 = r01.spot.ci_upper - r01.spot.ci_lower
        width_05 = r05.spot.ci_upper - r05.spot.ci_lower
        assert width_01 >= width_05

    def test_alpha_stored(self):
        effects = _make_effects(50, 5, 1.0, 1.0, 0.1)
        result = decompose_effects(effects, alpha=0.10)
        assert result.alpha == 0.10


# ---------------------------------------------------------------------------
# TestIntegrationWithCausalImpact
# ---------------------------------------------------------------------------


def _make_ci_data(t_pre=50, t_post=20, effect=5.0, seed=42):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 1, t_pre + t_post)) + 100
    y[t_pre:] += effect
    df = pd.DataFrame({"y": y})
    return df, [0, t_pre - 1], [t_pre, t_pre + t_post - 1]


class TestIntegrationWithCausalImpact:
    def test_returns_date_decomposition(self):
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.decompose()
        assert isinstance(result, DateDecomposition)

    def test_spot_shape_matches_tpost(self):
        from causal_impact import CausalImpact

        t_post = 15
        df, pre, post = _make_ci_data(t_post=t_post)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.decompose()
        assert result.spot.mean.shape == (t_post,)

    def test_stores_on_instance(self):
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        ci.decompose()
        assert ci._decomposition is not None

    def test_default_alpha(self):
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.decompose()
        assert result.alpha == 0.05

    def test_custom_alpha(self):
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data()
        ci = CausalImpact(
            df, pre, post, model_args={"niter": 100, "nwarmup": 50}, alpha=0.10
        )
        result = ci.decompose(alpha=0.01)
        assert result.alpha == 0.01

    def test_short_post_warns(self):
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data(t_post=2)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        with pytest.warns(UserWarning, match="T_post=2 < 3"):
            ci.decompose()


# ---------------------------------------------------------------------------
# TestPlotDecomposition
# ---------------------------------------------------------------------------


class TestPlotDecomposition:
    @pytest.fixture()
    def _ci_with_decomposition(self):
        import matplotlib

        matplotlib.use("Agg")
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data(t_post=20)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        ci.decompose()
        return ci

    def test_panel_exists(self, _ci_with_decomposition):
        ci = _ci_with_decomposition
        fig = ci.plot(metrics=["original", "pointwise", "cumulative", "decomposition"])
        assert len(fig.get_axes()) == 4

    def test_3_lines_with_trend(self, _ci_with_decomposition):
        ci = _ci_with_decomposition
        fig = ci.plot(metrics=["decomposition"])
        ax = fig.get_axes()[0]
        # 3 component lines + 1 hline + 1 axvline (intervention) = 5 lines
        assert len(ax.get_lines()) == 5

    def test_2_lines_without_trend(self):
        import matplotlib

        matplotlib.use("Agg")
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data(t_post=2)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        with pytest.warns(UserWarning):
            ci.decompose()
        fig = ci.plot(metrics=["decomposition"])
        ax = fig.get_axes()[0]
        # 2 component lines + 1 hline + 1 axvline (intervention) = 4 lines
        assert len(ax.get_lines()) == 4

    def test_ci_bands(self, _ci_with_decomposition):
        ci = _ci_with_decomposition
        fig = ci.plot(metrics=["decomposition"])
        ax = fig.get_axes()[0]
        # 3 fill_between collections (spot, persistent, trend)
        assert len(ax.collections) == 3

    def test_without_decompose_raises(self):
        import matplotlib

        matplotlib.use("Agg")
        from causal_impact import CausalImpact

        df, pre, post = _make_ci_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        with pytest.raises(ValueError, match="decompose"):
            ci.plot(metrics=["decomposition"])

    def test_mixed_metrics(self, _ci_with_decomposition):
        ci = _ci_with_decomposition
        fig = ci.plot(metrics=["pointwise", "decomposition"])
        assert len(fig.get_axes()) == 2
