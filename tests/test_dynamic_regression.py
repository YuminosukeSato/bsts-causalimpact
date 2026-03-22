"""Tests for dynamic regression (time-varying coefficients).

Dynamic regression allows β_t to vary over time as a random walk,
unlike static regression where β is constant. This is the key feature
for capturing structural changes in the pre-intervention relationship.
"""

import numpy as np
import pytest
from causal_impact import CausalImpact, ModelOptions
from causal_impact._core import run_gibbs_sampler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data_dynamic_k1_constant_beta(n=100, pre_frac=0.7, seed=42):
    """k=1, beta=2.0 constant. Dynamic should match static."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    return y, [x.tolist()], pre_end


def _make_data_dynamic_k1_structural_break(n=100, pre_frac=0.7, seed=42):
    """k=1, beta jumps from 1.0 to 3.0 midway. Dynamic should track."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x = rng.normal(0, 1, n)
    beta_true = np.where(np.arange(n) < n // 2, 1.0, 3.0)
    y = beta_true * x + rng.normal(0, 0.3, n)
    return y, [x.tolist()], pre_end


def _make_data_dynamic_k2(n=100, pre_frac=0.7, seed=42):
    """k=2, beta1=1.0 constant, beta2 gradually changes."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    beta2 = np.linspace(0.5, 2.5, n)
    y = 1.0 * x1 + beta2 * x2 + rng.normal(0, 0.3, n)
    return y, [x1.tolist(), x2.tolist()], pre_end


def _run_sampler_dynamic(y, x, pre_end, niter=500, nwarmup=250, seed=42):
    """Call run_gibbs_sampler with dynamic_regression=True."""
    return run_gibbs_sampler(
        y=y.tolist() if hasattr(y, "tolist") else list(y),
        x=x if x else None,
        pre_end=pre_end,
        niter=niter,
        nwarmup=nwarmup,
        nchains=1,
        seed=seed,
        prior_level_sd=0.01,
        expected_model_size=1.0,
        nseasons=None,
        season_duration=None,
        dynamic_regression=True,
    )


def _run_sampler_static(y, x, pre_end, niter=500, nwarmup=250, seed=42):
    """Call run_gibbs_sampler with dynamic_regression=False."""
    return run_gibbs_sampler(
        y=y.tolist() if hasattr(y, "tolist") else list(y),
        x=x if x else None,
        pre_end=pre_end,
        niter=niter,
        nwarmup=nwarmup,
        nchains=1,
        seed=seed,
        prior_level_sd=0.01,
        expected_model_size=1.0,
        nseasons=None,
        season_duration=None,
        dynamic_regression=False,
    )


# ---------------------------------------------------------------------------
# Option validation (2 tests)
# ---------------------------------------------------------------------------


class TestDynamicRegressionOptions:
    def test_default_is_false(self):
        opts = ModelOptions()
        assert opts.dynamic_regression is False

    def test_true_accepted(self):
        opts = ModelOptions(dynamic_regression=True)
        assert opts.dynamic_regression is True


# ---------------------------------------------------------------------------
# Basic behavior (3 tests)
# ---------------------------------------------------------------------------


class TestDynamicRegressionBasic:
    def test_predictions_shape_unchanged(self):
        y, x, pre_end = _make_data_dynamic_k1_constant_beta()
        result = _run_sampler_dynamic(y, x, pre_end, niter=50, nwarmup=25)
        preds = np.array(result.predictions)
        t_post = len(y) - pre_end
        assert preds.shape == (50, t_post)

    def test_posterior_inclusion_probs_none_when_dynamic(self):
        """Spike-and-slab is disabled when dynamic_regression=True."""
        y, x, pre_end = _make_data_dynamic_k1_constant_beta()
        result = _run_sampler_dynamic(y, x, pre_end, niter=50, nwarmup=25)
        # gamma should be empty (spike-and-slab disabled)
        assert result.gamma == [] or all(len(g) == 0 for g in result.gamma)

    def test_false_matches_existing_behavior(self):
        y, x, pre_end = _make_data_dynamic_k1_constant_beta()
        result_false = _run_sampler_static(y, x, pre_end, niter=50, nwarmup=25)
        result_default = run_gibbs_sampler(
            y=y.tolist(),
            x=x,
            pre_end=pre_end,
            niter=50,
            nwarmup=25,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
        )
        # Predictions must be identical
        np.testing.assert_array_equal(
            result_false.predictions, result_default.predictions
        )


# ---------------------------------------------------------------------------
# Boundary and edge cases (5 tests)
# ---------------------------------------------------------------------------


class TestDynamicRegressionBoundary:
    def test_k0_no_covariates_falls_back_gracefully(self):
        """k=0 with dynamic_regression=True should run like static."""
        rng = np.random.default_rng(42)
        y = rng.normal(10, 1, 30)
        result = _run_sampler_dynamic(y, [], 20, niter=20, nwarmup=10)
        preds = np.array(result.predictions)
        assert preds.shape == (20, 10)
        assert np.all(np.isfinite(preds))

    def test_k1_single_covariate_runs_without_error(self):
        y, x, pre_end = _make_data_dynamic_k1_constant_beta()
        result = _run_sampler_dynamic(y, x, pre_end, niter=100, nwarmup=50)
        preds = np.array(result.predictions)
        assert np.all(np.isfinite(preds))

    def test_k2_multiple_covariates_runs_without_error(self):
        y, x, pre_end = _make_data_dynamic_k2()
        result = _run_sampler_dynamic(y, x, pre_end, niter=100, nwarmup=50)
        preds = np.array(result.predictions)
        assert np.all(np.isfinite(preds))

    def test_minimum_tpre_2_does_not_crash(self):
        """T_pre=2 is the minimum for random walk (needs at least 1 diff)."""
        rng = np.random.default_rng(42)
        y = rng.normal(5, 0.5, 5)
        x = [rng.normal(0, 1, 5).tolist()]
        result = _run_sampler_dynamic(y, x, 2, niter=20, nwarmup=10)
        preds = np.array(result.predictions)
        assert np.all(np.isfinite(preds))

    def test_very_large_k_relative_to_tpre_runs_without_nan(self):
        """k=T_pre-1 (k=9, T_pre=10) for numerical stability check."""
        rng = np.random.default_rng(42)
        t_pre, k = 10, 9
        y = rng.normal(0, 1, 15)
        x = [rng.normal(0, 1, 15).tolist() for _ in range(k)]
        result = _run_sampler_dynamic(y, x, t_pre, niter=20, nwarmup=10)
        preds = np.array(result.predictions)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Statistical quality (3 tests)
# ---------------------------------------------------------------------------


class TestDynamicRegressionStatistical:
    def test_constant_beta_predictions_reasonable(self):
        """beta=2.0 constant data: predictions within +-20% of y_post."""
        y, x, pre_end = _make_data_dynamic_k1_constant_beta(n=200, seed=123)
        result = _run_sampler_dynamic(y, x, pre_end, niter=500, nwarmup=250, seed=123)
        preds = np.array(result.predictions)
        y_post = np.array(y[pre_end:])
        pred_mean = preds.mean(axis=0)
        y_post_mean = y_post.mean()
        pred_mean_overall = pred_mean.mean()
        assert abs(pred_mean_overall - y_post_mean) < 0.2 * abs(y_post_mean) + 0.5

    def test_structural_break_predictions_differ_from_static(self):
        """Structural break data: dynamic and static RMSE should differ."""
        y, x, pre_end = _make_data_dynamic_k1_structural_break(n=200, seed=99)
        result_dyn = _run_sampler_dynamic(
            y, x, pre_end, niter=500, nwarmup=250, seed=99
        )
        result_stat = _run_sampler_static(
            y, x, pre_end, niter=500, nwarmup=250, seed=99
        )
        y_post = np.array(y[pre_end:])
        rmse_dyn = np.sqrt(
            ((np.array(result_dyn.predictions).mean(axis=0) - y_post) ** 2).mean()
        )
        rmse_stat = np.sqrt(
            ((np.array(result_stat.predictions).mean(axis=0) - y_post) ** 2).mean()
        )
        # They should not be equal (dynamic adapts, static doesn't)
        assert rmse_dyn != pytest.approx(rmse_stat, rel=0.01)

    def test_post_period_predictions_no_nan(self):
        y, x, pre_end = _make_data_dynamic_k1_constant_beta()
        result = _run_sampler_dynamic(y, x, pre_end, niter=200, nwarmup=100)
        preds = np.array(result.predictions)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# Integration tests (3 tests)
# ---------------------------------------------------------------------------


class TestDynamicRegressionIntegration:
    def test_causal_impact_end_to_end(self):
        rng = np.random.default_rng(42)
        n = 80
        import pandas as pd

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 0.3, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x": x}, index=dates)
        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"dynamic_regression": True, "niter": 200, "nwarmup": 100},
        )
        assert ci.summary() is not None

    def test_causal_impact_summary_and_inferences(self):
        rng = np.random.default_rng(42)
        n = 80
        import pandas as pd

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 0.3, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x": x}, index=dates)
        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"dynamic_regression": True, "niter": 200, "nwarmup": 100},
        )
        summary = ci.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        inferences = ci.inferences
        assert isinstance(inferences, pd.DataFrame)
        assert len(inferences) > 0

    def test_causal_impact_plot_runs(self):
        rng = np.random.default_rng(42)
        n = 80
        import pandas as pd

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 0.3, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x": x}, index=dates)
        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"dynamic_regression": True, "niter": 200, "nwarmup": 100},
        )
        import matplotlib

        matplotlib.use("Agg")
        fig = ci.plot()
        assert fig is not None
