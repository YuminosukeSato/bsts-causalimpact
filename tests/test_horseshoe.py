"""Horseshoe prior specification tests.

Each test name encodes: test_horseshoe_{condition}_{expected_outcome}
These tests serve as the specification for the horseshoe prior feature.

Reference: Kohns & Bhattacharjee (2022), arXiv:2011.00938
           Makalic & Schmidt (2015), IEEE Signal Processing Letters 23(1).
"""

import numpy as np
import pandas as pd
import pytest
from causal_impact import CausalImpact, ModelOptions
from causal_impact._core import run_gibbs_sampler

# ---------------------------------------------------------------------------
# Fixtures: deterministic data generators
# ---------------------------------------------------------------------------


def _make_data_k1_strong(n=100, pre_frac=0.7, seed=42):
    """k=1, x strongly correlated with y."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x = rng.normal(5, 1, n)
    y = 2.0 * x + rng.normal(0, 0.3, n)
    return y.tolist(), [x.tolist()], pre_end


def _make_data_k2_signal_noise(n=100, pre_frac=0.7, seed=42):
    """k=2: x1 is signal (coeff=2.0), x2 is pure noise."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x1 = rng.normal(5, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 * x1 + rng.normal(0, 0.3, n)
    return y.tolist(), [x1.tolist(), x2.tolist()], pre_end


def _make_data_k3_all_noise(n=100, pre_frac=0.7, seed=42):
    """k=3: all covariates are independent noise."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
    return y.tolist(), [x1.tolist(), x2.tolist(), x3.tolist()], pre_end


def _make_data_k3_all_signal(n=100, pre_frac=0.7, seed=42):
    """k=3: all covariates contribute to y (dense DGP)."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x1 = rng.normal(5, 1, n)
    x2 = rng.normal(3, 1, n)
    x3 = rng.normal(1, 1, n)
    y = 1.0 * x1 + 0.8 * x2 + 0.6 * x3 + rng.normal(0, 0.3, n)
    return y.tolist(), [x1.tolist(), x2.tolist(), x3.tolist()], pre_end


def _make_data_k10_dense(n=200, pre_frac=0.7, seed=42):
    """k=10: 5 signal + 5 noise covariates (dense DGP)."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    xs = [rng.normal(0, 1, n) for _ in range(10)]
    coeffs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    y = sum(c * x for c, x in zip(coeffs, xs)) + rng.normal(0, 0.3, n)
    return y.tolist(), [x.tolist() for x in xs], pre_end


def _run_sampler_with_horseshoe(
    y, x, pre_end, niter=200, nwarmup=100, seed=42, nchains=1
):
    """Helper to call run_gibbs_sampler with prior_type='horseshoe'."""
    return run_gibbs_sampler(
        y=y,
        x=x if x else None,
        pre_end=pre_end,
        niter=niter,
        nwarmup=nwarmup,
        nchains=nchains,
        seed=seed,
        prior_level_sd=0.01,
        expected_model_size=1.0,
        prior_type="horseshoe",
    )


# ---------------------------------------------------------------------------
# Class 1: Output shape
# ---------------------------------------------------------------------------


class TestHorseshoeOutputShape:
    """Verify output shapes and value ranges of kappa_shrinkage."""

    def test_horseshoe_kappa_shrinkage_shape_is_niter_by_k(self):
        """Given k=3 data, kappa_shrinkage has shape (niter, k)."""
        y, x, pre_end = _make_data_k3_all_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa.shape == (200, 3)

    def test_horseshoe_kappa_shrinkage_values_are_between_0_and_1_exclusive(self):
        """Given k=3 data, all kappa values in (0, 1)."""
        y, x, pre_end = _make_data_k3_all_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        kappa = np.array(samples.kappa_shrinkage)
        assert np.all(kappa > 0)
        assert np.all(kappa < 1)

    def test_horseshoe_beta_shape_matches_niter_by_k(self):
        """Given k=2 data, beta has shape (niter, k)."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        beta = np.array(samples.beta)
        assert beta.shape == (200, 2)

    def test_horseshoe_gamma_is_empty_list(self):
        """Horseshoe does not use gamma (spike-and-slab disabled)."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        assert samples.gamma == []

    def test_horseshoe_predictions_shape_is_niter_by_post_length(self):
        """Given n=100, pre_end=70, predictions shape = (niter, 30)."""
        y, x, pre_end = _make_data_k1_strong(n=100)
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        assert len(samples.predictions) == 200
        assert len(samples.predictions[0]) == 30

    def test_horseshoe_k1_kappa_shape_is_niter_by_1(self):
        """Boundary: k=1 (minimum covariates)."""
        y, x, pre_end = _make_data_k1_strong()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa.shape == (200, 1)

    def test_horseshoe_nchains_2_kappa_shape_is_double_niter_by_k(self):
        """Boundary: nchains=2 doubles samples."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=100, nchains=2
        )

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa.shape == (200, 2)


# ---------------------------------------------------------------------------
# Class 2: Shrinkage behavior
# ---------------------------------------------------------------------------


class TestHorseshoeShrinkageBehavior:
    """Verify that horseshoe correctly shrinks noise and preserves signal."""

    def test_horseshoe_signal_covariate_has_low_mean_kappa(self):
        """Given k=2 with signal+noise, signal covariate has low kappa."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa[:, 0].mean() < 0.5

    def test_horseshoe_noise_covariate_has_high_mean_kappa(self):
        """Given k=2 with signal+noise, noise covariate has high kappa."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa[:, 1].mean() > 0.5

    def test_horseshoe_k3_all_noise_all_kappa_above_threshold(self):
        """Given k=3 all-noise DGP, all kappa should be above 0.3."""
        y, x, pre_end = _make_data_k3_all_noise()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        kappa = np.array(samples.kappa_shrinkage)
        for j in range(3):
            assert kappa[:, j].mean() > 0.3

    def test_horseshoe_k3_all_signal_all_kappa_below_threshold(self):
        """Given k=3 all-signal DGP, all kappa should be below 0.7."""
        y, x, pre_end = _make_data_k3_all_signal()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        kappa = np.array(samples.kappa_shrinkage)
        for j in range(3):
            assert kappa[:, j].mean() < 0.7

    def test_horseshoe_k1_strong_signal_kappa_near_zero(self):
        """Given k=1 strong signal (coeff=2.0), kappa should be near 0."""
        y, x, pre_end = _make_data_k1_strong()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa[:, 0].mean() < 0.3


# ---------------------------------------------------------------------------
# Class 3: Posterior shrinkage property
# ---------------------------------------------------------------------------


class TestHorseshoePosteriorShrinkage:
    """Verify posterior_shrinkage property via Python API."""

    def test_horseshoe_posterior_shrinkage_shape_is_k(self):
        """posterior_shrinkage has shape (k,)."""
        y, x, pre_end = _make_data_k3_all_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end)

        kappa = np.array(samples.kappa_shrinkage)
        shrinkage = kappa.mean(axis=0)
        assert shrinkage.shape == (3,)

    def test_horseshoe_posterior_shrinkage_values_between_0_and_1(self):
        """All posterior_shrinkage values in [0, 1]."""
        y, x, pre_end = _make_data_k3_all_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end)

        kappa = np.array(samples.kappa_shrinkage)
        shrinkage = kappa.mean(axis=0)
        assert np.all(shrinkage >= 0)
        assert np.all(shrinkage <= 1)

    def test_horseshoe_posterior_shrinkage_signal_lower_than_noise(self):
        """Signal covariate has lower shrinkage than noise."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        kappa = np.array(samples.kappa_shrinkage)
        shrinkage = kappa.mean(axis=0)
        assert shrinkage[0] < shrinkage[1]


# ---------------------------------------------------------------------------
# Class 4: Positivity
# ---------------------------------------------------------------------------


class TestHorseshoePositivity:
    """Verify variance parameters are positive via kappa output."""

    def test_horseshoe_kappa_all_positive(self):
        """All kappa values are strictly positive."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        kappa = np.array(samples.kappa_shrinkage)
        assert np.all(kappa > 0)

    def test_horseshoe_kappa_all_less_than_one(self):
        """All kappa values are strictly less than one."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        kappa = np.array(samples.kappa_shrinkage)
        assert np.all(kappa < 1)


# ---------------------------------------------------------------------------
# Class 5: Backward compatibility
# ---------------------------------------------------------------------------


class TestHorseshoeBackwardCompat:
    """Spike-and-slab behavior must be unchanged after horseshoe addition."""

    def test_spike_slab_default_kappa_shrinkage_is_empty(self):
        """Default (spike_slab) prior returns empty kappa_shrinkage."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = run_gibbs_sampler(
            y=y,
            x=x,
            pre_end=pre_end,
            niter=200,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
        )

        assert samples.kappa_shrinkage == []

    def test_spike_slab_default_gamma_unchanged_after_horseshoe_addition(self):
        """Spike-and-slab gamma shape and type are preserved."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = run_gibbs_sampler(
            y=y,
            x=x,
            pre_end=pre_end,
            niter=200,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
        )

        gamma = np.array(samples.gamma)
        assert gamma.shape == (200, 2)
        assert gamma.dtype == bool

    def test_no_covariates_kappa_shrinkage_is_empty_regardless_of_prior_type(self):
        """k=0 with horseshoe still returns empty kappa."""
        y = np.cumsum(np.random.default_rng(42).normal(0, 0.1, 50)) + 10.0
        samples = run_gibbs_sampler(
            y=y.tolist(),
            x=None,
            pre_end=35,
            niter=50,
            nwarmup=25,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            prior_type="horseshoe",
        )

        assert samples.kappa_shrinkage == []


# ---------------------------------------------------------------------------
# Class 6: Numerical stability
# ---------------------------------------------------------------------------


class TestHorseshoeNumericalStability:
    """Edge cases that might cause numerical issues."""

    def test_horseshoe_zero_variance_covariate_no_crash(self):
        """Constant covariate should not crash."""
        n = 50
        y = [10.0 + 0.1 * i for i in range(n)]
        x = [[5.0] * n]  # constant
        samples = _run_sampler_with_horseshoe(y, x, 35, niter=50, nwarmup=25)

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa.shape == (50, 1)
        assert np.all(np.isfinite(kappa))

    def test_horseshoe_near_zero_variance_covariate_no_crash(self):
        """Near-constant covariate should not crash or produce NaN."""
        n = 50
        y = [10.0 + 0.1 * i for i in range(n)]
        x = [[1.0 + 1e-15 * i for i in range(n)]]
        samples = _run_sampler_with_horseshoe(y, x, 35, niter=50, nwarmup=25)

        kappa = np.array(samples.kappa_shrinkage)
        assert np.all(np.isfinite(kappa))

    def test_horseshoe_pre_end_3_minimal_no_crash(self):
        """Minimal pre_end=3 should not crash."""
        y = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        samples = _run_sampler_with_horseshoe(y, x, 3, niter=50, nwarmup=25)

        assert len(samples.predictions) == 50

    def test_horseshoe_large_k_10_no_crash(self):
        """k=10 covariates should not crash."""
        y, x, pre_end = _make_data_k10_dense()
        samples = _run_sampler_with_horseshoe(y, x, pre_end, niter=200)

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa.shape == (200, 10)

    def test_horseshoe_very_small_sigma2_obs_no_nan(self):
        """Near-zero variance y should not produce NaN in kappa."""
        n = 50
        y = [10.0 + 1e-8 * i for i in range(n)]
        x = [[float(i) for i in range(n)]]
        samples = _run_sampler_with_horseshoe(y, x, 35, niter=50, nwarmup=25)

        kappa = np.array(samples.kappa_shrinkage)
        assert not np.any(np.isnan(kappa))
        assert np.all(np.isfinite(kappa))

    def test_horseshoe_multicollinear_covariates_no_crash(self):
        """Perfect collinearity: x2 = x1."""
        n = 50
        rng = np.random.default_rng(42)
        x1 = rng.normal(5, 1, n)
        y = (2.0 * x1 + rng.normal(0, 0.3, n)).tolist()
        x = [x1.tolist(), x1.tolist()]  # perfect collinearity
        samples = _run_sampler_with_horseshoe(y, x, 35, niter=50, nwarmup=25)

        kappa = np.array(samples.kappa_shrinkage)
        assert not np.any(np.isnan(kappa))


# ---------------------------------------------------------------------------
# Class 7: Validation
# ---------------------------------------------------------------------------


class TestHorseshoeValidation:
    """Input validation for horseshoe prior."""

    def test_model_options_invalid_prior_type_raises_value_error(self):
        with pytest.raises(ValueError, match="prior_type"):
            ModelOptions(prior_type="lasso")

    def test_model_options_prior_type_empty_string_raises(self):
        with pytest.raises(ValueError, match="prior_type"):
            ModelOptions(prior_type="")

    def test_model_options_prior_type_spike_slab_is_valid(self):
        opts = ModelOptions(prior_type="spike_slab")
        assert opts.prior_type == "spike_slab"

    def test_model_options_prior_type_horseshoe_is_valid(self):
        opts = ModelOptions(prior_type="horseshoe")
        assert opts.prior_type == "horseshoe"

    def test_model_options_prior_type_default_is_spike_slab(self):
        opts = ModelOptions()
        assert opts.prior_type == "spike_slab"

    def test_horseshoe_with_dynamic_regression_raises_value_error(self):
        y = [10.0 + 0.1 * i for i in range(20)]
        x = [[float(i) for i in range(20)]]
        with pytest.raises(ValueError, match="dynamic_regression"):
            run_gibbs_sampler(
                y=y,
                x=x,
                pre_end=15,
                niter=10,
                nwarmup=5,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
                expected_model_size=1.0,
                dynamic_regression=True,
                prior_type="horseshoe",
            )

    def test_model_options_horseshoe_with_dynamic_regression_raises(self):
        with pytest.raises(ValueError, match="dynamic_regression"):
            ModelOptions(prior_type="horseshoe", dynamic_regression=True)

    def test_horseshoe_with_retrospective_mode_raises_value_error(self):
        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x = rng.normal(5, 1, n)
        y = 1.0 * x + rng.normal(0, 0.5, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x1": x}, index=dates)
        pre = ["2020-01-01", "2020-02-25"]
        post = ["2020-02-26", "2020-03-20"]
        with pytest.raises(ValueError, match="retrospective"):
            CausalImpact(
                df,
                pre,
                post,
                model_args={
                    "niter": 100,
                    "nwarmup": 50,
                    "seed": 1,
                    "prior_type": "horseshoe",
                    "mode": "retrospective",
                },
            )

    def test_horseshoe_nan_in_pre_period_raises_error(self):
        y = [10.0, 11.0, float("nan"), 13.0, 14.0, 15.0, 16.0, 17.0]
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
        with pytest.raises(ValueError, match="NaN"):
            _run_sampler_with_horseshoe(y, x, 5, niter=10, nwarmup=5)


# ---------------------------------------------------------------------------
# Class 8: Python API integration
# ---------------------------------------------------------------------------


def _make_test_data_with_effect():
    """DataFrame with 1 covariate and a true effect of 3.0."""
    rng = np.random.default_rng(42)
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(5, 1, n)
    y = 1.0 * x + rng.normal(0, 0.5, n)
    y[56:] += 3.0
    df = pd.DataFrame({"y": y, "x1": x}, index=dates)
    pre = ["2020-01-01", "2020-02-25"]
    post = ["2020-02-26", "2020-03-20"]
    return df, pre, post


class TestHorseshoePythonAPI:
    """End-to-end tests via CausalImpact class."""

    def test_causal_impact_horseshoe_end_to_end_summary_not_none(self):
        df, pre, post = _make_test_data_with_effect()
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "horseshoe",
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
            },
        )

        assert ci.summary() is not None

    def test_causal_impact_horseshoe_posterior_shrinkage_shape_matches_covariates(
        self,
    ):
        df, pre, post = _make_test_data_with_effect()
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "horseshoe",
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
            },
        )

        shrinkage = ci.posterior_shrinkage
        assert shrinkage is not None
        assert shrinkage.shape == (1,)

    def test_causal_impact_horseshoe_posterior_shrinkage_in_unit_interval(self):
        df, pre, post = _make_test_data_with_effect()
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "horseshoe",
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
            },
        )

        shrinkage = ci.posterior_shrinkage
        assert np.all(shrinkage >= 0)
        assert np.all(shrinkage <= 1)

    def test_causal_impact_horseshoe_pip_is_none(self):
        """posterior_inclusion_probs should be None for horseshoe."""
        df, pre, post = _make_test_data_with_effect()
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "horseshoe",
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
            },
        )

        assert ci.posterior_inclusion_probs is None

    def test_causal_impact_horseshoe_detects_positive_effect(self):
        df, pre, post = _make_test_data_with_effect()
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "horseshoe",
                "niter": 200,
                "nwarmup": 100,
                "seed": 42,
            },
        )

        assert ci.summary_stats["point_effect_mean"] > 0

    def test_causal_impact_horseshoe_dict_and_model_options_produce_same_result(
        self,
    ):
        df, pre, post = _make_test_data_with_effect()
        params_dict = {
            "niter": 200,
            "nwarmup": 100,
            "seed": 42,
            "prior_type": "horseshoe",
        }
        params_opts = ModelOptions(
            niter=200, nwarmup=100, seed=42, prior_type="horseshoe"
        )

        ci_dict = CausalImpact(df, pre, post, model_args=params_dict)
        ci_opts = CausalImpact(df, pre, post, model_args=params_opts)

        assert ci_dict.summary_stats["point_effect_mean"] == pytest.approx(
            ci_opts.summary_stats["point_effect_mean"], rel=1e-10
        )

    def test_causal_impact_horseshoe_k0_posterior_shrinkage_is_none(self):
        """No covariates + horseshoe → posterior_shrinkage is None."""
        rng = np.random.default_rng(42)
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
        y[42:] += 2.0
        df = pd.DataFrame({"y": y}, index=dates)
        pre = ["2020-01-01", "2020-02-11"]
        post = ["2020-02-12", "2020-02-29"]
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "horseshoe",
                "niter": 100,
                "nwarmup": 50,
                "seed": 42,
            },
        )

        assert ci.posterior_shrinkage is None

    def test_causal_impact_spike_slab_posterior_shrinkage_is_none(self):
        """Spike-slab prior → posterior_shrinkage is None."""
        df, pre, post = _make_test_data_with_effect()
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={
                "prior_type": "spike_slab",
                "niter": 100,
                "nwarmup": 50,
                "seed": 42,
            },
        )

        assert ci.posterior_shrinkage is None


# ---------------------------------------------------------------------------
# Class 9: Dense DGP comparison
# ---------------------------------------------------------------------------


class TestHorseshoeDenseDGPComparison:
    """Compare horseshoe vs spike_slab on dense DGP scenarios."""

    def test_horseshoe_dense_dgp_mse_not_worse_than_spike_slab(self):
        """Horseshoe MSE should not be dramatically worse on dense DGP."""
        y, x, pre_end = _make_data_k10_dense()

        samples_hs = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )
        samples_ss = run_gibbs_sampler(
            y=y,
            x=x,
            pre_end=pre_end,
            niter=500,
            nwarmup=250,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
            prior_type="spike_slab",
        )

        y_post = np.array(y[pre_end:])
        pred_hs = np.array(samples_hs.predictions).mean(axis=0)
        pred_ss = np.array(samples_ss.predictions).mean(axis=0)
        mse_hs = np.mean((y_post - pred_hs) ** 2)
        mse_ss = np.mean((y_post - pred_ss) ** 2)

        assert mse_hs <= mse_ss * 1.5

    def test_horseshoe_dense_dgp_predictions_finite_and_within_3sd(self):
        """Dense DGP (5 signal + 5 noise): finite and reasonable."""
        y, x, pre_end = _make_data_k10_dense()
        samples = _run_sampler_with_horseshoe(
            y, x, pre_end, niter=500, nwarmup=250
        )

        pred = np.array(samples.predictions)
        assert np.all(np.isfinite(pred))

        y_post = np.array(y[pre_end:])
        pred_mean = pred.mean(axis=0)
        pred_sd = pred.std(axis=0)
        # Predictions should be within 3 SD of post-period mean
        assert np.abs(pred_mean.mean() - y_post.mean()) < 3 * pred_sd.mean()


# ---------------------------------------------------------------------------
# Class 10: Seasonal + horseshoe
# ---------------------------------------------------------------------------


class TestHorseshoeWithSeasonal:
    """Horseshoe + seasonal state-space model."""

    def test_horseshoe_with_seasonal_kappa_shape_correct(self):
        """Seasonal + horseshoe: kappa shape = (niter, k)."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(5, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2.0 * x1 + rng.normal(0, 0.3, n)
        samples = run_gibbs_sampler(
            y=y.tolist(),
            x=[x1.tolist(), x2.tolist()],
            pre_end=70,
            niter=200,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
            nseasons=7.0,
            season_duration=1.0,
            prior_type="horseshoe",
        )

        kappa = np.array(samples.kappa_shrinkage)
        assert kappa.shape == (200, 2)

    def test_horseshoe_with_seasonal_no_nan_in_predictions(self):
        """Seasonal + horseshoe: all predictions are finite."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(5, 1, n)
        y = 2.0 * x1 + rng.normal(0, 0.3, n)
        samples = run_gibbs_sampler(
            y=y.tolist(),
            x=[x1.tolist()],
            pre_end=70,
            niter=200,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
            nseasons=7.0,
            season_duration=1.0,
            prior_type="horseshoe",
        )

        pred = np.array(samples.predictions)
        assert np.all(np.isfinite(pred))
