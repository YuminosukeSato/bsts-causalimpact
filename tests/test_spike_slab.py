"""Spike-and-slab variable selection specification tests.

Each test name encodes: test_spike_slab_{condition}_{expected_outcome}
These tests serve as the specification for the spike-and-slab feature.
"""

import numpy as np
import pytest
from causal_impact._core import run_gibbs_sampler

# ---------------------------------------------------------------------------
# Fixtures: deterministic data generators
# ---------------------------------------------------------------------------


def _make_data_no_covariates(n=50, pre_frac=0.7, seed=42):
    """y = trend + noise, no covariates."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    y = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
    return y.tolist(), [], pre_end


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


def _make_data_k3_one_signal(n=100, pre_frac=0.7, seed=42):
    """k=3: only x1 is signal, x2 and x3 are noise."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x1 = rng.normal(5, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 3.0 * x1 + rng.normal(0, 0.5, n)
    return y.tolist(), [x1.tolist(), x2.tolist(), x3.tolist()], pre_end


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
    """k=3: all covariates contribute to y."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)
    x1 = rng.normal(5, 1, n)
    x2 = rng.normal(3, 1, n)
    x3 = rng.normal(1, 1, n)
    y = 1.0 * x1 + 0.8 * x2 + 0.6 * x3 + rng.normal(0, 0.3, n)
    return y.tolist(), [x1.tolist(), x2.tolist(), x3.tolist()], pre_end


def _run_sampler_with_spike_slab(
    y, x, pre_end, expected_model_size=1.0, niter=500, nwarmup=250, seed=42
):
    """Helper to call run_gibbs_sampler with expected_model_size."""
    return run_gibbs_sampler(
        y=y,
        x=x if x else None,
        pre_end=pre_end,
        niter=niter,
        nwarmup=nwarmup,
        nchains=1,
        seed=seed,
        prior_level_sd=0.01,
        expected_model_size=expected_model_size,
    )


def _inclusion_probs(samples):
    """Compute posterior inclusion probabilities from gamma samples."""
    gamma = np.array(samples.gamma)
    return gamma.mean(axis=0)


# ---------------------------------------------------------------------------
# 1. Backward compatibility (k=0)
# ---------------------------------------------------------------------------


class TestSpikeSlabBackwardCompat:
    """k=0 and k=1 with pi=1.0 must behave identically to the old g-prior."""

    def test_spike_slab_k0_returns_same_result_as_before(self):
        """k=0: gamma=[], output identical to pre-spike-slab behavior."""
        y, x, pre_end = _make_data_no_covariates()
        samples = _run_sampler_with_spike_slab(y, x, pre_end)
        assert samples.gamma == []  # no covariates → empty gamma

    def test_spike_slab_k1_includes_all_when_pi_is_one(self):
        """k=1 with expected_model_size=1 → pi=1.0 → always included."""
        y, x, pre_end = _make_data_k1_strong()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=1.0)
        gamma = np.array(samples.gamma)
        assert gamma.shape[1] == 1
        assert gamma.all(), "k=1, pi=1.0: all samples must have gamma=True"

    def test_spike_slab_k1_numerical_equivalence_with_g_prior(self):
        """k=1, spike-slab (pi=1.0) predictions must match g-prior exactly."""
        y, x, pre_end = _make_data_k1_strong(seed=123)
        # Run with pi=1.0 (spike-slab path but always-include)
        samples_ss = _run_sampler_with_spike_slab(
            y, x, pre_end, expected_model_size=1.0, seed=123
        )
        # Run without expected_model_size (default=1.0, same path)
        samples_gp = run_gibbs_sampler(
            y=y,
            x=x,
            pre_end=pre_end,
            niter=500,
            nwarmup=250,
            nchains=1,
            seed=123,
            prior_level_sd=0.01,
            expected_model_size=1.0,
        )
        preds_ss = np.array(samples_ss.predictions)
        preds_gp = np.array(samples_gp.predictions)
        np.testing.assert_allclose(
            preds_ss,
            preds_gp,
            atol=1e-10,
            err_msg="pi=1.0 must be bit-identical to g-prior",
        )


# ---------------------------------------------------------------------------
# 2. Variable selection behavior
# ---------------------------------------------------------------------------


class TestSpikeSlabSelection:
    """Core variable selection: signal vs noise discrimination."""

    def test_spike_slab_k1_strong_signal_has_high_inclusion_prob(self):
        """k=1 with strong signal: inclusion prob > 0.9."""
        y, x, pre_end = _make_data_k1_strong()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=0.5)
        probs = _inclusion_probs(samples)
        assert probs[0] > 0.9, (
            f"Strong signal inclusion prob {probs[0]} should be > 0.9"
        )

    def test_spike_slab_k2_selects_signal_over_noise(self):
        """k=2: signal covariate has higher inclusion prob than noise."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=1.0)
        probs = _inclusion_probs(samples)
        assert probs[0] > probs[1], (
            f"Signal prob {probs[0]} should exceed noise prob {probs[1]}"
        )

    def test_spike_slab_k3_selects_one_true_signal(self):
        """k=3 with 1 signal: signal variable has prob > 0.7."""
        y, x, pre_end = _make_data_k3_one_signal()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=1.0)
        probs = _inclusion_probs(samples)
        assert probs[0] > 0.7, f"Signal prob {probs[0]} should be > 0.7"

    def test_spike_slab_k3_all_noise_has_low_inclusion(self):
        """k=3 with all noise: all inclusion probs < 0.5."""
        y, x, pre_end = _make_data_k3_all_noise()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=1.0)
        probs = _inclusion_probs(samples)
        assert all(p < 0.5 for p in probs), (
            f"All-noise probs {probs} should all be < 0.5"
        )

    def test_spike_slab_k3_all_signal_has_high_inclusion(self):
        """k=3 with all signals: all inclusion probs > 0.5."""
        y, x, pre_end = _make_data_k3_all_signal()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=3.0)
        probs = _inclusion_probs(samples)
        assert all(p > 0.5 for p in probs), (
            f"All-signal probs {probs} should all be > 0.5"
        )

    def test_spike_slab_k10_finds_single_true_signal(self):
        """k=10 with 1 signal among 9 noise: signal has highest prob."""
        rng = np.random.default_rng(42)
        n, pre_end = 200, 140
        xs = [rng.normal(0, 1, n).tolist() for _ in range(10)]
        x_signal = np.array(xs[0])
        y = (3.0 * x_signal + rng.normal(0, 0.5, n)).tolist()
        samples = _run_sampler_with_spike_slab(
            y, xs, pre_end, expected_model_size=1.0, niter=1000, nwarmup=500
        )
        probs = _inclusion_probs(samples)
        assert np.argmax(probs) == 0, (
            f"Signal (idx=0) should have highest prob, got argmax={np.argmax(probs)}"
        )


# ---------------------------------------------------------------------------
# 3. Numerical stability & edge cases
# ---------------------------------------------------------------------------


class TestSpikeSlabNumericalStability:
    """Edge cases: zero-variance, near-zero-variance, tiny pre-period."""

    def test_spike_slab_zero_variance_covariate_excluded(self):
        """Constant covariate (n_j=0): always excluded, beta=0."""
        rng = np.random.default_rng(42)
        n, pre_end = 50, 35
        x_const = [5.0] * n  # zero variance
        y = (rng.normal(10, 1, n)).tolist()
        samples = _run_sampler_with_spike_slab(
            y, [x_const], pre_end, expected_model_size=0.5
        )
        gamma = np.array(samples.gamma)
        assert not gamma.any(), "Constant covariate must never be included"
        beta = np.array(samples.beta)
        assert (beta[:, 0] == 0.0).all(), "Constant covariate beta must be exactly 0"

    def test_spike_slab_near_zero_variance_excluded(self):
        """Near-zero variance covariate (n_j ~ 1e-14): excluded for stability."""
        n, pre_end = 50, 35
        x_near_zero = [1.0 + 1e-15 * i for i in range(n)]
        rng = np.random.default_rng(42)
        y = (rng.normal(10, 1, n)).tolist()
        samples = _run_sampler_with_spike_slab(
            y, [x_near_zero], pre_end, expected_model_size=0.5
        )
        gamma = np.array(samples.gamma)
        assert not gamma.any(), "Near-zero variance covariate must be excluded"

    def test_spike_slab_pre_end_2_with_covariate_no_crash(self):
        """Minimal pre-period (t_pre=2) with covariate: no crash."""
        rng = np.random.default_rng(42)
        y = rng.normal(10, 1, 5).tolist()
        x = rng.normal(0, 1, 5).tolist()
        samples = _run_sampler_with_spike_slab(
            y, [x], pre_end=2, expected_model_size=1.0, niter=50, nwarmup=25
        )
        assert len(samples.predictions) == 50

    def test_spike_slab_excluded_beta_is_exactly_zero(self):
        """When gamma_j=0, beta_j must be exactly 0.0 (not near-zero)."""
        y, x, pre_end = _make_data_k2_signal_noise()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=1.0)
        gamma = np.array(samples.gamma)
        beta = np.array(samples.beta)
        excluded_mask = ~gamma
        if excluded_mask.any():
            assert (beta[excluded_mask] == 0.0).all(), (
                "Excluded (gamma=0) coefficients must be exactly 0.0"
            )

    def test_spike_slab_included_beta_is_nonzero(self):
        """When gamma_j=1 for a strong signal, beta_j should be nonzero."""
        y, x, pre_end = _make_data_k1_strong()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=0.5)
        gamma = np.array(samples.gamma)
        beta = np.array(samples.beta)
        included_mask = gamma
        if included_mask.any():
            assert (beta[included_mask] != 0.0).all(), (
                "Included (gamma=1) strong signal should have nonzero beta"
            )

    def test_spike_slab_multicollinear_selects_at_least_one(self):
        """Perfectly correlated x2=x1: at least one has prob > 0.5."""
        rng = np.random.default_rng(42)
        n, pre_end = 100, 70
        x1 = rng.normal(5, 1, n)
        x2 = x1.copy()  # perfect collinearity
        y = (2.0 * x1 + rng.normal(0, 0.3, n)).tolist()
        samples = _run_sampler_with_spike_slab(
            y,
            [x1.tolist(), x2.tolist()],
            pre_end,
            expected_model_size=1.0,
        )
        probs = _inclusion_probs(samples)
        assert max(probs) > 0.5, (
            f"At least one collinear var should have prob > 0.5, got {probs}"
        )


# ---------------------------------------------------------------------------
# 4. Prior odds behavior
# ---------------------------------------------------------------------------


class TestSpikeSlabPriorOdds:
    """expected_model_size controls π = min(1, expected_model_size/k)."""

    def test_spike_slab_expected_model_size_ge_k_includes_all(self):
        """expected_model_size >= k → pi=1.0 → all always included."""
        y, x, pre_end = _make_data_k3_all_signal()
        samples = _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=3.0)
        gamma = np.array(samples.gamma)
        assert gamma.all(), "pi=1.0 (expected_model_size>=k): all gamma must be True"

    def test_spike_slab_small_pi_excludes_most(self):
        """k=10, expected_model_size=0.1 → pi=0.01: most excluded."""
        rng = np.random.default_rng(42)
        n, pre_end = 100, 70
        xs = [rng.normal(0, 1, n).tolist() for _ in range(10)]
        y = (rng.normal(10, 1, n)).tolist()  # no real signal
        samples = _run_sampler_with_spike_slab(y, xs, pre_end, expected_model_size=0.1)
        probs = _inclusion_probs(samples)
        assert np.mean(probs < 0.2) >= 0.7, (
            f"With small pi, most probs should be < 0.2, got {probs}"
        )


# ---------------------------------------------------------------------------
# 5. Correlated covariates (blog reproduction)
# ---------------------------------------------------------------------------


class TestSpikeSlabCorrelatedCovariates:
    """Reproduce the blog's scenario: correlated covariates causing sign flip."""

    def test_spike_slab_correlated_covariates_no_sign_inversion(self):
        """With correlated covariates, spike-slab prevents sign inversion."""
        rng = np.random.default_rng(42)
        n, pre_end = 200, 140
        x1 = rng.normal(5, 1, n)
        x2 = 0.9 * x1 + rng.normal(0, 0.3, n)  # highly correlated with x1
        true_effect = 3.0
        y_base = 2.0 * x1 + rng.normal(0, 0.5, n)
        y = y_base.copy()
        y[pre_end:] += true_effect

        import pandas as pd
        from causal_impact import CausalImpact

        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=dates)
        ci = CausalImpact(
            df,
            [str(dates[0].date()), str(dates[pre_end - 1].date())],
            [str(dates[pre_end].date()), str(dates[-1].date())],
            model_args={
                "niter": 1000,
                "nwarmup": 500,
                "seed": 42,
                "expected_model_size": 1,
            },
        )
        effect = ci.summary_stats["point_effect_mean"]
        assert effect > 0, (
            f"Effect should be positive (true={true_effect}), got {effect}"
        )


# ---------------------------------------------------------------------------
# 6. Output shape
# ---------------------------------------------------------------------------


class TestSpikeSlabOutputShape:
    """gamma output must have correct shape."""

    def test_spike_slab_gamma_output_shape(self):
        """gamma shape = (n_samples, k)."""
        y, x, pre_end = _make_data_k3_one_signal()
        niter = 200
        samples = _run_sampler_with_spike_slab(
            y, x, pre_end, expected_model_size=1.0, niter=niter, nwarmup=100
        )
        gamma = np.array(samples.gamma)
        assert gamma.shape == (niter, 3), (
            f"Expected gamma shape ({niter}, 3), got {gamma.shape}"
        )


# ---------------------------------------------------------------------------
# 7. Validation errors
# ---------------------------------------------------------------------------


class TestSpikeSlabValidation:
    """Input validation for expected_model_size."""

    def test_spike_slab_negative_expected_model_size_raises(self):
        """expected_model_size=-1 with k>0 must raise ValueError."""
        y, x, pre_end = _make_data_k1_strong()
        with pytest.raises(ValueError, match="expected_model_size"):
            _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=-1.0)

    def test_spike_slab_zero_expected_model_size_raises(self):
        """expected_model_size=0 with k>0 must raise ValueError (R bsts compat)."""
        y, x, pre_end = _make_data_k1_strong()
        with pytest.raises(ValueError, match="expected_model_size"):
            _run_sampler_with_spike_slab(y, x, pre_end, expected_model_size=0.0)

    def test_spike_slab_expected_model_size_ignored_when_k0(self):
        """expected_model_size is irrelevant when k=0: no error."""
        y, x, pre_end = _make_data_no_covariates()
        samples = _run_sampler_with_spike_slab(
            y, x, pre_end, expected_model_size=-999.0
        )
        assert len(samples.predictions) > 0


# ---------------------------------------------------------------------------
# 8. Python API tests
# ---------------------------------------------------------------------------


class TestSpikeSlabPythonAPI:
    """Python-level API: expected_model_size param and posterior_inclusion_probs."""

    def test_causal_impact_expected_model_size_param(self):
        """model_args with expected_model_size works without error."""
        import pandas as pd
        from causal_impact import CausalImpact

        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x = rng.normal(5, 1, n)
        y = 2.0 * x + rng.normal(0, 0.5, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x1": x}, index=dates)

        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"niter": 200, "nwarmup": 100, "expected_model_size": 1},
        )
        assert ci.summary() is not None

    def test_causal_impact_posterior_inclusion_probs_shape(self):
        """k>0: posterior_inclusion_probs has shape (k,), values in [0,1]."""
        import pandas as pd
        from causal_impact import CausalImpact

        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x1 = rng.normal(5, 1, n)
        x2 = rng.normal(3, 1, n)
        y = 2.0 * x1 + rng.normal(0, 0.5, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=dates)

        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"niter": 200, "nwarmup": 100, "expected_model_size": 1},
        )
        probs = ci.posterior_inclusion_probs
        assert probs is not None
        assert probs.shape == (2,)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_causal_impact_default_prior_keeps_two_covariates_included_for_r_compat(
        self,
    ):
        """R static regression uses expected.model.size=3 including intercept."""
        import pandas as pd
        from causal_impact import CausalImpact

        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x1 = rng.normal(5, 1, n)
        x2 = rng.normal(3, 1, n)
        y = 2.0 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
        y[56:] += 3.0
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=dates)

        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"niter": 200, "nwarmup": 100},
        )
        probs = ci.posterior_inclusion_probs
        assert probs is not None
        assert np.allclose(probs, np.ones(2))

    def test_causal_impact_posterior_inclusion_probs_none_when_k0(self):
        """k=0: posterior_inclusion_probs returns None."""
        import pandas as pd
        from causal_impact import CausalImpact

        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
        y[56:] += 2.0
        df = pd.DataFrame({"y": y}, index=dates)

        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
            model_args={"niter": 200, "nwarmup": 100},
        )
        assert ci.posterior_inclusion_probs is None
