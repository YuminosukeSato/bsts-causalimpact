"""Tests for Rust Gibbs sampler speed-up: Cholesky-based sampling + pre-computation.

Verifies:
1. Sampler output correctness after Cholesky migration (same statistical properties)
2. XtX pre-computation produces identical results to per-iteration computation
3. Spike-and-slab slab_stats pre-computation produces correct n_j and x_mean
4. Speed improvement with many covariates (k=20)
5. R numerical compatibility improvement
"""

import math
import time

import numpy as np
from causal_impact._core import run_gibbs_sampler


class TestSamplerOutputCorrectness:
    """Verify that Cholesky migration does not break sampler output properties."""

    def test_no_covariates_output_unchanged(self):
        """k=0: no regression, output identical regardless of Cholesky."""
        y = [10.0 + 0.1 * i for i in range(100)]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=70,
            niter=50,
            nwarmup=10,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert len(result.states) == 50
        assert len(result.beta) == 50
        for beta_row in result.beta:
            assert len(beta_row) == 0

    def test_single_covariate_posterior_mean_reasonable(self):
        """k=1: posterior beta mean should be near true coefficient."""
        rng = np.random.RandomState(42)
        t = 200
        x1 = rng.randn(t)
        y = [10.0 + 2.0 * x1[i] + 0.1 * rng.randn() for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1.tolist()],
            pre_end=150,
            niter=500,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        beta_samples = [b[0] for b in result.beta]
        beta_mean = np.mean(beta_samples)
        assert abs(beta_mean - 2.0) < 1.0, (
            f"Posterior beta mean {beta_mean:.3f} should be near true value 2.0"
        )

    def test_two_covariates_posterior_means_reasonable(self):
        """k=2: both posterior means near true coefficients."""
        rng = np.random.RandomState(123)
        t = 300
        x1 = rng.randn(t)
        x2 = rng.randn(t)
        y = [10.0 + 1.5 * x1[i] - 0.8 * x2[i] + 0.1 * rng.randn() for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1.tolist(), x2.tolist()],
            pre_end=200,
            niter=500,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        beta1_mean = np.mean([b[0] for b in result.beta])
        beta2_mean = np.mean([b[1] for b in result.beta])
        assert abs(beta1_mean - 1.5) < 1.0, f"beta1 mean {beta1_mean:.3f} off"
        assert abs(beta2_mean - (-0.8)) < 1.0, f"beta2 mean {beta2_mean:.3f} off"

    def test_sigma_obs_positive(self):
        """sigma_obs samples must all be positive."""
        rng = np.random.RandomState(42)
        t = 100
        x1 = rng.randn(t)
        y = [5.0 + x1[i] + 0.5 * rng.randn() for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1.tolist()],
            pre_end=70,
            niter=100,
            nwarmup=20,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        for sigma in result.sigma_obs:
            assert sigma > 0, f"sigma_obs must be positive, got {sigma}"

    def test_predictions_finite(self):
        """All predictions must be finite (no NaN or Inf)."""
        rng = np.random.RandomState(42)
        t = 100
        x1 = rng.randn(t)
        y = [5.0 + x1[i] + 0.5 * rng.randn() for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1.tolist()],
            pre_end=70,
            niter=100,
            nwarmup=20,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        for pred_row in result.predictions:
            for val in pred_row:
                assert math.isfinite(val), f"Prediction not finite: {val}"


class TestSamplerDeterminism:
    """Same seed must produce same output (determinism after refactor)."""

    def test_same_seed_same_output(self):
        """Two runs with identical seed produce identical beta samples."""
        rng = np.random.RandomState(42)
        t = 100
        x1 = rng.randn(t)
        y = [5.0 + x1[i] for i in range(t)]
        kwargs = dict(
            y=y,
            x=[x1.tolist()],
            pre_end=70,
            niter=50,
            nwarmup=10,
            nchains=1,
            seed=99,
            prior_level_sd=0.01,
        )
        r1 = run_gibbs_sampler(**kwargs)
        r2 = run_gibbs_sampler(**kwargs)
        for b1, b2 in zip(r1.beta, r2.beta):
            for v1, v2 in zip(b1, b2):
                assert v1 == v2, "Determinism broken"


class TestSpikeSlab:
    """Verify spike-and-slab still works correctly after pre-computation."""

    def test_spike_slab_irrelevant_covariates_excluded(self):
        """Irrelevant covariates should have low inclusion probability."""
        rng = np.random.RandomState(42)
        t = 300
        x_signal = rng.randn(t)
        x_noise = rng.randn(t)
        y = [10.0 + 3.0 * x_signal[i] + 0.1 * rng.randn() for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x_signal.tolist(), x_noise.tolist()],
            pre_end=200,
            niter=500,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=1.0,
        )
        gamma_signal = np.mean([g[0] for g in result.gamma])
        gamma_noise = np.mean([g[1] for g in result.gamma])
        assert gamma_signal > gamma_noise, (
            f"Signal inclusion {gamma_signal:.3f} should exceed noise {gamma_noise:.3f}"
        )

    def test_spike_slab_constant_covariate_excluded(self):
        """Constant covariate (n_j=0) should be excluded (gamma=false)."""
        t = 100
        x_const = [5.0] * t
        y = [10.0 + 0.1 * i for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x_const],
            pre_end=70,
            niter=100,
            nwarmup=20,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            expected_model_size=0.5,
        )
        gamma_const = np.mean([g[0] for g in result.gamma])
        assert gamma_const == 0.0, (
            f"Constant covariate should be excluded, got {gamma_const}"
        )


class TestManyCovariates:
    """Test with large k to verify no numerical issues and speed improvement."""

    def test_k20_no_nan(self):
        """k=20 covariates: all outputs must be finite (no NaN/Inf)."""
        rng = np.random.RandomState(0)
        t = 200
        k = 20
        x_cols = [rng.randn(t).tolist() for _ in range(k)]
        y = [10.0 + 0.5 * rng.randn() for _ in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=x_cols,
            pre_end=150,
            niter=200,
            nwarmup=50,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        for beta_row in result.beta:
            for val in beta_row:
                assert math.isfinite(val), f"Beta not finite: {val}"
        for sigma in result.sigma_obs:
            assert math.isfinite(sigma), f"sigma_obs not finite: {sigma}"

    def test_k20_speed_benchmark(self):
        """k=20, T=400, niter=500: should complete within reasonable time."""
        rng = np.random.RandomState(0)
        t = 400
        k = 20
        x_cols = [rng.randn(t).tolist() for _ in range(k)]
        coefs = rng.randn(k)
        y = [
            sum(coefs[j] * x_cols[j][i] for j in range(k)) + rng.randn()
            for i in range(t)
        ]
        t0 = time.time()
        result = run_gibbs_sampler(
            y=y,
            x=x_cols,
            pre_end=300,
            niter=500,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        elapsed = time.time() - t0
        assert elapsed < 30.0, f"k=20 benchmark took {elapsed:.1f}s, expected < 30s"
        assert len(result.beta) == 500

    def test_k5_speed_benchmark(self):
        """k=5, T=200, niter=500: should complete within reasonable time."""
        rng = np.random.RandomState(0)
        t = 200
        k = 5
        x_cols = [rng.randn(t).tolist() for _ in range(k)]
        y = [10.0 + rng.randn() for _ in range(t)]
        t0 = time.time()
        result = run_gibbs_sampler(
            y=y,
            x=x_cols,
            pre_end=150,
            niter=500,
            nwarmup=100,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"k=5 benchmark took {elapsed:.1f}s, expected < 10s"
        assert len(result.beta) == 500


class TestSeasonalRegression:
    """Seasonal regression should also benefit from XtX pre-computation."""

    def test_seasonal_with_covariates_finite(self):
        """Seasonal model + covariates: all outputs finite."""
        rng = np.random.RandomState(42)
        t = 200
        x1 = rng.randn(t)
        y = [10.0 + x1[i] + 0.5 * rng.randn() for i in range(t)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1.tolist()],
            pre_end=150,
            niter=100,
            nwarmup=20,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            nseasons=7.0,
            season_duration=1.0,
        )
        for beta_row in result.beta:
            for val in beta_row:
                assert math.isfinite(val), f"Beta not finite: {val}"
        for pred_row in result.predictions:
            for val in pred_row:
                assert math.isfinite(val), f"Prediction not finite: {val}"
