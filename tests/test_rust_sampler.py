"""Tests for the Rust Gibbs sampler via PyO3 binding."""

import numpy as np
import pytest
from causal_impact._core import GibbsSamples, run_gibbs_sampler


class TestSamplerShapes:
    """Output shape validation."""

    def test_sampler_returns_correct_shapes(self):
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
        assert isinstance(result, GibbsSamples)
        assert len(result.states) == 50
        assert len(result.states[0]) == 100
        assert len(result.sigma_obs) == 50
        assert len(result.sigma_level) == 50
        assert len(result.predictions) == 50
        assert len(result.predictions[0]) == 30  # 100 - 70

    def test_sampler_nchains_2(self):
        y = [10.0 + 0.1 * i for i in range(50)]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=35,
            niter=20,
            nwarmup=5,
            nchains=2,
            seed=42,
            prior_level_sd=0.01,
        )
        # niter * nchains = 40 samples
        assert len(result.states) == 40
        assert len(result.sigma_obs) == 40

    def test_sampler_keeps_post_period_states_non_constant_for_random_walk_uncertainty(
        self,
    ):
        y = [10.0 + 0.2 * i for i in range(40)]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=25,
            niter=5,
            nwarmup=2,
            nchains=1,
            seed=42,
            prior_level_sd=0.1,
        )
        post_states = result.states[0][25:]
        rounded_unique_states = {round(state, 9) for state in post_states}
        assert len(rounded_unique_states) > 1

    def test_sampler_with_seasonal_component_returns_same_public_shapes(self):
        y = [10.0 + [0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0][i % 7] for i in range(84)]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=56,
            niter=20,
            nwarmup=5,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            nseasons=7,
            season_duration=1,
        )
        assert len(result.states) == 20
        assert len(result.states[0]) == 84
        assert len(result.predictions[0]) == 28


class TestSamplerReproducibility:
    """Reproducibility with fixed seed."""

    def test_sampler_seed_reproducibility(self):
        y = [10.0 + 0.1 * i for i in range(50)]
        kwargs = dict(
            y=y,
            x=None,
            pre_end=35,
            niter=20,
            nwarmup=5,
            nchains=1,
            seed=123,
            prior_level_sd=0.01,
        )
        r1 = run_gibbs_sampler(**kwargs)
        r2 = run_gibbs_sampler(**kwargs)
        assert r1.states == r2.states
        assert r1.sigma_obs == r2.sigma_obs
        assert r1.predictions == r2.predictions

    def test_sampler_keeps_two_chain_outputs_deterministic_for_the_same_seed(self):
        y = [10.0 + 0.1 * i for i in range(50)]
        kwargs = dict(
            y=y,
            x=None,
            pre_end=35,
            niter=20,
            nwarmup=5,
            nchains=2,
            seed=123,
            prior_level_sd=0.01,
        )
        r1 = run_gibbs_sampler(**kwargs)
        r2 = run_gibbs_sampler(**kwargs)
        assert r1.states == r2.states
        assert r1.sigma_obs == r2.sigma_obs
        assert r1.sigma_level == r2.sigma_level
        assert r1.beta == r2.beta
        assert r1.predictions == r2.predictions

    def test_sampler_reports_niter_times_nchains_samples_for_many_chains(self):
        y = [10.0 + 0.1 * i for i in range(80)]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=60,
            niter=7,
            nwarmup=3,
            nchains=16,
            seed=123,
            prior_level_sd=0.01,
        )
        assert len(result.states) == 7 * 16
        assert len(result.sigma_obs) == 7 * 16
        assert len(result.sigma_level) == 7 * 16
        assert len(result.beta) == 7 * 16
        assert len(result.predictions) == 7 * 16
        assert all(len(prediction) == 20 for prediction in result.predictions)


class TestSamplerCovariates:
    """With and without covariates."""

    def test_sampler_no_covariates(self):
        y = [5.0] * 30
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=20,
            niter=10,
            nwarmup=5,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert len(result.beta) == 10
        assert len(result.beta[0]) == 0  # no covariates

    def test_sampler_single_covariate(self):
        y = [10.0 + 0.5 * i for i in range(30)]
        x1 = [0.1 * i for i in range(30)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1],
            pre_end=20,
            niter=10,
            nwarmup=5,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert len(result.beta[0]) == 1

    def test_sampler_accepts_numpy_arrays_because_the_python_api_should_hit_the_borrowed_fast_path_and_match_list_inputs(  # noqa: E501
        self,
    ):
        y = np.array([10.0 + 0.5 * i for i in range(30)], dtype=np.float64)
        x = np.ascontiguousarray([[0.1 * i for i in range(30)]], dtype=np.float64)
        kwargs = dict(
            pre_end=20,
            niter=10,
            nwarmup=5,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )

        result_from_numpy = run_gibbs_sampler(y=y, x=x, **kwargs)
        result_from_lists = run_gibbs_sampler(
            y=y.tolist(),
            x=[x[0].tolist()],
            **kwargs,
        )

        assert result_from_numpy.states == result_from_lists.states
        assert result_from_numpy.sigma_obs == result_from_lists.sigma_obs
        assert result_from_numpy.sigma_level == result_from_lists.sigma_level
        assert result_from_numpy.beta == result_from_lists.beta
        assert result_from_numpy.predictions == result_from_lists.predictions


class TestSamplerBoundary:
    """Boundary value tests."""

    def test_sampler_niter_1(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=3,
            niter=1,
            nwarmup=0,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert len(result.states) == 1

    def test_sampler_nwarmup_0(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=3,
            niter=5,
            nwarmup=0,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert len(result.states) == 5

    def test_sampler_pre_end_1(self):
        y = [10.0, 20.0, 30.0]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=1,
            niter=5,
            nwarmup=2,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert len(result.states) == 5
        assert len(result.predictions[0]) == 2

    def test_sampler_rejects_zero_chains_because_at_least_one_chain_is_required(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(ValueError, match="nchains"):
            run_gibbs_sampler(
                y=y,
                x=None,
                pre_end=3,
                niter=5,
                nwarmup=0,
                nchains=0,
                seed=42,
                prior_level_sd=0.01,
            )


class TestSamplerErrors:
    """Error case tests."""

    def test_sampler_pre_end_equals_T(self):
        """pre_end == len(y) is valid for retrospective mode (no post-period)."""
        y = [1.0, 2.0, 3.0]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=3,
            niter=10,
            nwarmup=5,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        for pred in result.predictions:
            assert len(pred) == 0

    def test_sampler_pre_end_exceeds_T(self):
        y = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="pre_end must be <= length of y"):
            run_gibbs_sampler(
                y=y,
                x=None,
                pre_end=4,
                niter=10,
                nwarmup=5,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
            )

    def test_sampler_empty_y(self):
        with pytest.raises(ValueError, match="empty"):
            run_gibbs_sampler(
                y=[],
                x=None,
                pre_end=0,
                niter=10,
                nwarmup=5,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
            )

    def test_sampler_nan_in_y_pre(self):
        y = [1.0, float("nan"), 3.0, 4.0, 5.0]
        with pytest.raises(ValueError, match="NaN"):
            run_gibbs_sampler(
                y=y,
                x=None,
                pre_end=3,
                niter=10,
                nwarmup=5,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
            )

    def test_sampler_rejects_non_integer_nseasons(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(ValueError, match="nseasons"):
            run_gibbs_sampler(
                y=y,
                x=None,
                pre_end=3,
                niter=5,
                nwarmup=0,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
                nseasons=7.5,
                season_duration=1,
            )

    def test_sampler_rejects_zero_season_duration(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(ValueError, match="season_duration"):
            run_gibbs_sampler(
                y=y,
                x=None,
                pre_end=3,
                niter=5,
                nwarmup=0,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
                nseasons=7,
                season_duration=0,
            )


class TestSamplerKappaShrinkage:
    """kappa_shrinkage field tests for horseshoe/spike-slab."""

    def test_sampler_horseshoe_returns_kappa_shrinkage_field(self):
        """R1: horseshoe returns kappa_shrinkage with correct length."""
        y = [10.0 + 0.1 * i for i in range(50)]
        x1 = [0.5 * i for i in range(50)]
        x2 = [0.3 * i for i in range(50)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1, x2],
            pre_end=35,
            niter=50,
            nwarmup=10,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            prior_type="horseshoe",
        )
        assert hasattr(result, "kappa_shrinkage")
        assert len(result.kappa_shrinkage) == 50
        assert len(result.kappa_shrinkage[0]) == 2

    def test_sampler_spike_slab_kappa_shrinkage_is_empty_list(self):
        """R2: spike_slab (default) returns empty kappa_shrinkage."""
        y = [10.0 + 0.1 * i for i in range(50)]
        x1 = [0.5 * i for i in range(50)]
        x2 = [0.3 * i for i in range(50)]
        result = run_gibbs_sampler(
            y=y,
            x=[x1, x2],
            pre_end=35,
            niter=50,
            nwarmup=10,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )
        assert result.kappa_shrinkage == []

    def test_sampler_no_covariates_horseshoe_kappa_is_empty(self):
        """R3: k=0 with horseshoe returns empty kappa_shrinkage."""
        y = [10.0 + 0.1 * i for i in range(50)]
        result = run_gibbs_sampler(
            y=y,
            x=None,
            pre_end=35,
            niter=50,
            nwarmup=10,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
            prior_type="horseshoe",
        )
        assert result.kappa_shrinkage == []


class TestSamplerCovariateValidation:
    """Covariate shape validation at PyO3 boundary."""

    def test_sampler_rejects_covariate_length_mismatch(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        x_wrong = [[0.1, 0.2, 0.3]]  # length 3, y length 5
        with pytest.raises(
            ValueError, match="covariate column 0 has length 3 but y has length 5"
        ):
            run_gibbs_sampler(
                y=y,
                x=x_wrong,
                pre_end=3,
                niter=5,
                nwarmup=0,
                nchains=1,
                seed=42,
                prior_level_sd=0.01,
            )


class TestSamplerConvergence:
    """Estimation accuracy on known signals."""

    def test_sampler_convergence_known_signal(self):
        """Fit a constant signal (y=10) and verify post-period predictions are close.

        The Rust sampler assumes standardized data
        (CausalImpact Python API standardizes internally).
        """
        import numpy as np

        rng = np.random.default_rng(42)
        n = 100
        true_level = 10.0
        y_raw = true_level + rng.normal(0, 0.5, n)

        # Same standardization as CausalImpact: (y - mean) / sd
        y_pre_mean = y_raw[:70].mean()
        y_pre_sd = y_raw[:70].std(ddof=1)
        y_std = ((y_raw - y_pre_mean) / y_pre_sd).tolist()

        result = run_gibbs_sampler(
            y=y_std,
            x=None,
            pre_end=70,
            niter=500,
            nwarmup=200,
            nchains=1,
            seed=42,
            prior_level_sd=0.01,
        )

        # Unstandardize and compare on the original scale
        preds = np.array(result.predictions) * y_pre_sd + y_pre_mean
        mean_pred = preds.mean()
        assert abs(mean_pred - true_level) < 2.0, (
            f"Mean prediction {mean_pred:.2f} should be near {true_level}"
        )
