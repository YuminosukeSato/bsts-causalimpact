"""Tests for the Rust Gibbs sampler via PyO3 binding."""

import pytest
from causal_impact._core import GibbsSamples, run_gibbs_sampler


class TestSamplerShapes:
    """出力サイズの検証."""

    def test_sampler_returns_correct_shapes(self):
        y = [10.0 + 0.1 * i for i in range(100)]
        result = run_gibbs_sampler(
            y=y, x=None, pre_end=70, niter=50, nwarmup=10, nchains=1,
            seed=42, prior_level_sd=0.01,
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
            y=y, x=None, pre_end=35, niter=20, nwarmup=5, nchains=2,
            seed=42, prior_level_sd=0.01,
        )
        # niter * nchains = 40 samples
        assert len(result.states) == 40
        assert len(result.sigma_obs) == 40


class TestSamplerReproducibility:
    """シード固定での再現性."""

    def test_sampler_seed_reproducibility(self):
        y = [10.0 + 0.1 * i for i in range(50)]
        kwargs = dict(
            y=y, x=None, pre_end=35, niter=20, nwarmup=5, nchains=1,
            seed=123, prior_level_sd=0.01,
        )
        r1 = run_gibbs_sampler(**kwargs)
        r2 = run_gibbs_sampler(**kwargs)
        assert r1.states == r2.states
        assert r1.sigma_obs == r2.sigma_obs
        assert r1.predictions == r2.predictions


class TestSamplerCovariates:
    """共変量の有無."""""

    def test_sampler_no_covariates(self):
        y = [5.0] * 30
        result = run_gibbs_sampler(
            y=y, x=None, pre_end=20, niter=10, nwarmup=5, nchains=1,
            seed=42, prior_level_sd=0.01,
        )
        assert len(result.beta) == 10
        assert len(result.beta[0]) == 0  # no covariates

    def test_sampler_single_covariate(self):
        y = [10.0 + 0.5 * i for i in range(30)]
        x1 = [0.1 * i for i in range(30)]
        result = run_gibbs_sampler(
            y=y, x=[x1], pre_end=20, niter=10, nwarmup=5, nchains=1,
            seed=42, prior_level_sd=0.01,
        )
        assert len(result.beta[0]) == 1


class TestSamplerBoundary:
    """境界値テスト."""

    def test_sampler_niter_1(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = run_gibbs_sampler(
            y=y, x=None, pre_end=3, niter=1, nwarmup=0, nchains=1,
            seed=42, prior_level_sd=0.01,
        )
        assert len(result.states) == 1

    def test_sampler_nwarmup_0(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = run_gibbs_sampler(
            y=y, x=None, pre_end=3, niter=5, nwarmup=0, nchains=1,
            seed=42, prior_level_sd=0.01,
        )
        assert len(result.states) == 5

    def test_sampler_pre_end_1(self):
        y = [10.0, 20.0, 30.0]
        result = run_gibbs_sampler(
            y=y, x=None, pre_end=1, niter=5, nwarmup=2, nchains=1,
            seed=42, prior_level_sd=0.01,
        )
        assert len(result.states) == 5
        assert len(result.predictions[0]) == 2


class TestSamplerErrors:
    """異常系テスト."""

    def test_sampler_pre_end_equals_T(self):
        y = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="post-period"):
            run_gibbs_sampler(
                y=y, x=None, pre_end=3, niter=10, nwarmup=5, nchains=1,
                seed=42, prior_level_sd=0.01,
            )

    def test_sampler_empty_y(self):
        with pytest.raises(ValueError, match="empty"):
            run_gibbs_sampler(
                y=[], x=None, pre_end=0, niter=10, nwarmup=5, nchains=1,
                seed=42, prior_level_sd=0.01,
            )

    def test_sampler_nan_in_y_pre(self):
        y = [1.0, float("nan"), 3.0, 4.0, 5.0]
        with pytest.raises(ValueError, match="NaN"):
            run_gibbs_sampler(
                y=y, x=None, pre_end=3, niter=10, nwarmup=5, nchains=1,
                seed=42, prior_level_sd=0.01,
            )


class TestSamplerConvergence:
    """既知信号での推定精度検証."""

    def test_sampler_convergence_known_signal(self):
        """定数信号(y=10)をフィットし、post-period予測が近い値を返す."""
        import numpy as np

        rng = np.random.default_rng(42)
        n = 100
        true_level = 10.0
        y = (true_level + rng.normal(0, 0.5, n)).tolist()

        result = run_gibbs_sampler(
            y=y, x=None, pre_end=70, niter=500, nwarmup=200, nchains=1,
            seed=42, prior_level_sd=0.01,
        )

        # 事後予測の平均がtrue_levelに近い
        preds = np.array(result.predictions)
        mean_pred = preds.mean()
        assert abs(mean_pred - true_level) < 2.0, (
            f"Mean prediction {mean_pred:.2f} should be near {true_level}"
        )
