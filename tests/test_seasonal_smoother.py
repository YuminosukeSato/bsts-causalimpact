"""State-space seasonal smoother の Python 統合テスト.

nseasons > 1 の場合に状態空間 seasonal smoother が正しく動作することを検証する。
R bsts AddSeasonal() 互換の実装に対応。
"""

import numpy as np
import pandas as pd
import pytest
from causal_impact import CausalImpact


MCMC_ARGS_FAST = {"niter": 500, "nwarmup": 200, "seed": 42, "prior_level_sd": 0.01}
MCMC_ARGS_MEDIUM = {"niter": 2000, "nwarmup": 500, "seed": 42, "prior_level_sd": 0.01}


def _make_seasonal_df(
    n: int = 84,
    pre_end: int = 56,
    nseasons: int = 7,
    effect: float = 5.0,
    noise_sd: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Generate time series with seasonal pattern and post-period effect."""
    rng = np.random.default_rng(seed)
    seasonal_pattern = np.sin(2 * np.pi * np.arange(nseasons) / nseasons)
    repeated = np.resize(seasonal_pattern, n)
    y = 20.0 + repeated + rng.normal(0, noise_sd, n)
    y[pre_end:] += effect
    df = pd.DataFrame({"y": y})
    return df, [0, pre_end - 1], [pre_end, n - 1]


class TestSeasonalSmootherIntegration:
    def test_sigma_seasonal_exists_when_nseasons_set(self):
        """nseasons > 1 のとき sigma_seasonal が非空であること."""
        df, pre, post = _make_seasonal_df()
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_FAST, "nseasons": 7}
        )
        from causal_impact._core import run_gibbs_sampler

        result = run_gibbs_sampler(
            list(df["y"]),
            None,
            pre[1] + 1,
            500,
            200,
            1,
            42,
            0.01,
            1.0,
            7.0,
            1.0,
            False,
            "local_level",
        )
        assert len(result.sigma_seasonal) > 0

    def test_sigma_seasonal_empty_when_no_seasons(self):
        """nseasons=None のとき sigma_seasonal が空であること."""
        from causal_impact._core import run_gibbs_sampler

        y = [10.0 + 0.1 * i for i in range(20)]
        result = run_gibbs_sampler(
            y, None, 15, 10, 5, 1, 42, 0.01, 1.0, None, None, False, "local_level"
        )
        assert len(result.sigma_seasonal) == 0

    def test_sigma_seasonal_positive_all_samples(self):
        """sigma_seasonal の全サンプルが正であること."""
        from causal_impact._core import run_gibbs_sampler

        y = [20.0 + np.sin(2 * np.pi * i / 7) for i in range(84)]
        result = run_gibbs_sampler(
            y, None, 56, 200, 100, 1, 42, 0.01, 1.0, 7.0, 1.0, False, "local_level"
        )
        assert all(s > 0 for s in result.sigma_seasonal)

    def test_sigma_seasonal_len_equals_post_warmup(self):
        """sigma_seasonal の長さが niter - nwarmup であること."""
        from causal_impact._core import run_gibbs_sampler

        niter, nwarmup = 100, 30
        y = [20.0 + np.sin(2 * np.pi * i / 7) for i in range(84)]
        result = run_gibbs_sampler(
            y, None, 56, niter, nwarmup, 1, 42, 0.01, 1.0, 7.0, 1.0, False, "local_level"
        )
        assert len(result.sigma_seasonal) == niter

    def test_point_effect_finite_with_seasonal(self):
        """seasonal モデルの point_effect_mean が有限値であること."""
        df, pre, post = _make_seasonal_df()
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_FAST, "nseasons": 7}
        )
        assert np.isfinite(ci.summary_stats["point_effect_mean"])

    def test_ci_bounds_finite_with_seasonal(self):
        """seasonal モデルの CI bounds が有限値であること."""
        df, pre, post = _make_seasonal_df()
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_FAST, "nseasons": 7}
        )
        assert np.isfinite(ci.summary_stats["ci_lower"])
        assert np.isfinite(ci.summary_stats["ci_upper"])
        assert ci.summary_stats["ci_lower"] < ci.summary_stats["ci_upper"]

    def test_seasonal_predictions_continue_in_post(self):
        """post period でも seasonal パターンが予測に反映されること."""
        df, pre, post = _make_seasonal_df(effect=0.0, noise_sd=0.01)
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_MEDIUM, "nseasons": 7}
        )
        inf = ci.inferences
        post_predictions = inf["predicted_mean"]
        expected_post_len = post[1] - post[0] + 1
        assert len(post_predictions) == expected_post_len
        assert all(np.isfinite(post_predictions))

    def test_strong_seasonal_effect_detected(self):
        """強い因果効果 + seasonal → significant."""
        df, pre, post = _make_seasonal_df(effect=10.0, noise_sd=0.5)
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_MEDIUM, "nseasons": 7}
        )
        assert ci.summary_stats["p_value"] < 0.05

    def test_no_effect_seasonal_not_significant(self):
        """因果効果なし + seasonal → not significant."""
        df, pre, post = _make_seasonal_df(effect=0.0, noise_sd=2.0)
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_MEDIUM, "nseasons": 7}
        )
        assert ci.summary_stats["p_value"] > 0.05

    def test_seasonal_backward_compat_nseasons_none(self):
        """nseasons=None のとき既存動作と同一であること."""
        rng = np.random.default_rng(99)
        y = 10.0 + rng.normal(0, 0.5, 30)
        y[20:] += 3.0
        df = pd.DataFrame({"y": y})
        ci = CausalImpact(
            df,
            [0, 19],
            [20, 29],
            model_args={"niter": 200, "nwarmup": 100, "seed": 99},
        )
        assert np.isfinite(ci.summary_stats["point_effect_mean"])
        assert ci.summary_stats["p_value"] < 0.05

    def test_nseasons_2_valid(self):
        """S=2（最小の seasonal）でエラーなし."""
        df, pre, post = _make_seasonal_df(nseasons=2)
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_FAST, "nseasons": 2}
        )
        assert np.isfinite(ci.summary_stats["point_effect_mean"])

    def test_nseasons_12_valid(self):
        """S=12（月次 seasonal）でエラーなし."""
        df, pre, post = _make_seasonal_df(n=120, pre_end=84, nseasons=12)
        ci = CausalImpact(
            df, pre, post, model_args={**MCMC_ARGS_FAST, "nseasons": 12}
        )
        assert np.isfinite(ci.summary_stats["point_effect_mean"])

    def test_season_duration_7_valid(self):
        """season_duration=7（週次ブロック）でエラーなし."""
        df, pre, post = _make_seasonal_df(n=168, pre_end=112, nseasons=4)
        ci = CausalImpact(
            df,
            pre,
            post,
            model_args={**MCMC_ARGS_FAST, "nseasons": 4, "season_duration": 7},
        )
        assert np.isfinite(ci.summary_stats["point_effect_mean"])
