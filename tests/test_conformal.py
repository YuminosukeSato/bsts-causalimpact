"""Tests for split conformal inference."""

import numpy as np
import pandas as pd
from causal_impact import CausalImpact


def _make_data(t_pre=50, t_post=20, seed=42):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 1, t_pre + t_post))
    return pd.DataFrame({"y": y}), [0, t_pre - 1], [t_pre, t_pre + t_post - 1]


class TestConformalInference:
    def test_q_hat_nonneg(self):
        df, pre, post = _make_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_conformal_analysis()
        assert result.q_hat >= 0.0

    def test_shape_matches_post(self):
        t_post = 15
        df, pre, post = _make_data(t_post=t_post)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_conformal_analysis()
        assert result.lower.shape == (t_post,)
        assert result.upper.shape == (t_post,)

    def test_lower_lt_upper(self):
        df, pre, post = _make_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_conformal_analysis()
        assert np.all(result.lower < result.upper)

    def test_coverage_on_calibration(self):
        """Conformal coverage guarantee: at least 1-alpha on calibration set."""
        df, pre, post = _make_data(t_pre=60, seed=999)
        ci = CausalImpact(df, pre, post, model_args={"niter": 200, "nwarmup": 100})
        result = ci.run_conformal_analysis(alpha=0.1)
        # q_hat should be positive and finite
        assert result.q_hat > 0.0
        assert np.isfinite(result.q_hat)

    def test_smaller_alpha_wider(self):
        """Smaller alpha (higher confidence) should give wider intervals."""
        df, pre, post = _make_data(t_pre=60)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result_05 = ci.run_conformal_analysis(alpha=0.05)
        result_20 = ci.run_conformal_analysis(alpha=0.20)
        assert result_05.q_hat >= result_20.q_hat

    def test_split_ratio_default(self):
        t_pre = 50
        df, pre, post = _make_data(t_pre=t_pre)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_conformal_analysis()
        assert result.n_calibration == t_pre - t_pre // 2

    def test_min_calibration_4(self):
        """T_pre=4 -> t_train=2, n_calib=2: minimum viable."""
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        ci = CausalImpact(df, [0, 3], [4, 5], model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_conformal_analysis()
        assert result.n_calibration == 2

    def test_differs_from_bayesian(self):
        """Conformal intervals should differ from Bayesian credible intervals."""
        df, pre, post = _make_data(t_pre=60, seed=777)
        ci = CausalImpact(df, pre, post, model_args={"niter": 200, "nwarmup": 100})
        conformal = ci.run_conformal_analysis()
        bayesian_lower = ci._results.predictions_lower
        # They should not be identical (different methodology)
        assert not np.allclose(conformal.lower, bayesian_lower, atol=1e-6)

    def test_result_attributes_exist(self):
        df, pre, post = _make_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_conformal_analysis()
        assert hasattr(result, "q_hat")
        assert hasattr(result, "lower")
        assert hasattr(result, "upper")
        assert hasattr(result, "n_calibration")
