"""Tests for placebo test functionality."""

import numpy as np
import pandas as pd
import pytest
from causal_impact import CausalImpact


def _make_no_effect_data(t_pre=50, t_post=20, seed=42):
    """Generate data with no causal effect (pure random walk)."""
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 1, t_pre + t_post))
    return pd.DataFrame({"y": y}), [0, t_pre - 1], [t_pre, t_pre + t_post - 1]


def _make_strong_effect_data(t_pre=50, t_post=20, effect=5.0, seed=42):
    """Generate data with a strong causal effect in the post-period."""
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 1, t_pre + t_post))
    y[t_pre:] += effect * np.std(y[:t_pre])
    return pd.DataFrame({"y": y}), [0, t_pre - 1], [t_pre, t_pre + t_post - 1]


class TestPlaceboTest:
    def test_p_value_in_0_1(self):
        df, pre, post = _make_no_effect_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_placebo_test(n_placebos=5, min_pre_length=10)
        assert 0.0 <= result.p_value <= 1.0

    def test_no_effect_high_p_value(self):
        df, pre, post = _make_no_effect_data(t_pre=60, seed=123)
        ci = CausalImpact(df, pre, post, model_args={"niter": 200, "nwarmup": 100})
        result = ci.run_placebo_test(n_placebos=10, min_pre_length=15)
        assert result.p_value >= 0.1

    def test_strong_effect_low_p_value(self):
        df, pre, post = _make_strong_effect_data(t_pre=60, effect=8.0, seed=456)
        ci = CausalImpact(df, pre, post, model_args={"niter": 200, "nwarmup": 100})
        result = ci.run_placebo_test(n_placebos=15, min_pre_length=15)
        assert result.p_value <= 0.5

    def test_distribution_length(self):
        df, pre, post = _make_no_effect_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_placebo_test(n_placebos=5, min_pre_length=10)
        assert len(result.effect_distribution) == result.n_placebos

    def test_rank_ratio_formula(self):
        df, pre, post = _make_no_effect_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_placebo_test(n_placebos=5, min_pre_length=10)
        n = result.n_placebos
        rank = sum(1 for e in result.effect_distribution if e >= result.real_effect)
        expected_ratio = rank / (n + 1)
        assert result.rank_ratio == pytest.approx(expected_ratio)

    def test_n_placebos_1_edge(self):
        df, pre, post = _make_no_effect_data(t_pre=30)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_placebo_test(n_placebos=1, min_pre_length=10)
        assert result.n_placebos == 1
        assert len(result.effect_distribution) == 1

    def test_too_short_pre_raises(self):
        df, pre, post = _make_no_effect_data(t_pre=5)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        with pytest.raises(ValueError, match="pre_end"):
            ci.run_placebo_test(min_pre_length=10)

    def test_reproducible(self):
        df, pre, post = _make_no_effect_data()
        ci = CausalImpact(
            df, pre, post, model_args={"niter": 100, "nwarmup": 50, "seed": 42}
        )
        r1 = ci.run_placebo_test(n_placebos=3, min_pre_length=10)
        r2 = ci.run_placebo_test(n_placebos=3, min_pre_length=10)
        np.testing.assert_array_equal(r1.effect_distribution, r2.effect_distribution)

    def test_real_effect_stored(self):
        df, pre, post = _make_no_effect_data()
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_placebo_test(n_placebos=3, min_pre_length=10)
        assert result.real_effect >= 0.0
        assert np.isfinite(result.real_effect)

    def test_none_uses_all_splits(self):
        t_pre = 30
        min_pre = 10
        df, pre, post = _make_no_effect_data(t_pre=t_pre)
        ci = CausalImpact(df, pre, post, model_args={"niter": 100, "nwarmup": 50})
        result = ci.run_placebo_test(n_placebos=None, min_pre_length=min_pre)
        expected_splits = t_pre - min_pre
        assert result.n_placebos == expected_splits
