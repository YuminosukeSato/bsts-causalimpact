"""Tests for ModelOptions typed configuration."""

import numpy as np
import pandas as pd
import pytest

from causal_impact import CausalImpact, ModelOptions


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestModelOptionsDefaults:
    """Default values must match R bsts defaults."""

    def test_defaults_match_r_bsts(self):
        opts = ModelOptions()
        assert opts.niter == 1000
        assert opts.nwarmup == 500
        assert opts.nchains == 1
        assert opts.seed == 0
        assert opts.standardize_data is True
        assert opts.prior_level_sd == 0.01
        assert opts.expected_model_size == 1

    def test_all_explicit(self):
        opts = ModelOptions(
            niter=2000,
            nwarmup=1000,
            nchains=4,
            seed=42,
            standardize_data=False,
            prior_level_sd=0.1,
            expected_model_size=3,
        )
        assert opts.niter == 2000
        assert opts.nwarmup == 1000
        assert opts.nchains == 4
        assert opts.seed == 42
        assert opts.standardize_data is False
        assert opts.prior_level_sd == 0.1
        assert opts.expected_model_size == 3


# ---------------------------------------------------------------------------
# Boundary: niter
# ---------------------------------------------------------------------------


class TestNiterBoundary:
    def test_niter_1_is_minimum_valid(self):
        opts = ModelOptions(niter=1)
        assert opts.niter == 1

    def test_niter_0_raises(self):
        with pytest.raises(ValueError, match="niter must be >= 1"):
            ModelOptions(niter=0)

    def test_niter_negative_raises(self):
        with pytest.raises(ValueError, match="niter must be >= 1"):
            ModelOptions(niter=-1)


# ---------------------------------------------------------------------------
# Boundary: nwarmup
# ---------------------------------------------------------------------------


class TestNwarmupBoundary:
    def test_nwarmup_0_is_valid(self):
        opts = ModelOptions(nwarmup=0)
        assert opts.nwarmup == 0

    def test_nwarmup_negative_raises(self):
        with pytest.raises(ValueError, match="nwarmup must be >= 0"):
            ModelOptions(nwarmup=-1)

    def test_nwarmup_equals_niter_is_valid(self):
        opts = ModelOptions(niter=100, nwarmup=100)
        assert opts.nwarmup == 100

    def test_nwarmup_exceeds_niter_is_valid(self):
        """R allows nwarmup >= niter (results in 0 post-warmup samples)."""
        opts = ModelOptions(niter=100, nwarmup=200)
        assert opts.nwarmup == 200


# ---------------------------------------------------------------------------
# Boundary: nchains
# ---------------------------------------------------------------------------


class TestNchainsBoundary:
    def test_nchains_1_minimum(self):
        opts = ModelOptions(nchains=1)
        assert opts.nchains == 1

    def test_nchains_0_raises(self):
        with pytest.raises(ValueError, match="nchains must be >= 1"):
            ModelOptions(nchains=0)


# ---------------------------------------------------------------------------
# Boundary: prior_level_sd
# ---------------------------------------------------------------------------


class TestPriorLevelSdBoundary:
    def test_very_small_positive(self):
        opts = ModelOptions(prior_level_sd=1e-10)
        assert opts.prior_level_sd == 1e-10

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="prior_level_sd must be > 0"):
            ModelOptions(prior_level_sd=0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="prior_level_sd must be > 0"):
            ModelOptions(prior_level_sd=-0.01)

    def test_very_large(self):
        opts = ModelOptions(prior_level_sd=1e6)
        assert opts.prior_level_sd == 1e6


# ---------------------------------------------------------------------------
# Boundary: expected_model_size
# ---------------------------------------------------------------------------


class TestExpectedModelSizeBoundary:
    def test_expected_model_size_very_small_positive(self):
        opts = ModelOptions(expected_model_size=0.01)
        assert opts.expected_model_size == 0.01

    def test_expected_model_size_zero_raises(self):
        with pytest.raises(ValueError, match="expected_model_size must be > 0"):
            ModelOptions(expected_model_size=0.0)

    def test_expected_model_size_negative_raises(self):
        with pytest.raises(ValueError, match="expected_model_size must be > 0"):
            ModelOptions(expected_model_size=-1.0)


# ---------------------------------------------------------------------------
# Immutability (frozen dataclass)
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_is_immutable(self):
        opts = ModelOptions()
        with pytest.raises(AttributeError):
            opts.niter = 2000  # type: ignore[misc]

    def test_equality_by_value(self):
        a = ModelOptions(niter=500, seed=42)
        b = ModelOptions(niter=500, seed=42)
        assert a == b

    def test_inequality(self):
        a = ModelOptions(niter=500)
        b = ModelOptions(niter=1000)
        assert a != b


# ---------------------------------------------------------------------------
# to_dict conversion
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_returns_all_keys(self):
        opts = ModelOptions()
        d = opts.to_dict()
        expected_keys = {
            "niter",
            "nwarmup",
            "nchains",
            "seed",
            "standardize_data",
            "prior_level_sd",
            "expected_model_size",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self):
        opts = ModelOptions(niter=2000, seed=42)
        d = opts.to_dict()
        assert d["niter"] == 2000
        assert d["seed"] == 42


# ---------------------------------------------------------------------------
# Backward compatibility: CausalImpact accepts dict, ModelOptions, or None
# ---------------------------------------------------------------------------


def _make_test_data():
    """Minimal synthetic data for backward compatibility tests."""
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


class TestBackwardCompatibility:
    def test_dict_model_args_still_works(self):
        df, pre, post = _make_test_data()
        ci = CausalImpact(
            df, pre, post, model_args={"niter": 100, "nwarmup": 50, "seed": 1}
        )
        assert ci.summary() is not None

    def test_model_options_instance(self):
        df, pre, post = _make_test_data()
        opts = ModelOptions(niter=100, nwarmup=50, seed=1)
        ci = CausalImpact(df, pre, post, model_args=opts)
        assert ci.summary() is not None

    def test_none_model_args_uses_defaults(self):
        df, pre, post = _make_test_data()
        ci = CausalImpact(df, pre, post, model_args=None)
        assert ci.summary() is not None

    def test_empty_dict_uses_defaults(self):
        df, pre, post = _make_test_data()
        ci = CausalImpact(df, pre, post, model_args={})
        assert ci.summary() is not None

    def test_dict_and_model_options_produce_same_result(self):
        """Same parameters via dict and ModelOptions must yield identical results."""
        df, pre, post = _make_test_data()
        params = {"niter": 200, "nwarmup": 100, "seed": 42}

        ci_dict = CausalImpact(df, pre, post, model_args=params)
        ci_opts = CausalImpact(
            df, pre, post, model_args=ModelOptions(**params)
        )

        assert ci_dict.summary_stats["point_effect_mean"] == pytest.approx(
            ci_opts.summary_stats["point_effect_mean"], rel=1e-10
        )
