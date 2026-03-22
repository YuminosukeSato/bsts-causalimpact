"""Shared fixtures for CausalImpact tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_df(rng):
    """DataFrame with 1 response + 1 covariate, 100 time points."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(0, 1, n)
    y = 1.2 * x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x1": x}, index=dates)


@pytest.fixture
def sample_df_no_cov(rng):
    """DataFrame with response only, no covariates."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    y = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
    return pd.DataFrame({"y": y}, index=dates)


@pytest.fixture
def pre_period():
    return ["2020-01-01", "2020-03-10"]


@pytest.fixture
def post_period():
    return ["2020-03-11", "2020-04-09"]


@pytest.fixture
def pre_period_int():
    return [0, 69]


@pytest.fixture
def post_period_int():
    return [70, 99]
