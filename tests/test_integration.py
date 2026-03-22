"""Integration tests: end-to-end and R numerical compatibility."""

import time

import numpy as np
import pandas as pd
from causal_impact import CausalImpact


def _make_causal_data(
    n=100, pre_frac=0.7, true_effect=3.0, noise_sd=0.5, seed=42
):
    """Generate synthetic data with a known causal effect."""
    rng = np.random.default_rng(seed)
    pre_end = int(n * pre_frac)

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.normal(5, 1, n)
    y = 1.0 * x + rng.normal(0, noise_sd, n)

    # Add causal effect in post-period
    y[pre_end:] += true_effect

    df = pd.DataFrame({"y": y, "x1": x}, index=dates)
    pre_period = [str(dates[0].date()), str(dates[pre_end - 1].date())]
    post_period = [str(dates[pre_end].date()), str(dates[-1].date())]

    return df, pre_period, post_period, true_effect


class TestEndToEnd:
    """E2Eテスト."""

    def test_basic_usage_end_to_end(self):
        df, pre_period, post_period, _ = _make_causal_data()
        ci = CausalImpact(df, pre_period, post_period)

        # All main APIs work
        summary = ci.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        report = ci.report()
        assert isinstance(report, str)
        assert len(report) > 50

        inferences = ci.inferences
        assert isinstance(inferences, pd.DataFrame)
        assert len(inferences) > 0

        stats = ci.summary_stats
        assert isinstance(stats, dict)
        assert "p_value" in stats

    def test_no_covariates_mode(self):
        rng = np.random.default_rng(42)
        n = 80
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = np.cumsum(rng.normal(0, 0.1, n)) + 10.0
        y[56:] += 2.0  # effect at 70% mark
        df = pd.DataFrame({"y": y}, index=dates)

        ci = CausalImpact(
            df,
            ["2020-01-01", "2020-02-25"],
            ["2020-02-26", "2020-03-20"],
        )
        assert ci.summary() is not None
        assert ci.inferences is not None



class TestPerformance:
    """非機能テスト."""

    def test_large_dataset_performance(self):
        """1000時点のデータが10秒以内に完了."""
        rng = np.random.default_rng(42)
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        x = rng.normal(0, 1, n)
        y = 1.5 * x + rng.normal(0, 0.5, n)
        y[700:] += 2.0
        df = pd.DataFrame({"y": y, "x1": x}, index=dates)

        start = time.time()
        ci = CausalImpact(
            df,
            ["2020-01-01", "2021-10-27"],
            ["2021-10-28", "2022-09-26"],
            model_args={"niter": 500, "nwarmup": 250, "seed": 42},
        )
        elapsed = time.time() - start
        assert elapsed < 10.0, f"Took {elapsed:.1f}s, should be < 10s"
        assert ci.summary() is not None
