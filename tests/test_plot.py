"""Tests for Plotter: matplotlib output."""

import numpy as np
from causal_impact.analysis import CausalImpactResults
from causal_impact.plot import Plotter


def _make_results_with_index():
    """Create results and time index for plotting."""
    import pandas as pd

    t_pre = 70
    t_post = 30
    t_total = t_pre + t_post

    y = np.random.default_rng(42).normal(10, 1, t_total)
    time_index = pd.date_range("2020-01-01", periods=t_total, freq="D")
    results = CausalImpactResults(
        point_effects=np.full(t_post, 2.0),
        ci_lower=1.0,
        ci_upper=3.0,
        point_effect_mean=2.0,
        cumulative_effect=np.cumsum(np.full(t_post, 2.0)),
        cumulative_effect_total=60.0,
        relative_effect_mean=0.2,
        p_value=0.01,
        predictions_mean=np.full(t_post, 10.0),
        predictions_lower=np.full(t_post, 9.0),
        predictions_upper=np.full(t_post, 11.0),
    )
    return results, y, time_index, t_pre


class TestPlot:
    """Plotのテスト."""

    def test_plot_3_panels(self):
        import matplotlib

        matplotlib.use("Agg")
        results, y, time_index, t_pre = _make_results_with_index()
        fig = Plotter.plot(results, y, time_index, t_pre)
        axes = fig.get_axes()
        assert len(axes) == 3

    def test_plot_single_metric(self):
        import matplotlib

        matplotlib.use("Agg")
        results, y, time_index, t_pre = _make_results_with_index()
        fig = Plotter.plot(results, y, time_index, t_pre, metrics=["original"])
        axes = fig.get_axes()
        assert len(axes) == 1

    def test_plot_vertical_line_at_intervention(self):
        import matplotlib

        matplotlib.use("Agg")
        results, y, time_index, t_pre = _make_results_with_index()
        fig = Plotter.plot(results, y, time_index, t_pre)
        # Check that vertical lines exist in each subplot
        axes = fig.get_axes()
        for ax in axes:
            lines = ax.get_lines()
            # At least the intervention vertical line should be present
            assert len(lines) >= 1
