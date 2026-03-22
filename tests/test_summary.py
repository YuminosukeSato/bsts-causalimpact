"""Tests for SummaryFormatter: summary and report output."""

import numpy as np
from causal_impact.analysis import CausalImpactResults
from causal_impact.summary import SummaryFormatter


def _make_results(effect=2.0, p_value=0.01):
    """Create a CausalImpactResults fixture."""
    t_post = 10
    return CausalImpactResults(
        actual=np.full(t_post, 12.0),
        point_effects=np.full(t_post, effect),
        point_effect_lower=np.full(t_post, effect * 0.75),
        point_effect_upper=np.full(t_post, effect * 1.25),
        ci_lower=effect * 0.5,
        ci_upper=effect * 1.5,
        point_effect_mean=effect,
        average_effect_sd=effect * 0.1,
        cumulative_effect=np.cumsum(np.full(t_post, effect)),
        cumulative_effect_lower=np.cumsum(np.full(t_post, effect * 0.75)),
        cumulative_effect_upper=np.cumsum(np.full(t_post, effect * 1.25)),
        cumulative_effect_total=effect * t_post,
        cumulative_effect_sd=effect,
        relative_effect_mean=effect / 10.0,
        relative_effect_sd=effect / 100.0,
        relative_effect_lower=effect / 20.0,
        relative_effect_upper=effect / 5.0,
        p_value=p_value,
        predictions_mean=np.full(t_post, 10.0),
        predictions_sd=np.full(t_post, 0.5),
        predictions_lower=np.full(t_post, 9.0),
        predictions_upper=np.full(t_post, 11.0),
        average_prediction_sd=0.5,
        average_prediction_lower=9.0,
        average_prediction_upper=11.0,
        cumulative_prediction_sd=5.0,
        cumulative_prediction_lower=90.0,
        cumulative_prediction_upper=110.0,
    )


class TestSummaryFormat:
    """summary出力フォーマット."""

    def test_summary_default_format(self):
        result = _make_results(effect=2.0, p_value=0.01)
        text = SummaryFormatter.summary(result, digits=2)
        assert "Average" in text
        assert "Cumulative" in text
        assert "2.0" in text or "2.00" in text

    def test_summary_includes_r_style_sections(self):
        """R互換summary: Actual/Prediction/Absolute/Relativeの各行を表示."""
        result = _make_results(effect=2.0, p_value=0.01)

        text = SummaryFormatter.summary(result, digits=2)
        lines = text.split("\n")

        assert "Actual                   12.00          120.00" in lines
        assert "Prediction (s.d.)        10.00 (0.50)   100.00 (5.00)" in lines
        assert "95% CI                   [9.00, 11.00]  [90.00, 110.00]" in lines
        assert "Absolute effect (s.d.)   2.00 (0.20)    20.00 (2.00)" in lines
        assert "95% CI                   [1.00, 3.00]   [15.00, 25.00]" in lines
        assert "Relative effect (s.d.)   20.00% (2.00%) 20.00% (2.00%)" in lines
        assert "95% CI                   [10.00%, 40.00%] [10.00%, 40.00%]" in lines

    def test_summary_report_format(self):
        result = _make_results(effect=2.0, p_value=0.01)
        text = SummaryFormatter.report(result)
        # Report should be natural language
        assert len(text) > 100
        assert "effect" in text.lower() or "impact" in text.lower()

    def test_summary_digits_0(self):
        result = _make_results(effect=2.345, p_value=0.01)
        text = SummaryFormatter.summary(result, digits=0)
        assert "2.345" not in text  # should be rounded

    def test_summary_digits_10(self):
        result = _make_results(effect=2.0, p_value=0.01)
        text = SummaryFormatter.summary(result, digits=10)
        assert isinstance(text, str)

    def test_summary_shows_cumulative_ci_in_95_percent_ci_row(self):
        """Absolute effect の 95% CI 行 cumulative 列には最終時点の累積CIを表示する."""
        result = _make_results(effect=2.0, p_value=0.01)
        text = SummaryFormatter.summary(result, digits=2)
        ci_line = text.split("\n")[8]
        assert "15.00" in ci_line
        assert "25.00" in ci_line


class TestReportContent:
    """レポート内容の検証."""

    def test_report_significant_effect(self):
        result = _make_results(effect=3.0, p_value=0.001)
        text = SummaryFormatter.report(result)
        assert "significant" in text.lower()

    def test_report_no_effect(self):
        result = _make_results(effect=0.1, p_value=0.45)
        text = SummaryFormatter.report(result)
        assert "not" in text.lower() or "no" in text.lower()
