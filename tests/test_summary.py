"""Tests for SummaryFormatter: summary and report output."""

import numpy as np
from causal_impact.analysis import CausalImpactResults
from causal_impact.summary import SummaryFormatter


def _make_results(effect=2.0, p_value=0.01):
    """Create a CausalImpactResults fixture."""
    t_post = 10
    return CausalImpactResults(
        point_effects=np.full(t_post, effect),
        ci_lower=effect * 0.5,
        ci_upper=effect * 1.5,
        point_effect_mean=effect,
        cumulative_effect=np.cumsum(np.full(t_post, effect)),
        cumulative_effect_total=effect * t_post,
        relative_effect_mean=effect / 10.0,
        p_value=p_value,
        predictions_mean=np.full(t_post, 10.0),
        predictions_lower=np.full(t_post, 9.0),
        predictions_upper=np.full(t_post, 11.0),
    )


class TestSummaryFormat:
    """summary出力フォーマット."""

    def test_summary_default_format(self):
        result = _make_results(effect=2.0, p_value=0.01)
        text = SummaryFormatter.summary(result, digits=2)
        assert "Average" in text
        assert "Cumulative" in text
        assert "2.0" in text or "2.00" in text

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
