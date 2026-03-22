"""Tests for CausalAnalysis: effect computation, CI, p-values."""

import numpy as np
from causal_impact.analysis import CausalAnalysis, CausalImpactResults


def _make_predictions(n_samples, t_post, base, effect=0.0, noise_sd=0.1, seed=42):
    """Helper: create synthetic predictions (counterfactual) and observed y_post."""
    rng = np.random.default_rng(seed)
    # Counterfactual predictions (no effect)
    predictions = base + rng.normal(0, noise_sd, (n_samples, t_post))
    # Observed = counterfactual + true effect
    y_post = base + effect + rng.normal(0, noise_sd, t_post)
    return predictions, y_post


class TestEffectDetection:
    """効果検出テスト."""

    def test_positive_effect_detected(self):
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=3.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        assert isinstance(result, CausalImpactResults)
        assert result.point_effect_mean > 0
        assert result.ci_lower > 0  # significant positive effect

    def test_negative_effect_detected(self):
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=-3.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        assert result.point_effect_mean < 0
        assert result.ci_upper < 0  # significant negative effect

    def test_no_effect_zero(self):
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=0.0, noise_sd=0.5
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        # CI should contain zero
        assert result.ci_lower <= 0 <= result.ci_upper


class TestCIAndPValue:
    """信頼区間とp値のテスト."""

    def test_ci_coverage_95(self):
        """95% CIが真の効果値を含むことを検証."""
        true_effect = 2.0
        predictions, y_post = _make_predictions(
            1000, 30, base=10.0, effect=true_effect, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        assert result.ci_lower <= true_effect <= result.ci_upper

    def test_p_value_significant(self):
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=5.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        assert result.p_value < 0.05

    def test_p_value_not_significant(self):
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=0.0, noise_sd=0.5
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        assert result.p_value > 0.05


class TestCumulativeAndRelative:
    """累積効果と相対効果."""""

    def test_cumulative_effect_monotone(self):
        """正の効果のみの場合、累積効果は単調増加."""
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=5.0, noise_sd=0.1
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        cum = result.cumulative_effect
        for i in range(1, len(cum)):
            assert cum[i] >= cum[i - 1] - 1e-10  # allow tiny float errors

    def test_relative_effect_percentage(self):
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=2.0, noise_sd=0.1
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        # Relative effect ≈ 2.0/10.0 = 20%
        assert abs(result.relative_effect_mean - 0.2) < 0.1


class TestBoundary:
    """境界値テスト."""

    def test_single_post_point(self):
        predictions, y_post = _make_predictions(
            500, 1, base=10.0, effect=1.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        assert len(result.point_effects) == 1
        assert len(result.cumulative_effect) == 1
