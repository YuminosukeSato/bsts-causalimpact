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

    def test_summary_stats_use_posterior_sample_aggregates(self):
        """summary行はposterior sample集約値(mean/sd/CI)を保持する."""
        y_post = np.array([10.0, 10.0])
        predictions = np.array(
            [
                [8.0, 8.0],
                [10.0, 6.0],
                [11.0, 7.0],
            ]
        )

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.0,
        )

        assert np.array_equal(result.actual, y_post)
        assert np.allclose(result.predictions_sd, np.std(predictions, axis=0, ddof=1))
        assert result.average_prediction_sd == np.sqrt(1.0 / 3.0)
        assert result.average_prediction_lower == 8.0
        assert result.average_prediction_upper == 9.0
        assert result.cumulative_prediction_sd == np.sqrt(4.0 / 3.0)
        assert result.cumulative_prediction_lower == 16.0
        assert result.cumulative_prediction_upper == 18.0
        assert result.average_effect_sd == np.sqrt(1.0 / 3.0)
        assert result.cumulative_effect_sd == np.sqrt(4.0 / 3.0)
        assert result.relative_effect_lower == 1.0 / 9.0
        assert result.relative_effect_upper == 0.25

    def test_single_sample_degenerates_sd_to_zero(self):
        """posterior sampleが1本ならs.d.はNaNではなく0に潰す."""
        y_post = np.array([10.0, 12.0, 14.0])
        predictions = np.array([[9.0, 11.0, 13.0]])

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.05,
        )

        assert np.array_equal(result.predictions_sd, np.zeros(3))
        assert result.average_prediction_sd == 0.0
        assert result.cumulative_prediction_sd == 0.0
        assert result.average_effect_sd == 0.0
        assert result.cumulative_effect_sd == 0.0
        assert result.relative_effect_sd == 0.0


class TestPointwiseCI:
    """各時点CI（R実装一致）テスト."""

    def test_pointwise_ci_bounds_shape(self):
        """point_effect_lower/upper の形状は (t_post,)."""
        predictions, y_post = _make_predictions(
            200, 15, base=10.0, effect=2.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        assert result.point_effect_lower.shape == (15,)
        assert result.point_effect_upper.shape == (15,)

    def test_pointwise_ci_lower_le_mean_le_upper(self):
        """各時点で lower <= point_effects <= upper が成立."""
        predictions, y_post = _make_predictions(
            200, 20, base=10.0, effect=2.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        assert np.all(result.point_effect_lower <= result.point_effects + 1e-10)
        assert np.all(result.point_effect_upper >= result.point_effects - 1e-10)

    def test_pointwise_lower_le_upper(self):
        """各時点で lower <= upper."""
        predictions, y_post = _make_predictions(
            200, 20, base=10.0, effect=0.0, noise_sd=0.5
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        assert np.all(result.point_effect_lower <= result.point_effect_upper + 1e-10)

    def test_summary_ci_uses_average_effect_quantiles_for_r_summary_table(self):
        """R summary の平均CIは pointwise CI 平均ではなく sample 平均の分位点を使う."""
        predictions, y_post = _make_predictions(
            300, 20, base=10.0, effect=3.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        sample_average_effects = (y_post[np.newaxis, :] - predictions).mean(axis=1)
        assert result.ci_lower == float(np.percentile(sample_average_effects, 2.5))
        assert result.ci_upper == float(np.percentile(sample_average_effects, 97.5))

    def test_alpha_01_is_wider_than_alpha_05_because_tail_mass_is_smaller(self):
        """alpha=0.01のCIはalpha=0.05より広い."""
        predictions, y_post = _make_predictions(
            500, 20, base=10.0, effect=2.0, noise_sd=0.3
        )
        r05 = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.05,
        )
        r01 = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.01,
        )
        width_05 = r05.ci_upper - r05.ci_lower
        width_01 = r01.ci_upper - r01.ci_lower
        assert width_01 > width_05

    def test_pointwise_quantile_average_differs_from_average_quantile(self):
        """pointwise CI 平均と summary CI は別定義なので一致しないことがある."""
        y_post = np.array([0.0, 0.0])
        predictions = np.array(
            [
                [4.0, 4.0],
                [4.0, 2.0],
                [4.0, 2.0],
                [2.0, 4.0],
            ]
        )

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.5,
        )
        legacy_ci_lower = float(
            np.percentile((y_post[np.newaxis, :] - predictions).mean(axis=1), 25.0)
        )

        assert result.ci_lower == legacy_ci_lower
        assert legacy_ci_lower == -3.25
        assert float(result.point_effect_lower.mean()) == -4.0
        assert float(result.point_effect_lower.mean()) != legacy_ci_lower

    def test_pointwise_and_summary_ci_diverge_exactly_in_two_sample_symmetric_case(
        self,
    ):
        """対称な2サンプル2時点では pointwise CI と summary CI の差を手計算できる."""
        y_post = np.array([10.0, 10.0])
        predictions = np.array(
            [
                [9.0, 11.0],
                [11.0, 9.0],
            ]
        )

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.05,
        )

        assert np.array_equal(result.point_effects, np.array([0.0, 0.0]))
        assert np.array_equal(result.point_effect_lower, np.array([-0.95, -0.95]))
        assert np.array_equal(result.point_effect_upper, np.array([0.95, 0.95]))
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0


class TestCumulativeCI:
    """累積効果CIテスト."""

    def test_cumulative_ci_shape(self):
        """cumulative_effect_lower/upper の形状は (t_post,)."""
        predictions, y_post = _make_predictions(
            200, 15, base=10.0, effect=2.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        assert result.cumulative_effect_lower.shape == (15,)
        assert result.cumulative_effect_upper.shape == (15,)

    def test_cumulative_ci_lower_le_mean_le_upper(self):
        """累積効果の各時点で lower <= cumulative_effect <= upper."""
        predictions, y_post = _make_predictions(
            200, 20, base=10.0, effect=2.0, noise_sd=0.3
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        assert np.all(
            result.cumulative_effect_lower <= result.cumulative_effect + 1e-10
        )
        assert np.all(
            result.cumulative_effect_upper >= result.cumulative_effect - 1e-10
        )

    def test_cumulative_lower_le_upper(self):
        """累積効果CIで lower <= upper."""
        predictions, y_post = _make_predictions(
            200, 20, base=10.0, effect=0.0, noise_sd=0.5
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        assert np.all(
            result.cumulative_effect_lower <= result.cumulative_effect_upper + 1e-10
        )

    def test_cumulative_ci_widens_over_time(self):
        """正の効果では累積CI幅は時間と共に広がる（単調増加の傾向）."""
        predictions, y_post = _make_predictions(
            1000, 20, base=10.0, effect=0.0, noise_sd=1.0, seed=0
        )
        result = CausalAnalysis.compute_effects(y_post=y_post, predictions=predictions)
        widths = result.cumulative_effect_upper - result.cumulative_effect_lower
        # 最終幅 > 最初の幅（累積誤差が積み上がるため）
        assert widths[-1] > widths[0]

    def test_cumulative_ci_uses_quantiles_of_cumulative_sample_paths(self):
        """累積CIは sample ごとの cumsum を作ってから各時点で分位点を取る."""
        y_post = np.array([10.0, 10.0])
        predictions = np.array(
            [
                [9.0, 11.0],
                [11.0, 9.0],
            ]
        )

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.05,
        )

        assert np.array_equal(result.cumulative_effect, np.array([0.0, 0.0]))
        assert np.array_equal(result.cumulative_effect_lower, np.array([-0.95, 0.0]))
        assert np.array_equal(result.cumulative_effect_upper, np.array([0.95, 0.0]))


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
        assert len(result.point_effect_lower) == 1
        assert len(result.point_effect_upper) == 1
        assert len(result.cumulative_effect_lower) == 1
        assert len(result.cumulative_effect_upper) == 1
        # 1時点の場合: ci == pointwise == cumulative の最終値
        assert abs(result.ci_lower - float(result.point_effect_lower[0])) < 1e-10

    def test_two_samples_minimum(self):
        """n_samples=2 の境界値: 分位点は2点の端値."""
        t_post = 5
        rng = np.random.default_rng(0)
        predictions = rng.normal(10.0, 0.5, (2, t_post))
        y_post = np.full(t_post, 12.0)
        result = CausalAnalysis.compute_effects(
            y_post=y_post, predictions=predictions, alpha=0.05
        )
        # CI は存在する
        assert result.ci_lower < result.ci_upper
        assert result.point_effect_lower.shape == (t_post,)

    def test_single_sample_collapses_all_ci_widths_to_zero(self):
        """サンプルが1本しかないと全CIは退化して平均値そのものになる."""
        y_post = np.array([10.0, 12.0, 14.0])
        predictions = np.array([[9.0, 11.0, 13.0]])

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.05,
        )

        assert np.array_equal(result.point_effects, np.array([1.0, 1.0, 1.0]))
        assert np.array_equal(result.point_effect_lower, result.point_effects)
        assert np.array_equal(result.point_effect_upper, result.point_effects)
        assert np.array_equal(result.cumulative_effect, np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(
            result.cumulative_effect_lower,
            result.cumulative_effect,
        )
        assert np.array_equal(
            result.cumulative_effect_upper,
            result.cumulative_effect,
        )
        assert result.ci_lower == result.ci_upper == 1.0
        assert result.p_value == 1.0

    def test_identical_samples_make_ci_width_zero_because_every_path_matches(self):
        """全サンプル同値なら分位点に幅は生じない."""
        y_post = np.array([10.0, 12.0, 14.0])
        predictions = np.tile(np.array([[9.0, 11.0, 13.0]]), (4, 1))

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.05,
        )

        assert np.array_equal(result.point_effect_lower, result.point_effect_upper)
        assert np.array_equal(
            result.cumulative_effect_lower,
            result.cumulative_effect_upper,
        )
        assert result.ci_lower == result.ci_upper == 1.0

    def test_alpha_zero_uses_full_range_for_pointwise_and_summary_quantiles(self):
        """alpha=0 では pointwise は各時点の端値、summary は sample 平均の端値を返す."""
        y_post = np.array([10.0, 10.0])
        predictions = np.array(
            [
                [9.0, 12.0],
                [13.0, 6.0],
            ]
        )

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=0.0,
        )

        assert np.array_equal(result.point_effect_lower, np.array([-3.0, -2.0]))
        assert np.array_equal(result.point_effect_upper, np.array([1.0, 4.0]))
        assert np.array_equal(result.cumulative_effect_lower, np.array([-3.0, -1.0]))
        assert np.array_equal(result.cumulative_effect_upper, np.array([1.0, 1.0]))
        assert result.ci_lower == -0.5
        assert result.ci_upper == 0.5

    def test_alpha_one_collapses_to_median_for_pointwise_and_summary_quantiles(self):
        """alpha=1 では lower=upper=中央値になりCI幅がゼロになる."""
        y_post = np.array([10.0, 10.0])
        predictions = np.array(
            [
                [9.0, 12.0],
                [13.0, 6.0],
            ]
        )

        result = CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=1.0,
        )

        assert np.array_equal(result.point_effect_lower, np.array([-1.0, 1.0]))
        assert np.array_equal(result.point_effect_upper, np.array([-1.0, 1.0]))
        assert np.array_equal(result.cumulative_effect_lower, np.array([-1.0, 0.0]))
        assert np.array_equal(result.cumulative_effect_upper, np.array([-1.0, 0.0]))
        assert result.ci_lower == result.ci_upper == 0.0
