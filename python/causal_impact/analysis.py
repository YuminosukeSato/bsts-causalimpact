"""Causal effect computation from Gibbs sampler output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CausalImpactResults:
    """Results of causal impact analysis."""

    actual: np.ndarray
    point_effects: np.ndarray
    point_effect_lower: np.ndarray
    point_effect_upper: np.ndarray
    ci_lower: float
    ci_upper: float
    point_effect_mean: float
    average_effect_sd: float
    cumulative_effect: np.ndarray
    cumulative_effect_lower: np.ndarray
    cumulative_effect_upper: np.ndarray
    cumulative_effect_total: float
    cumulative_effect_sd: float
    relative_effect_mean: float
    relative_effect_sd: float
    relative_effect_lower: float
    relative_effect_upper: float
    p_value: float
    predictions_mean: np.ndarray
    predictions_sd: np.ndarray
    predictions_lower: np.ndarray
    predictions_upper: np.ndarray
    average_prediction_sd: float
    average_prediction_lower: float
    average_prediction_upper: float
    cumulative_prediction_sd: float
    cumulative_prediction_lower: float
    cumulative_prediction_upper: float
    conformal_lower: np.ndarray | None = field(default=None)
    conformal_upper: np.ndarray | None = field(default=None)


class CausalAnalysis:
    """Compute causal effects from posterior samples."""

    @staticmethod
    def compute_effects(
        y_post: np.ndarray,
        predictions: np.ndarray,
        alpha: float = 0.05,
    ) -> CausalImpactResults:
        y_post = np.asarray(y_post, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)

        n_samples = predictions.shape[0]

        effects = y_post[np.newaxis, :] - predictions
        avg_effects = effects.mean(axis=1)
        point_effects = effects.mean(axis=0)

        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        point_effect_lower = np.percentile(effects, 100 * lower_q, axis=0)
        point_effect_upper = np.percentile(effects, 100 * upper_q, axis=0)
        ci_lower = float(np.percentile(avg_effects, 100 * lower_q))
        ci_upper = float(np.percentile(avg_effects, 100 * upper_q))

        point_effect_mean = float(avg_effects.mean())

        cumulative_effect = np.cumsum(point_effects)
        cum_effects_samples = np.cumsum(effects, axis=1)
        cumulative_effect_lower = np.percentile(
            cum_effects_samples,
            100 * lower_q,
            axis=0,
        )
        cumulative_effect_upper = np.percentile(
            cum_effects_samples,
            100 * upper_q,
            axis=0,
        )
        cumulative_effect_total = float(cumulative_effect[-1])

        actual = y_post.copy()

        if n_samples == 1:
            predictions_sd_arr = np.zeros(predictions.shape[1])
        else:
            predictions_sd_arr = np.std(predictions, axis=0, ddof=1)

        avg_pred_per_sample = predictions.mean(axis=1)
        cum_pred_per_sample = predictions.sum(axis=1)

        if n_samples == 1:
            average_prediction_sd = 0.0
            cumulative_prediction_sd = 0.0
        else:
            average_prediction_sd = float(np.std(avg_pred_per_sample, ddof=1))
            cumulative_prediction_sd = float(np.std(cum_pred_per_sample, ddof=1))

        average_prediction_lower = float(
            np.percentile(avg_pred_per_sample, 100 * lower_q)
        )
        average_prediction_upper = float(
            np.percentile(avg_pred_per_sample, 100 * upper_q)
        )
        cumulative_prediction_lower = float(
            np.percentile(cum_pred_per_sample, 100 * lower_q)
        )
        cumulative_prediction_upper = float(
            np.percentile(cum_pred_per_sample, 100 * upper_q)
        )

        cum_effects_per_sample = effects.sum(axis=1)

        if n_samples == 1:
            average_effect_sd = 0.0
            cumulative_effect_sd = 0.0
        else:
            average_effect_sd = float(np.std(avg_effects, ddof=1))
            cumulative_effect_sd = float(np.std(cum_effects_per_sample, ddof=1))

        avg_pred_per_sample_safe = np.where(
            np.abs(avg_pred_per_sample) > 1e-10,
            avg_pred_per_sample,
            np.nan,
        )
        rel_effects_per_sample = np.where(
            np.abs(avg_pred_per_sample) > 1e-10,
            avg_effects / avg_pred_per_sample_safe,
            0.0,
        )

        relative_effect_mean = float(rel_effects_per_sample.mean())

        if n_samples == 1:
            relative_effect_sd = 0.0
        else:
            relative_effect_sd = float(np.std(rel_effects_per_sample, ddof=1))

        relative_effect_lower = float(
            np.percentile(rel_effects_per_sample, 100 * lower_q)
        )
        relative_effect_upper = float(
            np.percentile(rel_effects_per_sample, 100 * upper_q)
        )

        if point_effect_mean >= 0:
            p_value = float(np.mean(avg_effects < 0))
        else:
            p_value = float(np.mean(avg_effects > 0))
        p_value = max(p_value, 1.0 / n_samples)

        predictions_mean = predictions.mean(axis=0)
        predictions_lower = np.percentile(predictions, 100 * lower_q, axis=0)
        predictions_upper = np.percentile(predictions, 100 * upper_q, axis=0)

        return CausalImpactResults(
            actual=actual,
            point_effects=point_effects,
            point_effect_lower=point_effect_lower,
            point_effect_upper=point_effect_upper,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            point_effect_mean=point_effect_mean,
            average_effect_sd=average_effect_sd,
            cumulative_effect=cumulative_effect,
            cumulative_effect_lower=cumulative_effect_lower,
            cumulative_effect_upper=cumulative_effect_upper,
            cumulative_effect_total=cumulative_effect_total,
            cumulative_effect_sd=cumulative_effect_sd,
            relative_effect_mean=relative_effect_mean,
            relative_effect_sd=relative_effect_sd,
            relative_effect_lower=relative_effect_lower,
            relative_effect_upper=relative_effect_upper,
            p_value=p_value,
            predictions_mean=predictions_mean,
            predictions_sd=predictions_sd_arr,
            predictions_lower=predictions_lower,
            predictions_upper=predictions_upper,
            average_prediction_sd=average_prediction_sd,
            average_prediction_lower=average_prediction_lower,
            average_prediction_upper=average_prediction_upper,
            cumulative_prediction_sd=cumulative_prediction_sd,
            cumulative_prediction_lower=cumulative_prediction_lower,
            cumulative_prediction_upper=cumulative_prediction_upper,
        )
