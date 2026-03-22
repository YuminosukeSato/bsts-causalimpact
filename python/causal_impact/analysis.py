"""Causal effect computation from Gibbs sampler output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalImpactResults:
    """Results of causal impact analysis."""

    point_effects: np.ndarray  # (T_post,) mean effect per time point
    ci_lower: float  # lower CI bound on average effect
    ci_upper: float  # upper CI bound on average effect
    point_effect_mean: float  # mean of point effects across time
    cumulative_effect: np.ndarray  # (T_post,) cumulative point effects
    cumulative_effect_total: float  # total cumulative effect
    relative_effect_mean: float  # relative effect (effect / predicted)
    p_value: float  # Bayesian one-sided tail probability
    predictions_mean: np.ndarray  # (T_post,) mean counterfactual
    predictions_lower: np.ndarray  # (T_post,) lower CI counterfactual
    predictions_upper: np.ndarray  # (T_post,) upper CI counterfactual


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

        # Effect per sample per time point: observed - counterfactual
        # predictions shape: (n_samples, t_post)
        effects = y_post[np.newaxis, :] - predictions  # (n_samples, t_post)

        # Average effect across time for each sample
        avg_effects = effects.mean(axis=1)  # (n_samples,)

        # Point effects: mean across samples at each time point
        point_effects = effects.mean(axis=0)  # (t_post,)

        # CI on average effect
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        ci_lower = float(np.percentile(avg_effects, 100 * lower_q))
        ci_upper = float(np.percentile(avg_effects, 100 * upper_q))

        # Mean effect
        point_effect_mean = float(avg_effects.mean())

        # Cumulative effect
        cumulative_effect = np.cumsum(point_effects)
        cumulative_effect_total = float(cumulative_effect[-1])

        # Relative effect
        pred_mean_total = predictions.mean()
        if abs(pred_mean_total) > 1e-10:
            relative_effect_mean = point_effect_mean / pred_mean_total
        else:
            relative_effect_mean = 0.0

        # p-value: proportion of samples where average effect has opposite sign
        if point_effect_mean >= 0:
            p_value = float(np.mean(avg_effects < 0))
        else:
            p_value = float(np.mean(avg_effects > 0))
        # Ensure minimum p-value of 1/n_samples
        p_value = max(p_value, 1.0 / n_samples)

        # Counterfactual prediction summaries
        predictions_mean = predictions.mean(axis=0)
        predictions_lower = np.percentile(predictions, 100 * lower_q, axis=0)
        predictions_upper = np.percentile(predictions, 100 * upper_q, axis=0)

        return CausalImpactResults(
            point_effects=point_effects,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            point_effect_mean=point_effect_mean,
            cumulative_effect=cumulative_effect,
            cumulative_effect_total=cumulative_effect_total,
            relative_effect_mean=relative_effect_mean,
            p_value=p_value,
            predictions_mean=predictions_mean,
            predictions_lower=predictions_lower,
            predictions_upper=predictions_upper,
        )
