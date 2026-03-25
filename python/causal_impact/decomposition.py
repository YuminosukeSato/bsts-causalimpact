"""DATE decomposition of pointwise causal effects.

Decomposes the pointwise treatment effect into three components:
- Spot: immediate impulse at the intervention time point
- Persistent: sustained level shift from intervention onward
- Trend: linearly growing/decaying effect over time

For each MCMC sample, OLS projects the pointwise effect vector onto a
design matrix D, yielding per-sample coefficients. Posterior summaries
(mean, credible intervals) are computed over these samples.

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836, Eq. (11e).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EffectComponent",
    "DateDecomposition",
    "decompose_effects",
    "build_design_matrix",
]


@dataclass(frozen=True)
class EffectComponent:
    """A single decomposed component of the causal effect."""

    name: str
    coefficient: float
    ci_lower: float
    ci_upper: float
    samples: np.ndarray
    coefficients: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class DateDecomposition:
    """Results of DATE decomposition.

    Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.
    """

    spot: EffectComponent
    persistent: EffectComponent
    trend: EffectComponent | None
    residual_mean: np.ndarray
    design_matrix: np.ndarray
    alpha: float


def build_design_matrix(t_post: int, *, include_trend: bool = True) -> np.ndarray:
    """Construct the DATE design matrix.

    Args:
        t_post: Number of post-intervention time points.
        include_trend: Include trend column. Forced False when t_post < 3.

    Returns:
        D: shape (t_post, n_cols) where n_cols is 2 or 3.

    Raises:
        ValueError: If t_post < 1.
    """
    if t_post < 1:
        msg = f"t_post must be >= 1, got {t_post}"
        raise ValueError(msg)

    n_cols = 3 if (include_trend and t_post >= 3) else 2
    D = np.zeros((t_post, n_cols), dtype=np.float64)
    D[0, 0] = 1.0
    D[:, 1] = 1.0
    if n_cols == 3:
        D[:, 2] = np.arange(t_post, dtype=np.float64)
    return D


def decompose_effects(
    effects: np.ndarray,
    alpha: float = 0.05,
) -> DateDecomposition:
    """Decompose pointwise effects into spot, persistent, and trend.

    Args:
        effects: shape (n_samples, T_post). Each row is a pointwise effect
            trajectory (y_post - prediction_s) for one MCMC sample.
        alpha: Significance level for credible intervals.

    Returns:
        DateDecomposition with per-component posterior summaries.

    Raises:
        ValueError: If effects is empty.

    Warns:
        UserWarning: When T_post < 3 (trend not identifiable) or T_post == 1
            (spot/persistent collinear).
    """
    effects = np.asarray(effects, dtype=np.float64)
    if effects.ndim == 1:
        effects = effects[np.newaxis, :]
    n_samples, t_post = effects.shape

    if n_samples == 0:
        msg = "effects array has 0 samples"
        raise ValueError(msg)
    if t_post == 0:
        msg = "effects array has 0 time points"
        raise ValueError(msg)

    include_trend = t_post >= 3
    if not include_trend:
        warnings.warn(
            f"T_post={t_post} < 3: trend not identifiable. "
            "Only spot and persistent are estimated.",
            UserWarning,
            stacklevel=2,
        )
    if t_post == 1:
        warnings.warn(
            "T_post=1: spot and persistent are collinear. "
            "Minimum-norm OLS solution (equal split). Interpret with caution.",
            UserWarning,
            stacklevel=2,
        )

    D = build_design_matrix(t_post, include_trend=include_trend)
    D_pinv = np.linalg.pinv(D)
    coefficients = effects @ D_pinv.T

    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    def _make(name: str, col: int) -> EffectComponent:
        coefs = coefficients[:, col]
        trajs = coefs[:, np.newaxis] * D[:, col]
        return EffectComponent(
            name=name,
            coefficient=float(np.mean(coefs)),
            ci_lower=float(np.quantile(coefs, lower_q)),
            ci_upper=float(np.quantile(coefs, upper_q)),
            samples=trajs,
            coefficients=coefs,
            mean=trajs.mean(axis=0),
            lower=np.quantile(trajs, lower_q, axis=0),
            upper=np.quantile(trajs, upper_q, axis=0),
        )

    spot = _make("spot", 0)
    persistent = _make("persistent", 1)
    trend = _make("trend", 2) if include_trend else None

    fitted = spot.samples + persistent.samples
    if trend is not None:
        fitted = fitted + trend.samples
    residual_mean = (effects - fitted).mean(axis=0)

    return DateDecomposition(
        spot=spot,
        persistent=persistent,
        trend=trend,
        residual_mean=residual_mean,
        design_matrix=D,
        alpha=alpha,
    )
