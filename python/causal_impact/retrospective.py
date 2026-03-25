"""Retrospective attribution mode for CausalImpact.

In retrospective mode, treatment indicator columns (spot, persistent, trend)
are added as covariates and the BSTS model is fit on the entire time series.
Treatment effects are extracted directly from the beta posteriors for the
treatment columns.

This avoids the counterfactual prediction approach of forward mode and instead
estimates treatment effects within a single model fit.

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.
"""

from __future__ import annotations

import numpy as np

from causal_impact.decomposition import (
    DateDecomposition,
    EffectComponent,
    build_design_matrix,
)

__all__ = ["build_treatment_design", "extract_treatment_effects"]


def build_treatment_design(n_total: int, intervention_idx: int) -> np.ndarray:
    """Build treatment indicator design matrix for the full time series.

    Args:
        n_total: Total number of time points.
        intervention_idx: Index of the first post-intervention time point.

    Returns:
        D: shape (n_total, 3) with columns [spot, persistent, trend].
            Pre-intervention rows are all zero.
    """
    D = np.zeros((n_total, 3), dtype=np.float64)
    D[intervention_idx, 0] = 1.0
    D[intervention_idx:, 1] = 1.0
    D[intervention_idx:, 2] = np.arange(n_total - intervention_idx, dtype=np.float64)
    return D


def extract_treatment_effects(
    beta_samples: np.ndarray,
    treatment_col_start: int,
    t_post: int,
    alpha: float = 0.05,
) -> DateDecomposition:
    """Extract treatment effect components from beta posterior samples.

    Args:
        beta_samples: shape (n_samples, k_total). Each row is a beta vector.
        treatment_col_start: Column index where treatment columns begin.
            The next 3 columns are [spot, persistent, trend].
        t_post: Number of post-intervention time points (for trajectory shapes).
        alpha: Significance level for credible intervals.

    Returns:
        DateDecomposition with treatment effect components.
    """
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    include_trend = t_post >= 3
    D = build_design_matrix(t_post, include_trend=include_trend)

    def _make(name: str, col_offset: int, d_col: int) -> EffectComponent:
        coefs = beta_samples[:, treatment_col_start + col_offset]
        trajs = coefs[:, np.newaxis] * D[:, d_col]
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

    spot = _make("spot", 0, 0)
    persistent = _make("persistent", 1, 1)
    trend = _make("trend", 2, 2) if include_trend else None

    # In retrospective mode, residual is zero by construction
    residual_mean = np.zeros(t_post)

    return DateDecomposition(
        spot=spot,
        persistent=persistent,
        trend=trend,
        residual_mean=residual_mean,
        design_matrix=D,
        alpha=alpha,
    )
