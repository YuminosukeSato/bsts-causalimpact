"""Split conformal inference for distribution-free prediction intervals.

Splits the pre-period into a training half and a calibration half.
Fits a Gibbs sampler on the training half, computes nonconformity scores
(absolute residuals) on the calibration half, and derives a conformal
quantile q_hat that guarantees marginal coverage >= 1 - alpha.

Reference: Vovk et al. (2005), "Algorithmic Learning in a Random World"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causal_impact._core import run_gibbs_sampler

if TYPE_CHECKING:
    from causal_impact.data import PreparedData


@dataclass(frozen=True)
class ConformalResults:
    """Results of split conformal inference.

    Attributes:
        q_hat: Conformal quantile threshold.
        lower: Lower conformal prediction interval for post-period.
        upper: Upper conformal prediction interval for post-period.
        n_calibration: Number of calibration points used.
    """

    q_hat: float
    lower: np.ndarray
    upper: np.ndarray
    n_calibration: int


def compute_conformal_intervals(
    prepared: PreparedData,
    model_args: dict,
    alpha: float = 0.05,
    post_predictions_destd: np.ndarray | None = None,
) -> ConformalResults:
    """Compute split conformal prediction intervals.

    Args:
        prepared: Validated and prepared data from DataProcessor.
        model_args: Gibbs sampler arguments (niter, nwarmup, seed, etc.).
        alpha: Significance level (default 0.05 for 95% coverage).
        post_predictions_destd: De-standardized post-period prediction means
            from the main model. If None, re-runs the sampler.

    Returns:
        ConformalResults with conformal intervals for the post-period.
    """
    t_pre = len(prepared.y_pre)
    t_train = t_pre // 2
    n_calib = t_pre - t_train

    if n_calib < 2:
        msg = (
            f"Not enough calibration points: n_calib={n_calib}. "
            f"Need at least 2. T_pre={t_pre}."
        )
        raise ValueError(msg)

    if t_pre < 4:
        warnings.warn(
            f"Very short pre-period (T_pre={t_pre}). "
            "Conformal intervals may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # Step 1: Run sampler on training split to get calibration predictions
    y_for_calib = np.ascontiguousarray(prepared.y_pre)

    x_cols_calib = None
    if prepared.X_pre is not None:
        x_cols_calib = np.ascontiguousarray(prepared.X_pre.T)

    samples = run_gibbs_sampler(
        y=y_for_calib,
        x=x_cols_calib,
        pre_end=t_train,
        niter=model_args["niter"],
        nwarmup=model_args["nwarmup"],
        nchains=1,
        seed=model_args["seed"] + 7919,
        prior_level_sd=model_args["prior_level_sd"],
        expected_model_size=float(model_args["expected_model_size"]),
        nseasons=model_args["nseasons"],
        season_duration=model_args["season_duration"],
        dynamic_regression=bool(model_args.get("dynamic_regression", False)),
        state_model=str(model_args["state_model"]),
    )

    predictions = np.array(samples.predictions)
    # predictions shape: (n_samples, n_calib) — post-period only
    calib_preds_mean = predictions.mean(axis=0)

    # Step 2: Compute nonconformity scores on calibration set (de-standardized)
    calib_actual = prepared.y_pre[t_train:] * prepared.y_sd + prepared.y_mean
    calib_predicted = calib_preds_mean * prepared.y_sd + prepared.y_mean
    scores = np.abs(calib_actual - calib_predicted)

    # Step 3: Conformal quantile
    level = min((n_calib + 1) * (1 - alpha) / n_calib, 1.0)
    q_hat = float(np.quantile(scores, level))

    # Step 4: Apply conformal interval to post-period predictions
    if post_predictions_destd is None:
        post_predictions_destd = _run_full_model_post_predictions(prepared, model_args)

    lower = post_predictions_destd - q_hat
    upper = post_predictions_destd + q_hat

    return ConformalResults(
        q_hat=q_hat,
        lower=lower,
        upper=upper,
        n_calibration=n_calib,
    )


def _run_full_model_post_predictions(
    prepared: PreparedData,
    model_args: dict,
) -> np.ndarray:
    """Re-run the full sampler to get de-standardized post predictions."""
    y_full = np.ascontiguousarray(np.concatenate([prepared.y_pre, prepared.y_post]))
    x_cols = None
    if prepared.X_pre is not None and prepared.X_post is not None:
        X_full = np.ascontiguousarray(np.vstack([prepared.X_pre, prepared.X_post]))
        x_cols = np.ascontiguousarray(X_full.T)

    full_samples = run_gibbs_sampler(
        y=y_full,
        x=x_cols,
        pre_end=len(prepared.y_pre),
        niter=model_args["niter"],
        nwarmup=model_args["nwarmup"],
        nchains=1,
        seed=model_args["seed"],
        prior_level_sd=model_args["prior_level_sd"],
        expected_model_size=float(model_args["expected_model_size"]),
        nseasons=model_args["nseasons"],
        season_duration=model_args["season_duration"],
        dynamic_regression=bool(model_args.get("dynamic_regression", False)),
        state_model=str(model_args["state_model"]),
    )

    full_predictions = np.array(full_samples.predictions)
    post_preds_mean = full_predictions.mean(axis=0)[len(prepared.y_pre) :]
    return post_preds_mean * prepared.y_sd + prepared.y_mean
