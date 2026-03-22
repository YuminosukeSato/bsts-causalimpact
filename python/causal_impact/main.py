"""CausalImpact main class: facade for the full analysis pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from causal_impact._core import run_gibbs_sampler
from causal_impact.analysis import CausalAnalysis, CausalImpactResults
from causal_impact.data import DataProcessor, PreparedData
from causal_impact.options import ModelOptions
from causal_impact.plot import Plotter
from causal_impact.summary import SummaryFormatter

if TYPE_CHECKING:
    from matplotlib.figure import Figure

DEFAULT_MODEL_ARGS = {
    "niter": 1000,
    "nwarmup": 500,
    "nchains": 1,
    "seed": 0,
    "standardize_data": True,
    "prior_level_sd": 0.01,
    "expected_model_size": 2,
    "dynamic_regression": False,
    "state_model": "local_level",
    "nseasons": None,
    "season_duration": None,
}


def _normalize_model_args(
    model_args: dict | ModelOptions | None,
) -> dict:
    if isinstance(model_args, ModelOptions):
        args = model_args.to_dict()
    else:
        args = dict(model_args or {})

    if "season.duration" in args:
        if "season_duration" in args:
            msg = "Use either season.duration or season_duration, not both"
            raise ValueError(msg)
        args["season_duration"] = args.pop("season.duration")

    normalized = {**DEFAULT_MODEL_ARGS, **args}
    if normalized["state_model"] not in {"local_level", "local_linear_trend"}:
        msg = (
            "state_model must be one of "
            "{'local_level', 'local_linear_trend'}, "
            f"got {normalized['state_model']}"
        )
        raise ValueError(msg)
    return normalized


class CausalImpact:
    """Causal inference using Bayesian structural time series.

    Args:
        data: DataFrame or ndarray. First column is response, rest are covariates.
        pre_period: [start, end] of pre-intervention period.
        post_period: [start, end] of post-intervention period.
        model_args: MCMC parameters (niter, nwarmup, nchains, seed, prior_level_sd).
        alpha: Significance level for credible intervals (default 0.05).
    """

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        pre_period: list[str | int | pd.Timestamp],
        post_period: list[str | int | pd.Timestamp],
        model_args: dict | ModelOptions | None = None,
        alpha: float = 0.05,
    ) -> None:
        args = _normalize_model_args(model_args)
        standardize = args.pop("standardize_data")

        self._prepared = DataProcessor.validate_and_prepare(
            data, pre_period, post_period, alpha=alpha, standardize=standardize
        )
        self._alpha = alpha
        self._data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self._pre_period = pre_period
        self._post_period = post_period

        self._samples = self._run_sampler(self._prepared, args)
        self._results = self._compute_results(self._prepared, self._samples)

    def _run_sampler(self, prepared: PreparedData, args: dict):
        y_full = np.concatenate([prepared.y_pre, prepared.y_post])

        x_cols = None
        if prepared.X_pre is not None and prepared.X_post is not None:
            X_full = np.vstack([prepared.X_pre, prepared.X_post])
            x_cols = [X_full[:, j].tolist() for j in range(X_full.shape[1])]

        return run_gibbs_sampler(
            y=y_full.tolist(),
            x=x_cols,
            pre_end=len(prepared.y_pre),
            niter=args["niter"],
            nwarmup=args["nwarmup"],
            nchains=args["nchains"],
            seed=args["seed"],
            prior_level_sd=args["prior_level_sd"],
            expected_model_size=float(args["expected_model_size"]),
            nseasons=args["nseasons"],
            season_duration=args["season_duration"],
            dynamic_regression=bool(args.get("dynamic_regression", False)),
            state_model=str(args["state_model"]),
        )

    def _compute_results(self, prepared: PreparedData, samples) -> CausalImpactResults:
        predictions = np.array(samples.predictions)

        # De-standardize predictions if needed
        if prepared.y_sd != 1.0 or prepared.y_mean != 0.0:
            predictions = predictions * prepared.y_sd + prepared.y_mean

        # De-standardize y_post
        y_post = prepared.y_post * prepared.y_sd + prepared.y_mean

        return CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=self._alpha,
        )

    def summary(self, output: str = "summary", digits: int = 2) -> str:
        if output == "report":
            return SummaryFormatter.report(self._results, alpha=self._alpha)
        return SummaryFormatter.summary(self._results, alpha=self._alpha, digits=digits)

    def report(self) -> str:
        return SummaryFormatter.report(self._results, alpha=self._alpha)

    def plot(self, metrics: list[str] | None = None) -> Figure:
        if isinstance(self._data, pd.DataFrame):
            y = self._data.iloc[:, 0].values
        else:
            arr = np.asarray(self._data)
            y = arr[:, 0] if arr.ndim > 1 else arr

        return Plotter.plot(
            self._results,
            y,
            self._prepared.time_index,
            len(self._prepared.y_pre),
            metrics=metrics,
        )

    @property
    def inferences(self) -> pd.DataFrame:
        t_post = len(self._results.point_effects)
        pre_end = len(self._prepared.y_pre)
        post_index = self._prepared.time_index[pre_end : pre_end + t_post]

        return pd.DataFrame(
            {
                "actual": self._results.actual,
                "predicted_mean": self._results.predictions_mean,
                "predicted_lower": self._results.predictions_lower,
                "predicted_upper": self._results.predictions_upper,
                "predictions_sd": self._results.predictions_sd,
                "point_effect": self._results.point_effects,
                "point_effect_lower": self._results.point_effect_lower,
                "point_effect_upper": self._results.point_effect_upper,
                "cumulative_effect": self._results.cumulative_effect,
                "cumulative_effect_lower": self._results.cumulative_effect_lower,
                "cumulative_effect_upper": self._results.cumulative_effect_upper,
            },
            index=post_index,
        )

    @property
    def posterior_inclusion_probs(self) -> np.ndarray | None:
        """Posterior inclusion probabilities for each covariate.

        Returns None when there are no covariates (k=0).
        """
        gamma = self._samples.gamma
        if not gamma or not gamma[0]:
            return None
        return np.array(gamma).mean(axis=0)

    @property
    def summary_stats(self) -> dict:
        return {
            "point_effect_mean": self._results.point_effect_mean,
            "ci_lower": self._results.ci_lower,
            "ci_upper": self._results.ci_upper,
            "cumulative_effect_total": self._results.cumulative_effect_total,
            "relative_effect_mean": self._results.relative_effect_mean,
            "p_value": self._results.p_value,
        }
