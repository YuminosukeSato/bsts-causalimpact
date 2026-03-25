"""CausalImpact main class: facade for the full analysis pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from causal_impact._core import py_run_placebo_test, run_gibbs_sampler
from causal_impact.analysis import CausalAnalysis, CausalImpactResults
from causal_impact.conformal import ConformalResults, compute_conformal_intervals
from causal_impact.data import DataProcessor, PreparedData
from causal_impact.decomposition import DateDecomposition, decompose_effects
from causal_impact.options import ModelOptions
from causal_impact.placebo import PlaceboTestResults
from causal_impact.plot import Plotter
from causal_impact.retrospective import (
    build_treatment_design,
    extract_treatment_effects,
)
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
    "prior_type": "spike_slab",
    "nseasons": None,
    "season_duration": None,
}


def _normalize_model_args(
    model_args: dict | ModelOptions | None,
) -> dict:
    if isinstance(model_args, ModelOptions):
        return model_args.to_dict()

    raw = dict(model_args or {})

    if "season.duration" in raw:
        if "season_duration" in raw:
            msg = "Use either season.duration or season_duration, not both"
            raise ValueError(msg)
        raw["season_duration"] = raw.pop("season.duration")

    # mode は ModelOptions のフィールドではないため先に抽出する
    mode = raw.pop("mode", None)

    _allowed = set(DEFAULT_MODEL_ARGS)
    unknown = set(raw) - _allowed
    if unknown:
        raise ValueError(f"Unknown model_args keys: {sorted(unknown)}")

    # ModelOptions を経由することで __post_init__ のバリデーションを全て通す
    merged = {**DEFAULT_MODEL_ARGS, **raw}
    opts = ModelOptions(**merged)

    result = opts.to_dict()
    if mode is not None:
        result["mode"] = mode
    return result


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
        mode = args.pop("mode", "forward")
        if mode not in {"forward", "retrospective"}:
            msg = f"mode must be 'forward' or 'retrospective', got '{mode}'"
            raise ValueError(msg)
        if mode == "retrospective" and args.get("dynamic_regression"):
            msg = (
                "dynamic_regression is not supported in retrospective mode. "
                "Use mode='forward' for time-varying coefficients."
            )
            raise ValueError(msg)
        if mode == "retrospective" and args.get("prior_type") == "horseshoe":
            msg = (
                "horseshoe prior is not supported in retrospective mode. "
                "Use prior_type='spike_slab' or mode='forward'."
            )
            raise ValueError(msg)

        self._prepared = DataProcessor.validate_and_prepare(
            data, pre_period, post_period, alpha=alpha, standardize=standardize
        )
        self._alpha = alpha
        self._data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self._pre_period = pre_period
        self._post_period = post_period
        self._mode = mode

        self._model_args = args
        self._decomposition: DateDecomposition | None = None

        if mode == "retrospective":
            self._samples, retro_predictions = self._run_retrospective(
                self._prepared,
                args,
            )
            self._results = self._compute_results_from_predictions(
                self._prepared,
                retro_predictions,
            )
        else:
            self._samples = self._run_sampler(self._prepared, args)
            self._results = self._compute_results(self._prepared, self._samples)

    def _run_sampler(self, prepared: PreparedData, args: dict):
        y_full = np.ascontiguousarray(np.concatenate([prepared.y_pre, prepared.y_post]))

        x_cols = None
        if prepared.X_pre is not None and prepared.X_post is not None:
            X_full = np.ascontiguousarray(np.vstack([prepared.X_pre, prepared.X_post]))
            x_cols = np.ascontiguousarray(X_full.T)

        return run_gibbs_sampler(
            y=y_full,
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
            prior_type=str(args.get("prior_type", "spike_slab")),
        )

    def _run_retrospective(self, prepared: PreparedData, args: dict):
        """Run sampler in retrospective mode: treatment indicators as covariates."""
        y_full = np.ascontiguousarray(np.concatenate([prepared.y_pre, prepared.y_post]))
        n_total = len(y_full)
        intervention_idx = len(prepared.y_pre)
        t_post = len(prepared.y_post)

        # Build treatment design columns
        # NOTE: treatment_X is NOT divided by y_sd. The sampler fits on
        # standardized y, so beta_treat = true_effect / y_sd. The
        # de-standardization step (beta * y_sd) recovers the original scale.
        treatment_X = build_treatment_design(n_total, intervention_idx)

        # Stack user covariates + treatment columns
        if prepared.X_pre is not None and prepared.X_post is not None:
            X_full = np.vstack([prepared.X_pre, prepared.X_post])
            X_full = np.hstack([X_full, treatment_X])
        else:
            X_full = treatment_X

        x_cols = np.ascontiguousarray(X_full.T)
        k_total = X_full.shape[1]

        # Force spike-and-slab off: set expected_model_size >= k_total
        expected_model_size = float(max(args["expected_model_size"], k_total))

        samples = run_gibbs_sampler(
            y=y_full,
            x=x_cols,
            pre_end=n_total,  # Fit on entire series
            niter=args["niter"],
            nwarmup=args["nwarmup"],
            nchains=args["nchains"],
            seed=args["seed"],
            prior_level_sd=args["prior_level_sd"],
            expected_model_size=expected_model_size,
            nseasons=args["nseasons"],
            season_duration=args["season_duration"],
            dynamic_regression=False,  # Not supported in retrospective
            state_model=str(args["state_model"]),
        )

        # Extract treatment effects from beta posteriors
        beta_samples = np.array(samples.beta)
        treatment_col_start = k_total - 3
        # De-standardize beta coefficients
        if prepared.y_sd != 1.0:
            beta_destd = beta_samples * prepared.y_sd
        else:
            beta_destd = beta_samples

        self._decomposition = extract_treatment_effects(
            beta_destd,
            treatment_col_start,
            t_post,
            alpha=self._alpha,
        )

        # Construct counterfactual predictions from treatment decomposition
        # counterfactual = actual - treatment_effect
        y_post = prepared.y_post * prepared.y_sd + prepared.y_mean
        treatment_effects = (
            self._decomposition.spot.samples + self._decomposition.persistent.samples
        )
        if self._decomposition.trend is not None:
            treatment_effects = treatment_effects + self._decomposition.trend.samples
        predictions = y_post[np.newaxis, :] - treatment_effects

        return samples, predictions

    def _compute_results_from_predictions(
        self,
        prepared: PreparedData,
        predictions: np.ndarray,
    ) -> CausalImpactResults:
        """Compute results from pre-built predictions (used by retrospective mode)."""
        y_post = prepared.y_post * prepared.y_sd + prepared.y_mean
        return CausalAnalysis.compute_effects(
            y_post=y_post,
            predictions=predictions,
            alpha=self._alpha,
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
            decomposition=self._decomposition,
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

        Only available for prior_type='spike_slab'. Returns None for
        horseshoe prior or when there are no covariates (k=0).
        """
        if self._model_args.get("prior_type", "spike_slab") == "horseshoe":
            return None
        gamma = self._samples.gamma
        if not gamma or not gamma[0]:
            return None
        return np.array(gamma).mean(axis=0)

    @property
    def posterior_shrinkage(self) -> np.ndarray | None:
        """Mean shrinkage factor per covariate (horseshoe prior only).

        kappa_j = E[1 / (1 + lambda_j^2 * tau^2)].
        Values close to 1 indicate strong shrinkage (covariate excluded).
        Values close to 0 indicate weak shrinkage (covariate included).

        Returns None for spike_slab prior or when there are no covariates.
        """
        if self._model_args.get("prior_type", "spike_slab") != "horseshoe":
            return None
        kappa = self._samples.kappa_shrinkage
        if not kappa or not kappa[0]:
            return None
        return np.array(kappa).mean(axis=0)

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

    def run_placebo_test(
        self,
        n_placebos: int | None = None,
        min_pre_length: int = 3,
    ) -> PlaceboTestResults:
        """Run a placebo test to validate the causal effect.

        Splits the pre-period at multiple points and compares
        the real effect against the placebo distribution.

        Args:
            n_placebos: Max number of placebo splits. None = use all valid splits.
            min_pre_length: Minimum pre-period length for each placebo split.

        Returns:
            PlaceboTestResults with p_value, effect_distribution, etc.
        """
        prepared = self._prepared
        args = self._model_args

        y_full = np.ascontiguousarray(np.concatenate([prepared.y_pre, prepared.y_post]))
        x_cols = None
        if prepared.X_pre is not None and prepared.X_post is not None:
            X_full = np.ascontiguousarray(np.vstack([prepared.X_pre, prepared.X_post]))
            x_cols = np.ascontiguousarray(X_full.T)

        result = py_run_placebo_test(
            y=y_full,
            x=x_cols,
            pre_end=len(prepared.y_pre),
            niter=args["niter"],
            nwarmup=args["nwarmup"],
            seed=args["seed"],
            prior_level_sd=args["prior_level_sd"],
            expected_model_size=float(args["expected_model_size"]),
            nseasons=args["nseasons"],
            season_duration=args["season_duration"],
            state_model=str(args["state_model"]),
            n_placebos=n_placebos,
            min_pre_length=min_pre_length,
        )

        return PlaceboTestResults(
            p_value=result.p_value,
            rank_ratio=result.rank_ratio,
            effect_distribution=np.array(result.effect_distribution),
            real_effect=result.real_effect,
            n_placebos=result.n_placebos,
        )

    def decompose(self, alpha: float | None = None) -> DateDecomposition:
        """Decompose pointwise effects into spot, persistent, and trend (DATE).

        Based on Schaffe-Odeleye et al. (2026), arXiv:2602.00836.

        In retrospective mode, the decomposition is computed during initialization
        from beta posteriors. If alpha matches, the cached result is returned.
        If alpha differs, the decomposition is recomputed from the stored betas.

        Args:
            alpha: Significance level. Defaults to self._alpha.

        Returns:
            DateDecomposition with per-component posterior summaries.
        """
        if alpha is None:
            alpha = self._alpha

        if self._mode == "retrospective":
            if self._decomposition is not None and self._decomposition.alpha == alpha:
                return self._decomposition
            # Recompute from stored beta posteriors with new alpha
            beta_samples = np.array(self._samples.beta)
            if self._prepared.y_sd != 1.0:
                beta_samples = beta_samples * self._prepared.y_sd
            t_post = len(self._prepared.y_post)
            k_total = beta_samples.shape[1]
            treatment_col_start = k_total - 3
            self._decomposition = extract_treatment_effects(
                beta_samples, treatment_col_start, t_post, alpha=alpha,
            )
            return self._decomposition

        predictions = np.array(self._samples.predictions)
        if self._prepared.y_sd != 1.0 or self._prepared.y_mean != 0.0:
            predictions = predictions * self._prepared.y_sd + self._prepared.y_mean
        y_post = self._prepared.y_post * self._prepared.y_sd + self._prepared.y_mean
        effects = y_post[np.newaxis, :] - predictions
        self._decomposition = decompose_effects(effects, alpha=alpha)
        return self._decomposition

    def run_conformal_analysis(
        self,
        alpha: float | None = None,
    ) -> ConformalResults:
        """Run split conformal inference to get distribution-free prediction intervals.

        Splits the pre-period into train and calibration halves,
        computes nonconformity scores on the calibration set, and derives
        a conformal quantile q_hat.

        Args:
            alpha: Significance level. Defaults to self._alpha.

        Returns:
            ConformalResults with q_hat, lower, upper bounds.
        """
        if alpha is None:
            alpha = self._alpha

        return compute_conformal_intervals(
            prepared=self._prepared,
            model_args=self._model_args,
            alpha=alpha,
            post_predictions_destd=self._results.predictions_mean,
        )
