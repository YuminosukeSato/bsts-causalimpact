"""Matplotlib plotting for CausalImpact results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from causal_impact.analysis import CausalImpactResults
    from causal_impact.decomposition import DateDecomposition


class Plotter:
    """Create matplotlib plots of CausalImpact results."""

    VALID_METRICS = ("original", "pointwise", "cumulative", "decomposition")
    DEFAULT_METRICS = ("original", "pointwise", "cumulative")

    @staticmethod
    def plot(
        results: CausalImpactResults,
        y: np.ndarray,
        time_index: pd.DatetimeIndex | pd.RangeIndex,
        pre_end: int,
        metrics: list[str] | None = None,
        decomposition: DateDecomposition | None = None,
    ) -> Figure:
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = list(Plotter.DEFAULT_METRICS)

        n_panels = len(metrics)
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        t_post = len(results.point_effects)
        intervention_idx = time_index[pre_end]
        post_index = time_index[pre_end : pre_end + t_post]

        for ax, metric in zip(axes, metrics):
            if metric == "original":
                Plotter._plot_original(ax, y, time_index, post_index, results)
            elif metric == "pointwise":
                Plotter._plot_pointwise(ax, post_index, results)
            elif metric == "cumulative":
                Plotter._plot_cumulative(ax, post_index, results)
            elif metric == "decomposition":
                if decomposition is None:
                    msg = (
                        "Call ci.decompose() before plotting "
                        "with 'decomposition' metric."
                    )
                    raise ValueError(msg)
                Plotter._plot_decomposition(ax, post_index, decomposition)

            ax.axvline(x=intervention_idx, color="gray", linestyle="--", alpha=0.8)

        plt.tight_layout()
        return fig

    @staticmethod
    def _plot_original(ax, y, time_index, post_index, results):
        ax.plot(time_index, y, color="black", linewidth=1, label="Observed")
        ax.plot(
            post_index,
            results.predictions_mean,
            color="blue",
            linestyle="--",
            label="Counterfactual",
        )
        ax.fill_between(
            post_index,
            results.predictions_lower,
            results.predictions_upper,
            alpha=0.2,
            color="blue",
        )
        ax.set_ylabel("Response")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Original")

    @staticmethod
    def _plot_pointwise(ax, post_index, results):
        ax.plot(post_index, results.point_effects, color="blue", linewidth=1)
        ax.fill_between(
            post_index,
            results.point_effect_lower,
            results.point_effect_upper,
            alpha=0.2,
            color="blue",
        )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_ylabel("Point Effect")
        ax.set_title("Pointwise")

    @staticmethod
    def _plot_cumulative(ax, post_index, results):
        ax.plot(post_index, results.cumulative_effect, color="blue", linewidth=1)
        ax.fill_between(
            post_index,
            results.cumulative_effect_lower,
            results.cumulative_effect_upper,
            alpha=0.2,
            color="blue",
        )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_ylabel("Cumulative Effect")
        ax.set_title("Cumulative")

    @staticmethod
    def _plot_decomposition(ax, post_index, decomposition):
        components = [
            (decomposition.spot, "#d62728", "Spot"),
            (decomposition.persistent, "#1f77b4", "Persistent"),
        ]
        if decomposition.trend is not None:
            components.append((decomposition.trend, "#2ca02c", "Trend"))

        for comp, color, label in components:
            ax.plot(post_index, comp.mean, color=color, linewidth=1, label=label)
            ax.fill_between(post_index, comp.lower, comp.upper, alpha=0.15, color=color)

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax.set_ylabel("Effect Component")
        ax.set_title("DATE Decomposition")
        ax.legend(loc="upper left", fontsize=8)
