"""Summary and report formatting for CausalImpact results."""

from __future__ import annotations

from causal_impact.analysis import CausalImpactResults


class SummaryFormatter:
    """Format CausalImpact results as text summary or natural language report."""

    @staticmethod
    def summary(results: CausalImpactResults, digits: int = 2) -> str:
        fmt = f".{digits}f"

        # Actual
        avg_actual = format(results.actual.mean(), fmt)
        cum_actual = format(results.actual.sum(), fmt)

        # Prediction
        avg_pred = format(results.predictions_mean.mean(), fmt)
        avg_pred_sd = format(results.average_prediction_sd, fmt)
        cum_pred = format(results.predictions_mean.sum(), fmt)
        cum_pred_sd = format(results.cumulative_prediction_sd, fmt)

        # Prediction CI
        avg_pred_ci = (
            f"[{format(results.average_prediction_lower, fmt)}, "
            f"{format(results.average_prediction_upper, fmt)}]"
        )
        cum_pred_ci = (
            f"[{format(results.cumulative_prediction_lower, fmt)}, "
            f"{format(results.cumulative_prediction_upper, fmt)}]"
        )

        # Absolute effect
        avg_eff = format(results.point_effect_mean, fmt)
        avg_eff_sd = format(results.average_effect_sd, fmt)
        cum_eff = format(results.cumulative_effect_total, fmt)
        cum_eff_sd = format(results.cumulative_effect_sd, fmt)

        # Absolute effect CI
        avg_eff_ci = (
            f"[{format(results.ci_lower, fmt)}, {format(results.ci_upper, fmt)}]"
        )
        cum_eff_ci = (
            f"[{format(results.cumulative_effect_lower[-1], fmt)}, "
            f"{format(results.cumulative_effect_upper[-1], fmt)}]"
        )

        # Relative effect
        rel_m = format(results.relative_effect_mean * 100, fmt)
        rel_sd = format(results.relative_effect_sd * 100, fmt)
        rel_lo = format(results.relative_effect_lower * 100, fmt)
        rel_hi = format(results.relative_effect_upper * 100, fmt)

        p_val = format(results.p_value, f".{max(digits, 3)}f")
        prob = format((1 - results.p_value) * 100, fmt)

        pred_row = (
            f"Prediction (s.d.)        "
            f"{avg_pred} ({avg_pred_sd})   "
            f"{cum_pred} ({cum_pred_sd})"
        )
        eff_row = (
            f"Absolute effect (s.d.)   "
            f"{avg_eff} ({avg_eff_sd})    "
            f"{cum_eff} ({cum_eff_sd})"
        )
        rel_row = f"Relative effect (s.d.)   {rel_m}% ({rel_sd}%) {rel_m}% ({rel_sd}%)"
        rel_ci_row = (
            f"95% CI                   [{rel_lo}%, {rel_hi}%] [{rel_lo}%, {rel_hi}%]"
        )

        lines = [
            "Posterior inference {CausalImpact}",
            "",
            "                         Average        Cumulative",
            f"Actual                   {avg_actual}          {cum_actual}",
            pred_row,
            f"95% CI                   {avg_pred_ci}  {cum_pred_ci}",
            "",
            eff_row,
            f"95% CI                   {avg_eff_ci}   {cum_eff_ci}",
            "",
            rel_row,
            rel_ci_row,
            "",
            f"Posterior tail-area probability p: {p_val}",
            f"Posterior prob. of a causal effect: {prob}%",
        ]

        return "\n".join(lines)

    @staticmethod
    def report(results: CausalImpactResults) -> str:
        is_significant = results.p_value < 0.05
        direction = "increase" if results.point_effect_mean >= 0 else "decrease"

        lines = [
            "Analysis report {CausalImpact}",
            "",
            f"During the post-intervention period, the response variable showed "
            f"a {direction} compared to what would have been expected without "
            f"the intervention.",
            "",
            f"The average causal effect was {results.point_effect_mean:.2f} "
            f"(95% CI [{results.ci_lower:.2f}, {results.ci_upper:.2f}]).",
            "",
            f"The cumulative effect over the entire post-period was "
            f"{results.cumulative_effect_total:.2f}.",
            "",
            f"The relative effect was {results.relative_effect_mean * 100:.1f}%.",
            "",
        ]

        p = results.p_value
        if is_significant:
            lines.append(
                f"This effect is statistically significant "
                f"(p = {p:.4f}). The probability of obtaining "
                f"an effect of this magnitude by chance is very "
                f"small. Hence, the causal effect can be "
                f"considered statistically significant."
            )
        else:
            lines.append(
                f"This effect is not statistically significant "
                f"(p = {p:.4f}). The apparent effect could be "
                f"the result of random fluctuations that are "
                f"not related to the intervention. This is "
                f"often the case when the intervention effect "
                f"is small relative to the noise level."
            )

        return "\n".join(lines)
