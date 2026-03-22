"""Summary and report formatting for CausalImpact results."""

from __future__ import annotations

from causal_impact.analysis import CausalImpactResults


class SummaryFormatter:
    """Format CausalImpact results as text summary or natural language report."""

    @staticmethod
    def summary(results: CausalImpactResults, digits: int = 2) -> str:
        fmt = f".{digits}f"

        avg_effect = format(results.point_effect_mean, fmt)
        avg_ci = f"[{format(results.ci_lower, fmt)}, {format(results.ci_upper, fmt)}]"
        cum_effect = format(results.cumulative_effect_total, fmt)
        cum_ci = (
            f"[{format(results.cumulative_effect_lower[-1], fmt)}, "
            f"{format(results.cumulative_effect_upper[-1], fmt)}]"
        )
        rel_effect = format(results.relative_effect_mean * 100, fmt)
        p_val = format(results.p_value, f".{max(digits, 3)}f")

        lines = [
            "Posterior inference {CausalImpact}",
            "",
            "                         Average        Cumulative",
            "Actual                   -              -",
            "Prediction (s.d.)        -              -",
            f"95% CI                   {avg_ci}       {cum_ci}",
            "",
            f"Absolute effect (mean)   {avg_effect}           {cum_effect}",
            f"Relative effect          {rel_effect}%",
            "",
            f"Posterior tail-area probability p: {p_val}",
        ]

        if results.p_value < 0.05:
            lines.append(
                "Posterior prob. of a causal effect: "
                f"{format((1 - results.p_value) * 100, fmt)}%"
            )
        else:
            lines.append("The effect is not statistically significant.")

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
