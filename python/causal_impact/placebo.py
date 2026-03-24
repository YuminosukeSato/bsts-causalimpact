"""Placebo test for CausalImpact results validation.

A placebo test artificially splits the pre-period into "fake pre" and "fake post"
at multiple split points and runs the Gibbs sampler for each split. The absolute
average effect from each split forms a null distribution. The real effect is ranked
against this distribution to produce a p-value.

If the real intervention had a genuine effect, its effect magnitude should be
unusually large compared to the placebo distribution (low p-value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaceboTestResults:
    """Results of a placebo test.

    Attributes:
        p_value: Fraction of placebo effects >= real effect.
        rank_ratio: rank / (n_placebos + 1), more conservative p-value estimate.
        effect_distribution: Absolute average effects from each placebo split.
        real_effect: Absolute average effect from the real intervention.
        n_placebos: Number of placebo splits evaluated.
    """

    p_value: float
    rank_ratio: float
    effect_distribution: np.ndarray
    real_effect: float
    n_placebos: int
