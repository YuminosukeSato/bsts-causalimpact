"""Control group selection using DTW (Dynamic Time Warping) distance.

Selects the top-k most similar control series to the target series
from the pre-intervention period.

Optimization layers:
1. z-score normalization (scale-invariant comparison)
2. LB_Keogh lower bound filter (prunes >90% candidates in O(T))
3. Sakoe-Chiba band + early abandoning in DTW (Rust)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_impact._core import (
    py_dtw_distance,
    py_lb_keogh_distance,
    py_lb_keogh_envelope,
)


def select_controls(
    df: pd.DataFrame,
    target_col: str,
    pre_period: list,
    top_k: int = 3,
    window: int | None = None,
) -> pd.DataFrame:
    """Select the top-k control series most similar to the target in the pre-period.

    Args:
        df: DataFrame with target and candidate columns.
        target_col: Name of the target (response) column.
        pre_period: [start, end] of the pre-intervention period.
        top_k: Number of controls to select (must be > 0).
        window: Sakoe-Chiba band width for DTW. None = unconstrained.

    Returns:
        DataFrame with target_col + top_k selected controls.

    Raises:
        ValueError: If target_col not in df, or top_k <= 0.
    """
    if target_col not in df.columns:
        msg = f"target_col '{target_col}' not found in DataFrame columns"
        raise ValueError(msg)
    if top_k <= 0:
        msg = f"top_k must be > 0, got {top_k}"
        raise ValueError(msg)

    candidates = [c for c in df.columns if c != target_col]
    if not candidates:
        return df[[target_col]]

    # Extract and z-score normalize the pre-period
    y_pre = _extract_and_normalize(df[target_col], df.index, pre_period)

    w = window if window is not None else len(y_pre)
    lo, hi = py_lb_keogh_envelope(y_pre.tolist(), w)

    best_so_far = float("inf")
    distances: list[tuple[float, str]] = []

    for col in candidates:
        xi_pre = _extract_and_normalize(df[col], df.index, pre_period)
        if len(xi_pre) != len(y_pre):
            continue

        lb = py_lb_keogh_distance(xi_pre.tolist(), lo, hi)
        if lb >= best_so_far and len(distances) >= top_k:
            continue

        d = py_dtw_distance(xi_pre.tolist(), y_pre.tolist(), window, best_so_far)
        distances.append((d, col))

        # Update best_so_far: the k-th smallest distance seen so far
        if len(distances) >= top_k:
            distances.sort(key=lambda x: x[0])
            best_so_far = distances[top_k - 1][0]

    distances.sort(key=lambda x: x[0])
    selected = [col for _, col in distances[:top_k]]
    return df[[target_col, *selected]]


def _extract_and_normalize(
    series: pd.Series,
    index: pd.Index,
    pre_period: list,
) -> np.ndarray:
    """Extract the pre-period slice and z-score normalize it."""
    start, end = pre_period[0], pre_period[1]
    if isinstance(index, pd.DatetimeIndex):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        mask = (index >= start) & (index <= end)
    else:
        mask = (index >= start) & (index <= end)
    values = series[mask].values.astype(np.float64)
    sd = np.std(values, ddof=1) if len(values) > 1 else 1.0
    if sd == 0:
        sd = 1.0
    return (values - np.mean(values)) / sd
