"""Data validation and preparation for CausalImpact."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    """Validated and prepared data for the Gibbs sampler."""

    y_pre: np.ndarray  # (T_pre,)
    y_post: np.ndarray  # (T_post,)
    X_pre: np.ndarray | None  # (T_pre, k) or None
    X_post: np.ndarray | None  # (T_post, k) or None
    y_mean: float  # standardization param
    y_sd: float
    X_mean: np.ndarray | None
    X_sd: np.ndarray | None
    time_index: pd.DatetimeIndex | pd.RangeIndex
    alpha: float


class DataProcessor:
    """Validates input data and prepares it for the sampler."""

    @staticmethod
    def validate_and_prepare(
        data: pd.DataFrame | np.ndarray,
        pre_period: list,
        post_period: list,
        alpha: float = 0.05,
        standardize: bool = True,
    ) -> PreparedData:
        # --- alpha validation ---
        if alpha <= 0 or alpha >= 1:
            msg = f"alpha must be in (0, 1), got {alpha}"
            raise ValueError(msg)

        # --- Convert input to DataFrame ---
        df, time_index = DataProcessor._to_dataframe(data)

        # --- Parse periods ---
        pre_start, pre_end, post_start, post_end = DataProcessor._parse_periods(
            pre_period, post_period, time_index
        )

        # --- Period validation ---
        DataProcessor._validate_periods(
            pre_start, pre_end, post_start, post_end, time_index
        )

        # --- Split data ---
        pre_mask = (time_index >= pre_start) & (time_index <= pre_end)
        post_mask = (time_index >= post_start) & (time_index <= post_end)

        y_col = df.iloc[:, 0]
        y_pre_raw = y_col[pre_mask].values.astype(np.float64)
        y_post_raw = y_col[post_mask].values.astype(np.float64)

        has_covariates = df.shape[1] > 1
        if has_covariates:
            X_all = df.iloc[:, 1:].values.astype(np.float64)
            X_pre_raw = X_all[pre_mask]
            X_post_raw = X_all[post_mask]
        else:
            X_pre_raw = None
            X_post_raw = None

        # --- NaN checks ---
        if np.any(np.isnan(y_pre_raw)):
            msg = "NaN values found in pre-period response variable"
            raise ValueError(msg)

        if has_covariates:
            all_X = df.iloc[:, 1:].values.astype(np.float64)
            if np.any(np.isnan(all_X)):
                msg = "NaN values found in covariates"
                raise ValueError(msg)

        # --- Warnings ---
        if len(y_pre_raw) <= 1:
            warnings.warn(
                "Pre-period contains a single observation. "
                "Model estimation may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        if np.std(y_pre_raw) == 0:
            warnings.warn(
                "Pre-period response has zero variance (constant values). "
                "Standardization will use sd=1.",
                UserWarning,
                stacklevel=2,
            )

        # --- Standardize ---
        if standardize:
            y_mean = float(np.mean(y_pre_raw))
            y_sd = float(np.std(y_pre_raw, ddof=1)) if len(y_pre_raw) > 1 else 1.0
            if y_sd == 0:
                y_sd = 1.0
            y_pre = (y_pre_raw - y_mean) / y_sd
            y_post = (y_post_raw - y_mean) / y_sd

            if has_covariates:
                X_mean = np.mean(X_pre_raw, axis=0)
                X_sd = np.std(X_pre_raw, axis=0, ddof=1)
                X_sd[X_sd == 0] = 1.0
                X_pre = (X_pre_raw - X_mean) / X_sd
                X_post = (X_post_raw - X_mean) / X_sd
            else:
                X_mean = None
                X_sd = None
                X_pre = None
                X_post = None
        else:
            y_mean = 0.0
            y_sd = 1.0
            y_pre = y_pre_raw
            y_post = y_post_raw
            X_pre = X_pre_raw
            X_post = X_post_raw
            X_mean = None
            X_sd = None

        return PreparedData(
            y_pre=y_pre,
            y_post=y_post,
            X_pre=X_pre,
            X_post=X_post,
            y_mean=y_mean,
            y_sd=y_sd,
            X_mean=X_mean,
            X_sd=X_sd,
            time_index=time_index,
            alpha=alpha,
        )

    @staticmethod
    def _to_dataframe(
        data: pd.DataFrame | np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex | pd.RangeIndex]:
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            df = pd.DataFrame(data)
            time_index = pd.RangeIndex(len(df))
            return df, time_index

        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.DatetimeIndex):
                time_index = data.index
            else:
                try:
                    time_index = pd.DatetimeIndex(data.index)
                except (ValueError, TypeError):
                    time_index = pd.RangeIndex(len(data))
            return data, time_index

        msg = f"data must be pd.DataFrame or np.ndarray, got {type(data)}"
        raise TypeError(msg)

    @staticmethod
    def _parse_periods(
        pre_period: list,
        post_period: list,
        time_index: pd.DatetimeIndex | pd.RangeIndex,
    ) -> tuple:
        if isinstance(time_index, pd.DatetimeIndex):
            pre_start = pd.Timestamp(pre_period[0])
            pre_end = pd.Timestamp(pre_period[1])
            post_start = pd.Timestamp(post_period[0])
            post_end = pd.Timestamp(post_period[1])
        else:
            pre_start = int(pre_period[0])
            pre_end = int(pre_period[1])
            post_start = int(post_period[0])
            post_end = int(post_period[1])
        return pre_start, pre_end, post_start, post_end

    @staticmethod
    def _validate_periods(pre_start, pre_end, post_start, post_end, time_index) -> None:
        idx_min = time_index.min()
        idx_max = time_index.max()

        if pre_start < idx_min or post_end > idx_max:
            msg = (
                f"Period bounds outside data range. "
                f"Data: [{idx_min}, {idx_max}], "
                f"pre: [{pre_start}, {pre_end}], post: [{post_start}, {post_end}]"
            )
            raise ValueError(msg)

        if pre_end >= post_start:
            msg = (
                f"Pre-period must end before post-period starts. "
                f"pre_end={pre_end}, post_start={post_start}"
            )
            raise ValueError(msg)

        if pre_start > pre_end:
            msg = "Pre-period start must be before or equal to pre-period end"
            raise ValueError(msg)

        if post_start > post_end:
            msg = "Post-period start must be before or equal to post-period end"
            raise ValueError(msg)
