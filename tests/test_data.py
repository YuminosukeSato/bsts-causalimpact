"""Tests for DataProcessor: validation, preparation, standardization."""

import numpy as np
import pandas as pd
import pytest

from causal_impact.data import DataProcessor, PreparedData


class TestValidateDataFrameInput:
    """DataFrameの入力バリデーション."""

    def test_validate_dataframe_valid(self, sample_df, pre_period, post_period):
        result = DataProcessor.validate_and_prepare(
            sample_df, pre_period, post_period
        )
        assert isinstance(result, PreparedData)

    def test_validate_numpy_array(self, rng, pre_period_int, post_period_int):
        arr = rng.normal(0, 1, (100, 2))
        result = DataProcessor.validate_and_prepare(
            arr, pre_period_int, post_period_int
        )
        assert isinstance(result, PreparedData)

    def test_validate_single_column(
        self, sample_df_no_cov, pre_period, post_period
    ):
        result = DataProcessor.validate_and_prepare(
            sample_df_no_cov, pre_period, post_period
        )
        assert result.X_pre is None
        assert result.X_post is None


class TestPeriodValidation:
    """期間指定のバリデーション."""

    def test_pre_period_overlap_post(self, sample_df):
        with pytest.raises(ValueError, match="overlap|before"):
            DataProcessor.validate_and_prepare(
                sample_df,
                ["2020-01-01", "2020-03-15"],
                ["2020-03-10", "2020-04-09"],
            )

    def test_pre_period_after_post(self, sample_df):
        with pytest.raises(ValueError, match="before|after"):
            DataProcessor.validate_and_prepare(
                sample_df,
                ["2020-03-11", "2020-04-09"],
                ["2020-01-01", "2020-03-10"],
            )

    def test_pre_period_outside_data(self, sample_df):
        with pytest.raises(ValueError, match="outside|range|bound"):
            DataProcessor.validate_and_prepare(
                sample_df,
                ["2019-01-01", "2019-06-01"],
                ["2019-06-02", "2019-12-31"],
            )

    def test_pre_period_one_point(self, sample_df):
        with pytest.warns(UserWarning, match="single|one|short"):
            DataProcessor.validate_and_prepare(
                sample_df,
                ["2020-01-01", "2020-01-01"],
                ["2020-01-02", "2020-04-09"],
            )

    def test_post_period_one_point(self, sample_df):
        # 1時点のpost期間はエラーにならない
        result = DataProcessor.validate_and_prepare(
            sample_df,
            ["2020-01-01", "2020-04-08"],
            ["2020-04-09", "2020-04-09"],
        )
        assert len(result.y_post) == 1


class TestNaNHandling:
    """NaN値の処理."""

    def test_nan_in_y_pre(self, sample_df, pre_period, post_period):
        df = sample_df.copy()
        df.iloc[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN|nan|missing"):
            DataProcessor.validate_and_prepare(df, pre_period, post_period)

    def test_nan_in_y_post(self, sample_df, pre_period, post_period):
        # R互換: post期間のNaNは許可
        df = sample_df.copy()
        df.iloc[80, 0] = np.nan
        result = DataProcessor.validate_and_prepare(df, pre_period, post_period)
        assert isinstance(result, PreparedData)

    def test_nan_in_covariates(self, sample_df, pre_period, post_period):
        df = sample_df.copy()
        df.iloc[5, 1] = np.nan
        with pytest.raises(ValueError, match="NaN|nan|missing"):
            DataProcessor.validate_and_prepare(df, pre_period, post_period)


class TestStandardization:
    """標準化と逆標準化."""

    def test_standardize_and_restore(self, sample_df, pre_period, post_period):
        result = DataProcessor.validate_and_prepare(
            sample_df, pre_period, post_period
        )
        # 標準化されたy_preの平均≈0, 標準偏差≈1
        assert abs(result.y_pre.mean()) < 0.3
        assert abs(result.y_pre.std() - 1.0) < 0.3
        # 逆標準化で元値に戻る
        original_y = sample_df["y"].values[:70]
        restored = result.y_pre * result.y_sd + result.y_mean
        np.testing.assert_allclose(restored, original_y, rtol=1e-10)

    def test_standardize_disabled(self, sample_df, pre_period, post_period):
        result = DataProcessor.validate_and_prepare(
            sample_df, pre_period, post_period, standardize=False
        )
        original_y = sample_df["y"].values[:70]
        np.testing.assert_allclose(result.y_pre, original_y, rtol=1e-10)
        assert result.y_mean == 0.0
        assert result.y_sd == 1.0


class TestEdgeCases:
    """境界値テスト."""

    def test_all_same_y(self, pre_period, post_period):
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({"y": np.ones(n)}, index=dates)
        with pytest.warns(UserWarning, match="constant|zero|same|variance"):
            DataProcessor.validate_and_prepare(df, pre_period, post_period)


class TestAlphaValidation:
    """alpha パラメータのバリデーション."""

    def test_alpha_0(self, sample_df, pre_period, post_period):
        with pytest.raises(ValueError, match="alpha"):
            DataProcessor.validate_and_prepare(
                sample_df, pre_period, post_period, alpha=0.0
            )

    def test_alpha_1(self, sample_df, pre_period, post_period):
        with pytest.raises(ValueError, match="alpha"):
            DataProcessor.validate_and_prepare(
                sample_df, pre_period, post_period, alpha=1.0
            )

    def test_alpha_negative(self, sample_df, pre_period, post_period):
        with pytest.raises(ValueError, match="alpha"):
            DataProcessor.validate_and_prepare(
                sample_df, pre_period, post_period, alpha=-0.1
            )

    def test_alpha_0_01(self, sample_df, pre_period, post_period):
        result = DataProcessor.validate_and_prepare(
            sample_df, pre_period, post_period, alpha=0.01
        )
        assert isinstance(result, PreparedData)


class TestIndexParsing:
    """インデックス解析."""

    def test_date_index_string(self, sample_df, pre_period, post_period):
        result = DataProcessor.validate_and_prepare(
            sample_df, pre_period, post_period
        )
        assert isinstance(result.time_index, pd.DatetimeIndex)

    def test_date_index_integer(self, rng, pre_period_int, post_period_int):
        arr = rng.normal(0, 1, (100, 2))
        result = DataProcessor.validate_and_prepare(
            arr, pre_period_int, post_period_int
        )
        assert isinstance(result.time_index, pd.RangeIndex)
