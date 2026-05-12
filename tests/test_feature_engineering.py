"""
Tests for app.ml.feature_engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.feature_engineering import (
    FeatureEngineer,
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)


class TestIndicatorPrimitives:
    def test_rsi_bounded(self, sample_ohlcv):
        rsi = compute_rsi(sample_ohlcv["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_returns_three_series(self, sample_ohlcv):
        macd, signal, hist = compute_macd(sample_ohlcv["Close"])
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_bollinger_upper_above_lower(self, sample_ohlcv):
        mid, upper, lower = compute_bollinger_bands(sample_ohlcv["Close"])
        valid = upper.dropna()
        valid_lower = lower.dropna()
        assert (valid.values >= valid_lower.dropna().values).all()

    def test_atr_positive(self, sample_ohlcv):
        atr = compute_atr(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        assert (atr.dropna() > 0).all()


class TestFeatureEngineer:
    def test_build_features_returns_dataframe(self, sample_ohlcv):
        fe = FeatureEngineer()
        df = fe.build_features(sample_ohlcv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_no_nan_after_build(self, sample_ohlcv):
        fe = FeatureEngineer()
        df = fe.build_features(sample_ohlcv)
        assert df.isnull().sum().sum() == 0

    def test_target_column_is_binary(self, sample_ohlcv):
        from configs.settings import settings

        fe = FeatureEngineer()
        df = fe.build_features(sample_ohlcv)
        assert set(df[settings.TARGET_COLUMN].unique()).issubset({0, 1})

    def test_no_inf_values(self, sample_ohlcv):
        fe = FeatureEngineer()
        df = fe.build_features(sample_ohlcv)
        feature_cols = fe.get_feature_columns(df)
        assert not np.isinf(df[feature_cols].values).any()

    def test_split_X_y_shapes_match(self, feature_df):
        fe = FeatureEngineer()
        X, y = fe.split_X_y(feature_df)
        assert len(X) == len(y)

    def test_target_not_in_X(self, feature_df):
        from configs.settings import settings

        fe = FeatureEngineer()
        X, _ = fe.split_X_y(feature_df)
        assert settings.TARGET_COLUMN not in X.columns

    def test_ohlcv_not_in_X(self, feature_df):
        fe = FeatureEngineer()
        X, _ = fe.split_X_y(feature_df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in X.columns
