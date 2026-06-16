"""
tests/test_sector_correlation_features.py
==========================================
Unit tests for SectorCorrelationFeatureEngineer and the FeatureEngineer
sector-correlation integration layer (Improvement 1).

All yfinance and network calls are mocked — no internet access required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml.feature_engineering import FeatureEngineer
from app.ml.sector_correlation_features import (
    _SECTOR_CORRELATION_FEATURE_NAMES,
    SECTOR_ETF_MAP,
    SectorCorrelationFeatureEngineer,
    _log_returns,
    _market_breadth_proxy,
    _relative_return_features,
    _resolve_sector_etf,
    _rolling_beta_feature,
    _rolling_correlation_feature,
    _sector_trend_regime,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def price_index() -> pd.DatetimeIndex:
    """500-day business-day index ending today."""
    return pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=500)


@pytest.fixture
def stock_close(price_index) -> pd.Series:
    """Synthetic stock price series using geometric Brownian motion."""
    np.random.seed(42)
    n = len(price_index)
    prices = [150.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * np.exp(np.random.normal(0.0002, 0.015)))
    return pd.Series(prices, index=price_index, name="Close")


@pytest.fixture
def etf_close(price_index) -> pd.Series:
    """Synthetic sector ETF price series (slightly correlated with stock)."""
    np.random.seed(99)
    n = len(price_index)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * np.exp(np.random.normal(0.0001, 0.012)))
    return pd.Series(prices, index=price_index, name="Close")


@pytest.fixture
def spy_close(price_index) -> pd.Series:
    """Synthetic SPY price series."""
    np.random.seed(7)
    n = len(price_index)
    prices = [400.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * np.exp(np.random.normal(0.0001, 0.010)))
    return pd.Series(prices, index=price_index, name="Close")


def _make_mock_engineer(etf_close: pd.Series, spy_close: pd.Series):
    """Return a SectorCorrelationFeatureEngineer with mocked ETF fetches."""
    eng = SectorCorrelationFeatureEngineer()

    def _mock_fetch(etf_ticker, price_index, period_years=5):
        if etf_ticker == "SPY":
            return spy_close.reindex(price_index, method="ffill")
        return etf_close.reindex(price_index, method="ffill")

    return eng, _mock_fetch


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR_ETF_MAP coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestSectorEtfMap:
    def test_all_eleven_gics_sectors_covered(self):
        canonical_sectors = {
            "Technology",
            "Health Care",
            "Financials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Energy",
            "Industrials",
            "Materials",
            "Real Estate",
            "Utilities",
            "Communication Services",
        }
        mapped = set(SECTOR_ETF_MAP.keys())
        # All canonical sectors must be in the map
        assert canonical_sectors.issubset(mapped)

    def test_etf_tickers_are_non_empty_strings(self):
        for sector, etf in SECTOR_ETF_MAP.items():
            assert isinstance(etf, str) and len(etf) >= 2, (
                f"Sector '{sector}' has invalid ETF ticker: {etf!r}"
            )

    def test_no_duplicate_etf_values_for_canonical_sectors(self):
        # Each canonical sector should map to a distinct ETF
        # (aliases like "Healthcare"/"Health Care" may share one)
        canonical = [
            SECTOR_ETF_MAP[s]
            for s in SECTOR_ETF_MAP
            if s
            in {
                "Technology",
                "Health Care",
                "Financials",
                "Consumer Discretionary",
                "Consumer Staples",
                "Energy",
                "Industrials",
                "Materials",
                "Real Estate",
                "Utilities",
                "Communication Services",
            }
        ]
        assert len(canonical) == len(set(canonical)), (
            "Canonical GICS sectors must map to distinct ETF tickers"
        )


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_sector_etf
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveSectorEtf:
    def _clear_cache(self):
        from app.ml.sector_correlation_features import _SECTOR_CACHE

        _SECTOR_CACHE.clear()

    def test_technology_resolves_to_xlk(self):
        self._clear_cache()
        with patch(
            "app.ml.sector_correlation_features._fetch_yfinance_info_sector",
            return_value="Technology",
        ):
            with patch("yfinance.Ticker") as mock_yf:
                mock_yf.return_value.info = {"sector": "Technology"}
                sector, etf = _resolve_sector_etf("AAPL")
        assert etf == "XLK"
        assert sector == "Technology"

    def test_unknown_sector_falls_back_to_spy(self):
        self._clear_cache()
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.info = {"sector": "Exotic Sector XYZ"}
            sector, etf = _resolve_sector_etf("AAPL")
        assert etf == "SPY"

    def test_yfinance_failure_falls_back_to_spy(self):
        self._clear_cache()
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            sector, etf = _resolve_sector_etf("AAPL")
        assert etf == "SPY"
        assert sector == ""

    def test_result_is_cached(self):
        self._clear_cache()
        call_count = [0]

        def _mock_ticker(t):
            call_count[0] += 1
            m = MagicMock()
            m.info = {"sector": "Technology"}
            return m

        with patch("yfinance.Ticker", side_effect=_mock_ticker):
            _resolve_sector_etf("AAPL")
            _resolve_sector_etf("AAPL")

        assert call_count[0] == 1, "Second call should be served from cache"


# ─────────────────────────────────────────────────────────────────────────────
# Primitive feature builders
# ─────────────────────────────────────────────────────────────────────────────


class TestLogReturns:
    def test_shape_matches_input(self, stock_close):
        rets = _log_returns(stock_close)
        assert len(rets) == len(stock_close)

    def test_first_value_is_nan(self, stock_close):
        rets = _log_returns(stock_close)
        assert np.isnan(rets.iloc[0])

    def test_no_inf_values(self, stock_close):
        rets = _log_returns(stock_close)
        assert not np.isinf(rets.dropna()).any()


class TestRelativeReturnFeatures:
    def test_columns_produced(self, stock_close, etf_close):
        df = _relative_return_features(stock_close, etf_close, prefix="sector")
        expected = {"sector_rel_ret_5d", "sector_rel_ret_21d", "sector_rel_ret_63d"}
        assert expected.issubset(df.columns)

    def test_shape_matches_input(self, stock_close, etf_close):
        df = _relative_return_features(stock_close, etf_close, prefix="sector")
        assert len(df) == len(stock_close)

    def test_outperforming_is_positive(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        # Stock +2% all days, ETF +1% all days → relative return should be positive
        stock = pd.Series([100.0 * (1.02**i) for i in range(100)], index=idx)
        etf = pd.Series([100.0 * (1.01**i) for i in range(100)], index=idx)
        df = _relative_return_features(stock, etf, prefix="sector")
        valid = df["sector_rel_ret_5d"].dropna()
        assert (valid > 0).all(), (
            "Outperforming stock should have positive relative returns"
        )

    def test_underperforming_is_negative(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        stock = pd.Series([100.0 * (1.01**i) for i in range(100)], index=idx)
        etf = pd.Series([100.0 * (1.02**i) for i in range(100)], index=idx)
        df = _relative_return_features(stock, etf, prefix="sector")
        valid = df["sector_rel_ret_5d"].dropna()
        assert (valid < 0).all(), (
            "Underperforming stock should have negative relative returns"
        )


class TestRollingBeta:
    def test_output_is_series(self, stock_close, spy_close):
        stock_rets = _log_returns(stock_close)
        market_rets = _log_returns(spy_close)
        beta = _rolling_beta_feature(stock_rets, market_rets)
        assert isinstance(beta, pd.Series)

    def test_shape_matches_input(self, stock_close, spy_close):
        stock_rets = _log_returns(stock_close)
        market_rets = _log_returns(spy_close)
        beta = _rolling_beta_feature(stock_rets, market_rets)
        assert len(beta) == len(stock_close)

    def test_beta_is_positive_for_correlated_series(self):
        """When stock moves with the market, beta should be positive."""
        idx = pd.bdate_range("2020-01-01", periods=200)
        np.random.seed(0)
        market_rets = pd.Series(np.random.normal(0, 0.01, 200), index=idx)
        # Stock returns = 1.5 * market + small noise → beta ≈ 1.5
        stock_rets = 1.5 * market_rets + pd.Series(
            np.random.normal(0, 0.001, 200), index=idx
        )
        beta = _rolling_beta_feature(stock_rets, market_rets, window=60)
        valid = beta.dropna()
        assert len(valid) > 0
        assert (valid > 0).all(), (
            "Beta must be positive for positively correlated series"
        )

    def test_first_window_minus_one_is_nan(self, stock_close, spy_close):
        stock_rets = _log_returns(stock_close)
        market_rets = _log_returns(spy_close)
        beta = _rolling_beta_feature(stock_rets, market_rets, window=60)
        # Rows before the window should be NaN
        assert beta.iloc[:59].isna().all()


class TestRollingCorrelation:
    def test_output_is_series(self, stock_close, etf_close):
        corr = _rolling_correlation_feature(
            _log_returns(stock_close), _log_returns(etf_close)
        )
        assert isinstance(corr, pd.Series)

    def test_values_bounded_between_minus_one_and_one(self, stock_close, etf_close):
        corr = _rolling_correlation_feature(
            _log_returns(stock_close), _log_returns(etf_close)
        )
        valid = corr.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_identical_series_has_correlation_one(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        np.random.seed(1)
        rets = pd.Series(np.random.normal(0, 0.01, 100), index=idx)
        corr = _rolling_correlation_feature(rets, rets, window=20)
        valid = corr.dropna()
        assert (valid.round(10) == 1.0).all()


class TestSectorTrendRegime:
    def test_columns_produced(self, etf_close):
        df = _sector_trend_regime(etf_close)
        expected = {
            "sector_sma50_200_cross",
            "sector_trend_regime",
            "sector_close_vs_sma50",
        }
        assert expected.issubset(df.columns)

    def test_cross_is_binary(self, etf_close):
        df = _sector_trend_regime(etf_close)
        valid = df["sector_sma50_200_cross"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_regime_is_in_minus_one_zero_one(self, etf_close):
        df = _sector_trend_regime(etf_close)
        valid = df["sector_trend_regime"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_strong_uptrend_gives_regime_one(self):
        """Monotonically rising series → SMA50 > SMA200 → regime = 1."""
        idx = pd.bdate_range("2015-01-01", periods=500)
        close = pd.Series([100.0 + i * 0.5 for i in range(500)], index=idx)
        df = _sector_trend_regime(close)
        # After 200-day warm-up, regime should be +1
        regime_after_warmup = df["sector_trend_regime"].iloc[210:]
        assert (regime_after_warmup == 1.0).all()


class TestMarketBreadthProxy:
    def test_columns_produced(self, spy_close):
        df = _market_breadth_proxy(spy_close)
        expected = {
            "market_spy_vs_sma200",
            "market_above_sma200",
            "market_spy_momentum_21d",
        }
        assert expected.issubset(df.columns)

    def test_above_sma200_is_binary(self, spy_close):
        df = _market_breadth_proxy(spy_close)
        valid = df["market_above_sma200"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_spy_vs_sma200_is_near_zero_for_flat_series(self):
        idx = pd.bdate_range("2015-01-01", periods=400)
        # Flat series — price equals SMA200 always → deviation ≈ 0
        close = pd.Series([100.0] * 400, index=idx)
        df = _market_breadth_proxy(close)
        valid = df["market_spy_vs_sma200"].dropna()
        assert valid.abs().max() < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# SectorCorrelationFeatureEngineer.build()
# ─────────────────────────────────────────────────────────────────────────────


class TestSectorCorrelationFeatureEngineerBuild:
    def test_returns_dataframe_with_price_index(
        self, stock_close, etf_close, spy_close, price_index
    ):
        eng = SectorCorrelationFeatureEngineer()
        with (
            patch("yfinance.Ticker") as mock_yf,
            patch(
                "app.ml.sector_correlation_features._fetch_etf_close",
                side_effect=lambda etf, idx, **kw: (
                    spy_close.reindex(idx, method="ffill")
                    if etf == "SPY"
                    else etf_close.reindex(idx, method="ffill")
                ),
            ),
        ):
            mock_yf.return_value.info = {"sector": "Technology"}
            result = eng.build("AAPL", stock_close, price_index)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(price_index)
        assert result.index.equals(price_index)

    def test_contains_expected_core_columns(
        self, stock_close, etf_close, spy_close, price_index
    ):
        eng = SectorCorrelationFeatureEngineer()
        with (
            patch("yfinance.Ticker") as mock_yf,
            patch(
                "app.ml.sector_correlation_features._fetch_etf_close",
                side_effect=lambda etf, idx, **kw: (
                    spy_close.reindex(idx, method="ffill")
                    if etf == "SPY"
                    else etf_close.reindex(idx, method="ffill")
                ),
            ),
        ):
            mock_yf.return_value.info = {"sector": "Technology"}
            result = eng.build("AAPL", stock_close, price_index)

        for col in [
            "sector_rel_ret_5d",
            "sector_rel_ret_21d",
            "market_beta",
            "sector_corr_20d",
            "sector_rsi_14",
            "sector_trend_regime",
            "market_spy_vs_sma200",
        ]:
            assert col in result.columns, f"Expected column missing: {col}"

    def test_no_rows_dropped(self, stock_close, etf_close, spy_close, price_index):
        """build() must not drop rows — that is the caller's responsibility."""
        eng = SectorCorrelationFeatureEngineer()
        with (
            patch("yfinance.Ticker") as mock_yf,
            patch(
                "app.ml.sector_correlation_features._fetch_etf_close",
                side_effect=lambda etf, idx, **kw: (
                    spy_close.reindex(idx, method="ffill")
                    if etf == "SPY"
                    else etf_close.reindex(idx, method="ffill")
                ),
            ),
        ):
            mock_yf.return_value.info = {"sector": "Technology"}
            result = eng.build("AAPL", stock_close, price_index)

        assert len(result) == len(price_index)

    def test_returns_empty_df_for_short_index(self, stock_close):
        eng = SectorCorrelationFeatureEngineer()
        tiny_index = pd.bdate_range("2026-01-01", periods=10)
        tiny_close = stock_close.reindex(tiny_index, method="ffill")
        result = eng.build("AAPL", tiny_close, tiny_index)
        assert result.empty or len(result) == len(tiny_index)

    def test_gracefully_handles_etf_fetch_failure(self, stock_close, price_index):
        """When ETF data is unavailable, should return empty or partial DataFrame."""
        eng = SectorCorrelationFeatureEngineer()
        with (
            patch("yfinance.Ticker") as mock_yf,
            patch(
                "app.ml.sector_correlation_features._fetch_etf_close",
                return_value=pd.Series(dtype=float, index=price_index),
            ),
        ):
            mock_yf.return_value.info = {"sector": "Technology"}
            result = eng.build("AAPL", stock_close, price_index)

        # Should not raise; result may be empty or partial
        assert isinstance(result, pd.DataFrame)

    def test_include_market_false_skips_spy_features(
        self, stock_close, etf_close, price_index
    ):
        eng = SectorCorrelationFeatureEngineer(include_market=False)
        with (
            patch("yfinance.Ticker") as mock_yf,
            patch(
                "app.ml.sector_correlation_features._fetch_etf_close",
                return_value=etf_close.reindex(price_index, method="ffill"),
            ),
        ):
            mock_yf.return_value.info = {"sector": "Technology"}
            result = eng.build("AAPL", stock_close, price_index)

        # SPY-specific columns must not appear
        spy_cols = [c for c in result.columns if c.startswith("market_")]
        assert len(spy_cols) == 0, (
            f"SPY/market columns should not be present with include_market=False: {spy_cols}"
        )

    def test_no_inf_values_in_output(
        self, stock_close, etf_close, spy_close, price_index
    ):
        eng = SectorCorrelationFeatureEngineer()
        with (
            patch("yfinance.Ticker") as mock_yf,
            patch(
                "app.ml.sector_correlation_features._fetch_etf_close",
                side_effect=lambda etf, idx, **kw: (
                    spy_close.reindex(idx, method="ffill")
                    if etf == "SPY"
                    else etf_close.reindex(idx, method="ffill")
                ),
            ),
        ):
            mock_yf.return_value.info = {"sector": "Technology"}
            result = eng.build("AAPL", stock_close, price_index)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result[numeric_cols].values).any()

    def test_get_feature_names(self):
        eng = SectorCorrelationFeatureEngineer()
        names = eng.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert names == list(_SECTOR_CORRELATION_FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer integration
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineerSectorIntegration:
    """
    Tests the FeatureEngineer.include_sector_correlation=True pathway,
    mocking SectorCorrelationFeatureEngineer.build() to avoid network calls.
    """

    def _make_sector_df(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a synthetic sector correlation feature DataFrame."""
        np.random.seed(123)
        n = len(index)
        return pd.DataFrame(
            {
                "sector_rel_ret_5d": np.random.normal(0, 0.01, n),
                "sector_rel_ret_21d": np.random.normal(0, 0.02, n),
                "market_beta": np.random.uniform(0.5, 1.5, n),
                "sector_corr_20d": np.random.uniform(0.4, 0.95, n),
                "sector_rsi_14": np.random.uniform(30, 70, n),
                "sector_trend_regime": np.random.choice([-1.0, 0.0, 1.0], n),
                "market_spy_vs_sma200": np.random.normal(0, 0.05, n),
                "market_above_sma200": np.random.choice([0, 1], n),
            },
            index=index,
        )

    def test_include_sector_correlation_false_does_not_add_columns(self, sample_ohlcv):
        """Default behaviour: no sector correlation columns."""
        eng = FeatureEngineer(include_sector_correlation=False)
        df = eng.build_features(sample_ohlcv)
        sector_cols = [
            c for c in df.columns if c.startswith("sector_") or c.startswith("market_")
        ]
        assert len(sector_cols) == 0

    def test_include_sector_correlation_true_merges_columns(self, sample_ohlcv):
        mock_sec_eng = MagicMock()

        def _build(ticker, stock_close, price_index):
            return self._make_sector_df(price_index)

        mock_sec_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker="AAPL",
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv)

        sector_cols = [c for c in df.columns if c.startswith("sector_")]
        assert len(sector_cols) > 0
        assert "sector_rel_ret_5d" in df.columns

    def test_sector_correlation_requires_ticker(self, sample_ohlcv):
        """Without a ticker, sector correlation is skipped with a warning (not an error)."""
        mock_sec_eng = MagicMock()
        mock_sec_eng.build.return_value = pd.DataFrame()

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker=None,
            sector_correlation_engineer=mock_sec_eng,
        )
        # Should not raise; sector features simply omitted
        df = eng.build_features(sample_ohlcv)
        assert not df.empty

    def test_sector_correlation_failure_does_not_crash_pipeline(self, sample_ohlcv):
        """If sector build raises, technical features should still be returned."""
        mock_sec_eng = MagicMock()
        mock_sec_eng.build.side_effect = RuntimeError("ETF fetch failed")

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker="AAPL",
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv)
        # Technical features must still be present
        assert "rsi_14" in df.columns
        assert "macd" in df.columns

    def test_ticker_can_be_passed_to_build_features(self, sample_ohlcv):
        """Ticker can be passed to build_features() instead of the constructor."""
        mock_sec_eng = MagicMock()
        captured_ticker = []

        def _build(ticker, stock_close, price_index):
            captured_ticker.append(ticker)
            return self._make_sector_df(price_index)

        mock_sec_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker=None,
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv, ticker="AAPL")

        assert captured_ticker == ["AAPL"]
        assert "sector_rel_ret_5d" in df.columns

    def test_feature_columns_includes_sector(self, sample_ohlcv):
        """get_feature_columns() must include sector_* and market_* columns."""
        mock_sec_eng = MagicMock()

        def _build(ticker, stock_close, price_index):
            return self._make_sector_df(price_index)

        mock_sec_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker="AAPL",
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv)
        feat_cols = eng.get_feature_columns(df)

        sector_in_features = [c for c in feat_cols if c.startswith("sector_")]
        assert len(sector_in_features) > 0

    def test_split_X_y_includes_sector_in_X(self, sample_ohlcv):
        """X matrix should contain sector features when built with them."""
        mock_sec_eng = MagicMock()

        def _build(ticker, stock_close, price_index):
            return self._make_sector_df(price_index)

        mock_sec_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker="AAPL",
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv)
        X, y = eng.split_X_y(df, horizon="1d")

        assert "sector_rel_ret_5d" in X.columns
        # Target must not be in X
        assert "target" not in X.columns
        assert "target_1d" not in X.columns

    def test_no_inf_in_merged_output(self, sample_ohlcv):
        """Merged feature matrix must not contain inf values."""
        mock_sec_eng = MagicMock()

        def _build(ticker, stock_close, price_index):
            df = self._make_sector_df(price_index)
            # Inject some NaN to verify imputation works
            df.loc[df.index[10:20], "sector_rel_ret_5d"] = np.nan
            return df

        mock_sec_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_sector_correlation=True,
            ticker="AAPL",
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv)
        feat_cols = eng.get_feature_columns(df)
        assert not np.isinf(df[feat_cols].values).any()

    def test_combined_fundamentals_and_sector_correlation(self, sample_ohlcv):
        """Verify both Improvement 4 and Improvement 1 can be enabled together."""
        mock_fund_eng = MagicMock()
        mock_sec_eng = MagicMock()

        def _build_fund(ticker, price_index):
            return pd.DataFrame(
                {"fund_trailing_pe": 28.5, "fund_profit_margin": 0.25},
                index=price_index,
            )

        def _build_sec(ticker, stock_close, price_index):
            return self._make_sector_df(price_index)

        mock_fund_eng.build.side_effect = _build_fund
        mock_sec_eng.build.side_effect = _build_sec

        eng = FeatureEngineer(
            include_fundamentals=True,
            include_sector_correlation=True,
            ticker="AAPL",
            fundamental_engineer=mock_fund_eng,
            sector_correlation_engineer=mock_sec_eng,
        )
        df = eng.build_features(sample_ohlcv)

        # Both sets of features must be present
        assert "fund_trailing_pe" in df.columns
        assert "sector_rel_ret_5d" in df.columns
        assert "rsi_14" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Feature name catalogue
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureNameCatalogue:
    def test_all_names_are_strings(self):
        for name in _SECTOR_CORRELATION_FEATURE_NAMES:
            assert isinstance(name, str) and len(name) > 0

    def test_no_duplicate_names(self):
        assert len(_SECTOR_CORRELATION_FEATURE_NAMES) == len(
            set(_SECTOR_CORRELATION_FEATURE_NAMES)
        )

    def test_prefixes_are_correct(self):
        for name in _SECTOR_CORRELATION_FEATURE_NAMES:
            assert name.startswith("sector_") or name.startswith("market_"), (
                f"Feature {name!r} must start with 'sector_' or 'market_'"
            )
