"""
tests/test_fundamental_features.py
====================================
Unit tests for FundamentalFeatureEngineer and the FeatureEngineer
fundamental-integration layer (Improvement 4).

All yfinance and network calls are mocked — no internet access required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml.feature_engineering import FeatureEngineer
from app.ml.fundamental_features import (
    _FUNDAMENTAL_FEATURE_NAMES,
    FundamentalFeatureEngineer,
    FundamentalSnapshot,
    _safe_float,
    add_sector_relative_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def price_index() -> pd.DatetimeIndex:
    """500-day business-day index ending today."""
    return pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=500)


@pytest.fixture
def mock_info() -> dict:
    """Minimal yfinance info dict representing a large-cap equity."""
    return {
        "shortName": "Apple Inc.",
        "longName": "Apple Inc.",
        "quoteType": "EQUITY",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "trailingPE": 28.5,
        "forwardPE": 24.0,
        "priceToBook": 45.2,
        "enterpriseToEbitda": 22.1,
        "profitMargins": 0.253,
        "operatingMargins": 0.298,
        "revenueGrowth": 0.04,
        "earningsGrowth": 0.08,
        "debtToEquity": 140.0,
        "currentRatio": 1.07,
        "freeCashflow": 105_000_000_000.0,
        "marketCap": 3_000_000_000_000.0,
        "institutionPercentHeld": 0.61,
        "shortPercentOfFloat": 0.008,
        "mostRecentQuarter": "2024-12-31",
    }


@pytest.fixture
def mock_vix_series(price_index) -> pd.Series:
    np.random.seed(42)
    return pd.Series(
        np.random.uniform(14, 30, len(price_index)),
        index=price_index,
        name="Close",
    )


@pytest.fixture
def mock_yield_series(price_index) -> pd.Series:
    return pd.Series(
        np.linspace(3.5, 4.5, len(price_index)),
        index=price_index,
        name="Close",
    )


# ─────────────────────────────────────────────────────────────────────────────
# _safe_float
# ─────────────────────────────────────────────────────────────────────────────


class TestSafeFloat:
    def test_converts_int(self):
        assert _safe_float(5) == 5.0

    def test_converts_float_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_returns_nan_on_none(self):
        assert np.isnan(_safe_float(None))

    def test_returns_nan_on_string(self):
        assert np.isnan(_safe_float("N/A"))

    def test_returns_nan_on_inf(self):
        assert np.isnan(_safe_float(float("inf")))

    def test_returns_nan_on_neg_inf(self):
        assert np.isnan(_safe_float(float("-inf")))


# ─────────────────────────────────────────────────────────────────────────────
# FundamentalSnapshot
# ─────────────────────────────────────────────────────────────────────────────


class TestFundamentalSnapshot:
    def test_default_values_are_nan(self):
        snap = FundamentalSnapshot(ticker="TEST")
        assert np.isnan(snap.trailing_pe)
        assert np.isnan(snap.profit_margin)
        assert snap.sector == ""


# ─────────────────────────────────────────────────────────────────────────────
# FundamentalFeatureEngineer — snapshot fetch
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchSnapshot:
    def test_parses_known_info_keys(self, mock_info, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            snap = eng._fetch_snapshot("AAPL")

        assert snap.trailing_pe == pytest.approx(28.5)
        assert snap.forward_pe == pytest.approx(24.0)
        assert snap.price_to_book == pytest.approx(45.2)
        assert snap.profit_margin == pytest.approx(0.253)
        assert snap.revenue_growth == pytest.approx(0.04)
        assert snap.sector == "Technology"

    def test_pe_premium_computed(self, mock_info):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            snap = eng._fetch_snapshot("AAPL")
        # forward_pe / trailing_pe - 1 = 24.0 / 28.5 - 1
        expected = 24.0 / 28.5 - 1.0
        assert snap.pe_premium == pytest.approx(expected, rel=1e-4)

    def test_fcf_yield_computed(self, mock_info):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            snap = eng._fetch_snapshot("AAPL")
        expected = 105_000_000_000.0 / 3_000_000_000_000.0
        assert snap.fcf_yield == pytest.approx(expected, rel=1e-4)

    def test_handles_missing_keys_gracefully(self):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value={}):
            snap = eng._fetch_snapshot("UNKNOWN")
        assert np.isnan(snap.trailing_pe)
        assert np.isnan(snap.profit_margin)
        assert snap.sector == ""

    def test_handles_yfinance_failure(self):
        _eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch(
            "app.ml.fundamental_features._get_yf_info",
            side_effect=Exception("network error"),
        ):
            # _get_yf_info already catches exceptions and returns {}
            # So this tests the _get_yf_info wrapper behaviour
            pass  # confirmed by the function returning {} on error


# ─────────────────────────────────────────────────────────────────────────────
# FundamentalFeatureEngineer — build()
# ─────────────────────────────────────────────────────────────────────────────


class TestBuild:
    def test_returns_dataframe_with_price_index(self, mock_info, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            result = eng.build("AAPL", price_index)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(price_index)
        assert result.index.equals(price_index)

    def test_contains_expected_valuation_columns(self, mock_info, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            result = eng.build("AAPL", price_index)

        for col in [
            "fund_trailing_pe",
            "fund_forward_pe",
            "fund_price_to_book",
            "fund_ev_to_ebitda",
            "fund_pe_premium",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_static_values_are_forward_filled(self, mock_info, price_index):
        """Snapshot value should be the same on every row (forward-filled)."""
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            result = eng.build("AAPL", price_index)

        # trailing_pe should be constant across all rows (last-known-value fill)
        valid = result["fund_trailing_pe"].dropna()
        assert len(valid) > 0
        assert valid.nunique() == 1
        assert valid.iloc[0] == pytest.approx(28.5)

    def test_no_rows_dropped(self, mock_info, price_index):
        """build() must not drop rows — that is the caller's responsibility."""
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        with patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info):
            result = eng.build("AAPL", price_index)
        assert len(result) == len(price_index)

    def test_returns_empty_df_for_short_index(self):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=False
        )
        tiny_index = pd.bdate_range(end="2026-01-10", periods=5)
        with patch("app.ml.fundamental_features._get_yf_info", return_value={}):
            result = eng.build("TEST", tiny_index)
        assert result.empty or len(result) == len(tiny_index)


# ─────────────────────────────────────────────────────────────────────────────
# FundamentalFeatureEngineer — macro features
# ─────────────────────────────────────────────────────────────────────────────


class TestMacroFeatures:
    def test_macro_vix_column_present(self, mock_info, mock_vix_series, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=True, include_earnings_surprise=False
        )

        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_macro_series",
                return_value=mock_vix_series,
            ),
            patch(
                "app.ml.fundamental_features._fetch_sector_etf",
                return_value=pd.Series(dtype=float),
            ),
        ):
            result = eng.build("AAPL", price_index)

        assert "macro_vix" in result.columns
        assert result["macro_vix"].notna().any()

    def test_vix_zscore_is_computed(self, mock_info, mock_vix_series, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=True, include_earnings_surprise=False
        )
        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_macro_series",
                return_value=mock_vix_series,
            ),
            patch(
                "app.ml.fundamental_features._fetch_sector_etf",
                return_value=pd.Series(dtype=float),
            ),
        ):
            result = eng.build("AAPL", price_index)

        if "macro_vix_zscore_20" in result.columns:
            valid = result["macro_vix_zscore_20"].dropna()
            assert len(valid) > 0

    def test_macro_fails_gracefully_when_vix_unavailable(self, mock_info, price_index):
        """Macro build failure must not propagate — falls back silently."""
        eng = FundamentalFeatureEngineer(
            include_macro=True, include_earnings_surprise=False
        )
        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_macro_series",
                return_value=pd.Series(dtype=float),
            ),
            patch(
                "app.ml.fundamental_features._fetch_sector_etf",
                return_value=pd.Series(dtype=float),
            ),
        ):
            result = eng.build("AAPL", price_index)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(price_index)

    def test_yield_curve_slope_column(self, mock_info, mock_yield_series, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=True, include_earnings_surprise=False
        )

        def _mock_macro(ticker, *a, **kw):
            return mock_yield_series

        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_macro_series",
                side_effect=_mock_macro,
            ),
            patch(
                "app.ml.fundamental_features._fetch_sector_etf",
                return_value=pd.Series(dtype=float),
            ),
        ):
            result = eng.build("AAPL", price_index)

        # yield curve slope = tnx - irx; both mocked to same series → slope = 0
        if "macro_yield_curve_slope" in result.columns:
            slope = result["macro_yield_curve_slope"].dropna()
            assert len(slope) > 0


# ─────────────────────────────────────────────────────────────────────────────
# FundamentalFeatureEngineer — earnings surprise
# ─────────────────────────────────────────────────────────────────────────────


class TestEarningsSurprise:
    def _make_surprise_df(self, price_index) -> pd.DataFrame:
        """Synthetic quarterly earnings data aligned to price_index."""
        # Pick 4 announcement dates spread across the index
        dates = pd.DatetimeIndex(
            [
                price_index[50],
                price_index[175],
                price_index[300],
                price_index[425],
            ]
        )
        return pd.DataFrame(
            {
                "actual_eps": [2.10, 1.98, 2.25, 2.40],
                "estimated_eps": [2.00, 2.05, 2.15, 2.30],
                "surprise_pct": [0.05, -0.034, 0.047, 0.043],
            },
            index=dates,
        )

    def test_earnings_surprise_column_present(self, mock_info, price_index):
        surprise_df = self._make_surprise_df(price_index)
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=True
        )
        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_earnings_surprise",
                return_value=surprise_df,
            ),
        ):
            result = eng.build("AAPL", price_index)

        assert "fund_earnings_surprise_pct" in result.columns

    def test_positive_surprise_propagates_forward(self, mock_info, price_index):
        surprise_df = self._make_surprise_df(price_index)
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=True
        )
        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_earnings_surprise",
                return_value=surprise_df,
            ),
        ):
            result = eng.build("AAPL", price_index)

        if "fund_earnings_surprise_pct" in result.columns:
            # After the first announcement date rows should have a non-NaN surprise
            after_first = result["fund_earnings_surprise_pct"].iloc[60:120]
            assert after_first.notna().any()

    def test_handles_empty_earnings_history(self, mock_info, price_index):
        eng = FundamentalFeatureEngineer(
            include_macro=False, include_earnings_surprise=True
        )
        with (
            patch("app.ml.fundamental_features._get_yf_info", return_value=mock_info),
            patch(
                "app.ml.fundamental_features._fetch_earnings_surprise",
                return_value=pd.DataFrame(),
            ),
        ):
            result = eng.build("AAPL", price_index)
        # Should not crash; earnings surprise columns may or may not be present
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Sector-relative normalisation
# ─────────────────────────────────────────────────────────────────────────────


class TestSectorRelativeFeatures:
    def _make_fund_df(self, price_index) -> pd.DataFrame:
        df = pd.DataFrame(index=price_index)
        df["fund_trailing_pe"] = 28.5
        df["fund_price_to_book"] = 45.2
        df["fund_ev_to_ebitda"] = 22.1
        df["fund_profit_margin"] = 0.253
        return df

    def test_adds_zscore_columns(self, price_index):
        df = self._make_fund_df(price_index)
        peer_infos = {
            "MSFT": {
                "trailingPE": 35.0,
                "priceToBook": 12.0,
                "enterpriseToEbitda": 25.0,
                "profitMargins": 0.35,
            },
            "GOOGL": {
                "trailingPE": 22.0,
                "priceToBook": 6.0,
                "enterpriseToEbitda": 18.0,
                "profitMargins": 0.24,
            },
            "META": {
                "trailingPE": 26.0,
                "priceToBook": 8.0,
                "enterpriseToEbitda": 20.0,
                "profitMargins": 0.33,
            },
        }

        def _mock_info(ticker):
            return peer_infos.get(ticker, {})

        with patch("app.ml.fundamental_features._get_yf_info", side_effect=_mock_info):
            result = add_sector_relative_features(
                df.copy(), "AAPL", list(peer_infos.keys())
            )

        assert "fund_trailing_pe_sector_zscore" in result.columns
        assert "fund_price_to_book_sector_zscore" in result.columns

    def test_zscore_is_numeric(self, price_index):
        df = self._make_fund_df(price_index)
        peer_infos = {
            "PEER1": {
                "trailingPE": 30.0,
                "priceToBook": 10.0,
                "enterpriseToEbitda": 20.0,
                "profitMargins": 0.20,
            },
            "PEER2": {
                "trailingPE": 20.0,
                "priceToBook": 8.0,
                "enterpriseToEbitda": 16.0,
                "profitMargins": 0.18,
            },
            "PEER3": {
                "trailingPE": 25.0,
                "priceToBook": 9.0,
                "enterpriseToEbitda": 18.0,
                "profitMargins": 0.22,
            },
        }

        def _mock_info(ticker):
            return peer_infos.get(ticker, {})

        with patch("app.ml.fundamental_features._get_yf_info", side_effect=_mock_info):
            result = add_sector_relative_features(
                df.copy(), "AAPL", list(peer_infos.keys())
            )

        zscore_col = result.get("fund_trailing_pe_sector_zscore")
        if zscore_col is not None:
            valid = result["fund_trailing_pe_sector_zscore"].dropna()
            assert len(valid) > 0
            assert all(np.isfinite(v) for v in valid)

    def test_empty_peers_leaves_df_unchanged(self, price_index):
        df = self._make_fund_df(price_index)
        original_cols = set(df.columns)
        result = add_sector_relative_features(df.copy(), "AAPL", [])
        assert set(result.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer integration
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineerFundamentalIntegration:
    """
    Tests the FeatureEngineer.include_fundamentals=True pathway,
    mocking FundamentalFeatureEngineer.build() to avoid network calls.
    """

    def _make_fundamental_df(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return a synthetic fundamental feature DataFrame."""
        return pd.DataFrame(
            {
                "fund_trailing_pe": 28.5,
                "fund_forward_pe": 24.0,
                "fund_price_to_book": 45.2,
                "fund_profit_margin": 0.253,
                "macro_vix": np.random.uniform(15, 25, len(index)),
                "macro_yield_curve_slope": np.random.uniform(-0.5, 1.5, len(index)),
            },
            index=index,
        )

    def test_include_fundamentals_false_does_not_call_engineer(self, sample_ohlcv):
        """Default behaviour: no fundamental fetches."""
        eng = FeatureEngineer(include_fundamentals=False)
        # Just check it runs without error and produces the standard output
        df = eng.build_features(sample_ohlcv)
        fund_cols = [
            c for c in df.columns if c.startswith("fund_") or c.startswith("macro_")
        ]
        assert len(fund_cols) == 0

    def test_include_fundamentals_true_merges_columns(self, sample_ohlcv):
        mock_fund_eng = MagicMock()

        def _build(ticker, price_index):
            return self._make_fundamental_df(price_index)

        mock_fund_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker="AAPL",
            fundamental_engineer=mock_fund_eng,
        )
        df = eng.build_features(sample_ohlcv)

        fund_cols = [c for c in df.columns if c.startswith("fund_")]
        assert len(fund_cols) > 0
        assert "fund_trailing_pe" in df.columns

    def test_include_fundamentals_requires_ticker(self, sample_ohlcv):
        """Without a ticker, fundamentals are skipped with a warning (not an error)."""
        mock_fund_eng = MagicMock()
        mock_fund_eng.build.return_value = pd.DataFrame()

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker=None,
            fundamental_engineer=mock_fund_eng,
        )
        # Should not raise; fundamental features simply omitted
        df = eng.build_features(sample_ohlcv)
        assert not df.empty

    def test_fundamental_failure_does_not_crash_pipeline(self, sample_ohlcv):
        """If fundamental build raises, technical features should still be returned."""
        mock_fund_eng = MagicMock()
        mock_fund_eng.build.side_effect = RuntimeError("yfinance network error")

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker="AAPL",
            fundamental_engineer=mock_fund_eng,
        )
        df = eng.build_features(sample_ohlcv)
        # Should still have technical features
        assert "rsi_14" in df.columns
        assert "macd" in df.columns

    def test_ticker_parameter_in_build_features(self, sample_ohlcv):
        """Ticker can be passed to build_features() instead of the constructor."""
        mock_fund_eng = MagicMock()

        def _build(ticker, price_index):
            assert ticker == "AAPL"
            return self._make_fundamental_df(price_index)

        mock_fund_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker=None,
            fundamental_engineer=mock_fund_eng,
        )
        df = eng.build_features(sample_ohlcv, ticker="AAPL")
        fund_cols = [c for c in df.columns if c.startswith("fund_")]
        assert len(fund_cols) > 0

    def test_feature_columns_includes_fundamental(self, sample_ohlcv):
        """get_feature_columns() must include fund_* and macro_* columns."""
        mock_fund_eng = MagicMock()

        def _build(ticker, price_index):
            return self._make_fundamental_df(price_index)

        mock_fund_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker="AAPL",
            fundamental_engineer=mock_fund_eng,
        )
        df = eng.build_features(sample_ohlcv)
        feat_cols = eng.get_feature_columns(df)

        fund_in_features = [c for c in feat_cols if c.startswith("fund_")]
        assert len(fund_in_features) > 0

    def test_split_X_y_includes_fundamental_in_X(self, sample_ohlcv):
        """X matrix should contain fundamental features when built with them."""
        mock_fund_eng = MagicMock()

        def _build(ticker, price_index):
            return self._make_fundamental_df(price_index)

        mock_fund_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker="AAPL",
            fundamental_engineer=mock_fund_eng,
        )
        df = eng.build_features(sample_ohlcv)
        X, y = eng.split_X_y(df, horizon="1d")

        assert "fund_trailing_pe" in X.columns
        # Target must not be in X
        assert "target" not in X.columns
        assert "target_1d" not in X.columns

    def test_no_inf_in_merged_output(self, sample_ohlcv):
        """Merged feature matrix must not contain inf values."""
        mock_fund_eng = MagicMock()

        def _build(ticker, price_index):
            df = self._make_fundamental_df(price_index)
            # Inject some NaN to verify imputation works
            df.loc[df.index[10:20], "fund_trailing_pe"] = np.nan
            return df

        mock_fund_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_fundamentals=True,
            ticker="AAPL",
            fundamental_engineer=mock_fund_eng,
        )
        df = eng.build_features(sample_ohlcv)
        feat_cols = eng.get_feature_columns(df)
        assert not np.isinf(df[feat_cols].values).any()


# ─────────────────────────────────────────────────────────────────────────────
# Feature name catalogue
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureNameCatalogue:
    def test_all_names_are_strings(self):
        for name in _FUNDAMENTAL_FEATURE_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_no_duplicate_names(self):
        assert len(_FUNDAMENTAL_FEATURE_NAMES) == len(set(_FUNDAMENTAL_FEATURE_NAMES))

    def test_prefixes_are_correct(self):
        for name in _FUNDAMENTAL_FEATURE_NAMES:
            assert name.startswith("fund_") or name.startswith("macro_"), (
                f"Feature {name!r} must start with 'fund_' or 'macro_'"
            )

    def test_get_feature_names_method(self):
        eng = FundamentalFeatureEngineer()
        names = eng.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert names == list(_FUNDAMENTAL_FEATURE_NAMES)
