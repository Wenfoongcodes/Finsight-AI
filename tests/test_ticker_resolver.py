"""
tests/test_ticker_resolver.py
==============================
Unit tests for TickerResolver and the asset-class-aware query builder
in NewsRetriever.

All yfinance.Ticker.info calls are mocked so no network access is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.ticker_resolver import (
    AssetProfile,
    TickerResolver,
    _strip_ticker_suffix,
    _heuristic_asset_class,
    _resolve_cached,
    resolve_ticker,
)
from app.services.news_intelligence import _build_queries


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_profile(
    ticker: str,
    display_name: str,
    asset_class: str,
    source: str = "yfinance",
) -> AssetProfile:
    return AssetProfile(
        ticker=ticker,
        display_name=display_name,
        asset_class=asset_class,
        source=source,
    )


def _mock_info(short_name: str = "", long_name: str = "", quote_type: str = "EQUITY"):
    """Return a minimal yfinance Ticker.info dict."""
    return {
        "shortName": short_name,
        "longName": long_name,
        "quoteType": quote_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# _strip_ticker_suffix
# ─────────────────────────────────────────────────────────────────────────────


class TestStripTickerSuffix:
    def test_btc_usd(self):
        assert _strip_ticker_suffix("BTC-USD") == "BTC"

    def test_eth_usd(self):
        assert _strip_ticker_suffix("ETH-USD") == "ETH"

    def test_sol_usdt(self):
        assert _strip_ticker_suffix("SOL-USDT") == "SOL"

    def test_ns_suffix(self):
        assert _strip_ticker_suffix("RELIANCE.NS") == "RELIANCE"

    def test_plain_equity_unchanged(self):
        assert _strip_ticker_suffix("AAPL") == "AAPL"

    def test_lowercase_input(self):
        assert _strip_ticker_suffix("btc-usd") == "BTC"


# ─────────────────────────────────────────────────────────────────────────────
# _heuristic_asset_class
# ─────────────────────────────────────────────────────────────────────────────


class TestHeuristicAssetClass:
    def test_btc_usd_is_crypto(self):
        assert _heuristic_asset_class("BTC-USD") == "CRYPTOCURRENCY"

    def test_eth_usdt_is_crypto(self):
        assert _heuristic_asset_class("ETH-USDT") == "CRYPTOCURRENCY"

    def test_plain_equity_is_equity(self):
        assert _heuristic_asset_class("AAPL") == "EQUITY"

    def test_spy_is_equity(self):
        # SPY has no currency suffix → falls back to EQUITY (correct for ETF)
        assert _heuristic_asset_class("SPY") == "EQUITY"


# ─────────────────────────────────────────────────────────────────────────────
# TickerResolver — yfinance path
# ─────────────────────────────────────────────────────────────────────────────


class TestTickerResolverYFinance:
    """Tests that exercise the yfinance.Ticker.info lookup path."""

    def _resolve(self, ticker: str, info: dict) -> AssetProfile:
        """Helper that patches yfinance and clears the LRU cache."""
        _resolve_cached.cache_clear()
        with patch(
            "app.services.ticker_resolver._fetch_yfinance_info", return_value=info
        ):
            return resolve_ticker(ticker)

    def test_btc_usd_resolves_to_bitcoin(self):
        profile = self._resolve(
            "BTC-USD",
            _mock_info("Bitcoin USD", "Bitcoin", "CRYPTOCURRENCY"),
        )
        assert profile.display_name == "Bitcoin USD"
        assert profile.asset_class == "CRYPTOCURRENCY"
        assert profile.source == "yfinance"
        assert profile.is_crypto

    def test_eth_usd_resolves_to_ethereum(self):
        profile = self._resolve(
            "ETH-USD",
            _mock_info("Ethereum USD", "Ethereum", "CRYPTOCURRENCY"),
        )
        assert profile.display_name == "Ethereum USD"
        assert profile.is_crypto

    def test_aapl_resolves_to_apple(self):
        profile = self._resolve(
            "AAPL",
            _mock_info("Apple Inc.", "Apple Inc.", "EQUITY"),
        )
        assert profile.display_name == "Apple Inc."
        assert profile.asset_class == "EQUITY"
        assert profile.is_equity

    def test_spy_resolves_as_etf(self):
        profile = self._resolve(
            "SPY",
            _mock_info("SPDR S&P 500 ETF Trust", "", "ETF"),
        )
        assert profile.display_name == "SPDR S&P 500 ETF Trust"
        assert profile.asset_class == "ETF"
        assert profile.is_equity  # ETF counts as equity-like

    def test_prefers_short_name_over_long_name(self):
        profile = self._resolve(
            "MSFT",
            _mock_info("Microsoft", "Microsoft Corporation", "EQUITY"),
        )
        assert profile.display_name == "Microsoft"

    def test_falls_back_to_long_name_when_short_absent(self):
        profile = self._resolve(
            "MSFT",
            _mock_info("", "Microsoft Corporation", "EQUITY"),
        )
        assert profile.display_name == "Microsoft Corporation"

    def test_heuristic_fallback_when_no_name(self):
        """When yfinance returns no name, strip the suffix and use the bare symbol."""
        profile = self._resolve(
            "BTC-USD",
            _mock_info("", "", "CRYPTOCURRENCY"),
        )
        assert profile.display_name == "BTC"
        assert profile.source == "heuristic"

    def test_heuristic_fallback_on_yfinance_failure(self):
        """When yfinance raises, the resolver must not propagate the exception."""
        _resolve_cached.cache_clear()
        with patch(
            "app.services.ticker_resolver._fetch_yfinance_info",
            side_effect=Exception("network error"),
        ):
            profile = resolve_ticker("BTC-USD")
        # Heuristic fallback: bare symbol
        assert profile.display_name == "BTC"
        assert profile.asset_class == "CRYPTOCURRENCY"
        assert profile.source == "heuristic"

    def test_unknown_quote_type_falls_back_to_heuristic(self):
        profile = self._resolve(
            "BTC-USD",
            _mock_info("Bitcoin", "", "UNKNOWN_TYPE"),
        )
        # quoteType not in map → heuristic: BTC-USD matches crypto pattern
        assert profile.asset_class == "CRYPTOCURRENCY"

    def test_result_is_cached(self):
        """Second call must not re-invoke _fetch_yfinance_info."""
        _resolve_cached.cache_clear()
        with patch(
            "app.services.ticker_resolver._fetch_yfinance_info",
            return_value=_mock_info("Apple Inc.", "", "EQUITY"),
        ) as mock_fetch:
            resolve_ticker("AAPL")
            resolve_ticker("AAPL")
        assert mock_fetch.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# _build_queries — query vocabulary correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildQueries:
    """
    Verifies that the right vocabulary is used per asset class and that
    the raw yfinance ticker (e.g. ``BTC-USD``) never appears in queries
    when a canonical name is available.
    """

    def test_crypto_uses_cryptocurrency_vocabulary(self):
        profile = _make_profile("BTC-USD", "Bitcoin", "CRYPTOCURRENCY")
        queries = _build_queries(profile)
        assert len(queries) == 3
        combined = " ".join(queries).lower()
        assert "cryptocurrency" in combined or "crypto" in combined or "blockchain" in combined
        # The raw ticker suffix must not appear in queries
        assert "btc-usd" not in combined

    def test_crypto_uses_display_name_not_raw_ticker(self):
        profile = _make_profile("ETH-USD", "Ethereum", "CRYPTOCURRENCY")
        queries = _build_queries(profile)
        combined = " ".join(queries)
        assert "Ethereum" in combined
        assert "ETH-USD" not in combined

    def test_equity_uses_earnings_vocabulary(self):
        profile = _make_profile("AAPL", "Apple Inc.", "EQUITY")
        queries = _build_queries(profile)
        combined = " ".join(queries).lower()
        assert "earnings" in combined or "stock" in combined

    def test_equity_includes_ticker_for_disambiguation(self):
        """Equity queries should include the raw ticker to avoid ambiguous name matches."""
        profile = _make_profile("META", "Meta Platforms, Inc.", "EQUITY")
        queries = _build_queries(profile)
        combined = " ".join(queries)
        assert "META" in combined

    def test_etf_uses_etf_vocabulary(self):
        profile = _make_profile("SPY", "SPDR S&P 500 ETF Trust", "ETF")
        queries = _build_queries(profile)
        combined = " ".join(queries).lower()
        assert "etf" in combined

    def test_mutual_fund_uses_fund_vocabulary(self):
        profile = _make_profile("VFIAX", "Vanguard 500 Index Fund", "MUTUALFUND")
        queries = _build_queries(profile)
        combined = " ".join(queries).lower()
        assert "fund" in combined

    def test_index_uses_macro_vocabulary(self):
        profile = _make_profile("^GSPC", "S&P 500", "INDEX")
        queries = _build_queries(profile)
        combined = " ".join(queries).lower()
        assert "index" in combined or "macro" in combined or "economic" in combined

    def test_currency_uses_forex_vocabulary(self):
        profile = _make_profile("EURUSD=X", "EUR/USD", "CURRENCY")
        queries = _build_queries(profile)
        combined = " ".join(queries).lower()
        assert "forex" in combined or "currency" in combined or "exchange rate" in combined

    def test_unknown_asset_class_falls_back_to_equity(self):
        """Unknown asset classes should produce valid equity-style queries."""
        profile = _make_profile("XYZ", "XYZ Corp", "WEIRD_TYPE")
        queries = _build_queries(profile)
        assert len(queries) == 3
        combined = " ".join(queries)
        assert "XYZ Corp" in combined


# ─────────────────────────────────────────────────────────────────────────────
# AssetProfile properties
# ─────────────────────────────────────────────────────────────────────────────


class TestAssetProfileProperties:
    def test_is_crypto(self):
        p = _make_profile("BTC-USD", "Bitcoin", "CRYPTOCURRENCY")
        assert p.is_crypto
        assert not p.is_equity
        assert not p.is_macro

    def test_is_equity(self):
        p = _make_profile("AAPL", "Apple Inc.", "EQUITY")
        assert p.is_equity
        assert not p.is_crypto

    def test_etf_is_equity_like(self):
        p = _make_profile("SPY", "SPDR S&P 500 ETF", "ETF")
        assert p.is_equity

    def test_mutualfund_is_equity_like(self):
        p = _make_profile("VFIAX", "Vanguard 500", "MUTUALFUND")
        assert p.is_equity

    def test_index_is_macro(self):
        p = _make_profile("^GSPC", "S&P 500", "INDEX")
        assert p.is_macro

    def test_future_is_macro(self):
        p = _make_profile("GC=F", "Gold Futures", "FUTURE")
        assert p.is_macro