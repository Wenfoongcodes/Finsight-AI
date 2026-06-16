"""
Fallback chain (most to least specific)
----------------------------------------
1. yfinance ``info["shortName"]``                 e.g. "Apple Inc."
2. yfinance ``info["longName"]``                  e.g. "Apple Inc."
3. Strip well-known suffixes from the raw ticker  e.g. "BTC-USD" → "BTC"
4. Raw ticker as-is

``quoteType`` fallback:
1. yfinance ``info["quoteType"]``                 "CRYPTOCURRENCY", "EQUITY", …
2. Heuristic: ``-USD`` / ``-EUR`` / ``-GBP`` suffix → "CRYPTOCURRENCY"
3. ``"EQUITY"`` (safest default)

Thread safety
-------------
The in-process LRU cache is protected by a module-level lock so concurrent
prediction pipeline requests for the same ticker do not race on the yfinance
HTTP call.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from app.core.logging_config import get_logger

logger = get_logger("ticker_resolver")

# ── Suffixes stripped when falling back to ticker-derived names ───────────────
# These are yfinance / exchange conventions that are never used by publishers.
_STRIP_SUFFIXES: tuple[str, ...] = (
    "-USD",
    "-USDT",
    "-EUR",
    "-GBP",
    "-BTC",
    "-PERP",
    "-SWAP",
    ".NS",
    ".BO",
    ".L",
    ".TO",
    ".AX",
)

# ── Currency-pair pattern: anything ending in a 3-letter fiat/crypto suffix ──
_CRYPTO_PAIR_RE = re.compile(
    r"^[A-Z0-9]{2,10}-(?:USD|USDT|USDC|EUR|GBP|BTC|ETH|BNB|SOL)$",
    re.IGNORECASE,
)

# ── Known quoteType → asset class normalization ───────────────────────────────
_QUOTE_TYPE_MAP: dict[str, str] = {
    "CRYPTOCURRENCY": "CRYPTOCURRENCY",
    "EQUITY": "EQUITY",
    "ETF": "ETF",
    "MUTUALFUND": "MUTUALFUND",
    "INDEX": "INDEX",
    "FUTURE": "FUTURE",
    "CURRENCY": "CURRENCY",
    "OPTION": "OPTION",
}

_resolver_lock = threading.Lock()


@dataclass(frozen=True)
class AssetProfile:
    """
    Canonical identity for a ticker symbol.

    Attributes
    ----------
    ticker      : Original ticker as passed by the caller (e.g. "BTC-USD").
    display_name: Market-facing name used in news queries (e.g. "Bitcoin").
    asset_class : Normalised asset class string (e.g. "CRYPTOCURRENCY").
    source      : Where the display_name came from ("yfinance", "heuristic").
    """

    ticker: str
    display_name: str
    asset_class: str
    source: str = "yfinance"

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "CRYPTOCURRENCY"

    @property
    def is_equity(self) -> bool:
        return self.asset_class in ("EQUITY", "ETF", "MUTUALFUND")

    @property
    def is_macro(self) -> bool:
        """True for indices, futures, and forex — excludes company-centric queries."""
        return self.asset_class in ("INDEX", "FUTURE", "CURRENCY")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _strip_ticker_suffix(ticker: str) -> str:
    """Remove exchange/pair suffixes to get a bare symbol (e.g. BTC-USD → BTC)."""
    upper = ticker.upper()
    for suffix in _STRIP_SUFFIXES:
        if upper.endswith(suffix.upper()):
            return upper[: -len(suffix)]
    # Also handle dotted suffixes: BRK.B → BRK
    return upper.split(".")[0]


def _heuristic_asset_class(ticker: str) -> str:
    """Return a best-guess asset class from the ticker string alone."""
    if _CRYPTO_PAIR_RE.match(ticker):
        return "CRYPTOCURRENCY"
    return "EQUITY"


def _fetch_yfinance_info(ticker: str) -> dict:
    """
    Call yfinance.Ticker(ticker).info with a short timeout.

    Returns an empty dict on any failure — callers must handle the missing-key
    case via the fallback chain.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception as exc:
        logger.debug("yfinance info fetch failed for %s: %s", ticker, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public resolver
# ─────────────────────────────────────────────────────────────────────────────


class TickerResolver:
    """
    Resolves a yfinance ticker to a canonical ``AssetProfile``.

    Results are cached in a module-level LRU cache so repeat calls
    within the same process lifetime are free.

    Usage
    -----
    ::

        resolver = TickerResolver()
        profile = resolver.resolve("BTC-USD")
        # AssetProfile(ticker='BTC-USD', display_name='Bitcoin',
        #              asset_class='CRYPTOCURRENCY', source='yfinance')

        profile = resolver.resolve("AAPL")
        # AssetProfile(ticker='AAPL', display_name='Apple Inc.',
        #              asset_class='EQUITY', source='yfinance')
    """

    def resolve(self, ticker: str) -> AssetProfile:
        """
        Resolve *ticker* to an ``AssetProfile``.

        The result is thread-safe and process-cached via ``_resolve_cached``.
        """
        with _resolver_lock:
            return _resolve_cached(ticker.upper().strip())


@lru_cache(maxsize=512)
def _resolve_cached(ticker: str) -> AssetProfile:
    """
    LRU-cached resolution — one yfinance call per unique ticker per process.

    Must only be called inside ``_resolver_lock`` to prevent concurrent
    cache-miss races on the same ticker.
    """
    info = _fetch_yfinance_info(ticker)

    # ── Display name ──────────────────────────────────────────────────────────
    display_name: Optional[str] = info.get("shortName") or info.get("longName")
    source = "yfinance"

    if not display_name or not display_name.strip():
        # Fallback: strip exchange suffixes and use the bare symbol.
        display_name = _strip_ticker_suffix(ticker)
        source = "heuristic"
        logger.debug(
            "TickerResolver: no yfinance name for %s — using heuristic %r",
            ticker,
            display_name,
        )
    else:
        display_name = display_name.strip()

    # ── Asset class ───────────────────────────────────────────────────────────
    raw_type: str = info.get("quoteType", "").upper().strip()
    asset_class = _QUOTE_TYPE_MAP.get(raw_type) or _heuristic_asset_class(ticker)

    logger.info(
        "TickerResolver: %s → display_name=%r asset_class=%s source=%s",
        ticker,
        display_name,
        asset_class,
        source,
    )

    return AssetProfile(
        ticker=ticker,
        display_name=display_name,
        asset_class=asset_class,
        source=source,
    )


# Module-level default instance — importers can use this directly.
_default_resolver = TickerResolver()


def resolve_ticker(ticker: str) -> AssetProfile:
    """Module-level convenience function wrapping the default resolver."""
    return _default_resolver.resolve(ticker)
