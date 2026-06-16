"""
app/ml/sector_correlation_features.py
=======================================
Sector and market correlation feature engineering for FinSight AI — Improvement 1.

Feature categories
------------------
1. Relative return features    — stock return minus sector ETF / SPY return over
                                 multiple windows (5d, 21d, 63d)
2. Rolling beta                — 60-day rolling OLS beta vs SPY (cov/var)
3. Correlation coefficient     — 20-day rolling Pearson r vs sector ETF
4. Sector momentum indicators  — RSI-14, 5d and 21d momentum on sector ETF
5. Sector trend regime         — SMA-50/200 cross, regime label, close vs SMA-50
6. Market breadth proxy        — SPY vs its own SMA-200 (deviation, binary, 21d mom)

Design principles
-----------------
* Fail-safe:   every fetcher is wrapped; failures return NaN-filled Series so
               the caller never crashes — it simply has no context for that day.
* Caching:     sector ETF data is fetched through the existing
               ``ingest_market_data()`` call, which provides full parquet
               caching at zero extra HTTP cost.
* Alignment:   stock and ETF DataFrames are aligned via .reindex() + ffill
               before any arithmetic.
* Stateless:   SectorCorrelationFeatureEngineer holds no mutable state and
               produces a plain pd.DataFrame that the caller merges.
* Testable:    _fetch_etf_close and _fetch_yfinance_info_sector are isolated
               functions that tests can patch at fine granularity.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger("sector_correlation_features")


# ── GICS sector -> SPDR Select Sector ETF ────────────────────────────────────
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Healthcare": "XLV",  # alternate yfinance spelling
    "Financials": "XLF",
    "Financial Services": "XLF",  # alternate yfinance spelling
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

_SPY = "SPY"

# Minimum rows in price_index before we attempt to build features
_MIN_ROWS: int = 63  # one quarter -- needed for rolling beta warm-up

# TTL for the in-process sector-lookup cache (seconds)
_INFO_CACHE_TTL_S: int = 3_600 * 6  # 6 hours

# Rolling windows for relative return calculations (trading days)
_REL_RETURN_WINDOWS: tuple[int, ...] = (5, 21, 63)

# Rolling windows for beta and correlation
_BETA_WINDOW: int = 60
_CORR_WINDOW: int = 20

# In-process cache:  ticker_upper -> (fetched_at_unix, sector_str)
_SECTOR_CACHE: dict[str, tuple[float, str]] = {}

# ── Feature name catalogue (for documentation / FeatureSelector awareness) ───
# Public name — use this in production code.
# The private alias ``_SECTOR_CORRELATION_FEATURE_NAMES`` is kept below for
# backward-compatibility with test imports.
SECTOR_CORRELATION_FEATURE_NAMES: tuple[str, ...] = (
    # Relative return vs sector ETF
    "sector_rel_ret_5d",
    "sector_rel_ret_21d",
    "sector_rel_ret_63d",
    # Relative return vs broad market (SPY)
    "market_rel_ret_5d",
    "market_rel_ret_21d",
    "market_rel_ret_63d",
    # Rolling beta vs SPY
    "market_beta",
    # Rolling Pearson correlation vs sector ETF
    "sector_corr_20d",
    # Sector ETF momentum
    "sector_rsi_14",
    "sector_momentum_5d",
    "sector_momentum_21d",
    # Sector ETF trend regime
    "sector_sma50_200_cross",
    "sector_trend_regime",
    "sector_close_vs_sma50",
    # Market breadth proxy (SPY-derived)
    "market_spy_vs_sma200",
    "market_above_sma200",
    "market_spy_momentum_21d",
)

# Private alias — referenced by test_sector_correlation_features.py imports.
# Both names point to the same object; neither copies the tuple.
_SECTOR_CORRELATION_FEATURE_NAMES = SECTOR_CORRELATION_FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Sector resolution helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_yfinance_info_sector(ticker: str) -> str:
    """
    Fetch the sector string from yfinance Ticker.info.

    Isolated into its own function so unit tests can patch it without
    touching the entire yfinance module.  Returns "" on any failure.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        return info.get("sector", "") or ""
    except Exception as exc:
        logger.debug("[%s] yfinance sector lookup failed: %s", ticker, exc)
        return ""


def _resolve_sector_etf(ticker: str) -> tuple[str, str]:
    """
    Return (sector_name, etf_ticker) for *ticker*.

    Lookup order
    ------------
    1. In-process TTL cache.
    2. _fetch_yfinance_info_sector() -> maps via SECTOR_ETF_MAP.
    3. Heuristic fallback -> ("", "SPY").

    The SPY fallback is safe: the model still gets broad-market context;
    it just lacks sector-specific context until the next successful lookup.
    """
    ticker_upper = ticker.upper().strip()
    now = time.time()

    cached = _SECTOR_CACHE.get(ticker_upper)
    if cached and (now - cached[0]) < _INFO_CACHE_TTL_S:
        sector = cached[1]
        etf = SECTOR_ETF_MAP.get(sector, _SPY)
        logger.debug("[%s] Sector from cache: %r -> %s", ticker_upper, sector, etf)
        return sector, etf

    sector = _fetch_yfinance_info_sector(ticker_upper)
    _SECTOR_CACHE[ticker_upper] = (now, sector)
    etf = SECTOR_ETF_MAP.get(sector, _SPY)

    logger.info(
        "[%s] Sector resolved: sector=%r etf=%s source=%s",
        ticker_upper,
        sector,
        etf,
        "yfinance" if sector else "heuristic_spy",
    )
    return sector, etf


# ─────────────────────────────────────────────────────────────────────────────
# ETF price fetcher  (reuses existing ingest infrastructure)
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_etf_close(
    etf_ticker: str,
    price_index: pd.DatetimeIndex,
    period_years: int = 5,
) -> pd.Series:
    """
    Fetch the daily Close series for *etf_ticker* aligned to *price_index*.

    Uses ingest_market_data() so the full parquet caching layer is reused;
    no extra HTTP calls are made if the ETF was already fetched today.

    Returns a Series indexed to *price_index*, NaN-filled on any failure.
    """
    try:
        from app.ml.data_ingestion import ingest_market_data

        df = ingest_market_data(
            etf_ticker,
            period_years=period_years,
            min_rows=_MIN_ROWS,
        )
        close: pd.Series = df["Close"].squeeze()
        close.index = pd.to_datetime(close.index)
        return close.reindex(price_index, method="ffill")
    except Exception as exc:
        logger.warning("[%s] ETF close fetch failed: %s", etf_ticker, exc)
        return pd.Series(np.nan, index=price_index, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Primitive feature builders
# ─────────────────────────────────────────────────────────────────────────────


def _log_returns(close: pd.Series) -> pd.Series:
    """Log returns, with +/-inf replaced by NaN."""
    ret = np.log(close / close.shift(1))
    return ret.replace([np.inf, -np.inf], np.nan)


def _relative_return_features(
    stock_close: pd.Series,
    etf_close: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    """
    Compute stock_pct_change(w) - etf_pct_change(w) for each window.

    Positive -> stock outperforming.  Negative -> stock underperforming.
    Columns: {prefix}_rel_ret_5d, _21d, _63d.
    """
    rows: dict[str, pd.Series] = {}
    for w in _REL_RETURN_WINDOWS:
        stock_ret = stock_close.pct_change(w)
        etf_ret = etf_close.pct_change(w)
        rows[f"{prefix}_rel_ret_{w}d"] = stock_ret - etf_ret
    return pd.DataFrame(rows, index=stock_close.index)


def _rolling_beta(
    stock_rets: pd.Series,
    market_rets: pd.Series,
    window: int = _BETA_WINDOW,
) -> pd.Series:
    """
    Rolling OLS beta = Cov(stock, market) / Var(market) over *window* days.

    Beta > 1 -> amplifies market moves.
    Beta < 1 -> defensive.
    Computed entirely via pandas rolling ops -- O(n), no Python loop.
    """
    cov = stock_rets.rolling(window, min_periods=window).cov(market_rets)
    var = market_rets.rolling(window, min_periods=window).var().replace(0, np.nan)
    beta = cov / var
    beta.name = "market_beta"
    return beta


def _rolling_corr(
    stock_rets: pd.Series,
    etf_rets: pd.Series,
    window: int = _CORR_WINDOW,
    col_name: str = "sector_corr_20d",
) -> pd.Series:
    """
    Rolling Pearson correlation between stock and sector ETF log returns.

    High correlation (>0.7) is normal.
    A sudden drop may signal idiosyncratic news decoupling the stock.
    """
    corr = stock_rets.rolling(window, min_periods=window).corr(etf_rets)
    corr.name = col_name
    return corr


def _sector_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI of the sector ETF close price, reusing production compute_rsi."""
    from app.ml.feature_engineering import compute_rsi

    return compute_rsi(close, period)


def _sector_momentum(close: pd.Series, window: int) -> pd.Series:
    """Rate-of-change momentum on the sector ETF."""
    return close.pct_change(window)


def _sector_trend_regime(close: pd.Series) -> pd.DataFrame:
    """
    Classify the sector ETF into a trend regime via SMA-50 vs SMA-200.

    Columns
    -------
    sector_sma50_200_cross : 1 if SMA-50 > SMA-200 (golden cross), else 0
    sector_trend_regime    : +1 uptrend, -1 downtrend, 0 sideways
                             (thresholds: +-2% diff between the two SMAs)
    sector_close_vs_sma50  : % deviation of close from SMA-50
    """
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()

    cross = (sma50 > sma200).astype(int)
    diff_pct = (sma50 - sma200) / sma200.replace(0, np.nan)

    regime = pd.Series(0.0, index=close.index)
    regime[diff_pct > 0.02] = 1.0
    regime[diff_pct < -0.02] = -1.0

    close_vs_sma50 = close / sma50.replace(0, np.nan) - 1.0

    return pd.DataFrame(
        {
            "sector_sma50_200_cross": cross,
            "sector_trend_regime": regime,
            "sector_close_vs_sma50": close_vs_sma50,
        },
        index=close.index,
    )


def _market_breadth_proxy(spy_close: pd.Series) -> pd.DataFrame:
    """
    Approximate market breadth using SPY vs its 200-day SMA.

    True breadth (% of S&P 500 above their 200-day) requires 500 fetches.
    SPY's own SMA deviation is a strong proxy and costs nothing extra since
    SPY is already fetched for market-relative return features.

    Columns
    -------
    market_spy_vs_sma200    : (SPY / SMA200) - 1  (+ = healthy, - = bear)
    market_above_sma200     : 1 if SPY > SMA200, else 0
    market_spy_momentum_21d : 21-day SPY price momentum
    """
    sma200 = spy_close.rolling(200, min_periods=200).mean()
    vs_sma200 = spy_close / sma200.replace(0, np.nan) - 1.0
    above = (spy_close > sma200).astype(int)
    mom21 = spy_close.pct_change(21)

    return pd.DataFrame(
        {
            "market_spy_vs_sma200": vs_sma200,
            "market_above_sma200": above,
            "market_spy_momentum_21d": mom21,
        },
        index=spy_close.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


# Function aliases — used by test_sector_correlation_features.py imports.
# Both names point to the same function object (zero overhead).
_rolling_beta_feature = _rolling_beta  # noqa: E221
_rolling_correlation_feature = _rolling_corr  # noqa: E221


class SectorCorrelationFeatureEngineer:
    """
    Builds sector and market correlation features aligned to a daily price index.

    Parameters
    ----------
    period_years : int
        Years of history to fetch for ETF data (should match stock data period).
    include_market : bool
        Whether to fetch SPY data for broad-market features.  Default True.

    Usage
    -----
    ::

        eng = SectorCorrelationFeatureEngineer()
        corr_df = eng.build(
            ticker="AAPL",
            stock_close=raw_df["Close"],
            price_index=feature_df.index,
        )
        merged = feature_df.join(corr_df, how="left")
    """

    def __init__(
        self,
        period_years: int = 5,
        include_market: bool = True,
    ) -> None:
        self.period_years = period_years
        self.include_market = include_market

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        ticker: str,
        stock_close: pd.Series,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Build all sector and market correlation features for *ticker*.

        Parameters
        ----------
        ticker :
            Stock ticker symbol (e.g. "AAPL").
        stock_close :
            Daily Close prices for the target stock, indexed by date.
        price_index :
            The DatetimeIndex of the technical feature matrix.
            Every output column will be indexed exactly to this.

        Returns
        -------
        pd.DataFrame
            One row per date in *price_index*.  Cells are NaN when data is
            unavailable.  Rows are never dropped -- that is the caller's job.
        """
        ticker = ticker.upper().strip()

        if len(price_index) < _MIN_ROWS:
            logger.warning(
                "[%s] Price index too short (%d rows < %d) -- "
                "returning empty DataFrame.",
                ticker,
                len(price_index),
                _MIN_ROWS,
            )
            return pd.DataFrame(index=price_index)

        logger.info("[%s] Building sector correlation features", ticker)

        # ── 1. Resolve sector ETF ─────────────────────────────────────────────
        sector_name, sector_etf = _resolve_sector_etf(ticker)

        # ── 2. Fetch ETF closes (both via ingest_market_data cache) ───────────
        # period_years is passed as a keyword argument so that test mocks using
        # lambda (etf, idx, **kw) can absorb it without a positional-arg error.
        sector_close = _fetch_etf_close(
            sector_etf, price_index, period_years=self.period_years
        )
        market_close = (
            _fetch_etf_close(_SPY, price_index, period_years=self.period_years)
            if self.include_market
            else pd.Series(np.nan, index=price_index, dtype=float)
        )

        # Align stock_close to price_index
        stock_close_aln = stock_close.reindex(price_index, method="ffill")

        # ── 3. Log-return series ──────────────────────────────────────────────
        stock_rets = _log_returns(stock_close_aln)
        sector_rets = _log_returns(sector_close)
        market_rets = _log_returns(market_close)

        frames: list[pd.DataFrame] = []

        # ── 4. Relative returns vs sector ETF ─────────────────────────────────
        if not sector_close.isna().all():
            try:
                frames.append(
                    _relative_return_features(stock_close_aln, sector_close, "sector")
                )
            except Exception as exc:
                logger.warning("[%s] Sector relative return failed: %s", ticker, exc)

        # ── 5. Relative returns vs SPY ────────────────────────────────────────
        if self.include_market and not market_close.isna().all():
            try:
                frames.append(
                    _relative_return_features(stock_close_aln, market_close, "market")
                )
            except Exception as exc:
                logger.warning("[%s] Market relative return failed: %s", ticker, exc)

        # ── 6. Rolling beta vs SPY ────────────────────────────────────────────
        if self.include_market and not market_rets.isna().all():
            try:
                frames.append(_rolling_beta(stock_rets, market_rets).to_frame())
            except Exception as exc:
                logger.warning("[%s] Rolling beta failed: %s", ticker, exc)

        # ── 7. Rolling correlation vs sector ETF ──────────────────────────────
        if not sector_rets.isna().all():
            try:
                frames.append(_rolling_corr(stock_rets, sector_rets).to_frame())
            except Exception as exc:
                logger.warning("[%s] Rolling correlation failed: %s", ticker, exc)

        # ── 8. Sector ETF momentum indicators ────────────────────────────────
        if not sector_close.isna().all():
            try:
                frames.append(
                    pd.DataFrame(
                        {
                            "sector_rsi_14": _sector_rsi(sector_close),
                            "sector_momentum_5d": _sector_momentum(sector_close, 5),
                            "sector_momentum_21d": _sector_momentum(sector_close, 21),
                        },
                        index=price_index,
                    )
                )
            except Exception as exc:
                logger.warning("[%s] Sector momentum failed: %s", ticker, exc)

        # ── 9. Sector ETF trend regime ────────────────────────────────────────
        if not sector_close.isna().all():
            try:
                frames.append(_sector_trend_regime(sector_close))
            except Exception as exc:
                logger.warning("[%s] Sector trend regime failed: %s", ticker, exc)

        # ── 10. Market breadth proxy (SPY-derived) ────────────────────────────
        if self.include_market and not market_close.isna().all():
            try:
                frames.append(_market_breadth_proxy(market_close))
            except Exception as exc:
                logger.warning("[%s] Market breadth proxy failed: %s", ticker, exc)

        if not frames:
            logger.warning(
                "[%s] All sector correlation builders failed -- empty DataFrame.",
                ticker,
            )
            return pd.DataFrame(index=price_index)

        result = pd.concat(frames, axis=1).reindex(price_index)

        logger.info(
            "[%s] Sector correlation matrix: %d features (%d with data) x %d rows "
            "[sector=%r etf=%s]",
            ticker,
            result.shape[1],
            result.notna().any(axis=0).sum(),
            len(result),
            sector_name or "unknown",
            sector_etf,
        )
        return result

    def get_feature_names(self) -> list[str]:
        """Return the canonical list of sector correlation feature names."""
        return list(SECTOR_CORRELATION_FEATURE_NAMES)
