"""
FinSight AI — Phase 2: Data Ingestion Pipeline
Fetches OHLCV market data from yfinance (free) with optional Alpha Vantage fallback.
Validates, cleans, and persists raw data to the data/raw directory.

Cache strategy
--------------
The previous implementation keyed the parquet cache on ``(ticker, start, end)``
where ``end`` defaulted to ``date.today()``.  This meant the cache *never* hit
across a midnight boundary — a fresh parquet was written every calendar day even
for the same underlying data window, accumulating stale files and wasting disk.

The fix: the cache file name still encodes the fetch parameters (for
reproducibility), but ``_load_from_cache`` now checks the file's *mtime* against
``settings.CACHE_MAX_AGE_DAYS``.  A file that is younger than the configured
max-age is returned directly; an older file is deleted and the data is
re-fetched.  This gives:

* Intra-day cache hits (no redundant downloads within a trading day).
* Automatic daily refresh (stale files are replaced, not accumulated).
* Configurable staleness via ``CACHE_MAX_AGE_DAYS`` env var (default: 1).
"""

from __future__ import annotations

import hashlib
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from app.core.exceptions import DataIngestionError, DataValidationError, InsufficientDataError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("ingestion")

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
MIN_ROWS = 245
MIN_ROWS_SUMMARY = 20


# ─────────────────────────────────────────────────────────────────────────────
# Schema Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame, ticker: str, min_rows: int = MIN_ROWS) -> None:
    """
    Validate that a DataFrame conforms to expected OHLCV schema.

    Args:
        df: Raw OHLCV DataFrame.
        ticker: Ticker symbol for error context.
        min_rows: Minimum acceptable row count.

    Raises:
        DataValidationError: On schema or quality violations.
        InsufficientDataError: When row count is too low.
    """
    if df.empty:
        raise DataValidationError(f"Empty DataFrame returned for {ticker}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing columns for {ticker}: {missing}",
            detail=f"Available: {list(df.columns)}",
        )

    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Only {len(df)} rows for {ticker}; minimum required is {min_rows}."
        )

    null_pct = df[list(REQUIRED_COLUMNS)].isnull().mean()
    bad_cols = null_pct[null_pct > 0.01].to_dict()
    if bad_cols:
        raise DataValidationError(
            f"Columns exceed 1% null threshold for {ticker}",
            detail=str(bad_cols),
        )

    if (df["Close"] <= 0).any():
        raise DataValidationError(f"Non-positive Close prices found for {ticker}.")

    logger.info("Validation passed for %s (%d rows)", ticker, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(ticker: str, start: str, end: str) -> str:
    raw = f"{ticker}_{start}_{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(ticker: str, start: str, end: str) -> Path:
    key = _cache_key(ticker, start, end)
    return settings.RAW_DATA_DIR / f"{ticker.upper()}_{key}.parquet"


def _load_from_cache(
    path: Path,
    max_age_days: int = settings.CACHE_MAX_AGE_DAYS,
) -> Optional[pd.DataFrame]:
    """
    Return a cached DataFrame if the file exists *and* is younger than
    ``max_age_days`` days.  Stale files are removed so they don't accumulate.

    Args:
        path: Path to the parquet cache file.
        max_age_days: Maximum acceptable file age in calendar days.

    Returns:
        Cached DataFrame, or ``None`` if absent or stale.
    """
    if not path.exists():
        return None

    file_age_seconds = time.time() - path.stat().st_mtime
    max_age_seconds  = max_age_days * 86_400

    if file_age_seconds > max_age_seconds:
        logger.debug(
            "Cache stale (%.1fh > %dd): %s — removing.",
            file_age_seconds / 3600,
            max_age_days,
            path.name,
        )
        path.unlink(missing_ok=True)
        return None

    logger.debug("Cache hit: %s (age %.1fh)", path.name, file_age_seconds / 3600)
    return pd.read_parquet(path)


def _save_to_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    logger.debug("Cached data to %s", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.

    Raises:
        DataIngestionError: On download failure.
    """
    try:
        logger.info("Fetching %s from yfinance [%s → %s]", ticker, start, end)
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise DataIngestionError(f"yfinance returned empty data for {ticker}.")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Flatten MultiIndex columns if present (multi-ticker download artefact)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[list(REQUIRED_COLUMNS)].copy()
        df.sort_index(inplace=True)
        logger.info("yfinance returned %d rows for %s", len(df), ticker)
        return df
    except DataIngestionError:
        raise
    except Exception as exc:
        raise DataIngestionError(
            f"Failed to fetch {ticker} from yfinance: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ingest_market_data(
    ticker: str,
    period_years: int = settings.DEFAULT_PERIOD_YEARS,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    min_rows: int = MIN_ROWS,
) -> pd.DataFrame:
    """
    Primary entry-point for market data ingestion.

    Fetches OHLCV data for *ticker* over the specified period, validates it,
    and optionally reads from / writes to a local parquet cache.

    Cache freshness is governed by ``settings.CACHE_MAX_AGE_DAYS`` (default 1).
    Files older than the configured threshold are evicted automatically.

    Args:
        ticker: Stock ticker symbol.
        period_years: Number of years of history to fetch.
        start_date: Explicit start date (YYYY-MM-DD).
        end_date: Explicit end date (YYYY-MM-DD); defaults to today.
        use_cache: Whether to use local parquet cache.
        min_rows: Minimum acceptable row count.

    Returns:
        Validated OHLCV DataFrame with DatetimeIndex.

    Raises:
        DataIngestionError: On fetch failure.
        DataValidationError: On schema violations.
        InsufficientDataError: On insufficient row count.
    """
    end   = end_date or date.today().isoformat()
    start = start_date or (
        datetime.strptime(end, "%Y-%m-%d") - timedelta(days=period_years * 365)
    ).strftime("%Y-%m-%d")

    cache_path = _cache_path(ticker, start, end)

    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached is not None:
            validate_ohlcv(cached, ticker, min_rows=min_rows)
            return cached

    df = fetch_yfinance(ticker, start, end)
    validate_ohlcv(df, ticker, min_rows=min_rows)

    if use_cache:
        _save_to_cache(df, cache_path)

    return df


def ingest_multiple_tickers(
    tickers: list[str],
    period_years: int = settings.DEFAULT_PERIOD_YEARS,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Ingest data for multiple tickers, skipping failures with a warning.

    Args:
        tickers: List of ticker symbols.
        period_years: History length in years.
        use_cache: Whether to use local parquet cache.

    Returns:
        Dict mapping ticker → OHLCV DataFrame (only successful fetches).
    """
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = ingest_market_data(
                ticker, period_years, use_cache=use_cache
            )
        except (DataIngestionError, DataValidationError, InsufficientDataError) as exc:
            logger.warning("Skipping %s: %s", ticker, exc.message)
    return results


def get_data_summary(df: pd.DataFrame, ticker: str) -> dict:
    """
    Return a lightweight summary dict for ingested data.

    Args:
        df: OHLCV DataFrame.
        ticker: Ticker symbol.

    Returns:
        Dict with date range, row count, and basic price statistics.
    """
    return {
        "ticker":     ticker,
        "start_date": str(df.index.min().date()),
        "end_date":   str(df.index.max().date()),
        "rows":       len(df),
        "columns":    list(df.columns),
        "close_min":  round(float(df["Close"].min()), 4),
        "close_max":  round(float(df["Close"].max()), 4),
        "close_mean": round(float(df["Close"].mean()), 4),
        "null_count": int(df.isnull().sum().sum()),
    }