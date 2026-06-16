"""
app/ml/fundamental_features.py
================================
Fundamental and macroeconomic feature engineering for FinSight AI.

Feature categories
------------------
1. Valuation ratios        — P/E (trailing + forward), P/B, EV/EBITDA
2. Profitability metrics   — profit margin, operating margin
3. Growth rates            — revenue growth, earnings growth (QoQ)
4. Financial health        — debt/equity, current ratio, free cash flow
5. Market sentiment        — institutional ownership %, short interest ratio
6. Earnings surprise       — actual vs. consensus estimate delta
7. Macroeconomic context   — VIX, yield curve slope, sector ETF relative perf

Design principles
-----------------
* Staleness handling: fundamental data is quarterly; each feature carries
  the most recent available value via forward-fill ("last known value").
* Cross-sectional normalization: absolute value + sector-relative Z-score.
* Rate of change: QoQ delta for key metrics (acceleration signal).
* Fail-safe: every yfinance call is wrapped; missing fields return NaN
  rather than raising. The caller can drop NaN columns or impute.
* No side effects: FundamentalFeatureEngineer is stateless and produces
  a plain pd.DataFrame that the caller merges with the technical matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger("fundamental_features")

# ── Macro tickers available via yfinance ─────────────────────────────────────
_VIX_TICKER = "^VIX"
_TNX_TICKER = "^TNX"  # 10-year US Treasury yield
_IRX_TICKER = "^IRX"  # 13-week Treasury bill (proxy for 2-year)

# ── Sector ETF universe for relative-performance calculation ─────────────────
# Maps GICS sector name → liquid ETF ticker
SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
_SPY_TICKER = "SPY"  # Broad market benchmark

# ── yfinance info keys we harvest ─────────────────────────────────────────────
_INFO_KEYS: list[str] = [
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToEbitda",
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "debtToEquity",
    "currentRatio",
    "freeCashflow",
    "marketCap",
    "sharesOutstanding",
    "floatShares",
    "institutionPercentHeld",
    "shortPercentOfFloat",
    "nextEarningsDate",
    "mostRecentQuarter",
    "sector",
    "industry",
]

# Minimum number of price rows required before attempting fundamental build
_MIN_PRICE_ROWS = 20

# Maximum age (days) for a cached info dict to be considered fresh
_INFO_CACHE_TTL_S = 3600 * 6  # 6 hours


# ─────────────────────────────────────────────────────────────────────────────
# In-process cache helpers
# ─────────────────────────────────────────────────────────────────────────────

# ticker → (fetched_at_unix, info_dict)
_INFO_CACHE: dict[str, tuple[float, dict]] = {}


def _get_yf_info(ticker: str) -> dict:
    """
    Fetch yfinance Ticker.info with a simple TTL cache.

    Returns an empty dict on any failure — callers handle missing keys.
    """
    now = time.time()
    cached = _INFO_CACHE.get(ticker)
    if cached and (now - cached[0]) < _INFO_CACHE_TTL_S:
        return cached[1]

    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        _INFO_CACHE[ticker] = (now, info)
        return info
    except Exception as exc:
        logger.debug("yfinance info fetch failed for %s: %s", ticker, exc)
        return {}


def _safe_float(value) -> float:
    """Convert a value to float; return NaN on failure."""
    try:
        f = float(value)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Earnings-surprise helper
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_earnings_surprise(ticker: str) -> pd.DataFrame:
    """
    Fetch historical earnings estimates vs. actuals via yfinance.

    Returns a DataFrame with columns:
        ['date', 'actual_eps', 'estimated_eps', 'surprise_pct']
    indexed by the earnings announcement date.

    Returns an empty DataFrame when data is unavailable.
    """
    try:
        import yfinance as yf

        tkr = yf.Ticker(ticker)

        # yfinance >= 0.2.x exposes earnings_history
        hist = getattr(tkr, "earnings_history", None)
        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            # Older yfinance: try earnings_dates attribute
            hist = getattr(tkr, "earnings_dates", None)

        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            return pd.DataFrame()

        df = hist.copy()
        # Normalise column names across yfinance versions
        col_map = {}
        for c in df.columns:
            lc = c.lower().replace(" ", "_")
            if "actual" in lc:
                col_map[c] = "actual_eps"
            elif "estimate" in lc or "expected" in lc:
                col_map[c] = "estimated_eps"
            elif "surprise" in lc and "%" not in c:
                col_map[c] = "surprise_pct"
        df = df.rename(columns=col_map)

        # Ensure both actual and estimated columns exist
        if "actual_eps" not in df.columns or "estimated_eps" not in df.columns:
            return pd.DataFrame()

        df["actual_eps"] = pd.to_numeric(df["actual_eps"], errors="coerce")
        df["estimated_eps"] = pd.to_numeric(df["estimated_eps"], errors="coerce")

        if "surprise_pct" not in df.columns:
            denom = df["estimated_eps"].replace(0, np.nan).abs()
            df["surprise_pct"] = (df["actual_eps"] - df["estimated_eps"]) / denom

        df = df[["actual_eps", "estimated_eps", "surprise_pct"]].dropna(
            subset=["actual_eps", "estimated_eps"]
        )
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df.sort_index()
        return df

    except Exception as exc:
        logger.debug("Earnings surprise fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Macro data helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_macro_series(
    ticker: str,
    start: str,
    end: str,
    column: str = "Close",
) -> pd.Series:
    """
    Download a single daily price/yield series from yfinance.

    Returns an empty Series on failure (caller forward-fills or imputes).
    """
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        series = df[column].squeeze()
        series.index = pd.to_datetime(series.index)
        return series.sort_index()
    except Exception as exc:
        logger.debug("Macro series fetch failed (%s): %s", ticker, exc)
        return pd.Series(dtype=float)


def _fetch_sector_etf(ticker: str, start: str, end: str) -> pd.Series:
    """Return the daily Close price series for a sector ETF."""
    return _fetch_macro_series(ticker, start, end, column="Close")


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FundamentalSnapshot:
    """
    Point-in-time fundamental snapshot for a single ticker.

    All values represent the most recently available quarterly data.
    NaN indicates the metric is unavailable for this ticker.
    """

    ticker: str
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sector: str = ""
    industry: str = ""

    # ── Valuation ─────────────────────────────────────────────────────────────
    trailing_pe: float = np.nan
    forward_pe: float = np.nan
    price_to_book: float = np.nan
    ev_to_ebitda: float = np.nan
    pe_premium: float = np.nan  # forward_pe / trailing_pe − 1

    # ── Profitability ─────────────────────────────────────────────────────────
    profit_margin: float = np.nan
    operating_margin: float = np.nan

    # ── Growth ────────────────────────────────────────────────────────────────
    revenue_growth: float = np.nan
    earnings_growth: float = np.nan

    # ── Financial health ──────────────────────────────────────────────────────
    debt_to_equity: float = np.nan
    current_ratio: float = np.nan
    free_cash_flow: float = np.nan  # raw, in millions
    fcf_yield: float = np.nan  # FCF / market cap

    # ── Market sentiment ──────────────────────────────────────────────────────
    institutional_ownership_pct: float = np.nan
    short_interest_ratio: float = np.nan

    # ── Earnings surprise (most recent quarter) ───────────────────────────────
    earnings_surprise_pct: float = np.nan
    prev_earnings_surprise_pct: float = np.nan  # QoQ change signal


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class FundamentalFeatureEngineer:
    """
    Builds fundamental and macroeconomic features aligned to a daily price index.

    Parameters
    ----------
    include_macro : bool
        Whether to fetch and include VIX, yield curve, and sector ETF features.
        Adds ~3 yfinance HTTP calls. Default True.
    include_earnings_surprise : bool
        Whether to fetch earnings history for surprise features. Default True.
    sector_etf_lookback_days : int
        Lookback window for sector ETF relative performance calculation.
    """

    def __init__(
        self,
        include_macro: bool = True,
        include_earnings_surprise: bool = True,
        sector_etf_lookback_days: int = 21,
    ) -> None:
        self.include_macro = include_macro
        self.include_earnings_surprise = include_earnings_surprise
        self.sector_etf_lookback_days = sector_etf_lookback_days

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        ticker: str,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Build a fundamental feature DataFrame aligned to *price_index*.

        Parameters
        ----------
        ticker :
            Stock ticker symbol (e.g. "AAPL").
        price_index :
            The DatetimeIndex of the technical feature matrix. Fundamental
            features will be reindexed and forward-filled to this index.

        Returns
        -------
        pd.DataFrame
            One row per date in *price_index*, columns are fundamental
            features. Cells are NaN when a feature is unavailable.
            Caller is responsible for merging with the technical matrix.
        """
        ticker = ticker.upper().strip()
        if len(price_index) < _MIN_PRICE_ROWS:
            logger.warning(
                "[%s] Price index too short (%d rows) for fundamental build — "
                "returning empty DataFrame.",
                ticker,
                len(price_index),
            )
            return pd.DataFrame(index=price_index)

        start = str(price_index.min().date())
        end = str(price_index.max().date())

        logger.info("[%s] Building fundamental features (%s → %s)", ticker, start, end)

        frames: list[pd.DataFrame] = []

        # ── 1. Static fundamentals (from yfinance info dict) ──────────────────
        snapshot = self._fetch_snapshot(ticker)
        static_df = self._snapshot_to_daily(snapshot, price_index)
        frames.append(static_df)

        # ── 2. Earnings surprise (quarterly, aligned by announcement date) ────
        if self.include_earnings_surprise:
            try:
                surprise_df = self._build_earnings_surprise_features(
                    ticker, price_index
                )
                if not surprise_df.empty:
                    frames.append(surprise_df)
            except Exception as exc:
                logger.warning("[%s] Earnings surprise build failed: %s", ticker, exc)

        # ── 3. Macroeconomic features (VIX, yield curve, sector ETF) ─────────
        if self.include_macro:
            try:
                macro_df = self._build_macro_features(
                    ticker, snapshot.sector, price_index, start, end
                )
                if not macro_df.empty:
                    frames.append(macro_df)
            except Exception as exc:
                logger.warning("[%s] Macro feature build failed: %s", ticker, exc)

        if not frames:
            return pd.DataFrame(index=price_index)

        result = pd.concat(frames, axis=1)
        # Forward-fill fundamental/macro data to every trading day.
        # This is the "last known value" strategy: a P/E from six months ago
        # is still the best available information until the next report.
        result = result.reindex(price_index).ffill()

        logger.info(
            "[%s] Fundamental feature matrix: %d features × %d rows",
            ticker,
            result.shape[1],
            result.shape[0],
        )
        return result

    def get_feature_names(self) -> list[str]:
        """Return the list of fundamental feature column names (for documentation)."""
        return list(_FUNDAMENTAL_FEATURE_NAMES)

    # ── Internal: static snapshot ─────────────────────────────────────────────

    def _fetch_snapshot(self, ticker: str) -> FundamentalSnapshot:
        """Pull the latest fundamental snapshot from yfinance info."""
        info = _get_yf_info(ticker)
        snap = FundamentalSnapshot(ticker=ticker)

        snap.sector = info.get("sector", "") or ""
        snap.industry = info.get("industry", "") or ""

        snap.trailing_pe = _safe_float(info.get("trailingPE"))
        snap.forward_pe = _safe_float(info.get("forwardPE"))
        snap.price_to_book = _safe_float(info.get("priceToBook"))
        snap.ev_to_ebitda = _safe_float(info.get("enterpriseToEbitda"))

        # PE premium: forward vs trailing — positive means market expects growth
        if (
            np.isfinite(snap.forward_pe)
            and np.isfinite(snap.trailing_pe)
            and snap.trailing_pe != 0
        ):
            snap.pe_premium = snap.forward_pe / snap.trailing_pe - 1.0

        snap.profit_margin = _safe_float(info.get("profitMargins"))
        snap.operating_margin = _safe_float(info.get("operatingMargins"))
        snap.revenue_growth = _safe_float(info.get("revenueGrowth"))
        snap.earnings_growth = _safe_float(info.get("earningsGrowth"))
        snap.debt_to_equity = _safe_float(info.get("debtToEquity"))
        snap.current_ratio = _safe_float(info.get("currentRatio"))

        fcf_raw = _safe_float(info.get("freeCashflow"))
        snap.free_cash_flow = fcf_raw / 1e6 if np.isfinite(fcf_raw) else np.nan

        market_cap = _safe_float(info.get("marketCap"))
        if np.isfinite(fcf_raw) and np.isfinite(market_cap) and market_cap > 0:
            snap.fcf_yield = fcf_raw / market_cap

        snap.institutional_ownership_pct = _safe_float(
            info.get("institutionPercentHeld")
        )
        snap.short_interest_ratio = _safe_float(info.get("shortPercentOfFloat"))

        logger.debug(
            "[%s] Snapshot: sector=%s trailing_pe=%.2f forward_pe=%.2f "
            "profit_margin=%.3f revenue_growth=%.3f",
            ticker,
            snap.sector,
            snap.trailing_pe if np.isfinite(snap.trailing_pe) else -1,
            snap.forward_pe if np.isfinite(snap.forward_pe) else -1,
            snap.profit_margin if np.isfinite(snap.profit_margin) else -1,
            snap.revenue_growth if np.isfinite(snap.revenue_growth) else -1,
        )
        return snap

    def _snapshot_to_daily(
        self, snap: FundamentalSnapshot, price_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Convert a static snapshot to a single-row DataFrame, then reindex.

        The row is placed at the earliest date in price_index. After reindex
        + ffill, all dates will carry the same value (last-known-value fill).
        """
        row = {
            "fund_trailing_pe": snap.trailing_pe,
            "fund_forward_pe": snap.forward_pe,
            "fund_price_to_book": snap.price_to_book,
            "fund_ev_to_ebitda": snap.ev_to_ebitda,
            "fund_pe_premium": snap.pe_premium,
            "fund_profit_margin": snap.profit_margin,
            "fund_operating_margin": snap.operating_margin,
            "fund_revenue_growth": snap.revenue_growth,
            "fund_earnings_growth": snap.earnings_growth,
            "fund_debt_to_equity": snap.debt_to_equity,
            "fund_current_ratio": snap.current_ratio,
            "fund_fcf_yield": snap.fcf_yield,
            "fund_institutional_pct": snap.institutional_ownership_pct,
            "fund_short_interest_ratio": snap.short_interest_ratio,
        }
        anchor_date = price_index.min()
        df = pd.DataFrame([row], index=pd.DatetimeIndex([anchor_date]))
        return df.reindex(price_index).ffill()

    # ── Internal: earnings surprise ───────────────────────────────────────────

    def _build_earnings_surprise_features(
        self, ticker: str, price_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Build earnings-surprise features aligned to the announcement dates.

        For each trading day, the feature value is the surprise from the most
        recently completed quarter (forward-filled). A positive surprise (actual
        > estimate) is associated with post-earnings announcement drift (PEAD).
        """
        surprise_raw = _fetch_earnings_surprise(ticker)
        if surprise_raw.empty:
            return pd.DataFrame(index=price_index)

        # Sort ascending so ffill propagates correctly
        surprise_raw = surprise_raw.sort_index()

        # Align to price_index: place values on announcement dates, then ffill
        df = pd.DataFrame(index=price_index)
        df["fund_earnings_surprise_pct"] = np.nan
        df["fund_prev_earnings_surprise_pct"] = np.nan

        for dt, row in surprise_raw.iterrows():
            # Find the first price index date on or after the announcement
            future_dates = price_index[price_index >= dt]
            if future_dates.empty:
                continue
            align_dt = future_dates[0]
            df.at[align_dt, "fund_earnings_surprise_pct"] = _safe_float(
                row.get("surprise_pct", np.nan)
            )

        # Create the previous quarter's surprise (QoQ change in surprise — drift signal)
        df["fund_prev_earnings_surprise_pct"] = df["fund_earnings_surprise_pct"].shift(
            1, freq=None
        )

        df = df.ffill()
        return df

    # ── Internal: macro features ──────────────────────────────────────────────

    def _build_macro_features(
        self,
        ticker: str,
        sector: str,
        price_index: pd.DatetimeIndex,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Build daily macroeconomic features:
        1. VIX level and 20-day z-score
        2. Yield curve slope (10yr − 2yr Treasury)
        3. Sector ETF performance relative to SPY
        """
        df = pd.DataFrame(index=price_index)

        # ── VIX ───────────────────────────────────────────────────────────────
        vix = _fetch_macro_series(_VIX_TICKER, start, end)
        if not vix.empty:
            vix_aligned = vix.reindex(price_index, method="ffill")
            df["macro_vix"] = vix_aligned
            vix_roll_mean = vix_aligned.rolling(20, min_periods=5).mean()
            vix_roll_std = (
                vix_aligned.rolling(20, min_periods=5).std().replace(0, np.nan)
            )
            df["macro_vix_zscore_20"] = (vix_aligned - vix_roll_mean) / vix_roll_std
            df["macro_high_vix"] = (vix_aligned > 25).astype(int)
            logger.debug("[%s] VIX: %d data points", ticker, vix.notna().sum())

        # ── Yield curve slope ────────────────────────────────────────────────
        tnx = _fetch_macro_series(_TNX_TICKER, start, end)  # 10yr yield
        irx = _fetch_macro_series(_IRX_TICKER, start, end)  # short-term proxy

        if not tnx.empty and not irx.empty:
            tnx_a = tnx.reindex(price_index, method="ffill")
            irx_a = irx.reindex(price_index, method="ffill")
            slope = tnx_a - irx_a
            df["macro_yield_curve_slope"] = slope
            df["macro_yield_curve_inverted"] = (slope < 0).astype(int)
            logger.debug("[%s] Yield curve: %d data points", ticker, tnx.notna().sum())
        elif not tnx.empty:
            # Fall back to 10-year yield alone
            df["macro_10yr_yield"] = tnx.reindex(price_index, method="ffill")

        # ── Sector ETF relative performance ───────────────────────────────────
        sector_etf_ticker = SECTOR_ETFS.get(sector)
        if sector_etf_ticker:
            try:
                sector_prices = _fetch_sector_etf(sector_etf_ticker, start, end)
                spy_prices = _fetch_sector_etf(_SPY_TICKER, start, end)

                if not sector_prices.empty and not spy_prices.empty:
                    sector_ret = sector_prices.reindex(
                        price_index, method="ffill"
                    ).pct_change(self.sector_etf_lookback_days)
                    spy_ret = spy_prices.reindex(
                        price_index, method="ffill"
                    ).pct_change(self.sector_etf_lookback_days)
                    df["macro_sector_rel_perf"] = sector_ret - spy_ret
                    df["macro_sector_outperforming"] = (sector_ret > spy_ret).astype(
                        int
                    )
                    logger.debug(
                        "[%s] Sector ETF (%s) vs SPY: %d data points",
                        ticker,
                        sector_etf_ticker,
                        sector_prices.notna().sum(),
                    )
            except Exception as exc:
                logger.debug(
                    "[%s] Sector ETF feature failed (%s): %s",
                    ticker,
                    sector_etf_ticker,
                    exc,
                )
        else:
            logger.debug("[%s] No sector ETF mapping for sector=%r", ticker, sector)

        return df


# ── Feature name catalogue (for documentation and selector integration) ───────

_FUNDAMENTAL_FEATURE_NAMES: tuple[str, ...] = (
    # Valuation
    "fund_trailing_pe",
    "fund_forward_pe",
    "fund_price_to_book",
    "fund_ev_to_ebitda",
    "fund_pe_premium",
    # Profitability
    "fund_profit_margin",
    "fund_operating_margin",
    # Growth
    "fund_revenue_growth",
    "fund_earnings_growth",
    # Financial health
    "fund_debt_to_equity",
    "fund_current_ratio",
    "fund_fcf_yield",
    # Market sentiment
    "fund_institutional_pct",
    "fund_short_interest_ratio",
    # Earnings surprise
    "fund_earnings_surprise_pct",
    "fund_prev_earnings_surprise_pct",
    # Macro: VIX
    "macro_vix",
    "macro_vix_zscore_20",
    "macro_high_vix",
    # Macro: yield curve
    "macro_yield_curve_slope",
    "macro_yield_curve_inverted",
    "macro_10yr_yield",
    # Macro: sector
    "macro_sector_rel_perf",
    "macro_sector_outperforming",
)


# ─────────────────────────────────────────────────────────────────────────────
# Sector-relative normalisation helper
# ─────────────────────────────────────────────────────────────────────────────


def add_sector_relative_features(
    df: pd.DataFrame,
    ticker: str,
    sector_peers: list[str],
    valuation_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Augment *df* with sector-relative Z-scores for valuation columns.

    The Z-score measures how many standard deviations above or below the
    sector median the current stock's ratio sits.

    Parameters
    ----------
    df :
        DataFrame already containing fundamental features (output of
        ``FundamentalFeatureEngineer.build()``).
    ticker :
        Target ticker (used for labelling and logging).
    sector_peers :
        List of peer ticker symbols to use as the sector cross-section.
        Pass an empty list to skip sector normalisation.
    valuation_cols :
        Which fundamental columns to normalise. Defaults to the four
        primary valuation ratios.

    Returns
    -------
    pd.DataFrame
        Original *df* with additional ``{col}_sector_zscore`` columns.
        NaN is returned for tickers where sector peers cannot be fetched.
    """
    if not sector_peers:
        return df

    valuation_cols = valuation_cols or [
        "fund_trailing_pe",
        "fund_price_to_book",
        "fund_ev_to_ebitda",
        "fund_profit_margin",
    ]

    # Collect the latest snapshot value for each peer
    peer_values: dict[str, list[float]] = {col: [] for col in valuation_cols}

    for peer in sector_peers:
        if peer.upper() == ticker.upper():
            continue
        info = _get_yf_info(peer.upper())
        key_map = {
            "fund_trailing_pe": "trailingPE",
            "fund_price_to_book": "priceToBook",
            "fund_ev_to_ebitda": "enterpriseToEbitda",
            "fund_profit_margin": "profitMargins",
        }
        for col in valuation_cols:
            if col in key_map:
                v = _safe_float(info.get(key_map[col]))
                if np.isfinite(v):
                    peer_values[col].append(v)

    for col in valuation_cols:
        zscore_col = f"{col}_sector_zscore"
        peers = peer_values.get(col, [])
        if len(peers) < 2 or col not in df.columns:
            df[zscore_col] = np.nan
            continue

        sector_median = float(np.median(peers))
        sector_std = float(np.std(peers, ddof=1)) or np.nan
        if np.isfinite(sector_std) and sector_std > 0:
            df[zscore_col] = (df[col] - sector_median) / sector_std
        else:
            df[zscore_col] = np.nan

    logger.info(
        "[%s] Sector-relative Z-scores computed from %d peers",
        ticker,
        len(sector_peers),
    )
    return df
