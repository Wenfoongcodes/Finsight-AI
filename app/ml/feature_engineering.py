from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from app.core.exceptions import FeatureEngineeringError, InsufficientDataError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("features")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Safety sentinel — keep False in production; set True only in scripts/notebooks ──
_HURST_ALLOWED: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Prediction horizons
# ─────────────────────────────────────────────────────────────────────────────

HORIZONS: dict[str, int] = {
    "1d": 1,
    "7d": 5,  # trading days
    "1m": 21,
    "6m": 126,
}


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Primitives (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def compute_rsi(series: pd.Series, period: int = settings.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast: int = settings.MACD_FAST,
    slow: int = settings.MACD_SLOW,
    signal: int = settings.MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, Signal line, Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_bollinger_bands(
    series: pd.Series,
    period: int = settings.BB_PERIOD,
    std_dev: float = settings.BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle (SMA), upper, lower."""
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    return middle, middle + std_dev * std, middle - std_dev * std


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = settings.ATR_PERIOD,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Garman-Klass volatility estimator."""
    hl_log = np.log(high / low.replace(0, np.nan))
    co_log = np.log(close / open_.replace(0, np.nan))
    gk = 0.5 * hl_log**2 - (2 * np.log(2) - 1) * co_log**2
    return (
        gk.rolling(period, min_periods=period)
        .mean()
        .apply(lambda x: np.sqrt(max(x, 0) * 252))
    )


def compute_hurst(series: pd.Series, lags: int = 20) -> pd.Series:
    """
    Rolling Hurst exponent (R/S method, window=lags*2).

    **Performance note:** ~2-4 seconds per 1 000 rows. Guarded by
    ``_HURST_ALLOWED`` — never call from a live API request handler.
    """
    log_ret = np.log(series / series.shift(1))

    def _hurst_single(arr: np.ndarray) -> float:
        if len(arr) < lags or np.isnan(arr).any():
            return np.nan
        lags_range = range(2, lags)
        tau = [np.std(np.subtract(arr[lag:], arr[:-lag])) for lag in lags_range]
        if any(t <= 0 for t in tau):
            return np.nan
        poly = np.polyfit(np.log(list(lags_range)), np.log(tau), 1)
        return poly[0]

    return log_ret.rolling(lags * 2, min_periods=lags * 2).apply(
        _hurst_single, raw=True
    )


def compute_amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Amihud (2002) illiquidity ratio."""
    abs_ret = close.pct_change().abs()
    dollar_vol = close * volume
    illiq = (abs_ret / dollar_vol.replace(0, np.nan)) * 1e6
    return illiq.rolling(period, min_periods=period).mean()


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Cumulative VWAP approximation using typical price × volume."""
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def compute_momentum(series: pd.Series, period: int) -> pd.Series:
    """Rate-of-change momentum."""
    return series.pct_change(periods=period)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_vol_regime(realized_vol: pd.Series, lookback: int = 252) -> pd.Series:
    """Volatility regime percentile rank over ``lookback`` bars."""
    return realized_vol.rolling(lookback, min_periods=60).rank(pct=True)


def compute_trend_regime(
    close: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """Simple trend-regime indicator: +1 (uptrend), -1 (downtrend), 0 (neutral)."""
    sma_fast = compute_sma(close, fast)
    sma_slow = compute_sma(close, slow)
    diff = (sma_fast - sma_slow) / sma_slow.replace(0, np.nan)
    regime = pd.Series(0.0, index=close.index)
    regime[diff > 0.02] = 1.0
    regime[diff < -0.02] = -1.0
    return regime


def compute_volume_imbalance(
    close: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Volume imbalance — buy pressure proxy."""
    body_pct = (close - open_) / (close + open_).replace(0, np.nan)
    imbalance = body_pct * volume
    return imbalance.rolling(period, min_periods=period).mean()


def compute_rolling_skew(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness of log-returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window, min_periods=window).skew()


def compute_rolling_kurt(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling excess kurtosis of log-returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window, min_periods=window).kurt()


def compute_momentum_persistence(
    close: pd.Series,
    short: int = 5,
    long: int = 20,
) -> pd.Series:
    """Momentum persistence: ratio of short-term to long-term momentum."""
    mom_short = compute_momentum(close, short)
    mom_long = compute_momentum(close, long)
    return mom_short / mom_long.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Selector (unchanged)
# ─────────────────────────────────────────────────────────────────────────────


class FeatureSelector:
    """
    Multi-stage feature selection pipeline.

    Stages (applied in order):
    1. Variance thresholding — remove near-constant features.
    2. Correlation filtering — remove one of each highly-correlated pair.
    3. Mutual-information ranking — keep top-N features by MI with target.

    With 100-135 features (technical + fundamental + sector correlation),
    the selector is more important than ever.
    """

    def __init__(
        self,
        variance_threshold: float = 1e-5,
        correlation_threshold: float = 0.95,
        mi_top_n: Optional[int] = 60,
        random_state: int = settings.RANDOM_SEED,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mi_top_n = mi_top_n
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        cols = list(X.columns)
        logger.info("FeatureSelector.fit: starting with %d features", len(cols))

        # Stage 1: Variance threshold
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X)
        cols = [c for c, s in zip(cols, vt.get_support()) if s]
        logger.info("After variance threshold: %d features", len(cols))

        # Stage 2: Correlation filter
        X_sub = X[cols]
        corr = X_sub.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = {
            col for col in upper.columns if any(upper[col] > self.correlation_threshold)
        }
        cols = [c for c in cols if c not in to_drop]
        logger.info(
            "After correlation filter (%.2f): %d features",
            self.correlation_threshold,
            len(cols),
        )

        # Stage 3: Mutual information
        if self.mi_top_n and len(cols) > self.mi_top_n:
            X_mi = X[cols].fillna(0)
            mi = mutual_info_classif(X_mi, y, random_state=self.random_state)
            mi_df = pd.Series(mi, index=cols).sort_values(ascending=False)
            cols = list(mi_df.head(self.mi_top_n).index)
            logger.info("After MI top-%d: %d features", self.mi_top_n, len(cols))

        self.selected_features_ = cols
        self._is_fitted = True
        logger.info("FeatureSelector fitted: %d final features", len(cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Call fit() before transform().")
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineer  (Improvement 4: fundamental integration)
#                   (Improvement 1: sector correlation integration)
#                   (Improvement 2: options market / implied volatility integration)
# ─────────────────────────────────────────────────────────────────────────────


class FeatureEngineer:
    """
    Stateful feature engineering pipeline.

    Builds a comprehensive feature matrix from raw OHLCV data.
    Optionally merges fundamental + macro features (Improvement 4), sector &
    market correlation features (Improvement 1), and options-market /
    implied-volatility features (Improvement 2).

    Parameters
    ----------
    rolling_windows : list[int]
        Window sizes for SMA, EMA, and range statistics.
    compute_hurst_exp : bool
        Whether to compute the Hurst exponent (expensive; disabled by default).
    include_fundamentals : bool
        Whether to fetch and merge fundamental + macro features (Improvement 4).
        Default False — backward-compatible with all existing call sites.
    ticker : str | None
        Ticker symbol — required when ``include_fundamentals=True``,
        ``include_sector_correlation=True``, or ``include_options=True``.
        Ignored otherwise.
    fundamental_engineer : FundamentalFeatureEngineer | None
        Pre-constructed fundamental engineer. If None and
        ``include_fundamentals=True``, one is created automatically.
    include_sector_correlation : bool
        Whether to fetch and merge sector ETF and market correlation features
        (Improvement 1).  Adds ~18 features (relative returns, rolling beta,
        sector RSI/momentum, trend regime, market breadth proxy).
        Default False — backward-compatible with all existing call sites.
        Requires internet; adds at most 2 yfinance calls per unique ticker per
        process lifetime (sector ETF + SPY), both cached via parquet after the
        first fetch.
    sector_correlation_engineer : SectorCorrelationFeatureEngineer | None
        Pre-constructed sector correlation engineer. If None and
        ``include_sector_correlation=True``, one is created automatically.
    include_options : bool
        Whether to fetch and merge options-market / implied-volatility
        features (Improvement 2): ATM IV, constant-maturity IV, IV rank,
        IV change, the IV/realized-vol spread, put/call ratios, and the
        VIX/VIX9D/VIX3M term-structure features. Default False —
        backward-compatible with all existing call sites. The snapshot-
        derived columns (``opt_*``) depend on a daily snapshot cache
        populated by ``scripts/warm_options_cache.py`` — they degrade
        gracefully to NaN (imputed away) when that cache hasn't been warmed
        for a given ticker yet. The VIX-family columns (``vix_*``) work
        immediately since they're regular historical yfinance series.
    options_engineer : OptionsFeatureEngineer | None
        Pre-constructed options engineer. If None and ``include_options=True``,
        one is created automatically.
    """

    def __init__(
        self,
        rolling_windows: Optional[list[int]] = None,
        compute_hurst_exp: bool = False,
        include_fundamentals: bool = False,
        ticker: Optional[str] = None,
        fundamental_engineer=None,  # FundamentalFeatureEngineer | None
        include_sector_correlation: bool = False,
        sector_correlation_engineer=None,  # SectorCorrelationFeatureEngineer | None
        include_options: bool = False,
        options_engineer=None,  # OptionsFeatureEngineer | None
    ) -> None:
        self.rolling_windows = rolling_windows or settings.ROLLING_WINDOWS
        self.include_fundamentals = include_fundamentals
        self.include_sector_correlation = include_sector_correlation
        self.include_options = include_options
        self.ticker = ticker.upper().strip() if ticker else None

        # Enforce the production guard on Hurst
        if compute_hurst_exp and not _HURST_ALLOWED:
            logger.warning(
                "compute_hurst_exp=True was requested but _HURST_ALLOWED is False "
                "(production guard). Hurst computation will be skipped."
            )
            self.compute_hurst_exp = False
        else:
            self.compute_hurst_exp = compute_hurst_exp

        # Lazily import to avoid mandatory dependency when fundamentals are off
        self._fundamental_engineer = fundamental_engineer
        if self.include_fundamentals and self._fundamental_engineer is None:
            from app.ml.fundamental_features import FundamentalFeatureEngineer

            self._fundamental_engineer = FundamentalFeatureEngineer()

        # Lazily import to avoid mandatory dependency when sector correlation is off
        self._sector_correlation_engineer = sector_correlation_engineer
        if (
            self.include_sector_correlation
            and self._sector_correlation_engineer is None
        ):
            from app.ml.sector_correlation_features import (
                SectorCorrelationFeatureEngineer,
            )

            self._sector_correlation_engineer = SectorCorrelationFeatureEngineer()

        # Lazily import to avoid mandatory dependency when options are off
        self._options_engineer = options_engineer
        if self.include_options and self._options_engineer is None:
            from app.ml.options_features import OptionsFeatureEngineer

            self._options_engineer = OptionsFeatureEngineer()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(
        self,
        df: pd.DataFrame,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute all features from a raw OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        ticker : str | None
            Ticker symbol. Required only when ``include_fundamentals=True``,
            ``include_sector_correlation=True``, or ``include_options=True``
            and not already set in the constructor. Constructor value takes
            precedence when both are supplied.

        Returns
        -------
        pd.DataFrame
            Feature matrix with NaN warm-up rows dropped.

        Raises
        ------
        FeatureEngineeringError : On computation failure.
        InsufficientDataError   : If result is empty after NaN drop.
        """
        try:
            feat = df.copy()
            feat = self._price_features(feat)
            feat = self._momentum_features(feat)
            feat = self._volatility_features(feat)
            feat = self._volume_features(feat)
            feat = self._moving_average_features(feat)
            feat = self._bollinger_features(feat)
            feat = self._regime_features(feat)
            feat = self._microstructure_features(feat)
            feat = self._higher_moment_features(feat)
            feat = self._candlestick_features(feat)
            feat = self._target_labels(feat)

            # ── Fundamental + macro features (Improvement 4) ──────────────────
            if self.include_fundamentals and self._fundamental_engineer is not None:
                resolved_ticker = self.ticker or (
                    ticker.upper().strip() if ticker else None
                )
                if resolved_ticker:
                    feat = self._merge_fundamental_features(feat, resolved_ticker)
                else:
                    logger.warning(
                        "include_fundamentals=True but no ticker supplied — "
                        "fundamental features skipped."
                    )

            # ── Sector & market correlation features (Improvement 1) ──────────
            if (
                self.include_sector_correlation
                and self._sector_correlation_engineer is not None
            ):
                resolved_ticker = self.ticker or (
                    ticker.upper().strip() if ticker else None
                )
                if resolved_ticker:
                    feat = self._merge_sector_correlation_features(
                        feat, resolved_ticker
                    )
                else:
                    logger.warning(
                        "include_sector_correlation=True but no ticker supplied — "
                        "sector correlation features skipped."
                    )

            # ── Options market / implied volatility features (Improvement 2) ──
            if self.include_options and self._options_engineer is not None:
                resolved_ticker = self.ticker or (
                    ticker.upper().strip() if ticker else None
                )
                if resolved_ticker:
                    feat = self._merge_options_features(feat, resolved_ticker)
                else:
                    logger.warning(
                        "include_options=True but no ticker supplied — "
                        "options features skipped."
                    )

            logger.info(
                "Feature matrix built: %d features × %d rows",
                feat.shape[1],
                feat.shape[0],
            )

            feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            feat.dropna(inplace=True)

            if feat.empty:
                raise InsufficientDataError(
                    "Feature matrix is empty after dropping NaN rows."
                )

            logger.info("After NaN drop: %d rows remain", len(feat))
            return feat

        except (InsufficientDataError, FeatureEngineeringError):
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Feature engineering failed: {exc}") from exc

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Return feature column names (excludes OHLCV + all target columns).

        Fundamental columns (``fund_*``, ``macro_*``), sector correlation
        columns (``sector_*``, ``market_*``), and options/IV columns
        (``opt_*``, ``vix_*``) are automatically included because they are
        simply present in the DataFrame and not in the exclude set.
        """
        horizon_targets = {f"target_{h}" for h in HORIZONS}
        exclude = (
            {"Open", "High", "Low", "Close", "Volume"}
            | horizon_targets
            | {settings.TARGET_COLUMN}
        )
        return [c for c in df.columns if c not in exclude]

    def split_X_y(
        self,
        df: pd.DataFrame,
        horizon: str = "1d",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split feature matrix into X and y for the specified prediction horizon.

        Parameters
        ----------
        df :
            Output of build_features().
        horizon :
            One of '1d', '7d', '1m', '6m'.

        Returns
        -------
        (X DataFrame, y Series)
        """
        target_col = f"target_{horizon}"
        if target_col not in df.columns:
            target_col = settings.TARGET_COLUMN

        feature_cols = self.get_feature_columns(df)
        feature_cols = [c for c in feature_cols if not c.startswith("target_")]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        return X, y

    # ── Fundamental merge (Improvement 4) ────────────────────────────────────

    def _merge_fundamental_features(
        self, feat: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Fetch, align, and merge fundamental + macro features into *feat*.

        The merge is a left join on the DatetimeIndex so that:
        * All technical rows are preserved.
        * Fundamental values are forward-filled (last-known-value).
        * Rows with no fundamental data yet (pre-IPO lookback) get NaN,
          which is later dropped by the ``dropna`` call in build_features.

        Fundamental NaN columns are imputed with the column median so that
        tickers with partial fundamental coverage (e.g. ETFs that have no
        P/E) do not cause entire rows to be dropped.
        """
        try:
            fund_df = self._fundamental_engineer.build(
                ticker=ticker,
                price_index=feat.index,
            )
            if fund_df.empty:
                logger.warning(
                    "[%s] Fundamental build returned empty DataFrame — skipping merge.",
                    ticker,
                )
                return feat

            # Align to the same index (should already match, but be safe)
            fund_df = fund_df.reindex(feat.index)

            # Impute fundamental NaNs with column median so partial coverage
            # doesn't eliminate trading days that otherwise have full tech data.
            for col in fund_df.columns:
                if fund_df[col].isna().all():
                    continue
                med = fund_df[col].median()
                fund_df[col] = fund_df[col].fillna(med)

            # Left-join: keep all rows from feat, add fundamental columns
            merged = feat.join(fund_df, how="left")

            n_fund = fund_df.shape[1]
            n_valid = fund_df.notna().any(axis=0).sum()
            logger.info(
                "[%s] Merged %d fundamental/macro features (%d with data)",
                ticker,
                n_fund,
                n_valid,
            )
            return merged

        except Exception as exc:
            logger.warning(
                "[%s] Fundamental feature merge failed (%s) — "
                "proceeding with technical features only.",
                ticker,
                exc,
            )
            return feat

    # ── Sector & market correlation merge (Improvement 1) ────────────────────

    def _merge_sector_correlation_features(
        self, feat: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """
        Fetch, align, and merge sector & market correlation features into *feat*.

        Uses SectorCorrelationFeatureEngineer.build() which internally calls
        ingest_market_data() for the sector ETF and SPY.  Both calls go
        through the full parquet caching layer so there is zero extra HTTP
        cost after the first call on a given trading day.

        Merge strategy
        --------------
        Left join on DatetimeIndex — all technical rows are preserved.
        NaN cells produced by the sector builder (e.g. the first 200 rows
        before the SMA-200 warms up) are imputed with the column median so
        that partial-coverage rows do not get eliminated by the dropna() call
        that immediately follows in build_features().

        Requirements
        ------------
        *feat* must still contain a "Close" column at call time.  This is
        always the case because _target_labels() keeps OHLCV columns in the
        DataFrame; they are excluded from the X matrix via get_feature_columns().
        """
        try:
            stock_close = feat.get("Close")
            if stock_close is None:
                logger.warning(
                    "[%s] 'Close' column absent in feat — "
                    "sector correlation features skipped.",
                    ticker,
                )
                return feat

            sector_df = self._sector_correlation_engineer.build(
                ticker=ticker,
                stock_close=stock_close,
                price_index=feat.index,
            )

            if sector_df.empty:
                logger.warning(
                    "[%s] Sector correlation build returned empty DataFrame — "
                    "skipping merge.",
                    ticker,
                )
                return feat

            sector_df = sector_df.reindex(feat.index)

            # Impute NaNs with column median so warm-up rows (e.g. first 200 days
            # before SMA-200 is defined) do not cascade into row drops.
            for col in sector_df.columns:
                if sector_df[col].isna().all():
                    continue
                sector_df[col] = sector_df[col].fillna(sector_df[col].median())

            merged = feat.join(sector_df, how="left")

            logger.info(
                "[%s] Merged %d sector/market correlation features (%d with data)",
                ticker,
                sector_df.shape[1],
                sector_df.notna().any(axis=0).sum(),
            )
            return merged

        except Exception as exc:
            logger.warning(
                "[%s] Sector correlation merge failed (%s) — "
                "proceeding with existing features only.",
                ticker,
                exc,
            )
            return feat

    # ── Options / implied-volatility merge (Improvement 2) ───────────────────

    def _merge_options_features(self, feat: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Fetch, align, and merge options-market / implied-volatility features
        into *feat*.

        Mirrors ``_merge_sector_correlation_features`` and
        ``_merge_fundamental_features``: left join on the DatetimeIndex, NaN
        columns imputed with the column median so partial coverage (e.g. a
        ticker that only recently became optionable, or has no options at
        all) does not eliminate otherwise-valid technical rows via the
        ``dropna()`` call that follows in ``build_features()``.

        The 21-day realized volatility column — already computed earlier in
        the pipeline by ``_volatility_features`` — is passed through so the
        options engineer can also produce the IV/realized-vol volatility-
        risk-premium spread feature (``opt_iv_rv_spread``).

        Requirements
        ------------
        The snapshot-derived columns (``opt_*``) depend on
        ``OptionsHistoryStore`` already having at least one stored snapshot
        for *ticker* (populated via ``scripts/warm_options_cache.py``).
        Until then, this merge simply contributes the VIX-family columns
        (``vix_*``), which have genuine historical depth via yfinance.
        """
        try:
            realized_vol = feat.get("realized_vol_21d")
            opt_df = self._options_engineer.build(
                ticker=ticker,
                price_index=feat.index,
                realized_vol_21d=realized_vol,
            )

            if opt_df.empty:
                logger.warning(
                    "[%s] Options feature build returned empty DataFrame — "
                    "skipping merge.",
                    ticker,
                )
                return feat

            opt_df = opt_df.reindex(feat.index)

            # Impute NaNs with column median so missing snapshot history
            # (e.g. cache not warmed yet, or ticker not optionable) does not
            # cascade into row drops for an otherwise valid trading day.
            for col in opt_df.columns:
                if opt_df[col].isna().all():
                    continue
                opt_df[col] = opt_df[col].fillna(opt_df[col].median())

            merged = feat.join(opt_df, how="left")

            logger.info(
                "[%s] Merged %d options/IV features (%d with data)",
                ticker,
                opt_df.shape[1],
                opt_df.notna().any(axis=0).sum(),
            )
            return merged

        except Exception as exc:
            logger.warning(
                "[%s] Options feature merge failed (%s) — "
                "proceeding with existing features only.",
                ticker,
                exc,
            )
            return feat

    # ── Private feature builders (all unchanged from original) ───────────────

    def _price_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        feat["returns_1d"] = close.pct_change(1)
        feat["returns_3d"] = close.pct_change(3)
        feat["returns_5d"] = close.pct_change(5)
        feat["returns_10d"] = close.pct_change(10)
        feat["log_returns"] = np.log(close / close.shift(1))
        feat["overnight_gap"] = (feat["Open"] - close.shift(1)) / close.shift(1)
        return feat

    def _momentum_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        feat["rsi_14"] = compute_rsi(close, 14)
        feat["rsi_7"] = compute_rsi(close, 7)
        feat["rsi_21"] = compute_rsi(close, 21)
        feat["rsi_overbought"] = (feat["rsi_14"] > 70).astype(int)
        feat["rsi_oversold"] = (feat["rsi_14"] < 30).astype(int)

        macd, sig, hist = compute_macd(close)
        feat["macd"] = macd
        feat["macd_signal"] = sig
        feat["macd_histogram"] = hist
        feat["macd_bullish"] = (macd > sig).astype(int)

        for p in [5, 10, 21, 63]:
            feat[f"momentum_{p}d"] = compute_momentum(close, p)

        feat["momentum_persistence_5_20"] = compute_momentum_persistence(close, 5, 20)
        feat["momentum_persistence_10_60"] = compute_momentum_persistence(close, 10, 60)
        return feat

    def _volatility_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        high = feat["High"]
        low = feat["Low"]

        feat["atr_14"] = compute_atr(high, low, close, 14)
        feat["atr_pct"] = feat["atr_14"] / close

        for w in [5, 10, 21, 63]:
            feat[f"realized_vol_{w}d"] = feat["log_returns"].rolling(
                w, min_periods=w
            ).std() * np.sqrt(252)

        feat["gk_vol_20"] = compute_garman_klass_vol(
            feat["Open"], high, low, close, period=20
        )

        if self.compute_hurst_exp:
            feat["hurst_30"] = compute_hurst(close, lags=15)

        return feat

    def _volume_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        volume = feat["Volume"]

        feat["volume_sma20"] = compute_sma(volume, 20)
        feat["volume_ratio"] = volume / feat["volume_sma20"]
        feat["obv"] = compute_obv(close, volume)
        feat["obv_sma20"] = compute_sma(feat["obv"], 20)
        feat["obv_momentum"] = feat["obv"].pct_change(5)

        vwap = compute_vwap(feat["High"], feat["Low"], close, volume)
        feat["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)

        feat["volume_imbalance_20"] = compute_volume_imbalance(
            close, feat["Open"], volume, 20
        )
        return feat

    def _moving_average_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        for w in self.rolling_windows:
            feat[f"sma_{w}"] = compute_sma(close, w)
            feat[f"ema_{w}"] = compute_ema(close, w)
            feat[f"close_vs_sma_{w}"] = close / feat[f"sma_{w}"] - 1

        feat["sma_5_20_cross"] = (
            compute_sma(close, 5) > compute_sma(close, 20)
        ).astype(int)
        feat["sma_20_50_cross"] = (
            compute_sma(close, 20) > compute_sma(close, 50)
        ).astype(int)
        feat["ema_12_26_cross"] = (
            compute_ema(close, 12) > compute_ema(close, 26)
        ).astype(int)
        return feat

    def _bollinger_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        bb_mid, bb_up, bb_low = compute_bollinger_bands(close)
        feat["bb_middle"] = bb_mid
        feat["bb_upper"] = bb_up
        feat["bb_lower"] = bb_low
        feat["bb_width"] = (bb_up - bb_low) / bb_mid.replace(0, np.nan)
        feat["bb_pct"] = (close - bb_low) / (bb_up - bb_low).replace(0, np.nan)
        feat["bb_squeeze"] = (
            feat["bb_width"] < feat["bb_width"].rolling(20).quantile(0.2)
        ).astype(int)
        return feat

    def _regime_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        rv20 = feat.get(
            "realized_vol_21d",
            feat["log_returns"].rolling(21).std() * np.sqrt(252),
        )
        feat["vol_regime_pct"] = compute_vol_regime(rv20, lookback=252)
        feat["high_vol_regime"] = (feat["vol_regime_pct"] > 0.75).astype(int)
        feat["low_vol_regime"] = (feat["vol_regime_pct"] < 0.25).astype(int)
        feat["trend_regime"] = compute_trend_regime(close)
        feat["in_uptrend"] = (feat["trend_regime"] > 0).astype(int)
        feat["in_downtrend"] = (feat["trend_regime"] < 0).astype(int)
        return feat

    def _microstructure_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        volume = feat["Volume"]
        high = feat["High"]
        low = feat["Low"]

        feat["amihud_illiq_20"] = compute_amihud_illiquidity(close, volume, 20)
        feat["hl_spread_pct"] = (high - low) / close.replace(0, np.nan)
        feat["hl_spread_ma10"] = feat["hl_spread_pct"].rolling(10).mean()

        for w in [5, 20]:
            feat[f"rolling_max_{w}"] = close.rolling(w).max()
            feat[f"rolling_min_{w}"] = close.rolling(w).min()
            feat[f"rolling_range_{w}"] = (
                feat[f"rolling_max_{w}"] - feat[f"rolling_min_{w}"]
            ) / close.replace(0, np.nan)
            feat[f"pct_from_high_{w}"] = close / feat[f"rolling_max_{w}"] - 1
            feat[f"pct_from_low_{w}"] = close / feat[f"rolling_min_{w}"] - 1
        return feat

    def _higher_moment_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        feat["rolling_skew_20"] = compute_rolling_skew(close, 20)
        feat["rolling_skew_60"] = compute_rolling_skew(close, 60)
        feat["rolling_kurt_20"] = compute_rolling_kurt(close, 20)
        feat["rolling_kurt_60"] = compute_rolling_kurt(close, 60)
        return feat

    def _candlestick_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        close = feat["Close"]
        high = feat["High"]
        low = feat["Low"]
        open_ = feat["Open"]
        rng = (high - low).replace(0, np.nan)

        feat["candle_body"] = (close - open_).abs() / rng
        feat["upper_shadow"] = (high - close.clip(lower=open_)) / rng
        feat["lower_shadow"] = (close.clip(upper=open_) - low) / rng
        feat["candle_dir"] = (close > open_).astype(int)
        return feat

    def _target_labels(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Compute binary directional labels for all prediction horizons."""
        close = feat["Close"]
        for horizon_name, horizon_days in HORIZONS.items():
            feat[f"target_{horizon_name}"] = (
                close.shift(-horizon_days) > close
            ).astype(int)

        feat[settings.TARGET_COLUMN] = feat["target_1d"]
        return feat


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper around FeatureEngineer.build_features()."""
    return FeatureEngineer().build_features(df)
