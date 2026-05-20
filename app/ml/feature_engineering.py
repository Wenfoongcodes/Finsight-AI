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
# This cannot be modified via an API request parameter, only by direct module access.
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
# Indicator Primitives
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
    """
    Garman-Klass volatility estimator.

    More efficient than close-to-close realized vol because it uses the
    full OHLC price path.  Formula:
        σ² = 0.5(ln(H/L))² − (2ln2−1)(ln(C/O))²
    """
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

    H > 0.5 → trending (momentum persists)
    H < 0.5 → mean-reverting
    H ≈ 0.5 → random walk

    **Performance note:** This is an O(n × lags) Python-level rolling
    computation — approximately 2-4 seconds per 1 000 rows.  It must never
    be called from a live API request handler.  Guard via ``_HURST_ALLOWED``
    or use the ``compute_hurst_exp=True`` constructor argument only from
    offline scripts and notebooks.
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
    """
    Amihud (2002) illiquidity ratio — |return| / dollar_volume.

    Higher values indicate more price impact per dollar traded.
    Scaled by 1e6 for readability.
    """
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
    """
    Volatility regime percentile rank over ``lookback`` bars.

    Returns a value in [0, 1]:
      - Near 0 → historically low volatility regime
      - Near 1 → historically high volatility regime
    """
    return realized_vol.rolling(lookback, min_periods=60).rank(pct=True)


def compute_trend_regime(
    close: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> pd.Series:
    """
    Simple trend-regime indicator: +1 (uptrend), -1 (downtrend), 0 (neutral).
    """
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
    """
    Volume imbalance — buy pressure proxy.

    Positive close relative to open suggests buyer-initiated volume.
    """
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
    """
    Momentum persistence: ratio of short-term to long-term momentum.

    > 1 → short-term outperforming long-term (momentum accelerating)
    < 1 → deceleration / reversal pressure
    """
    mom_short = compute_momentum(close, short)
    mom_long = compute_momentum(close, long)
    return mom_short / mom_long.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Selector
# ─────────────────────────────────────────────────────────────────────────────


class FeatureSelector:
    """
    Multi-stage feature selection pipeline.

    Stages (applied in order):
    1. Variance thresholding — remove near-constant features.
    2. Correlation filtering — remove one of each highly-correlated pair.
    3. Mutual-information ranking — keep top-N features by MI with target.
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
# Feature Engineer
# ─────────────────────────────────────────────────────────────────────────────


class FeatureEngineer:
    """
    Stateful feature engineering pipeline.

    Builds a comprehensive feature matrix from raw OHLCV data.

    Parameters
    ----------
    rolling_windows: list[int]
        Window sizes for SMA, EMA, and range statistics.
    compute_hurst_exp: bool
        Whether to compute the Hurst exponent.

        **Important:** This is an expensive O(n×lags) Python-level rolling
        computation (~2-4 s per 1 000 rows).  It is disabled by default and
        is additionally guarded by the module-level ``_HURST_ALLOWED``
        sentinel.  When ``_HURST_ALLOWED`` is ``False`` (production default),
        setting ``compute_hurst_exp=True`` is silently ignored with a warning
        so that no API request can accidentally trigger the computation.

        To enable it for offline scripts/notebooks::

            import app.ml.feature_engineering as fe_mod
            fe_mod._HURST_ALLOWED = True
            engineer = FeatureEngineer(compute_hurst_exp=True)
    """

    def __init__(
        self,
        rolling_windows: Optional[list[int]] = None,
        compute_hurst_exp: bool = False,
    ) -> None:
        self.rolling_windows = rolling_windows or settings.ROLLING_WINDOWS

        # Enforce the production guard — runtime enablement via API is blocked.
        if compute_hurst_exp and not _HURST_ALLOWED:
            logger.warning(
                "compute_hurst_exp=True was requested but _HURST_ALLOWED is False "
                "(production guard). Hurst computation will be skipped. "
                "Set app.ml.feature_engineering._HURST_ALLOWED = True in offline "
                "scripts to enable."
            )
            self.compute_hurst_exp = False
        else:
            self.compute_hurst_exp = compute_hurst_exp

    # ── Public API ────────────────────────────────────────────────────────────

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from a raw OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Feature matrix with NaN warm-up rows dropped.

        Raises:
            FeatureEngineeringError: On computation failure.
            InsufficientDataError:   If result is empty after NaN drop.
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
        """Return feature column names (excludes OHLCV + all target columns)."""
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

        Args:
            df:      Output of build_features().
            horizon: One of '1d', '7d', '1m', '6m'.

        Returns:
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

    # ── Private feature builders ──────────────────────────────────────────────

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

        # Hurst is guarded by both the constructor flag and _HURST_ALLOWED.
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
        """
        Compute binary directional labels for all prediction horizons.

        Label = 1 if price at horizon is strictly above current price.
        Shift forward so each row's label is knowable only at that future date.
        """
        close = feat["Close"]
        for horizon_name, horizon_days in HORIZONS.items():
            feat[f"target_{horizon_name}"] = (
                close.shift(-horizon_days) > close
            ).astype(int)

        feat[settings.TARGET_COLUMN] = feat["target_1d"]
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper around FeatureEngineer.build_features()."""
    return FeatureEngineer().build_features(df)
