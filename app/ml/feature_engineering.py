"""
FinSight AI — Phase 3: Feature Engineering Pipeline
Computes technical, volatility, and momentum indicators from raw OHLCV data.
Produces a clean feature matrix ready for ML training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.core.exceptions import FeatureEngineeringError, InsufficientDataError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("features")


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
    """
    MACD line, Signal line, and Histogram.

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series,
    period: int = settings.BB_PERIOD,
    std_dev: float = settings.BB_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: middle (SMA), upper, lower.

    Returns:
        (middle, upper, lower)
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = settings.ATR_PERIOD
) -> pd.Series:
    """Average True Range — volatility measure."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_momentum(series: pd.Series, period: int = settings.MOMENTUM_PERIOD) -> pd.Series:
    """Rate-of-change momentum."""
    return series.pct_change(periods=period)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def compute_vwap_approx(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Approximate VWAP using (H+L+C)/3 × Volume / cumulative Volume."""
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix Builder
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Stateful feature engineering pipeline.

    Attributes:
        rolling_windows: List of window sizes for rolling statistics.
    """

    def __init__(self, rolling_windows: Optional[list[int]] = None) -> None:
        self.rolling_windows = rolling_windows or settings.ROLLING_WINDOWS

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from a raw OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Feature matrix DataFrame (NaN rows from warm-up periods dropped).

        Raises:
            FeatureEngineeringError: On computation failure.
            InsufficientDataError: If result is empty after NaN drop.
        """
        try:
            feat = df.copy()
            close = feat["Close"]
            high = feat["High"]
            low = feat["Low"]
            volume = feat["Volume"]

            # ── Price-based features ──────────────────────────────────────────
            feat["returns_1d"] = close.pct_change(1)
            feat["returns_5d"] = close.pct_change(5)
            feat["log_returns"] = np.log(close / close.shift(1))

            # ── RSI ───────────────────────────────────────────────────────────
            feat["rsi"] = compute_rsi(close)
            feat["rsi_overbought"] = (feat["rsi"] > 70).astype(int)
            feat["rsi_oversold"] = (feat["rsi"] < 30).astype(int)

            # ── MACD ──────────────────────────────────────────────────────────
            macd_line, signal_line, histogram = compute_macd(close)
            feat["macd"] = macd_line
            feat["macd_signal"] = signal_line
            feat["macd_histogram"] = histogram
            feat["macd_bullish"] = (macd_line > signal_line).astype(int)

            # ── Bollinger Bands ───────────────────────────────────────────────
            bb_mid, bb_up, bb_low = compute_bollinger_bands(close)
            feat["bb_middle"] = bb_mid
            feat["bb_upper"] = bb_up
            feat["bb_lower"] = bb_low
            feat["bb_width"] = (bb_up - bb_low) / bb_mid
            feat["bb_pct"] = (close - bb_low) / (bb_up - bb_low)

            # ── ATR ───────────────────────────────────────────────────────────
            feat["atr"] = compute_atr(high, low, close)
            feat["atr_pct"] = feat["atr"] / close  # normalized

            # ── Moving Averages ───────────────────────────────────────────────
            for w in self.rolling_windows:
                feat[f"sma_{w}"] = compute_sma(close, w)
                feat[f"ema_{w}"] = compute_ema(close, w)
                feat[f"close_vs_sma_{w}"] = close / feat[f"sma_{w}"] - 1

            # ── Momentum ──────────────────────────────────────────────────────
            for p in [5, 10, 20]:
                feat[f"momentum_{p}d"] = compute_momentum(close, p)

            # ── Volume features ───────────────────────────────────────────────
            feat["volume_sma20"] = compute_sma(volume, 20)
            feat["volume_ratio"] = volume / feat["volume_sma20"]
            feat["obv"] = compute_obv(close, volume)
            feat["obv_sma20"] = compute_sma(feat["obv"], 20)

            # ── Volatility ────────────────────────────────────────────────────
            for w in [5, 10, 20]:
                feat[f"realized_vol_{w}d"] = (
                    feat["log_returns"].rolling(w).std() * np.sqrt(252)
                )

            # ── Rolling statistics ────────────────────────────────────────────
            for w in self.rolling_windows:
                feat[f"rolling_max_{w}"] = close.rolling(w).max()
                feat[f"rolling_min_{w}"] = close.rolling(w).min()
                feat[f"rolling_range_{w}"] = (
                    feat[f"rolling_max_{w}"] - feat[f"rolling_min_{w}"]
                ) / close

            # ── Candlestick patterns (body/shadow) ────────────────────────────
            feat["candle_body"] = (close - df["Open"]).abs() / (high - low + 1e-9)
            feat["upper_shadow"] = (high - close.clip(lower=df["Open"])) / (high - low + 1e-9)
            feat["lower_shadow"] = (close.clip(upper=df["Open"]) - low) / (high - low + 1e-9)

            # ── Target variable (next-day direction) ─────────────────────────
            feat[settings.TARGET_COLUMN] = (close.shift(-1) > close).astype(int)

            logger.info("Feature matrix built: %d features, %d rows", feat.shape[1], feat.shape[0])

            feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            feat.dropna(inplace=True)

            if feat.empty:
                raise InsufficientDataError("Feature matrix is empty after dropping NaN rows.")

            logger.info("After NaN drop: %d rows remain", len(feat))
            return feat

        except (InsufficientDataError, FeatureEngineeringError):
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Feature engineering failed: {exc}") from exc

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return all feature column names (excludes OHLCV + target)."""
        exclude = {"Open", "High", "Low", "Close", "Volume", settings.TARGET_COLUMN}
        return [c for c in df.columns if c not in exclude]

    def split_X_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split feature matrix into X and y.

        Args:
            df: Output of build_features().

        Returns:
            (X DataFrame, y Series)
        """
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        y = df[settings.TARGET_COLUMN].copy()
        return X, y


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper around FeatureEngineer.build_features()."""
    return FeatureEngineer().build_features(df)
