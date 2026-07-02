"""Human-readable descriptions for engineered ML features.

Used to translate raw feature names (e.g. ``vwap_deviation``,
``rsi_14``) surfaced by SHAP explanations into plain-English text for
non-technical users. Currently consumed by the Streamlit dashboard's
SHAP transparency panel; kept in ``app.core`` (rather than
``app.frontend``) so it can also ground the AI Agent's natural-language
explanations if/when that's wired up, keeping definitions consistent
across the UI and the LLM-generated commentary.

Extending the glossary: add a new ``"feature_name": "description"``
entry below. Lookup falls back to prefix-matching (see
``describe_shap_feature``), so a single entry like ``"rsi_14"`` will
also match closely related derived names if no exact key exists.
"""

from __future__ import annotations

SHAP_GLOSSARY: dict[str, str] = {
    "rsi_14": "Relative Strength Index (14-day) — measures whether the stock is overbought (>70) or oversold (<30) based on recent price changes.",
    "rsi_7": "Short-term RSI (7-day) — a faster version of RSI that reacts more quickly to recent price moves.",
    "rsi_21": "Slower RSI (21-day) — a smoother momentum indicator over a wider window.",
    "rsi_overbought": "Flag = 1 when RSI is above 70, signalling the stock may be overbought.",
    "rsi_oversold": "Flag = 1 when RSI is below 30, signalling the stock may be oversold.",
    "macd_histogram": "MACD Histogram — the gap between the MACD line and its signal line; positive = accelerating uptrend, negative = accelerating downtrend.",
    "macd_bullish": "Flag = 1 when the MACD line has crossed above its signal line (bullish crossover).",
    "macd": "Moving Average Convergence Divergence — difference between 12-day and 26-day exponential moving averages.",
    "momentum_5d": "5-day price momentum — how much the stock has moved over the last 5 trading days as a percentage.",
    "momentum_10d": "10-day price momentum — percentage price change over the last two weeks.",
    "momentum_21d": "21-day (monthly) price momentum — measures the monthly trend direction.",
    "momentum_63d": "63-day (quarterly) price momentum — measures the 3-month trend direction.",
    "momentum_persistence": "Ratio of short-term to long-term momentum; >1 means the recent move is accelerating.",
    "realized_vol_5d": "5-day realized volatility — annualised standard deviation of daily returns over 5 days. High = more uncertain.",
    "realized_vol_10d": "10-day realized volatility — same concept over two weeks.",
    "realized_vol_21d": "21-day realized volatility — monthly volatility estimate.",
    "realized_vol_63d": "3-month realized volatility — quarterly volatility estimate.",
    "gk_vol_20": "Garman-Klass volatility — a precise intraday volatility estimate using open, high, low, and close prices.",
    "atr_pct": "ATR as a % of price — Average True Range divided by the current price; shows how much the stock typically moves per day relative to its price.",
    "atr_14": "Average True Range (14-day) — average of daily price swings; a higher ATR means larger typical daily moves.",
    "hurst_30": "Hurst Exponent — >0.5 means the stock is trending; <0.5 means it tends to reverse; ≈0.5 means random behaviour.",
    "volume_ratio": "Today's volume divided by the 20-day average. >1 means unusually high activity.",
    "volume_imbalance_20": "Buy/sell volume imbalance — positive means more buyer-initiated trades; negative means more selling pressure.",
    "obv_momentum": "On-Balance Volume momentum — rate of change in cumulative volume trend over 5 days.",
    "obv_sma20": "OBV 20-day moving average — the smoothed trend of cumulative volume flow.",
    "obv": "On-Balance Volume — running total of volume; rising OBV confirms upward price moves.",
    "vwap_deviation": "Price deviation from VWAP (volume-weighted average price). Positive = trading above the session average; negative = below.",
    "bb_pct": "Bollinger Band position — 0 = at lower band, 0.5 = middle, 1 = at upper band. Shows where price sits within its recent range.",
    "bb_width": "Bollinger Band width — wider bands mean higher volatility; narrow bands (squeeze) often precede big moves.",
    "bb_squeeze": "Flag = 1 when the bands are unusually narrow, signalling a potential breakout is building.",
    "close_vs_sma_5": "How far the price is above or below its 5-day moving average, as a percentage.",
    "close_vs_sma_10": "How far the price is above or below its 10-day moving average.",
    "close_vs_sma_20": "How far the price is above or below its 20-day moving average.",
    "close_vs_sma_50": "How far the price is above or below its 50-day moving average.",
    "sma_5_20_cross": "Flag = 1 when the 5-day average is above the 20-day average (short-term uptrend).",
    "sma_20_50_cross": "Flag = 1 when the 20-day average is above the 50-day average (medium-term uptrend).",
    "ema_12_26_cross": "Flag = 1 when the 12-day EMA is above the 26-day EMA — the same crossover used by MACD.",
    "returns_1d": "Yesterday's return — did the stock go up or down since the previous close?",
    "returns_3d": "3-day return — cumulative price change over the last 3 sessions.",
    "returns_5d": "5-day (weekly) return — how the stock performed over the last week.",
    "returns_10d": "10-day return — two-week cumulative performance.",
    "log_returns": "Logarithmic daily return — a mathematically symmetric measure of daily price change.",
    "overnight_gap": "Overnight gap — how much the opening price differed from the previous close (news-driven moves often appear here).",
    "pct_from_high_5": "How far below the 5-day high the price is. Near 0 = price is at a recent peak.",
    "pct_from_high_20": "How far below the 20-day high the price is.",
    "pct_from_low_5": "How far above the 5-day low the price is. Near 0 = price is at a recent trough.",
    "pct_from_low_20": "How far above the 20-day low the price is.",
    "rolling_range_5": "5-day price range as a fraction of price — measures how much the stock swung over the last week.",
    "rolling_range_20": "20-day price range as a fraction of price.",
    "hl_spread_pct": "High-minus-low spread as a percentage of price — today's intraday volatility.",
    "amihud_illiq_20": "Amihud illiquidity — how much price moves per dollar traded. Higher = harder to trade without moving the price.",
    "rolling_skew_20": "Return skewness (20-day) — positive means occasional large up-days; negative means occasional large crashes.",
    "rolling_skew_60": "Return skewness (60-day) — same concept over a 3-month window.",
    "rolling_kurt_20": "Return kurtosis (20-day) — measures how 'fat' the tails of daily returns are; high kurtosis means more surprise moves.",
    "rolling_kurt_60": "Return kurtosis (60-day) — 3-month tail-risk indicator.",
    "vol_regime_pct": "Volatility regime percentile — 0 = historically calm market, 1 = historically turbulent.",
    "high_vol_regime": "Flag = 1 when the stock is in the top 25% of its historical volatility range.",
    "low_vol_regime": "Flag = 1 when the stock is in the bottom 25% of its historical volatility range.",
    "trend_regime": "Trend regime: +1 = confirmed uptrend, -1 = confirmed downtrend, 0 = sideways.",
    "in_uptrend": "Flag = 1 when price is in a confirmed uptrend (50-day avg above 200-day avg).",
    "in_downtrend": "Flag = 1 when price is in a confirmed downtrend.",
    "candle_body": "Candlestick body size — how large today's open-to-close move is relative to the full high-low range.",
    "upper_shadow": "Upper wick — how much the price reached above the open/close range; large upper wick can signal rejection.",
    "lower_shadow": "Lower wick — how much the price fell below the open/close range; large lower wick can signal support.",
    "candle_dir": "Candlestick direction: 1 = close > open (bullish candle), 0 = bearish candle.",
}


def describe_shap_feature(feature_name: str) -> str:
    """Return a plain-English description for a SHAP feature name.

    Tries an exact match first, then falls back to prefix-matching
    against the glossary keys (handles suffixed/derived variants of a
    known base feature). If nothing matches, generates a generic
    fallback description from the raw feature name rather than raising.
    """
    if feature_name in SHAP_GLOSSARY:
        return SHAP_GLOSSARY[feature_name]
    for key, desc in SHAP_GLOSSARY.items():
        if feature_name.startswith(key):
            return desc
    clean = feature_name.replace("_", " ").strip()
    return f"Quantitative indicator derived from price and volume data ({clean})."
