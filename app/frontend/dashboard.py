"""
FinSight AI — Dashboard (v7)

Changes vs v6
-------------

1.  **Environment-aware API base URL**

    ``API_BASE`` is now read from the ``FRONTEND_API_BASE`` environment
    variable at startup, with ``http://localhost:8000/api/v1`` as the
    local-dev default.  Set ``FRONTEND_API_BASE=https://api.yourdomain.com/api/v1``
    in your production ``.env`` / ``docker-compose.yml`` and the dashboard
    will point at the correct host without any code changes.

2.  **Optional API key injection**

    When ``FINSIGHT_API_KEY`` is set in the environment, every request to
    the backend includes ``X-API-Key: <key>`` automatically.  This is the
    correct approach — never hardcode API keys; read them from the environment
    at runtime.

3.  **Resilient connection handling**

    ``api_post`` and ``api_get`` now include:
    - Exponential-backoff retry (1 attempt by default; configurable).
    - A human-readable error card instead of a raw Streamlit error banner,
      with actionable guidance on what to check.
    - The ``X-Request-ID`` from the response header is displayed when an
      error occurs so users can correlate failures with server logs.

4.  **Connection status indicator hardened**

    The sidebar health-check now shows the ``features`` dict returned by
    the v2 health endpoint, giving a quick view of whether LLM / auth /
    rate-limiting are active.

All visual design (CSS, components, charts) is identical to v6.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — environment-driven
# ─────────────────────────────────────────────────────────────────────────────

# Read from env so the same Docker image works locally and in production.
# Override via:  FRONTEND_API_BASE=https://api.yourdomain.com/api/v1
API_BASE: str = os.environ.get(
    "FRONTEND_API_BASE",
    "http://localhost:8000/api/v1",
).rstrip("/")

# Optional shared API key.  Set FINSIGHT_API_KEY in the environment.
# Never put the real key in source code.
_API_KEY: Optional[str] = os.environ.get("FINSIGHT_API_KEY")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "GS", "SPY",
]

HORIZON_OPTIONS = {
    "Next Day (1d)": "1d",
    "Next Week (7d)": "7d",
    "Next Month (1m)": "1m",
    "Next 6 Months (6m)": "6m",
}

# ── SHAP glossary ──────────────────────────────────────────────────────────────
_SHAP_GLOSSARY: dict[str, str] = {
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


def _shap_feature_description(feature_name: str) -> str:
    if feature_name in _SHAP_GLOSSARY:
        return _SHAP_GLOSSARY[feature_name]
    for key, desc in _SHAP_GLOSSARY.items():
        if feature_name.startswith(key):
            return desc
    clean = feature_name.replace("_", " ").strip()
    return f"Quantitative indicator derived from price and volume data ({clean})."


st.set_page_config(
    page_title="FinSight AI",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System (identical to v6)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-base:        #080c10;
    --bg-surface:     #0d1117;
    --bg-elevated:    #131920;
    --bg-card:        #161d26;
    --border:         #1e2d3d;
    --border-bright:  #253545;
    --accent-cyan:    #00d4ff;
    --accent-green:   #00e676;
    --accent-red:     #ff3d57;
    --accent-amber:   #ffc107;
    --accent-purple:  #b388ff;
    --text-primary:   #e8edf2;
    --text-secondary: #7a8fa8;
    --text-muted:     #3d5068;
    --font-display:   'Syne', sans-serif;
    --font-mono:      'DM Mono', monospace;
    --font-body:      'Inter', sans-serif;
}

.stApp { background: var(--bg-base) !important; }
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }
html, body, .stApp * { font-family: var(--font-body) !important; color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] { background: var(--bg-surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important; border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important; color: var(--text-primary) !important;
    font-family: var(--font-mono) !important; font-size: 0.85rem !important;
}

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; color: var(--text-secondary) !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; padding: 0.75rem 1.5rem !important; margin: 0 !important; }
.stTabs [aria-selected="true"] { color: var(--accent-cyan) !important; border-bottom: 2px solid var(--accent-cyan) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; }

.stButton > button { background: transparent !important; border: 1px solid var(--accent-cyan) !important; color: var(--accent-cyan) !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; padding: 0.5rem 1.5rem !important; border-radius: 4px !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: rgba(0, 212, 255, 0.08) !important; box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important; }

[data-testid="metric-container"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 1rem 1.2rem !important; }
[data-testid="metric-container"] label { font-family: var(--font-mono) !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--text-secondary) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: var(--font-mono) !important; font-size: 1.5rem !important; font-weight: 500 !important; color: var(--text-primary) !important; }

.stAlert { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
.stSuccess { border-left: 3px solid var(--accent-green) !important; }
.stWarning { border-left: 3px solid var(--accent-amber) !important; }
.stError   { border-left: 3px solid var(--accent-red)   !important; }
.stInfo    { border-left: 3px solid var(--accent-cyan)  !important; }

.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important; border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important; color: var(--text-primary) !important;
    font-family: var(--font-mono) !important; font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: var(--accent-cyan) !important; box-shadow: 0 0 0 1px rgba(0,212,255,0.3) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

.finsight-wordmark { font-family: var(--font-display); font-weight: 800; font-size: 1.6rem; letter-spacing: -0.02em; color: var(--text-primary); display: flex; align-items: center; gap: 0.5rem; }
.finsight-wordmark .triangle { color: var(--accent-cyan); }
.finsight-tagline { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-top: 0.15rem; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; animation: pulse 2s ease-in-out infinite; }
.status-online  { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
.status-offline { background: var(--accent-red); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.fused-card { border-radius: 10px; padding: 1.8rem 2rem; text-align: center; position: relative; overflow: hidden; margin-bottom: 1rem; }
.fused-bullish { background: linear-gradient(135deg, rgba(0,230,118,0.10) 0%, rgba(0,230,118,0.03) 100%); border: 1px solid rgba(0,230,118,0.40); }
.fused-bearish { background: linear-gradient(135deg, rgba(255,61,87,0.10) 0%, rgba(255,61,87,0.03) 100%); border: 1px solid rgba(255,61,87,0.40); }
.fused-neutral { background: linear-gradient(135deg, rgba(255,193,7,0.10) 0%, rgba(255,193,7,0.03) 100%); border: 1px solid rgba(255,193,7,0.40); }
.fused-label { font-family: var(--font-display); font-size: 2.6rem; font-weight: 800; letter-spacing: 0.04em; margin: 0; line-height: 1; }
.fused-bullish .fused-label { color: var(--accent-green); }
.fused-bearish .fused-label { color: var(--accent-red); }
.fused-neutral .fused-label { color: var(--accent-amber); }
.fused-sublabel { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-secondary); margin-top: 0.4rem; }
.fused-conf-badge { display: inline-block; margin-top: 0.7rem; padding: 0.25rem 0.75rem; border-radius: 4px; font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 500; }
.conf-high     { background: rgba(0,230,118,0.15); color: var(--accent-green); border: 1px solid rgba(0,230,118,0.3); }
.conf-moderate { background: rgba(255,193,7,0.15); color: var(--accent-amber); border: 1px solid rgba(255,193,7,0.3); }
.conf-low      { background: rgba(255,61,87,0.12); color: var(--accent-red);   border: 1px solid rgba(255,61,87,0.25); }

.synthesis-block { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-purple); border-radius: 0 6px 6px 0; padding: 1.2rem 1.5rem; font-size: 0.9rem; line-height: 1.75; color: var(--text-primary); margin-bottom: 1rem; }
.synthesis-label { font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--accent-purple); margin-bottom: 0.4rem; }

.ml-signal-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
.ml-stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1rem; }
.ml-stat-label { font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem; }
.ml-stat-value { font-family: var(--font-mono); font-size: 1.1rem; font-weight: 500; color: var(--text-primary); }

.ml-narrative { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-cyan); border-radius: 0 6px 6px 0; padding: 0.9rem 1.2rem; font-size: 0.88rem; line-height: 1.7; color: var(--text-primary); margin-bottom: 1rem; }

.model-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.2); border-radius: 4px; padding: 0.2rem 0.7rem; font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-cyan); margin-bottom: 1rem; }
.horizon-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(179,136,255,0.07); border: 1px solid rgba(179,136,255,0.2); border-radius: 4px; padding: 0.2rem 0.7rem; font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-purple); margin-bottom: 1rem; margin-left: 0.5rem; }

.news-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.news-title { font-size: 0.85rem; font-weight: 500; color: var(--text-primary); margin-bottom: 0.3rem; }
.news-snippet { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; }
.news-url { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); margin-top: 0.25rem; }

.prob-bar-wrap { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.2rem; }
.prob-bar-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem; }
.prob-bar-track { height: 8px; background: var(--bg-elevated); border-radius: 4px; display: flex; overflow: hidden; }
.prob-bar-bull { height: 100%; background: var(--accent-green); border-radius: 4px 0 0 4px; transition: flex 0.4s ease; }
.prob-bar-bear { height: 100%; background: var(--accent-red); border-radius: 0 4px 4px 0; transition: flex 0.4s ease; }
.prob-bar-labels { display: flex; justify-content: space-between; margin-top: 0.4rem; font-family: var(--font-mono); font-size: 0.72rem; }

.shap-glossary-row { display: grid; grid-template-columns: 220px 1fr; gap: 0.5rem; padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border); font-size: 0.82rem; line-height: 1.5; }
.shap-glossary-row:last-child { border-bottom: none; }
.shap-feat-name { font-family: var(--font-mono); color: var(--accent-cyan); font-size: 0.78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.shap-feat-desc { color: var(--text-secondary); }

.section-label { font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }

.narrative-block {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 6px 6px 0;
    padding: 1.2rem 1.5rem;
    font-family: var(--font-body) !important;
    font-size: 0.9rem;
    font-style: normal !important;
    font-weight: 400 !important;
    line-height: 1.7;
    color: var(--text-primary);
}
.narrative-block em, .narrative-block i    { font-style: normal !important; }
.narrative-block strong, .narrative-block b    { font-weight: 400 !important; color: var(--text-primary) !important; }
.narrative-block h1, .narrative-block h2, .narrative-block h3   { font-size: 0.9rem !important; font-weight: 400 !important; font-family: var(--font-body) !important; margin: 0.25rem 0 !important; }
.narrative-block ul, .narrative-block ol   { margin: 0 !important; padding: 0 !important; list-style: none !important; }
.narrative-block li   { font-size: 0.9rem !important; line-height: 1.7 !important; }

.agent-heading { display: block; font-family: var(--font-mono) !important; font-size: 0.7rem !important; font-weight: 500 !important; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-muted) !important; margin-top: 0.9rem; margin-bottom: 0.2rem; font-style: normal !important; }
.agent-bold { font-family: var(--font-body) !important; font-size: 0.9rem !important; font-weight: 600 !important; color: var(--text-primary) !important; font-style: normal !important; }
.agent-bullet { font-family: var(--font-mono) !important; font-size: 0.75rem !important; font-weight: 500 !important; color: var(--accent-cyan) !important; margin-right: 0.35rem; font-style: normal !important; }

.chat-scroll-area { max-height: 420px; overflow-y: auto; padding-right: 0.5rem; margin-bottom: 1rem; }
.chat-bubble-user { background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.15); border-radius: 0 10px 10px 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-family: var(--font-body) !important; font-size: 0.88rem; font-style: normal !important; font-weight: 400 !important; }
.chat-bubble-ai { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px 10px 10px 0; padding: 0.8rem 1rem; margin: 0.5rem 0; font-family: var(--font-body) !important; font-size: 0.88rem; font-style: normal !important; font-weight: 400 !important; border-left: 2px solid var(--accent-cyan); line-height: 1.65; }
.chat-bubble-ai em, .chat-bubble-ai i    { font-style: normal !important; }
.chat-bubble-ai strong, .chat-bubble-ai b    { font-weight: 500 !important; color: var(--text-primary) !important; }
.chat-bubble-ai h1, .chat-bubble-ai h2, .chat-bubble-ai h3   { font-size: 0.88rem !important; font-weight: 500 !important; font-family: var(--font-body) !important; margin: 0.2rem 0 !important; }
.chat-bubble-ai ul, .chat-bubble-ai ol   { margin: 0 !important; padding-left: 1.2rem !important; }
.chat-bubble-ai li   { font-size: 0.88rem !important; line-height: 1.65 !important; }

.chat-role { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.3rem; }
.tool-chip { display: inline-block; background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.25); border-radius: 4px; padding: 0.2rem 0.6rem; font-family: var(--font-mono); font-size: 0.75rem; color: var(--accent-cyan); margin: 0.15rem; }
.market-stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.market-stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.2rem; }
.market-stat-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem; }
.market-stat-value { font-family: var(--font-mono); font-size: 1.2rem; font-weight: 500; color: var(--text-primary); }
.placeholder-state { text-align: center; padding: 4rem 2rem; border: 1px dashed var(--border-bright); border-radius: 10px; margin: 2rem 0; }
.placeholder-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.4; }
.placeholder-text { font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.05em; }
.sidebar-section-title { font-family: var(--font-mono) !important; font-size: 0.68rem !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; color: var(--text-muted) !important; margin: 1.2rem 0 0.6rem 0 !important; padding-bottom: 0.4rem !important; border-bottom: 1px solid var(--border) !important; }
.ingest-result { background: var(--bg-elevated); border: 1px solid rgba(0,230,118,0.3); border-left: 3px solid var(--accent-green); border-radius: 0 6px 6px 0; padding: 0.8rem 1rem; font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem; line-height: 1.6; }
.ingest-result .ingest-title { color: var(--text-primary); font-weight: 500; margin-bottom: 0.2rem; }

/* Error card */
.error-card { background: rgba(255,61,87,0.06); border: 1px solid rgba(255,61,87,0.3); border-left: 3px solid var(--accent-red); border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6; margin: 0.5rem 0; }
.error-card .error-title { color: var(--accent-red); font-weight: 500; font-size: 0.85rem; margin-bottom: 0.4rem; }
.error-card .error-detail { color: var(--text-secondary); }
.error-card .error-rid { color: var(--text-muted); font-size: 0.72rem; margin-top: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# API Helpers — resilient, env-aware, auth-injecting
# ─────────────────────────────────────────────────────────────────────────────


def _build_headers() -> dict[str, str]:
    """
    Build the standard request headers.

    Injects ``X-API-Key`` when ``FINSIGHT_API_KEY`` is set so every
    outgoing request is authenticated transparently.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _API_KEY:
        headers["X-API-Key"] = _API_KEY
    return headers


def api_post(
    endpoint: str,
    payload: dict,
    timeout: int = 180,
    retries: int = 1,
) -> Optional[dict]:
    """
    POST to the FinSight API with auth injection, retry, and user-friendly errors.

    Parameters
    ----------
    endpoint : Path relative to ``API_BASE`` (e.g. ``"/predict/"``).
    payload  : JSON-serialisable dict.
    timeout  : Per-attempt timeout in seconds.
    retries  : Number of retry attempts on transient errors (5xx / timeout).
    """
    url = f"{API_BASE}{endpoint}"
    headers = _build_headers()
    last_error: Optional[str] = None
    request_id: Optional[str] = None

    for attempt in range(retries + 1):
        if attempt > 0:
            wait = 2 ** attempt
            time.sleep(wait)

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            request_id = resp.headers.get("X-Request-ID")

            if resp.status_code == 401:
                _show_error_card(
                    "Authentication required",
                    "The API requires an API key. Set FINSIGHT_API_KEY in your environment.",
                    request_id,
                )
                return None

            if resp.status_code == 429:
                _show_error_card(
                    "Rate limit exceeded",
                    "Too many requests. Please wait a moment and try again.",
                    request_id,
                )
                return None

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.ConnectionError:
            last_error = (
                f"Cannot reach API at {API_BASE}. "
                "Check that the server is running and that FRONTEND_API_BASE is correct."
            )
        except requests.exceptions.Timeout:
            last_error = (
                "Request timed out. The server may be training models for the first "
                "time — this can take 1–3 minutes. Please wait and try again."
            )
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            # Don't retry client errors (4xx)
            if e.response.status_code < 500:
                _show_error_card(
                    f"API error {e.response.status_code}",
                    detail,
                    e.response.headers.get("X-Request-ID"),
                )
                return None
            last_error = f"Server error {e.response.status_code}: {detail}"

    _show_error_card("Request failed", last_error or "Unknown error.", request_id)
    return None


def api_get(endpoint: str, timeout: int = 10) -> Optional[dict]:
    """
    GET from the FinSight API (used for the health check).
    Silently returns None on failure — the sidebar handles offline display.
    """
    try:
        base = API_BASE.rsplit("/api/v1", 1)[0]
        resp = requests.get(
            f"{base}{endpoint}",
            headers=_build_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _show_error_card(title: str, detail: Optional[str], request_id: Optional[str] = None) -> None:
    rid_html = (
        f'<div class="error-rid">Request ID: {request_id}</div>'
        if request_id
        else ""
    )
    st.markdown(
        f'<div class="error-card">'
        f'<div class="error-title">⚠ {title}</div>'
        f'<div class="error-detail">{detail or ""}</div>'
        f"{rid_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("chat_history", []),
    ("last_prediction", None),
    ("market_summary", None),
    ("session_id", str(uuid.uuid4())),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div class="finsight-wordmark"><span class="triangle">▲</span> FinSight</div>
        <div class="finsight-tagline">Explainable Financial AI</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    health = api_get("/health")
    if health:
        features = health.get("features", {})
        feature_pills = ""
        if features.get("llm"):
            feature_pills += '<span style="color:var(--accent-green);font-size:0.65rem;">● LLM</span> &nbsp;'
        if features.get("auth"):
            feature_pills += '<span style="color:var(--accent-amber);font-size:0.65rem;">● AUTH</span> &nbsp;'
        if features.get("rate_limiting"):
            feature_pills += '<span style="color:var(--accent-cyan);font-size:0.65rem;">● RATE-LIMITED</span>'
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.75rem;color:#7a8fa8;margin-bottom:0.5rem;">'
            f'<span class="status-dot status-online"></span>'
            f"API v{health.get('version', '?')} &nbsp;·&nbsp; {health.get('environment', '?').upper()}</div>"
            f'<div style="margin-bottom:1.5rem;">{feature_pills}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.75rem;color:#7a8fa8;margin-bottom:0.5rem;">'
            f'<span class="status-dot status-offline"></span>API OFFLINE</div>'
            f'<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-bottom:1.5rem;">'
            f'Connecting to: {API_BASE}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True
    )
    selected_ticker = st.selectbox(
        "Ticker", TICKERS, index=0, label_visibility="collapsed"
    )
    custom_ticker = (
        st.text_input("Custom ticker", placeholder="e.g. NFLX").upper().strip()
    )
    if custom_ticker:
        selected_ticker = custom_ticker

    st.markdown(
        '<div class="sidebar-section-title">Prediction Horizon</div>',
        unsafe_allow_html=True,
    )
    horizon_label = st.radio(
        "Horizon",
        options=list(HORIZON_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Select the forward-looking window for the prediction signal.",
    )
    selected_horizon = HORIZON_OPTIONS[horizon_label]

    st.markdown(
        '<div class="sidebar-section-title">Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    kb_tab_text, kb_tab_url = st.tabs(["Paste Text", "From URL"])

    with kb_tab_text:
        ingest_text = st.text_area(
            "Document text",
            placeholder="Paste a financial news snippet, earnings summary, or research note…",
            height=90,
            label_visibility="collapsed",
        )
        if st.button("Ingest Text", use_container_width=True, key="btn_ingest_text"):
            if ingest_text.strip():
                result = api_post(
                    "/rag/ingest",
                    {
                        "source_type": "text",
                        "texts": [ingest_text],
                        "source": "user_input",
                    },
                )
                if result:
                    st.success(result.get("message", "Ingested."))
            else:
                st.warning("Enter text before ingesting.")

    with kb_tab_url:
        article_url = st.text_input(
            "Article URL",
            placeholder="https://www.reuters.com/markets/…",
            label_visibility="collapsed",
        )
        if st.button("Fetch & Ingest", use_container_width=True, key="btn_ingest_url"):
            url_val = article_url.strip()
            if url_val:
                if not url_val.startswith(("http://", "https://")):
                    st.error("URL must start with http:// or https://")
                else:
                    with st.spinner("Fetching article…"):
                        result = api_post(
                            "/rag/ingest", {"source_type": "url", "url": url_val}
                        )
                    if result:
                        if result.get("duplicate"):
                            st.info(result.get("message", "Already ingested."))
                        else:
                            title = result.get("title", "")
                            char_count = result.get("char_count", 0)
                            chunks = result.get("chunks_added", 0)
                            st.markdown(
                                f'<div class="ingest-result">'
                                f'<div class="ingest-title">{title or "Article ingested"}</div>'
                                f"{char_count:,} chars &nbsp;·&nbsp; {chunks} chunks indexed"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.warning("Enter a URL before fetching.")

    st.markdown(
        """
    <div style="position:fixed;bottom:1.5rem;left:0;width:260px;text-align:center;
                font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">
        Not investment advice · For research use only
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────────────────────────────────────

col_title, _ = st.columns([5, 1])
with col_title:
    st.markdown(
        f"""
    <div style="margin-bottom:0.25rem;">
        <span style="font-family:var(--font-mono);font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--text-muted);">DASHBOARD</span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;letter-spacing:-0.02em;color:var(--text-primary);line-height:1.1;margin-bottom:0.25rem;">
        {selected_ticker}
        <span style="color:var(--text-muted);font-weight:400;font-size:1.1rem;">&nbsp;/&nbsp; AI-Driven Signal Fusion</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="height:1px;background:var(--border);margin:0.5rem 0 1.5rem 0;"></div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_predict, tab_market, tab_chat, tab_agent = st.tabs(
    ["Signal", "Market Data", "AI Chat", "AI Agent"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

with tab_predict:
    run_col, _ = st.columns([2, 6])
    with run_col:
        run_clicked = st.button(
            "▶  Analyse Signal", type="primary", use_container_width=True
        )

    if run_clicked:
        with st.spinner(f"Running analysis for {selected_ticker} / {horizon_label}…"):
            result = api_post(
                "/predict/",
                {
                    "ticker": selected_ticker,
                    "horizon": selected_horizon,
                    "use_cache": True,
                },
            )
            if result:
                st.session_state.last_prediction = result

    pred = st.session_state.last_prediction

    if pred and (
        pred.get("ticker") != selected_ticker or pred.get("horizon") != selected_horizon
    ):
        st.markdown(
            """
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Ticker or horizon changed — click Analyse Signal to refresh</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        pred = None

    if pred and pred.get("ticker") == selected_ticker:
        model_used = pred.get("model_name", "unknown").replace("_", " ").title()
        horizon_display = pred.get("horizon", "1d")
        st.markdown(
            f'<div style="display:flex;gap:0;">'
            f'<div class="model-badge">⚙ Auto-selected: <strong>{model_used}</strong></div>'
            f'<div class="horizon-badge">⏱ Horizon: <strong>{horizon_display}</strong></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-label">Fused Signal — ML + News Synthesis</div>',
            unsafe_allow_html=True,
        )

        fused_dir = pred.get("fused_direction", "UNKNOWN")
        fused_conf = pred.get("fused_confidence", "LOW").upper()
        fused_prob = pred.get("fused_probability", 0.5)
        fusion_nar = pred.get("fusion_narrative", "")
        news_sent = pred.get("news_sentiment", "neutral")

        card_cls_map = {"BULLISH": "fused-bullish", "BEARISH": "fused-bearish", "NEUTRAL": "fused-neutral"}
        arrow_map = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "↔"}
        conf_css = {"HIGH": "conf-high", "MODERATE": "conf-moderate", "LOW": "conf-low"}.get(fused_conf, "conf-low")
        card_css = card_cls_map.get(fused_dir, "fused-neutral")
        arrow = arrow_map.get(fused_dir, "↔")

        c_fused, c_fused_prob = st.columns([3, 2])

        with c_fused:
            st.markdown(
                f'<div class="fused-card {card_css}">'
                f'<div class="fused-label">{arrow} {fused_dir}</div>'
                f'<div class="fused-sublabel">Fused direction · {horizon_label}</div>'
                f'<div class="fused-conf-badge {conf_css}">{fused_conf} CONFIDENCE</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with c_fused_prob:
            fused_prob = float(pred.get("fused_probability", 0.5))
            fused_prob = max(0.0, min(1.0, fused_prob))

            if fused_dir == "BULLISH":
                bull_prob = fused_prob
                bear_prob = 1.0 - fused_prob
            elif fused_dir == "BEARISH":
                bear_prob = fused_prob
                bull_prob = 1.0 - fused_prob
            else:
                bull_prob = 0.5
                bear_prob = 0.5

            bull_pct = round(bull_prob * 100, 1)
            bear_pct = round(bear_prob * 100, 1)

            sent_color = {
                "positive": "var(--accent-green)",
                "negative": "var(--accent-red)",
                "neutral": "var(--text-secondary)",
            }.get(news_sent, "var(--text-secondary)")

            st.markdown(
                '<div class="prob-bar-wrap" style="margin-bottom:0.75rem;">'
                '<div class="prob-bar-label">Fused Bull / Bear Probability</div>'
                '<div class="prob-bar-track">'
                f'<div class="prob-bar-bull" style="flex:{bull_prob}; min-width:2px;"></div>'
                f'<div class="prob-bar-bear" style="flex:{bear_prob}; min-width:2px;"></div>'
                "</div>"
                '<div class="prob-bar-labels">'
                f'<span style="color:var(--accent-green);">▲ {bull_pct}%</span>'
                f'<span style="color:var(--accent-red);">▼ {bear_pct}%</span>'
                "</div>"
                "</div>"
                '<div style="background:var(--bg-card);border:1px solid var(--border);'
                'border-radius:8px;padding:0.9rem 1rem;text-align:center;">'
                '<div style="font-family:var(--font-mono);font-size:0.65rem;letter-spacing:0.15em;'
                'text-transform:uppercase;color:var(--text-muted);margin-bottom:0.3rem;">News Sentiment</div>'
                f'<div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:500;color:{sent_color};">'
                f"{news_sent.upper()}</div></div>",
                unsafe_allow_html=True,
            )

        if fusion_nar:
            st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="synthesis-label">AI Synthesis Reasoning</div>'
                f'<div class="synthesis-block">{fusion_nar}</div>',
                unsafe_allow_html=True,
            )

        news_items = pred.get("news_items", [])
        if news_items:
            with st.expander(f"📰 News sources used in analysis ({len(news_items)} articles)"):
                for item in news_items:
                    st.markdown(
                        f'<div class="news-item">'
                        f'<div class="news-title">{item.get("title", "")}</div>'
                        f'<div class="news-snippet">{item.get("snippet", "")}</div>'
                        f'<div class="news-url">{item.get("url", "")}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">Quantitative Signal — ML Model Only</div>',
            unsafe_allow_html=True,
        )

        ml_dir = pred.get("prediction_label", "—")
        p_bull = pred.get("p_bullish", 0.5)
        p_bear = pred.get("p_bearish", 0.5)
        ml_conf = pred.get("confidence_label", "—").upper()
        close = pred.get("latest_close", 0.0)
        ml_dir_color = {"BULLISH": "var(--accent-green)", "BEARISH": "var(--accent-red)"}.get(
            ml_dir, "var(--text-secondary)"
        )

        st.markdown(
            f'<div class="ml-signal-row">'
            f'<div class="ml-stat"><div class="ml-stat-label">ML Direction</div>'
            f'<div class="ml-stat-value" style="color:{ml_dir_color};">{ml_dir}</div></div>'
            f'<div class="ml-stat"><div class="ml-stat-label">P(Bull) / P(Bear)</div>'
            f'<div class="ml-stat-value">{p_bull:.1%} / {p_bear:.1%}</div></div>'
            f'<div class="ml-stat"><div class="ml-stat-label">ML Confidence</div>'
            f'<div class="ml-stat-value">{ml_conf}</div></div>'
            f'<div class="ml-stat"><div class="ml-stat-label">Last Close</div>'
            f'<div class="ml-stat-value">${close:,.2f}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        narrative = pred.get("narrative", "")
        if narrative:
            st.markdown(f'<div class="ml-narrative">{narrative}</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">SHAP Feature Attribution</div>', unsafe_allow_html=True)
        features = pred.get("top_features", [])
        if features:
            df_shap = pd.DataFrame(features)
            colors = [
                "rgba(0,230,118,0.85)" if v > 0 else "rgba(255,61,87,0.85)"
                for v in df_shap["shap_value"]
            ]
            fig = go.Figure(
                go.Bar(
                    x=df_shap["shap_value"],
                    y=df_shap["feature"],
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:+.4f}" for v in df_shap["shap_value"]],
                    textfont=dict(family="DM Mono, monospace", size=11, color="#7a8fa8"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
                )
            )
            fig.update_layout(
                margin=dict(l=10, r=60, t=10, b=10),
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(30,45,61,0.8)",
                    zeroline=True, zerolinecolor="rgba(58,80,107,0.9)", zerolinewidth=1.5,
                    tickfont=dict(family="DM Mono, monospace", size=10),
                    title=dict(text="SHAP Value  (← bearish  ·  bullish →)", font=dict(size=10, color="#3d5068")),
                ),
                yaxis=dict(
                    autorange="reversed", showgrid=False,
                    tickfont=dict(family="DM Mono, monospace", size=11, color="#a0b4c8"),
                ),
                hoverlabel=dict(bgcolor="#131920", bordercolor="#1e2d3d", font=dict(family="DM Mono, monospace", size=11)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with st.expander("📖 What do these features mean?", expanded=False):
                st.markdown(
                    '<div style="font-family:var(--font-mono);font-size:0.68rem;'
                    "letter-spacing:0.15em;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.75rem;\">"
                    "Green bars push toward BULLISH · Red bars push toward BEARISH · "
                    "Longer bars = stronger influence on this prediction</div>",
                    unsafe_allow_html=True,
                )
                rows_html = ""
                for _, row in df_shap.iterrows():
                    feat_name = row["feature"]
                    direction = "▲" if row["shap_value"] > 0 else "▼"
                    dir_color = "var(--accent-green)" if row["shap_value"] > 0 else "var(--accent-red)"
                    desc = _shap_feature_description(feat_name)
                    rows_html += (
                        f'<div class="shap-glossary-row">'
                        f'<div class="shap-feat-name"><span style="color:{dir_color};margin-right:4px;">{direction}</span>{feat_name}</div>'
                        f'<div class="shap-feat-desc">{desc}</div>'
                        f"</div>"
                    )
                st.markdown(
                    f'<div style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:6px;padding:0.5rem 0.75rem;">{rows_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-family:var(--font-mono);font-size:0.8rem;color:var(--text-muted);'
                'padding:1.5rem;text-align:center;border:1px dashed var(--border);border-radius:6px;">'
                "SHAP explanation unavailable for this prediction.</div>",
                unsafe_allow_html=True,
            )

    elif not pred:
        st.markdown(
            """
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Select a ticker and horizon, then click Analyse Signal</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET DATA
# ═════════════════════════════════════════════════════════════════════════════

with tab_market:
    load_col, _ = st.columns([2, 6])
    with load_col:
        load_clicked = st.button("↓  Load Market Data", use_container_width=True)

    if load_clicked:
        with st.spinner("Fetching market data…"):
            summary = api_post("/market/summary", {"ticker": selected_ticker, "period_years": 1})
            if summary:
                st.session_state.market_summary = summary

    mkt = st.session_state.market_summary

    if mkt and mkt.get("ticker") == selected_ticker:
        st.markdown('<div class="section-label">12-Month Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="market-stat-row">'
            f'<div class="market-stat"><div class="market-stat-label">Trading Days</div><div class="market-stat-value">{mkt["rows"]}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">52W Low</div><div class="market-stat-value" style="color:var(--accent-red);">${mkt["close_min"]:,.2f}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">52W High</div><div class="market-stat-value" style="color:var(--accent-green);">${mkt["close_max"]:,.2f}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">Mean Close</div><div class="market-stat-value">${mkt["close_mean"]:,.2f}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        pct = (mkt["close_mean"] - mkt["close_min"]) / max(mkt["close_max"] - mkt["close_min"], 0.01) * 100
        st.markdown('<div class="section-label">52-Week Price Position</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:1.2rem 1.5rem;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent-red);">${mkt["close_min"]:,.2f}</span>'
            f'<span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted);">Mean ${mkt["close_mean"]:,.2f}</span>'
            f'<span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent-green);">${mkt["close_max"]:,.2f}</span>'
            f"</div>"
            f'<div style="background:var(--bg-elevated);border-radius:4px;height:6px;position:relative;">'
            f'<div style="position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;background:linear-gradient(90deg,var(--accent-red),var(--accent-cyan));border-radius:4px;"></div>'
            f'<div style="position:absolute;left:{pct:.1f}%;top:-3px;width:12px;height:12px;border-radius:50%;background:var(--accent-cyan);box-shadow:0 0 8px var(--accent-cyan);transform:translateX(-50%);"></div>'
            f"</div>"
            f'<div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);text-align:right;margin-top:0.5rem;">Mean at {pct:.1f}th percentile of 52W range</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">52-Week Closing Price</div>', unsafe_allow_html=True)
        try:
            import yfinance as yf
            df_hist = yf.download(selected_ticker, period="1y", auto_adjust=True, progress=False)
            if not df_hist.empty:
                closes = df_hist["Close"].squeeze()
                fig_spark = go.Figure(
                    go.Scatter(
                        x=closes.index, y=closes.values, mode="lines",
                        line=dict(color="#00d4ff", width=1.5),
                        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                        hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
                    )
                )
                fig_spark.update_layout(
                    height=200, margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(family="DM Mono, monospace", size=10, color="#3d5068")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(30,45,61,0.6)", tickfont=dict(family="DM Mono, monospace", size=10, color="#3d5068"), tickprefix="$"),
                    hoverlabel=dict(bgcolor="#131920", bordercolor="#1e2d3d", font=dict(family="DM Mono, monospace", size=11)),
                )
                st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("Price chart unavailable.")

        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);margin-top:0.5rem;text-align:right;">'
            f"{mkt['start_date']} → {mkt['end_date']} &nbsp;·&nbsp; {len(mkt.get('columns', []))} columns &nbsp;·&nbsp; {mkt['null_count']} nulls</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="placeholder-state"><div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Click Load Market Data to fetch 12-month statistics</div></div>""",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI CHAT
# ═════════════════════════════════════════════════════════════════════════════

with tab_chat:
    top_row, rag_toggle_col = st.columns([5, 1])
    with top_row:
        st.markdown('<div class="section-label">Financial AI Assistant</div>', unsafe_allow_html=True)
    with rag_toggle_col:
        use_rag = st.toggle("RAG", value=True, help="Inject knowledge base context")

    history_html = ""
    if not st.session_state.chat_history:
        history_html = (
            '<div style="text-align:center;padding:2rem 1rem;font-family:var(--font-mono);'
            'font-size:0.78rem;color:var(--text-muted);letter-spacing:0.05em;">'
            "Ask anything about financial markets, indicators, or model predictions.</div>"
        )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                history_html += (
                    f'<div class="chat-bubble-user"><div class="chat-role">You</div>{msg["content"]}</div>'
                )
            else:
                history_html += (
                    f'<div class="chat-bubble-ai"><div class="chat-role">FinSight AI</div>{msg["content"]}</div>'
                )

    st.markdown(f'<div class="chat-scroll-area">{history_html}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([7, 1])
        with input_col:
            user_input = st.text_input(
                "Message",
                placeholder="What does RSI divergence indicate in a downtrend?",
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner(""):
            result = api_post(
                "/rag/chat",
                {"query": user_input, "use_rag": use_rag, "session_id": st.session_state.session_id},
            )
            if result:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result.get("response", "No response received.")}
                )
        st.rerun()

    if st.session_state.chat_history:
        clear_col, _ = st.columns([1, 7])
        with clear_col:
            if st.button("Clear history", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI AGENT
# ═════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown('<div class="section-label">Autonomous Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:var(--font-body);font-size:0.875rem;color:var(--text-secondary);'
        'margin-bottom:1.5rem;line-height:1.6;max-width:680px;">'
        "The agent autonomously plans, selects, and chains tools — prediction, SHAP explanation, "
        "sentiment analysis, and knowledge retrieval — to answer complex multi-step financial queries.</div>",
        unsafe_allow_html=True,
    )

    EXAMPLES = [
        "Predict AAPL and explain the key drivers behind the signal.",
        "What is the sentiment of: 'Fed signals rate pause as inflation cools'?",
        "Explain RSI divergence and retrieve relevant context from the knowledge base.",
        f"Get a market summary for {selected_ticker} and predict its next-day direction.",
    ]

    ex_col, _ = st.columns([4, 4])
    with ex_col:
        selected_example = st.selectbox(
            "Example queries", ["— select an example —"] + EXAMPLES, label_visibility="collapsed"
        )

    agent_query = st.text_area(
        "Query",
        value="" if selected_example.startswith("—") else selected_example,
        placeholder="Ask a multi-step financial question…",
        height=90,
        label_visibility="collapsed",
    )

    run_agent_col, _ = st.columns([2, 6])
    with run_agent_col:
        agent_clicked = st.button("▶  Run Agent", type="primary", use_container_width=True)

    if agent_clicked:
        if agent_query.strip():
            with st.spinner("Agent planning and executing tools…"):
                result = api_post("/agent/run", {"query": agent_query})

            if result:
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

                if result.get("tools_used"):
                    st.markdown('<div class="section-label">Tools Invoked</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="tool-chip">{t}</span>' for t in result["tools_used"])
                    st.markdown(f'<div style="margin-bottom:1rem;">{chips}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label">Agent Response</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="narrative-block">{result["response"]}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Raw tool results"):
                    import json
                    st.code(json.dumps(result.get("tool_results", []), indent=2), language="json")
        else:
            st.warning("Enter a query to run the agent.")