from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Streamlit only adds this script's own directory (app/frontend/) to
# sys.path, not the project root — so `app.*` imports fail unless we
# add it explicitly. Same pattern as scripts/train_model.py etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import markdown as md
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from portfolio_tab import render_portfolio_tab
from streaming_signal import render_streaming_signal_tab

from app.core.shap_glossary import describe_shap_feature

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — environment-driven
# ─────────────────────────────────────────────────────────────────────────────

API_BASE: str = os.environ.get(
    "FRONTEND_API_BASE",
    "http://localhost:8000/api/v1",
).rstrip("/")

_API_KEY: Optional[str] = os.environ.get("FINSIGHT_API_KEY")

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
]

HORIZON_OPTIONS = {
    "Next Day (1d)": "1d",
    "Next Week (7d)": "7d",
    "Next Month (1m)": "1m",
    "Next 6 Months (6m)": "6m",
}

# ── Custom ticker persistence (survives page refresh / process restart) ─────
_CUSTOM_TICKERS_PATH = Path(__file__).parent / "custom_tickers.json"


def _load_custom_tickers() -> list[str]:
    """Read persisted custom tickers from disk. Tolerant of a missing or
    corrupted file — worst case we just start from an empty list rather
    than crashing the dashboard."""
    try:
        if _CUSTOM_TICKERS_PATH.exists():
            data = json.loads(_CUSTOM_TICKERS_PATH.read_text())
            if isinstance(data, list):
                return [str(t).upper().strip() for t in data if t]
    except Exception:
        pass
    return []


def _save_custom_tickers(tickers: list[str]) -> None:
    """Persist the current custom ticker list to disk. Failures are
    non-fatal — the ticker still works for the rest of the session even
    if the write fails (e.g. read-only filesystem)."""
    try:
        _CUSTOM_TICKERS_PATH.write_text(json.dumps(tickers, indent=2))
    except Exception:
        pass


@st.cache_data(ttl=3600, show_spinner=False)
def _is_valid_ticker(symbol: str) -> bool:
    """Check whether a ticker symbol resolves to real market data.

    Cached for an hour per symbol so re-selecting the same custom ticker
    (or re-running the script on an unrelated widget interaction) doesn't
    trigger a repeat network call to yfinance.
    """
    if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
        return False
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="5d")
        return not hist.empty
    except Exception:
        return False


# ── SHAP glossary ──────────────────────────────────────────────────────────────
# Moved to app/core/shap_glossary.py — shared reference data, not UI logic.
def _shap_feature_description(feature_name: str) -> str:
    return describe_shap_feature(feature_name)


st.set_page_config(
    page_title="FinSight AI",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System
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
.main .block-container { padding: 0.5rem 2rem 3rem 2rem !important; max-width: 1400px !important; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 0.5rem !important; }
div[data-testid="stMainBlockContainer"] { padding-top: 0.5rem !important; }
html, body, .stApp * { font-family: var(--font-body) !important; color: var(--text-primary); }
[data-testid="stIconMaterial"],
[data-testid="collapsedControl"] span,
.material-symbols-outlined,
.material-symbols-rounded {
    font-family: 'Material Symbols Outlined', 'Material Symbols Rounded' !important;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
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

[data-testid="stRadio"] label { align-items: flex-start !important; }
[data-testid="stRadio"] label > div:last-child { margin-top: 0.1rem !important; }

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
.narrative-block strong, .narrative-block b    { font-weight: 600 !important; color: var(--text-primary) !important; }
.narrative-block h1, .narrative-block h2, .narrative-block h3   { font-size: 0.9rem !important; font-weight: 400 !important; font-family: var(--font-body) !important; margin: 0.25rem 0 !important; }
.narrative-block ul, .narrative-block ol   { margin: 0 !important; padding: 0 !important; list-style: none !important; }
.narrative-block li   { font-size: 0.9rem !important; line-height: 1.7 !important; }
.narrative-block table { width: 100%; border-collapse: collapse; margin: 0.8rem 0; font-size: 0.85rem; }
.narrative-block th, .narrative-block td { border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; color: var(--text-primary); }
.narrative-block th { background: var(--bg-card); font-family: var(--font-mono) !important; font-size: 0.72rem !important; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-muted) !important; font-weight: 500 !important; }
.narrative-block tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
.narrative-block code { background: var(--bg-card); border: 1px solid var(--border); border-radius: 3px; padding: 0.1rem 0.35rem; font-family: var(--font-mono) !important; font-size: 0.82rem; color: var(--accent-cyan); }
.narrative-block p { margin: 0.4rem 0 !important; }

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

.error-card { background: rgba(255,61,87,0.06); border: 1px solid rgba(255,61,87,0.3); border-left: 3px solid var(--accent-red); border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6; margin: 0.5rem 0; }
.error-card .error-title { color: var(--accent-red); font-weight: 500; font-size: 0.85rem; margin-bottom: 0.4rem; }
.error-card .error-detail { color: var(--text-secondary); }
.error-card .error-rid { color: var(--text-muted); font-size: 0.72rem; margin-top: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_headers() -> dict[str, str]:
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
    url = f"{API_BASE}{endpoint}"
    headers = _build_headers()
    last_error: Optional[str] = None
    request_id: Optional[str] = None

    for attempt in range(retries + 1):
        if attempt > 0:
            wait = 2**attempt
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


def _show_error_card(
    title: str, detail: Optional[str], request_id: Optional[str] = None
) -> None:
    rid_html = (
        f'<div class="error-rid">Request ID: {request_id}</div>' if request_id else ""
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
    ("custom_tickers", None),  # lazily loaded from disk below
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.custom_tickers is None:
    st.session_state.custom_tickers = _load_custom_tickers()


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
            f"Connecting to: {API_BASE}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True
    )
    ticker_options = TICKERS + [
        t for t in st.session_state.custom_tickers if t not in TICKERS
    ]
    selected_ticker = st.selectbox(
        "Ticker", ticker_options, index=0, label_visibility="collapsed"
    )
    custom_ticker = (
        st.text_input("Custom ticker", placeholder="e.g. NFLX").upper().strip()
    )
    if custom_ticker:
        selected_ticker = custom_ticker
        if custom_ticker not in ticker_options:
            if _is_valid_ticker(custom_ticker):
                st.session_state.custom_tickers.append(custom_ticker)
                _save_custom_tickers(st.session_state.custom_tickers)
                st.rerun()
            else:
                st.warning(f"'{custom_ticker}' isn't a recognized ticker.")

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
        if st.button("Ingest Text", width="stretch", key="btn_ingest_text"):
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
        if st.button("Fetch & Ingest", width="stretch", key="btn_ingest_url"):
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

tab_predict, tab_market, tab_chat, tab_agent, tab_portfolio = st.tabs(
    ["Signal", "Market Data", "AI Chat", "AI Agent", "Portfolio"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

with tab_predict:
    render_streaming_signal_tab(
        selected_ticker=selected_ticker,
        selected_horizon=selected_horizon,
        horizon_label=horizon_label,
        api_base=API_BASE,
        api_key=_API_KEY,
    )

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

        card_cls_map = {
            "BULLISH": "fused-bullish",
            "BEARISH": "fused-bearish",
            "NEUTRAL": "fused-neutral",
        }
        arrow_map = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "↔"}
        conf_css = {
            "HIGH": "conf-high",
            "MODERATE": "conf-moderate",
            "LOW": "conf-low",
        }.get(fused_conf, "conf-low")
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
            with st.expander(
                f"📰 News sources used in analysis ({len(news_items)} articles)"
            ):
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
        ml_dir_color = {
            "BULLISH": "var(--accent-green)",
            "BEARISH": "var(--accent-red)",
        }.get(ml_dir, "var(--text-secondary)")

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

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">SHAP Feature Attribution</div>',
            unsafe_allow_html=True,
        )

        # ── SHAP chart + feature selection panel ──────────────────────────────
        # Both blocks are guarded by `if features:` so they only render when
        # the model returned SHAP data.  The `else:` at the same indent level
        # is the sole "unavailable" fallback.
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
                    textfont=dict(
                        family="DM Mono, monospace", size=11, color="#7a8fa8"
                    ),
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
                    showgrid=True,
                    gridcolor="rgba(30,45,61,0.8)",
                    zeroline=True,
                    zerolinecolor="rgba(58,80,107,0.9)",
                    zerolinewidth=1.5,
                    tickfont=dict(family="DM Mono, monospace", size=10),
                    title=dict(
                        text="SHAP Value  (← bearish  ·  bullish →)",
                        font=dict(size=10, color="#3d5068"),
                    ),
                ),
                yaxis=dict(
                    autorange="reversed",
                    showgrid=False,
                    tickfont=dict(
                        family="DM Mono, monospace", size=11, color="#a0b4c8"
                    ),
                ),
                hoverlabel=dict(
                    bgcolor="#131920",
                    bordercolor="#1e2d3d",
                    font=dict(family="DM Mono, monospace", size=11),
                ),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            with st.expander("📖 What do these features mean?", expanded=False):
                st.markdown(
                    '<div style="font-family:var(--font-mono);font-size:0.68rem;'
                    'letter-spacing:0.15em;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.75rem;">'
                    "Green bars push toward BULLISH · Red bars push toward BEARISH · "
                    "Longer bars = stronger influence on this prediction</div>",
                    unsafe_allow_html=True,
                )
                rows_html = ""
                for _, row in df_shap.iterrows():
                    feat_name = row["feature"]
                    direction = "▲" if row["shap_value"] > 0 else "▼"
                    dir_color = (
                        "var(--accent-green)"
                        if row["shap_value"] > 0
                        else "var(--accent-red)"
                    )
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

            # ── How the model chose its indicators (Improvement 4) ───────────
            # Expander always renders — content depends on whether metadata exists.
            fs_meta = pred.get("feature_selection_meta") or {}
            with st.expander("🎯 How the model chose its indicators", expanded=False):
                if fs_meta and fs_meta.get("n_input_features", 0) > 0:
                    n_in = fs_meta.get("n_input_features", 0)
                    n_out = fs_meta.get("n_output_features", 0)
                    n_folds_ev = fs_meta.get("n_folds_evaluated", 0)
                    min_stab = fs_meta.get("min_stability_threshold", 0)
                    d_var = fs_meta.get("dropped_low_variance_count", 0)
                    d_corr = fs_meta.get("dropped_high_correlation_count", 0)
                    d_mi = fs_meta.get("dropped_low_mi_count", 0)
                    d_unst = fs_meta.get("dropped_unstable_count", 0)
                    mi_top = fs_meta.get("mi_scores_top10", {})
                    stab_scores = fs_meta.get("stability_scores", {})
                    n_dropped = n_in - n_out

                    # ── Plain-English summary sentence ────────────────────────
                    st.markdown(
                        f'<p style="font-family:var(--font-body);font-size:0.9rem;'
                        f'color:var(--text-primary);line-height:1.75;margin-bottom:1rem;">'
                        f"Out of <strong>{n_in}</strong> available indicators, the model "
                        f'identified <strong style="color:var(--accent-green);">{n_out}</strong> '
                        f"that reliably predicted this stock's direction across "
                        f"{n_folds_ev} different historical periods. "
                        f"The remaining <strong>{n_dropped}</strong> were set aside because "
                        f"they were either redundant, inconsistent, or added no useful signal."
                        f"</p>",
                        unsafe_allow_html=True,
                    )

                    # ── Plain-English filtering breakdown ─────────────────────
                    st.markdown(
                        f'<div style="background:var(--bg-elevated);border:1px solid var(--border);'
                        f"border-left:3px solid var(--accent-purple);border-radius:0 6px 6px 0;"
                        f'padding:0.9rem 1.2rem;margin-bottom:1.1rem;font-size:0.87rem;line-height:1.85;">'
                        f'<div style="font-family:var(--font-body);color:var(--text-secondary);">'
                        f'<strong style="color:var(--text-primary);">{d_var}</strong> indicators were '
                        f"removed because they barely changed — they carried no useful information.<br>"
                        f'<strong style="color:var(--text-primary);">{d_corr}</strong> were removed '
                        f"because they were measuring the same thing as another indicator already kept.<br>"
                        f'<strong style="color:var(--text-primary);">{d_mi}</strong> were removed '
                        f"because they showed no meaningful connection to future price direction.<br>"
                        f'<strong style="color:var(--text-primary);">{d_unst}</strong> were removed '
                        f"because they only worked in some time periods, not consistently."
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                    # ── Most influential indicators (no raw scores) ───────────
                    if mi_top:
                        with st.expander(
                            "📊 Most influential indicators for this stock",
                            expanded=False,
                        ):
                            st.markdown(
                                '<div style="font-family:var(--font-body);font-size:0.8rem;'
                                'color:var(--text-muted);margin-bottom:0.75rem;">'
                                "Longer bar = stronger connection to future price direction "
                                "for this specific stock.</div>",
                                unsafe_allow_html=True,
                            )
                            mi_items = sorted(mi_top.items(), key=lambda kv: -kv[1])
                            bar_max = max(v for _, v in mi_items) or 1.0
                            rows = ""
                            for feat_name, mi_val in mi_items:
                                bar_w = round(mi_val / bar_max * 100, 1)
                                friendly = feat_name.replace("_", " ").title()
                                sc = stab_scores.get(feat_name, 0)
                                always_on = sc >= n_folds_ev
                                badge_color = (
                                    "var(--accent-green)"
                                    if sc >= min_stab
                                    else "var(--accent-red)"
                                )
                                badge_label = (
                                    "always consistent"
                                    if always_on
                                    else f"consistent {sc}/{n_folds_ev} periods"
                                )
                                rows += (
                                    '<div style="margin-bottom:0.65rem;">'
                                    + '<div style="display:flex;justify-content:space-between;'
                                    + 'font-size:0.82rem;margin-bottom:0.25rem;">'
                                    + f'<span style="color:var(--text-primary);font-family:var(--font-body);">{friendly}</span>'
                                    + '<span style="font-family:var(--font-mono);font-size:0.7rem;'
                                    + f'color:{badge_color};">{badge_label}</span></div>'
                                    + '<div style="background:var(--bg-elevated);border-radius:3px;height:6px;">'
                                    + f'<div style="height:100%;width:{bar_w}%;background:var(--accent-purple);'
                                    + 'border-radius:3px;opacity:0.85;"></div></div></div>'
                                )
                            st.markdown(
                                f'<div style="padding:0.25rem 0;">{rows}</div>',
                                unsafe_allow_html=True,
                            )

                    # ── Consistency breakdown ─────────────────────────────────
                    if stab_scores:
                        with st.expander(
                            "🔍 How consistent were these indicators?",
                            expanded=False,
                        ):
                            st.markdown(
                                '<div style="font-family:var(--font-body);font-size:0.8rem;'
                                'color:var(--text-muted);margin-bottom:0.75rem;">'
                                "An indicator is only kept if it proved useful across "
                                "multiple different time periods — not just one lucky stretch.</div>",
                                unsafe_allow_html=True,
                            )
                            from collections import Counter

                            dist = Counter(stab_scores.values())
                            rows = ""
                            for fc in sorted(dist.keys(), reverse=True):
                                cnt = dist[fc]
                                ok = fc >= min_stab
                                clr = (
                                    "var(--accent-green)" if ok else "var(--accent-red)"
                                )
                                if fc == n_folds_ev:
                                    consistency = "Worked every time"
                                elif fc >= min_stab:
                                    consistency = "Worked most of the time  ✓ kept"
                                elif fc == 1:
                                    consistency = "Only worked once  — excluded"
                                else:
                                    consistency = f"Inconsistent ({fc}/{n_folds_ev} periods)  — excluded"
                                rows += (
                                    '<div style="display:flex;justify-content:space-between;'
                                    + "padding:0.4rem 0;border-bottom:1px solid var(--border);"
                                    + 'font-size:0.83rem;">'
                                    + f'<span style="color:{clr};font-family:var(--font-body);">{consistency}</span>'
                                    + '<span style="font-family:var(--font-mono);font-size:0.78rem;'
                                    + f'color:var(--text-secondary);">{cnt} indicators</span></div>'
                                )
                            st.markdown(
                                '<div style="background:var(--bg-elevated);border:1px solid var(--border);'
                                + f'border-radius:6px;padding:0.5rem 0.75rem;">{rows}</div>',
                                unsafe_allow_html=True,
                            )

                else:
                    # Shown when the cached prediction pre-dates Improvement 4
                    # or the model hasn't been retrained yet with feature selection.
                    st.markdown(
                        '<div style="font-family:var(--font-body);font-size:0.85rem;'
                        'color:var(--text-muted);padding:0.25rem 0;line-height:1.7;">'
                        "Run a fresh prediction to see how the model chose its indicators. "
                        "This detail becomes available after the model has been trained "
                        "with the automated indicator selection pipeline."
                        "</div>",
                        unsafe_allow_html=True,
                    )
            # ── END indicator selection panel ─────────────────────────────────

        else:
            # ── FIX: this else belongs to `if features:` (8-space indent),
            # NOT to `if stab_scores:` (16-space indent) as it was before ────
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
        load_clicked = st.button("↓  Load Market Data", width="stretch")

    if load_clicked:
        with st.spinner("Fetching market data…"):
            summary = api_post(
                "/market/summary", {"ticker": selected_ticker, "period_years": 1}
            )
            if summary:
                st.session_state.market_summary = summary

    mkt = st.session_state.market_summary

    if mkt and mkt.get("ticker") == selected_ticker:
        st.markdown(
            '<div class="section-label">12-Month Summary</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="market-stat-row">'
            f'<div class="market-stat"><div class="market-stat-label">Trading Days</div><div class="market-stat-value">{mkt["rows"]}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">52W Low</div><div class="market-stat-value" style="color:var(--accent-red);">${mkt["close_min"]:,.2f}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">52W High</div><div class="market-stat-value" style="color:var(--accent-green);">${mkt["close_max"]:,.2f}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">Mean Close</div><div class="market-stat-value">${mkt["close_mean"]:,.2f}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        pct = (
            (mkt["close_mean"] - mkt["close_min"])
            / max(mkt["close_max"] - mkt["close_min"], 0.01)
            * 100
        )
        st.markdown(
            '<div class="section-label">52-Week Price Position</div>',
            unsafe_allow_html=True,
        )
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
        st.markdown(
            '<div class="section-label">52-Week Closing Price</div>',
            unsafe_allow_html=True,
        )
        try:
            import yfinance as yf

            df_hist = yf.download(
                selected_ticker, period="1y", auto_adjust=True, progress=False
            )
            if not df_hist.empty:
                closes = df_hist["Close"].squeeze()
                fig_spark = go.Figure(
                    go.Scatter(
                        x=closes.index,
                        y=closes.values,
                        mode="lines",
                        line=dict(color="#00d4ff", width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(0,212,255,0.05)",
                        hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
                    )
                )
                fig_spark.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=False,
                        showticklabels=True,
                        tickfont=dict(
                            family="DM Mono, monospace", size=10, color="#3d5068"
                        ),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(30,45,61,0.6)",
                        tickfont=dict(
                            family="DM Mono, monospace", size=10, color="#3d5068"
                        ),
                        tickprefix="$",
                    ),
                    hoverlabel=dict(
                        bgcolor="#131920",
                        bordercolor="#1e2d3d",
                        font=dict(family="DM Mono, monospace", size=11),
                    ),
                )
                st.plotly_chart(
                    fig_spark,
                    width="stretch",
                    config={"displayModeBar": False},
                )
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


def render_chat_markdown(text: str) -> str:
    """Convert LLM-generated markdown (bold, lists, tables, etc.) to HTML
    so it renders correctly inside the custom chat-bubble div, which is
    injected via unsafe_allow_html and therefore bypasses Streamlit's
    native markdown parser."""
    return md.markdown(text, extensions=["extra", "sane_lists"])


with tab_chat:
    top_row, rag_toggle_col = st.columns([5, 1])
    with top_row:
        st.markdown(
            '<div class="section-label">Financial AI Assistant</div>',
            unsafe_allow_html=True,
        )
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
                history_html += f'<div class="chat-bubble-user"><div class="chat-role">You</div>{msg["content"]}</div>'
            else:
                rendered_content = render_chat_markdown(msg["content"])
                history_html += f'<div class="chat-bubble-ai"><div class="chat-role">FinSight AI</div>{rendered_content}</div>'

    st.markdown(
        f'<div class="chat-scroll-area">{history_html}</div>', unsafe_allow_html=True
    )

    with st.form("chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([7, 1])
        with input_col:
            user_input = st.text_input(
                "Message",
                placeholder="What does RSI divergence indicate in a downtrend?",
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("Send", width="stretch")

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner(""):
            result = api_post(
                "/rag/chat",
                {
                    "query": user_input,
                    "use_rag": use_rag,
                    "session_id": st.session_state.session_id,
                },
            )
            if result:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result.get("response", "No response received."),
                    }
                )
        st.rerun()

    if st.session_state.chat_history:
        clear_col, _ = st.columns([1, 7])
        with clear_col:
            if st.button("Clear history", width="stretch"):
                st.session_state.chat_history = []
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI AGENT
# ═════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown(
        '<div class="section-label">Autonomous Agent</div>', unsafe_allow_html=True
    )
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
            "Example queries",
            ["— select an example —"] + EXAMPLES,
            label_visibility="collapsed",
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
        agent_clicked = st.button("▶  Run Agent", type="primary", width="stretch")

    if agent_clicked:
        if agent_query.strip():
            with st.spinner("Agent planning and executing tools…"):
                result = api_post("/agent/run", {"query": agent_query})

            if result:
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

                if result.get("tools_used"):
                    st.markdown(
                        '<div class="section-label">Tools Invoked</div>',
                        unsafe_allow_html=True,
                    )
                    chips = "".join(
                        f'<span class="tool-chip">{t}</span>'
                        for t in result["tools_used"]
                    )
                    st.markdown(
                        f'<div style="margin-bottom:1rem;">{chips}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="section-label">Agent Response</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="narrative-block">{render_chat_markdown(result["response"])}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Raw tool results"):
                    import json

                    st.code(
                        json.dumps(result.get("tool_results", []), indent=2),
                        language="json",
                    )
        else:
            st.warning("Enter a query to run the agent.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    render_portfolio_tab(api_base=API_BASE, api_key=_API_KEY)
