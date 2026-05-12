"""
FinSight AI — Dashboard (v3 — fixed)

Bugs fixed vs v2
----------------
1. Model Selection sidebar section removed entirely.
   It was rendered between the Instrument section and the Knowledge Base
   section.  The block and its ``st.markdown`` call are gone.

2. Horizon selector added to the sidebar (below Instrument).
   Four radio options: Next Day / Next Week / Next Month / Next 6 Months.
   The selected value is mapped to the API horizon key ('1d', '7d', '1m', '6m')
   and included in every ``/predict/`` request body.

3. Intelligence brief rendered in the Signal tab.
   ``pred["intelligence_brief"]`` is now checked and a dedicated section
   "Market Intelligence" is rendered below the fused signal card when present,
   showing:
   - Situation summary paragraph
   - Bullish catalysts list (green)
   - Bearish headwinds list (red)
   - Source quality note
   The field name matches the API response exactly (``intelligence_brief``
   → ``situation_summary``, ``bullish_catalysts``, ``bearish_catalysts``).

4. Confidence-degraded banner added.
   When ``pred["confidence_degraded"]`` is True and
   ``pred["selection_reason"] != "leaderboard"``, an amber info banner
   explains that the model quality is below the reliability threshold.
"""

from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "GS", "SPY"]

HORIZON_OPTIONS = {
    "Next Day (1d)":      "1d",
    "Next Week (7d)":     "7d",
    "Next Month (1m)":    "1m",
    "Next 6 Months (6m)": "6m",
}

st.set_page_config(
    page_title="FinSight AI",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
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

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
}

[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

.stAlert { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
.stSuccess { border-left: 3px solid var(--accent-green) !important; }
.stWarning { border-left: 3px solid var(--accent-amber) !important; }
.stError   { border-left: 3px solid var(--accent-red)   !important; }
.stInfo    { border-left: 3px solid var(--accent-cyan)  !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.3) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

/* ── Core component styles ── */
.finsight-wordmark { font-family: var(--font-display); font-weight: 800; font-size: 1.6rem; letter-spacing: -0.02em; color: var(--text-primary); display: flex; align-items: center; gap: 0.5rem; }
.finsight-wordmark .triangle { color: var(--accent-cyan); }
.finsight-tagline { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-top: 0.15rem; }
.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; animation: pulse 2s ease-in-out infinite; }
.status-online  { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
.status-offline { background: var(--accent-red); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── Fused signal card ── */
.fused-card { border-radius: 10px; padding: 1.8rem 2rem; text-align: center; position: relative; overflow: hidden; margin-bottom: 1rem; }
.fused-bullish { background: linear-gradient(135deg, rgba(0,230,118,0.10) 0%, rgba(0,230,118,0.03) 100%); border: 1px solid rgba(0,230,118,0.40); }
.fused-bearish { background: linear-gradient(135deg, rgba(255,61,87,0.10) 0%, rgba(255,61,87,0.03) 100%); border: 1px solid rgba(255,61,87,0.40); }
.fused-neutral { background: linear-gradient(135deg, rgba(255,193,7,0.10) 0%, rgba(255,193,7,0.03) 100%); border: 1px solid rgba(255,193,7,0.40); }
.fused-label { font-family: var(--font-display); font-size: 2.6rem; font-weight: 800; letter-spacing: 0.04em; margin: 0; line-height: 1; }
.fused-bullish  .fused-label { color: var(--accent-green); }
.fused-bearish  .fused-label { color: var(--accent-red);   }
.fused-neutral  .fused-label { color: var(--accent-amber); }
.fused-sublabel { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-secondary); margin-top: 0.4rem; }
.fused-conf-badge { display: inline-block; margin-top: 0.7rem; padding: 0.25rem 0.75rem; border-radius: 4px; font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 500; }
.conf-high     { background: rgba(0,230,118,0.15); color: var(--accent-green); border: 1px solid rgba(0,230,118,0.3); }
.conf-moderate { background: rgba(255,193,7,0.15); color: var(--accent-amber); border: 1px solid rgba(255,193,7,0.3); }
.conf-low      { background: rgba(255,61,87,0.12); color: var(--accent-red);   border: 1px solid rgba(255,61,87,0.25); }

/* ── Synthesis narrative ── */
.synthesis-block { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-purple); border-radius: 0 6px 6px 0; padding: 1.2rem 1.5rem; font-size: 0.9rem; line-height: 1.75; color: var(--text-primary); margin-bottom: 1rem; }
.synthesis-label { font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--accent-purple); margin-bottom: 0.4rem; }

/* ── Intelligence brief ── */
.intel-block { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-cyan); border-radius: 0 6px 6px 0; padding: 1.2rem 1.5rem; font-size: 0.88rem; line-height: 1.7; color: var(--text-primary); margin-bottom: 0.75rem; }
.catalyst-list { margin: 0.4rem 0 0 0; padding: 0; list-style: none; }
.catalyst-bull { color: var(--accent-green); font-size: 0.85rem; padding: 0.15rem 0; }
.catalyst-bull::before { content: "▲  "; font-size: 0.7rem; }
.catalyst-bear { color: var(--accent-red); font-size: 0.85rem; padding: 0.15rem 0; }
.catalyst-bear::before { content: "▼  "; font-size: 0.7rem; }
.source-note { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); margin-top: 0.6rem; }

/* ── ML sub-signal ── */
.ml-signal-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 0.5rem; }
.ml-stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1rem; }
.ml-stat-label { font-family: var(--font-mono); font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem; }
.ml-stat-value { font-family: var(--font-mono); font-size: 1.1rem; font-weight: 500; color: var(--text-primary); }

/* ── Model badge ── */
.model-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.2); border-radius: 4px; padding: 0.2rem 0.7rem; font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-cyan); margin-bottom: 1rem; }

/* ── Horizon badge ── */
.horizon-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(179,136,255,0.07); border: 1px solid rgba(179,136,255,0.2); border-radius: 4px; padding: 0.2rem 0.7rem; font-family: var(--font-mono); font-size: 0.72rem; color: var(--accent-purple); margin-bottom: 1rem; margin-left: 0.5rem; }

/* ── Degraded banner ── */
.degraded-banner { background: rgba(255,193,7,0.07); border: 1px solid rgba(255,193,7,0.25); border-radius: 6px; padding: 0.6rem 1rem; font-family: var(--font-mono); font-size: 0.75rem; color: var(--accent-amber); margin-bottom: 1rem; }
.model-quality-warning { background: rgba(255,61,87,0.07); border: 1px solid rgba(255,61,87,0.25); border-radius: 6px; padding: 0.6rem 1rem; font-family: var(--font-mono); font-size: 0.75rem; color: var(--accent-red); margin-bottom: 1rem; }

/* ── News items ── */
.news-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.news-title { font-size: 0.85rem; font-weight: 500; color: var(--text-primary); margin-bottom: 0.3rem; }
.news-snippet { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5; }
.news-url { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); margin-top: 0.25rem; }

/* ── Prob bar ── */
.prob-bar-wrap { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.2rem; }
.prob-bar-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem; }
.prob-bar-track { height: 8px; background: var(--bg-elevated); border-radius: 4px; position: relative; overflow: hidden; }
.prob-bar-fill-bull { position: absolute; left: 0; top: 0; height: 100%; background: var(--accent-green); border-radius: 4px 0 0 4px; }
.prob-bar-fill-bear { position: absolute; right: 0; top: 0; height: 100%; background: var(--accent-red); border-radius: 0 4px 4px 0; }
.prob-bar-labels { display: flex; justify-content: space-between; margin-top: 0.4rem; font-family: var(--font-mono); font-size: 0.72rem; }

.section-label { font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }
.narrative-block { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-cyan); border-radius: 0 6px 6px 0; padding: 1.2rem 1.5rem; font-size: 0.9rem; line-height: 1.7; color: var(--text-primary); }
.chat-scroll-area { max-height: 420px; overflow-y: auto; padding-right: 0.5rem; margin-bottom: 1rem; }
.chat-bubble-user { background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.15); border-radius: 0 10px 10px 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; }
.chat-bubble-ai { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px 10px 10px 0; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; border-left: 2px solid var(--accent-cyan); }
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the FinSight API. Ensure the server is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API error {e.response.status_code}: {detail}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be training a model — try again shortly.")
        return None


def api_get(endpoint: str) -> Optional[dict]:
    try:
        base = API_BASE.replace("/api/v1", "")
        resp = requests.get(f"{base}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("chat_history",    []),
    ("last_prediction", None),
    ("market_summary",  None),
    ("session_id",      str(uuid.uuid4())),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div class="finsight-wordmark"><span class="triangle">▲</span> FinSight</div>
        <div class="finsight-tagline">Explainable Financial AI</div>
    </div>
    """, unsafe_allow_html=True)

    health = api_get("/health")
    if health:
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.75rem;'
            f'color:#7a8fa8;margin-bottom:1.5rem;">'
            f'<span class="status-dot status-online"></span>'
            f'API v{health.get("version","?")} &nbsp;·&nbsp; '
            f'{health.get("environment","?").upper()}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.75rem;'
            'color:#7a8fa8;margin-bottom:1.5rem;">'
            '<span class="status-dot status-offline"></span>API OFFLINE</div>',
            unsafe_allow_html=True,
        )

    # ── Instrument ──────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True)
    selected_ticker = st.selectbox("Ticker", TICKERS, index=0, label_visibility="collapsed")
    custom_ticker   = st.text_input("Custom ticker", placeholder="e.g. NFLX").upper().strip()
    if custom_ticker:
        selected_ticker = custom_ticker

    # ── Prediction Horizon ───────────────────────────────────────────────────
    # BUG FIX 2: Horizon selector — was completely absent in v2
    st.markdown('<div class="sidebar-section-title">Prediction Horizon</div>', unsafe_allow_html=True)
    horizon_label    = st.radio(
        "Horizon",
        options=list(HORIZON_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Select the forward-looking window for the prediction signal.",
    )
    selected_horizon = HORIZON_OPTIONS[horizon_label]

    # NOTE: Model Selection section intentionally removed (BUG FIX 1)
    # The system auto-selects the best model per ticker/horizon.
    # The selected model name is shown in the prediction results badge.

    # ── Knowledge Base ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)

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
                    {"source_type": "text", "texts": [ingest_text], "source": "user_input"},
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
                            "/rag/ingest",
                            {"source_type": "url", "url": url_val},
                        )
                    if result:
                        if result.get("duplicate"):
                            st.info(result.get("message", "Already ingested."))
                        else:
                            title      = result.get("title", "")
                            char_count = result.get("char_count", 0)
                            chunks     = result.get("chunks_added", 0)
                            st.markdown(
                                f'<div class="ingest-result">'
                                f'<div class="ingest-title">{title or "Article ingested"}</div>'
                                f'{char_count:,} chars &nbsp;·&nbsp; {chunks} chunks indexed'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
            else:
                st.warning("Enter a URL before fetching.")

    st.markdown("""
    <div style="position:fixed;bottom:1.5rem;left:0;width:260px;text-align:center;
                font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">
        Not investment advice · For research use only
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────────────────────────────────────

col_title, _ = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <div style="margin-bottom:0.25rem;">
        <span style="font-family:var(--font-mono);font-size:0.72rem;
                     letter-spacing:0.2em;text-transform:uppercase;color:var(--text-muted);">
            DASHBOARD
        </span>
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                letter-spacing:-0.02em;color:var(--text-primary);line-height:1.1;
                margin-bottom:0.25rem;">
        {selected_ticker}
        <span style="color:var(--text-muted);font-weight:400;font-size:1.1rem;">
            &nbsp;/&nbsp; AI-Driven Signal Fusion
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div style="height:1px;background:var(--border);margin:0.5rem 0 1.5rem 0;"></div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_predict, tab_market, tab_chat, tab_agent = st.tabs([
    "Signal", "Market Data", "AI Chat", "AI Agent"
])


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
        with st.spinner(f"Running ML inference for {selected_ticker} / {horizon_label}…"):
            result = api_post(
                "/predict/",
                {
                    "ticker":    selected_ticker,
                    "horizon":   selected_horizon,   # ← BUG FIX: wired in
                    "use_cache": True,
                },
            )
            if result:
                st.session_state.last_prediction = result

    pred = st.session_state.last_prediction

    # Invalidate stale results when ticker OR horizon changed
    if pred and (
        pred.get("ticker") != selected_ticker
        or pred.get("horizon") != selected_horizon
    ):
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Ticker or horizon changed — click Analyse Signal to refresh</div>
        </div>
        """, unsafe_allow_html=True)
        pred = None

    if pred and pred.get("ticker") == selected_ticker:

        # ── Model + horizon badges ────────────────────────────────────────────
        model_used = pred.get("model_name", "unknown").replace("_", " ").title()
        horizon_display = pred.get("horizon", "1d")
        st.markdown(
            f'<div style="display:flex;gap:0;">'
            f'<div class="model-badge">⚙ Auto-selected: <strong>{model_used}</strong></div>'
            f'<div class="horizon-badge">⏱ Horizon: <strong>{horizon_display}</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Model quality warning (BUG FIX 1 — visible consequence) ──────────
        confidence_degraded = pred.get("confidence_degraded", False)
        selection_reason    = pred.get("selection_reason", "leaderboard")
        auto_trained        = pred.get("auto_trained", False)

        if auto_trained:
            st.markdown(
                '<div class="degraded-banner">'
                '⚙ Model was auto-trained this run (no prior artifact existed). '
                'Consider running a dedicated training job for better performance.'
                '</div>',
                unsafe_allow_html=True,
            )
        elif confidence_degraded and selection_reason == "best_below_threshold":
            st.markdown(
                '<div class="model-quality-warning">'
                '⚠ Selected model is below the reliability threshold (AUC &lt; 0.52). '
                'Predictions may be unreliable. Retrain with more data or HPO.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Fusion availability banner ────────────────────────────────────────
        fusion_applied = pred.get("fusion_applied", False)
        if not fusion_applied:
            st.markdown(
                '<div class="degraded-banner">'
                '⚠ News fusion unavailable — showing ML-only signal. '
                'Set OPENAI_API_KEY to enable full signal fusion.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ════════════════════════════════════════════════════════════════════
        # PRIMARY: FUSED SIGNAL
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-label">Fused Signal — ML + News Synthesis</div>',
                    unsafe_allow_html=True)

        fused_dir  = pred.get("fused_direction", "UNKNOWN")
        fused_conf = pred.get("fused_confidence", "LOW").upper()
        fused_prob = pred.get("fused_probability", 0.5)
        fusion_nar = pred.get("fusion_narrative", "")
        news_sent  = pred.get("news_sentiment", "neutral")

        card_cls_map = {"BULLISH": "fused-bullish", "BEARISH": "fused-bearish", "NEUTRAL": "fused-neutral"}
        arrow_map    = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "↔"}
        conf_css     = {"HIGH": "conf-high", "MODERATE": "conf-moderate", "LOW": "conf-low"}.get(fused_conf, "conf-low")
        card_css     = card_cls_map.get(fused_dir, "fused-neutral")
        arrow        = arrow_map.get(fused_dir, "↔")

        c_fused, c_fused_prob = st.columns([3, 2])

        with c_fused:
            st.markdown(
                f'<div class="fused-card {card_css}">'
                f'<div class="fused-label">{arrow} {fused_dir}</div>'
                f'<div class="fused-sublabel">Fused direction · {horizon_label}</div>'
                f'<div class="fused-conf-badge {conf_css}">{fused_conf} CONFIDENCE</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c_fused_prob:
            bull_pct   = round(fused_prob * 100, 1)
            bear_pct   = round((1 - fused_prob) * 100, 1)
            sent_color = {
                "positive": "var(--accent-green)",
                "negative": "var(--accent-red)",
                "neutral":  "var(--text-secondary)",
            }.get(news_sent, "var(--text-secondary)")

            st.markdown(
                f'<div class="prob-bar-wrap" style="margin-bottom:0.75rem;">'
                f'<div class="prob-bar-label">Fused Bull / Bear Probability</div>'
                f'<div class="prob-bar-track">'
                f'  <div class="prob-bar-fill-bull" style="width:{bull_pct}%;"></div>'
                f'  <div class="prob-bar-fill-bear" style="width:{bear_pct}%;"></div>'
                f'</div>'
                f'<div class="prob-bar-labels">'
                f'  <span style="color:var(--accent-green);">▲ {bull_pct}%</span>'
                f'  <span style="color:var(--accent-red);">▼ {bear_pct}%</span>'
                f'</div>'
                f'</div>'
                f'<div style="background:var(--bg-card);border:1px solid var(--border);'
                f'border-radius:8px;padding:0.9rem 1rem;text-align:center;">'
                f'<div style="font-family:var(--font-mono);font-size:0.65rem;'
                f'letter-spacing:0.15em;text-transform:uppercase;'
                f'color:var(--text-muted);margin-bottom:0.3rem;">News Sentiment</div>'
                f'<div style="font-family:var(--font-mono);font-size:1.1rem;'
                f'font-weight:500;color:{sent_color};">{news_sent.upper()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── LLM Synthesis Narrative ──────────────────────────────────────────
        if fusion_nar:
            st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="synthesis-label">AI Synthesis Reasoning</div>'
                f'<div class="synthesis-block">{fusion_nar}</div>',
                unsafe_allow_html=True,
            )

        # ════════════════════════════════════════════════════════════════════
        # MARKET INTELLIGENCE BRIEF  ← BUG FIX 2: was never rendered in v2
        # ════════════════════════════════════════════════════════════════════
        intel = pred.get("intelligence_brief")
        if intel:
            st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-label">Market Intelligence Brief</div>',
                unsafe_allow_html=True,
            )

            # Situation summary
            situation = intel.get("situation_summary", "")
            if situation:
                st.markdown(
                    f'<div class="intel-block">{situation}</div>',
                    unsafe_allow_html=True,
                )

            # Catalysts
            bulls = intel.get("bullish_catalysts", [])
            bears = intel.get("bearish_catalysts", [])
            if bulls or bears:
                cat_col_l, cat_col_r = st.columns(2)
                with cat_col_l:
                    if bulls:
                        bull_items = "".join(
                            f'<li class="catalyst-bull">{c}</li>' for c in bulls
                        )
                        st.markdown(
                            f'<div style="font-family:var(--font-mono);font-size:0.68rem;'
                            f'letter-spacing:0.15em;text-transform:uppercase;'
                            f'color:var(--accent-green);margin-bottom:0.4rem;">Bullish Catalysts</div>'
                            f'<ul class="catalyst-list">{bull_items}</ul>',
                            unsafe_allow_html=True,
                        )
                with cat_col_r:
                    if bears:
                        bear_items = "".join(
                            f'<li class="catalyst-bear">{c}</li>' for c in bears
                        )
                        st.markdown(
                            f'<div style="font-family:var(--font-mono);font-size:0.68rem;'
                            f'letter-spacing:0.15em;text-transform:uppercase;'
                            f'color:var(--accent-red);margin-bottom:0.4rem;">Bearish Headwinds</div>'
                            f'<ul class="catalyst-list">{bear_items}</ul>',
                            unsafe_allow_html=True,
                        )

            # Source quality note
            src_note = intel.get("source_quality_note", "")
            if src_note:
                st.markdown(
                    f'<div class="source-note">📰 {src_note}</div>',
                    unsafe_allow_html=True,
                )

        # ── News Sources ─────────────────────────────────────────────────────
        news_items = pred.get("news_items", [])
        if news_items:
            with st.expander(f"📰 News sources used in fusion ({len(news_items)} articles)"):
                for item in news_items:
                    st.markdown(
                        f'<div class="news-item">'
                        f'<div class="news-title">{item.get("title","")}</div>'
                        f'<div class="news-snippet">{item.get("snippet","")}</div>'
                        f'<div class="news-url">{item.get("url","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ════════════════════════════════════════════════════════════════════
        # SECONDARY: RAW ML SIGNAL
        # ════════════════════════════════════════════════════════════════════
        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">Quantitative Signal — ML Model Only</div>',
            unsafe_allow_html=True,
        )

        ml_dir   = pred.get("prediction_label", "—")
        ml_prob  = pred.get("probability", 0.0)
        ml_conf  = pred.get("confidence_label", "—").upper()
        p_bull   = pred.get("p_bullish", 0.5)
        p_bear   = pred.get("p_bearish", 0.5)
        close    = pred.get("latest_close", 0.0)

        ml_dir_color = {
            "BULLISH": "var(--accent-green)",
            "BEARISH": "var(--accent-red)",
        }.get(ml_dir, "var(--text-secondary)")

        st.markdown(
            f'<div class="ml-signal-row">'
            f'<div class="ml-stat">'
            f'  <div class="ml-stat-label">ML Direction</div>'
            f'  <div class="ml-stat-value" style="color:{ml_dir_color};">{ml_dir}</div>'
            f'</div>'
            f'<div class="ml-stat">'
            f'  <div class="ml-stat-label">P(Bull) / P(Bear)</div>'
            f'  <div class="ml-stat-value">{p_bull:.1%} / {p_bear:.1%}</div>'
            f'</div>'
            f'<div class="ml-stat">'
            f'  <div class="ml-stat-label">ML Confidence</div>'
            f'  <div class="ml-stat-value">{ml_conf}</div>'
            f'</div>'
            f'<div class="ml-stat">'
            f'  <div class="ml-stat-label">Last Close</div>'
            f'  <div class="ml-stat-value">${close:,.2f}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="narrative-block">{pred.get("narrative","")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # ── SHAP Chart ───────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-label">SHAP Feature Attribution</div>',
            unsafe_allow_html=True,
        )
        features = pred.get("top_features", [])
        if features:
            df_shap = pd.DataFrame(features)
            colors  = [
                "rgba(0,230,118,0.85)" if v > 0 else "rgba(255,61,87,0.85)"
                for v in df_shap["shap_value"]
            ]
            fig = go.Figure(go.Bar(
                x=df_shap["shap_value"],
                y=df_shap["feature"],
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:+.4f}" for v in df_shap["shap_value"]],
                textfont=dict(family="DM Mono, monospace", size=11, color="#7a8fa8"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
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
                    title=dict(
                        text="SHAP Value  (← bearish  ·  bullish →)",
                        font=dict(size=10, color="#3d5068"),
                    ),
                ),
                yaxis=dict(
                    autorange="reversed", showgrid=False,
                    tickfont=dict(family="DM Mono, monospace", size=11, color="#a0b4c8"),
                ),
                hoverlabel=dict(
                    bgcolor="#131920", bordercolor="#1e2d3d",
                    font=dict(family="DM Mono, monospace", size=11),
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    elif not pred:
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Select a ticker and horizon, then click Analyse Signal</div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET DATA
# ═════════════════════════════════════════════════════════════════════════════

with tab_market:
    load_col, _ = st.columns([2, 6])
    with load_col:
        load_clicked = st.button("↓  Load Market Data", use_container_width=True)

    if load_clicked:
        with st.spinner("Fetching market data…"):
            summary = api_post(
                "/market/summary", {"ticker": selected_ticker, "period_years": 1}
            )
            if summary:
                st.session_state.market_summary = summary

    mkt = st.session_state.market_summary

    if mkt and mkt.get("ticker") == selected_ticker:
        st.markdown('<div class="section-label">12-Month Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="market-stat-row">'
            f'<div class="market-stat"><div class="market-stat-label">Trading Days</div>'
            f'<div class="market-stat-value">{mkt["rows"]}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">52W Low</div>'
            f'<div class="market-stat-value" style="color:var(--accent-red);">'
            f'${mkt["close_min"]:,.2f}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">52W High</div>'
            f'<div class="market-stat-value" style="color:var(--accent-green);">'
            f'${mkt["close_max"]:,.2f}</div></div>'
            f'<div class="market-stat"><div class="market-stat-label">Mean Close</div>'
            f'<div class="market-stat-value">${mkt["close_mean"]:,.2f}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        pct = (mkt["close_mean"] - mkt["close_min"]) / max(
            mkt["close_max"] - mkt["close_min"], 0.01
        ) * 100
        st.markdown('<div class="section-label">52-Week Price Position</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:var(--bg-card);border:1px solid var(--border);'
            f'border-radius:8px;padding:1.2rem 1.5rem;">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent-red);">'
            f'${mkt["close_min"]:,.2f}</span>'
            f'<span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-muted);">'
            f'Mean ${mkt["close_mean"]:,.2f}</span>'
            f'<span style="font-family:var(--font-mono);font-size:0.75rem;color:var(--accent-green);">'
            f'${mkt["close_max"]:,.2f}</span>'
            f'</div>'
            f'<div style="background:var(--bg-elevated);border-radius:4px;height:6px;position:relative;">'
            f'<div style="position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;'
            f'background:linear-gradient(90deg,var(--accent-red),var(--accent-cyan));border-radius:4px;"></div>'
            f'<div style="position:absolute;left:{pct:.1f}%;top:-3px;width:12px;height:12px;'
            f'border-radius:50%;background:var(--accent-cyan);box-shadow:0 0 8px var(--accent-cyan);'
            f'transform:translateX(-50%);"></div>'
            f'</div>'
            f'<div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);'
            f'text-align:right;margin-top:0.5rem;">Mean at {pct:.1f}th percentile of 52W range</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">52-Week Closing Price</div>', unsafe_allow_html=True)
        try:
            import yfinance as yf
            df_hist = yf.download(selected_ticker, period="1y", auto_adjust=True, progress=False)
            if not df_hist.empty:
                closes = df_hist["Close"].squeeze()
                fig_spark = go.Figure(go.Scatter(
                    x=closes.index, y=closes.values, mode="lines",
                    line=dict(color="#00d4ff", width=1.5),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                    hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
                ))
                fig_spark.update_layout(
                    height=200, margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=True,
                               tickfont=dict(family="DM Mono, monospace", size=10, color="#3d5068")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(30,45,61,0.6)",
                               tickfont=dict(family="DM Mono, monospace", size=10, color="#3d5068"),
                               tickprefix="$"),
                    hoverlabel=dict(bgcolor="#131920", bordercolor="#1e2d3d",
                                    font=dict(family="DM Mono, monospace", size=11)),
                )
                st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.caption("Price chart unavailable.")

        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.72rem;'
            f'color:var(--text-muted);margin-top:0.5rem;text-align:right;">'
            f'{mkt["start_date"]} → {mkt["end_date"]}'
            f' &nbsp;·&nbsp; {len(mkt.get("columns", []))} columns'
            f' &nbsp;·&nbsp; {mkt["null_count"]} nulls'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Click Load Market Data to fetch 12-month statistics</div>
        </div>
        """, unsafe_allow_html=True)


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
            '<div style="text-align:center;padding:2rem 1rem;'
            'font-family:var(--font-mono);font-size:0.78rem;'
            'color:var(--text-muted);letter-spacing:0.05em;">'
            "Ask anything about financial markets, indicators, or model predictions."
            "</div>"
        )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                history_html += (
                    f'<div class="chat-bubble-user">'
                    f'<div class="chat-role">You</div>{msg["content"]}</div>'
                )
            else:
                history_html += (
                    f'<div class="chat-bubble-ai">'
                    f'<div class="chat-role">FinSight AI</div>{msg["content"]}</div>'
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
                {
                    "query":      user_input,
                    "use_rag":    use_rag,
                    "session_id": st.session_state.session_id,
                },
            )
            if result:
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": result.get("response", "No response received."),
                })
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
        '<div style="font-family:var(--font-body);font-size:0.875rem;'
        'color:var(--text-secondary);margin-bottom:1.5rem;line-height:1.6;max-width:680px;">'
        "The agent autonomously plans, selects, and chains tools — prediction, "
        "SHAP explanation, sentiment analysis, and knowledge retrieval — to answer "
        "complex multi-step financial queries."
        "</div>",
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
        agent_clicked = st.button("▶  Run Agent", type="primary", use_container_width=True)

    if agent_clicked:
        if agent_query.strip():
            with st.spinner("Agent planning and executing tools…"):
                result = api_post("/agent/run", {"query": agent_query})

            if result:
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
                if result.get("tools_used"):
                    st.markdown('<div class="section-label">Tools Invoked</div>', unsafe_allow_html=True)
                    chips = "".join(
                        f'<span class="tool-chip">{t}</span>'
                        for t in result["tools_used"]
                    )
                    st.markdown(f'<div style="margin-bottom:1rem;">{chips}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-label">Agent Response</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="narrative-block">{result["response"]}</div>',
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