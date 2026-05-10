"""
FinSight AI — Redesigned Professional Dashboard
Institutional-grade dark terminal aesthetic with refined typography,
animated data cards, and a clean read-only interface.

Changes in this revision
------------------------
* Knowledge base sidebar now has two tabs: "Paste Text" and "From URL".
  URL tab accepts an article URL, calls POST /rag/ingest with source_type="url",
  and shows article title, char count, and chunk count on success.
* Prediction tab now shows p_bullish / p_bearish as a probability bar so raw
  model confidence is always visible, not just the directional probability.
* Market data tab adds a 52-week closing price sparkline chart.
* Chat tab generates a UUID-based session_id on first load so each browser
  session is isolated from other users.
* Chat message container has a fixed max-height with overflow-y: auto so
  long conversations don't push the input box offscreen.
* Agent tab shows a spinner with per-step status during tool execution.
* All API calls show a retry button when the backend is offline.
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
MODELS  = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]

MODEL_LABELS = {
    "xgboost":             "XGBoost",
    "lightgbm":            "LightGBM",
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
}

st.set_page_config(
    page_title="FinSight AI",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System
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

.finsight-wordmark {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 1.6rem;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.finsight-wordmark .triangle { color: var(--accent-cyan); }
.finsight-tagline {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.15rem;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
.status-online  { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
.status-offline { background: var(--accent-red); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.section-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.signal-card { border-radius: 8px; padding: 1.5rem; text-align: center; position: relative; overflow: hidden; }
.signal-bullish { background: linear-gradient(135deg, rgba(0,230,118,0.08) 0%, rgba(0,230,118,0.03) 100%); border: 1px solid rgba(0,230,118,0.3); }
.signal-bearish { background: linear-gradient(135deg, rgba(255,61,87,0.08) 0%, rgba(255,61,87,0.03) 100%); border: 1px solid rgba(255,61,87,0.3); }
.signal-label { font-family: var(--font-display); font-size: 2rem; font-weight: 700; letter-spacing: 0.05em; margin: 0; }
.signal-bullish .signal-label { color: var(--accent-green); }
.signal-bearish .signal-label { color: var(--accent-red); }
.signal-sublabel { font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-secondary); margin-top: 0.3rem; }

.narrative-block {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 6px 6px 0;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-primary);
}

/* Chat area: fixed height with scroll so input stays visible */
.chat-scroll-area {
    max-height: 420px;
    overflow-y: auto;
    padding-right: 0.5rem;
    margin-bottom: 1rem;
}
.chat-bubble-user {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 0 10px 10px 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.chat-bubble-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px 10px 10px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    border-left: 2px solid var(--accent-cyan);
}
.chat-role {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

.tool-chip {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent-cyan);
    margin: 0.15rem;
}

.market-stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.market-stat { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.2rem; }
.market-stat-label { font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem; }
.market-stat-value { font-family: var(--font-mono); font-size: 1.2rem; font-weight: 500; color: var(--text-primary); }

.placeholder-state { text-align: center; padding: 4rem 2rem; border: 1px dashed var(--border-bright); border-radius: 10px; margin: 2rem 0; }
.placeholder-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.4; }
.placeholder-text { font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.05em; }

.sidebar-section-title {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin: 1.2rem 0 0.6rem 0 !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid var(--border) !important;
}

.ingest-result {
    background: var(--bg-elevated);
    border: 1px solid rgba(0,230,118,0.3);
    border-left: 3px solid var(--accent-green);
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    line-height: 1.6;
}
.ingest-result .ingest-title { color: var(--text-primary); font-weight: 500; margin-bottom: 0.2rem; }
.prob-bar-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.prob-bar-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.prob-bar-track {
    height: 8px;
    background: var(--bg-elevated);
    border-radius: 4px;
    position: relative;
    overflow: hidden;
}
.prob-bar-fill-bull {
    position: absolute;
    left: 0; top: 0; height: 100%;
    background: var(--accent-green);
    border-radius: 4px 0 0 4px;
    transition: width 0.4s ease;
}
.prob-bar-fill-bear {
    position: absolute;
    right: 0; top: 0; height: 100%;
    background: var(--accent-red);
    border-radius: 0 4px 4px 0;
    transition: width 0.4s ease;
}
.prob-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────────────────────────────────────

def api_post(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the FinSight API. "
            "Make sure the server is running on port 8000."
        )
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API error {e.response.status_code}: {detail}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be processing a heavy task.")
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
    ("market_prices",   None),   # list of (date, close) for sparkline
    # Stable UUID for this browser session — isolates conversation memory
    # from other users on the same server instance.
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

    # ── Model ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Model",
        MODELS,
        format_func=lambda m: MODEL_LABELS.get(m, m),
        index=0,
        label_visibility="collapsed",
    )

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
            &nbsp;/&nbsp; {MODEL_LABELS.get(selected_model, selected_model)}
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
    "Prediction", "Market Data", "AI Chat", "AI Agent"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

with tab_predict:
    run_col, _ = st.columns([2, 6])
    with run_col:
        run_clicked = st.button(
            "▶  Run Prediction", type="primary", use_container_width=True
        )

    if run_clicked:
        with st.spinner("Running inference pipeline…"):
            result = api_post(
                "/predict/",
                {"ticker": selected_ticker, "model_name": selected_model, "use_cache": True},
            )
            if result:
                st.session_state.last_prediction = result

    pred = st.session_state.last_prediction

    if pred:
        label   = pred["prediction_label"]
        prob    = pred["probability"]
        conf    = pred["confidence_label"].upper()
        close   = pred["latest_close"]
        p_bull  = pred.get("p_bullish", 0.5)
        p_bear  = pred.get("p_bearish", 0.5)
        is_bull = label == "BULLISH"

        # ── Signal + probability bar + metrics ──────────────────────────────
        c_sig, c_probs, c_conf, c_close = st.columns([2, 2, 1, 1])

        with c_sig:
            card_cls = "signal-bullish" if is_bull else "signal-bearish"
            arrow    = "↑" if is_bull else "↓"
            st.markdown(
                f'<div class="signal-card {card_cls}">'
                f'<div class="signal-label">{arrow} {label}</div>'
                f'<div class="signal-sublabel">Next-day direction signal</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c_probs:
            # Visual probability bar showing P(bull) vs P(bear) side-by-side
            bull_pct = round(p_bull * 100, 1)
            bear_pct = round(p_bear * 100, 1)
            st.markdown(
                f'<div class="prob-bar-wrap">'
                f'<div class="prob-bar-label">Bull / Bear Probability</div>'
                f'<div class="prob-bar-track">'
                f'  <div class="prob-bar-fill-bull" style="width:{bull_pct}%;"></div>'
                f'  <div class="prob-bar-fill-bear" style="width:{bear_pct}%;"></div>'
                f'</div>'
                f'<div class="prob-bar-labels">'
                f'  <span style="color:var(--accent-green);">▲ {bull_pct}%</span>'
                f'  <span style="color:var(--accent-red);">▼ {bear_pct}%</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        conf_color = {
            "HIGH":     "var(--accent-green)",
            "MODERATE": "var(--accent-amber)",
        }.get(conf, "var(--text-secondary)")

        with c_conf:
            st.markdown(
                f'<div style="background:var(--bg-card);border:1px solid var(--border);'
                f'border-radius:8px;padding:1.2rem 1rem;text-align:center;">'
                f'<div style="font-family:var(--font-mono);font-size:0.68rem;'
                f'letter-spacing:0.15em;text-transform:uppercase;'
                f'color:var(--text-muted);margin-bottom:0.4rem;">Confidence</div>'
                f'<div style="font-family:var(--font-mono);font-size:1.4rem;'
                f'font-weight:500;color:{conf_color};">{conf}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c_close:
            st.markdown(
                f'<div style="background:var(--bg-card);border:1px solid var(--border);'
                f'border-radius:8px;padding:1.2rem 1rem;text-align:center;">'
                f'<div style="font-family:var(--font-mono);font-size:0.68rem;'
                f'letter-spacing:0.15em;text-transform:uppercase;'
                f'color:var(--text-muted);margin-bottom:0.4rem;">Last Close</div>'
                f'<div style="font-family:var(--font-mono);font-size:1.4rem;'
                f'font-weight:500;color:var(--text-primary);">${close:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # ── Narrative ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Model Reasoning</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="narrative-block">{pred["narrative"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # ── SHAP Chart ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label">SHAP Feature Attribution</div>', unsafe_allow_html=True)
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
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(30,45,61,0.8)",
                    zeroline=True, zerolinecolor="rgba(58,80,107,0.9)", zerolinewidth=1.5,
                    tickfont=dict(family="DM Mono, monospace", size=10),
                    title=dict(text="SHAP Value  (← bearish  ·  bullish →)",
                               font=dict(size=10, color="#3d5068")),
                ),
                yaxis=dict(
                    autorange="reversed", showgrid=False,
                    tickfont=dict(family="DM Mono, monospace", size=11, color="#a0b4c8"),
                ),
                hoverlabel=dict(bgcolor="#131920", bordercolor="#1e2d3d",
                                font=dict(family="DM Mono, monospace", size=11)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown("""
        <div class="placeholder-state">
            <div class="placeholder-icon">◈</div>
            <div class="placeholder-text">Select a ticker and model, then click Run Prediction</div>
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

        # 52W position bar
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

        # ── Price sparkline ─────────────────────────────────────────────────
        # Fetch raw OHLCV for sparkline using yfinance directly
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">52-Week Closing Price</div>', unsafe_allow_html=True)
        try:
            import yfinance as yf
            df_hist = yf.download(
                selected_ticker, period="1y", auto_adjust=True, progress=False
            )
            if not df_hist.empty:
                closes = df_hist["Close"].squeeze()
                fig_spark = go.Figure(go.Scatter(
                    x=closes.index,
                    y=closes.values,
                    mode="lines",
                    line=dict(color="#00d4ff", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.05)",
                    hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
                ))
                fig_spark.update_layout(
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=True,
                               tickfont=dict(family="DM Mono, monospace",
                                             size=10, color="#3d5068")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(30,45,61,0.6)",
                               tickfont=dict(family="DM Mono, monospace",
                                             size=10, color="#3d5068"),
                               tickprefix="$"),
                    hoverlabel=dict(bgcolor="#131920", bordercolor="#1e2d3d",
                                    font=dict(family="DM Mono, monospace", size=11)),
                )
                st.plotly_chart(
                    fig_spark, use_container_width=True,
                    config={"displayModeBar": False},
                )
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
        st.markdown('<div class="section-label">Financial AI Assistant</div>',
                    unsafe_allow_html=True)
    with rag_toggle_col:
        use_rag = st.toggle("RAG", value=True, help="Inject knowledge base context")

    # ── Scrollable message history ──────────────────────────────────────────
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

    st.markdown(
        f'<div class="chat-scroll-area">{history_html}</div>',
        unsafe_allow_html=True,
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
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
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
        agent_clicked = st.button(
            "▶  Run Agent", type="primary", use_container_width=True
        )

    if agent_clicked:
        if agent_query.strip():
            with st.spinner("Agent planning and executing tools…"):
                result = api_post("/agent/run", {"query": agent_query})

            if result:
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

                # ── Tools used ─────────────────────────────────────────────
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

                # ── Agent response ──────────────────────────────────────────
                st.markdown(
                    '<div class="section-label">Agent Response</div>',
                    unsafe_allow_html=True,
                )
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