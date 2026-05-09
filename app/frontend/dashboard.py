"""
FinSight AI — Redesigned Professional Dashboard
Institutional-grade dark terminal aesthetic with refined typography,
animated data cards, and a clean read-only interface.
"""

from __future__ import annotations

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
    "xgboost":            "XGBoost",
    "lightgbm":           "LightGBM",
    "random_forest":      "Random Forest",
    "logistic_regression":"Logistic Regression",
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
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── CSS Variables ─────────────────────────────────────────────────────────── */
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

/* ── Global Reset ─────────────────────────────────────────────────────────── */
.stApp { background: var(--bg-base) !important; }
.main .block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }
html, body, .stApp * { font-family: var(--font-body) !important; color: var(--text-primary); }

/* ── Hide Streamlit chrome ────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}

/* ── Selectbox dropdown ───────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
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
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
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
.stButton [kind="primary"] > button,
button[kind="primary"] {
    background: rgba(0, 212, 255, 0.1) !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
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

/* ── Alerts / Info ────────────────────────────────────────────────────────── */
.stAlert {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stSuccess { border-left: 3px solid var(--accent-green) !important; }
.stWarning { border-left: 3px solid var(--accent-amber) !important; }
.stError   { border-left: 3px solid var(--accent-red)   !important; }
.stInfo    { border-left: 3px solid var(--accent-cyan)  !important; }

/* ── Text Input / Text Area ───────────────────────────────────────────────── */
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

/* ── Spinner ─────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Toggle ──────────────────────────────────────────────────────────────── */
.stToggle label { font-family: var(--font-mono) !important; font-size: 0.8rem !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

/* ── Custom Components ────────────────────────────────────────────────────── */
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
@keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}
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
.signal-card {
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.signal-bullish {
    background: linear-gradient(135deg, rgba(0,230,118,0.08) 0%, rgba(0,230,118,0.03) 100%);
    border: 1px solid rgba(0,230,118,0.3);
}
.signal-bearish {
    background: linear-gradient(135deg, rgba(255,61,87,0.08) 0%, rgba(255,61,87,0.03) 100%);
    border: 1px solid rgba(255,61,87,0.3);
}
.signal-label {
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0;
}
.signal-bullish .signal-label { color: var(--accent-green); }
.signal-bearish .signal-label { color: var(--accent-red); }
.signal-sublabel {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}
.narrative-block {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 6px 6px 0;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text-primary);
    font-family: var(--font-body);
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
.market-stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.market-stat {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.market-stat-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.market-stat-value {
    font-family: var(--font-mono);
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--text-primary);
}
.placeholder-state {
    text-align: center;
    padding: 4rem 2rem;
    border: 1px dashed var(--border-bright);
    border-radius: 10px;
    margin: 2rem 0;
}
.placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.placeholder-text {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
}
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
        st.error("Cannot connect to backend — ensure the API server is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"API error: {detail}")
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
    ("chat_history", []),
    ("last_prediction", None),
    ("market_summary", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Wordmark
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div class="finsight-wordmark">
            <span class="triangle">▲</span> FinSight
        </div>
        <div class="finsight-tagline">Explainable Financial AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    health = api_get("/health")
    if health:
        st.markdown(
            f'<div style="font-family:var(--font-mono);font-size:0.75rem;color:#7a8fa8;margin-bottom:1.5rem;">'
            f'<span class="status-dot status-online"></span>'
            f'API v{health.get("version","?")} &nbsp;·&nbsp; {health.get("environment","?").upper()}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.75rem;color:#7a8fa8;margin-bottom:1.5rem;">'
            '<span class="status-dot status-offline"></span>API OFFLINE</div>',
            unsafe_allow_html=True,
        )

    # ── Instrument Selection ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Instrument</div>', unsafe_allow_html=True)
    selected_ticker = st.selectbox("Ticker", TICKERS, index=0, label_visibility="collapsed")
    custom_ticker = st.text_input("Custom ticker", placeholder="e.g. NFLX").upper().strip()
    if custom_ticker:
        selected_ticker = custom_ticker

    # ── Model Selection ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Model",
        MODELS,
        format_func=lambda m: MODEL_LABELS.get(m, m),
        index=0,
        label_visibility="collapsed",
    )

    # ── RAG Knowledge Base ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
    ingest_text = st.text_area(
        "Add document",
        placeholder="Paste a financial news snippet, earnings summary, or research note...",
        height=90,
        label_visibility="collapsed",
    )
    if st.button("Ingest Document", use_container_width=True):
        if ingest_text.strip():
            result = api_post("/rag/ingest", {"texts": [ingest_text], "source": "user_input"})
            if result:
                st.success(f"Ingested — {result.get('ingested_count', 1)} document(s) added.")
        else:
            st.warning("Enter text before ingesting.")

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="position:fixed;bottom:1.5rem;left:0;width:260px;text-align:center;
                font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">
        Not investment advice · For research use only
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_ticker_badge = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <div style="margin-bottom:0.25rem;">
        <span style="font-family:var(--font-mono);font-size:0.72rem;
                     letter-spacing:0.2em;text-transform:uppercase;
                     color:var(--text-muted);">DASHBOARD</span>
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

st.markdown('<div style="height:1px;background:var(--border);margin:0.5rem 0 1.5rem 0;"></div>',
            unsafe_allow_html=True)


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
        run_clicked = st.button("▶  Run Prediction", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running inference pipeline..."):
            result = api_post("/predict/", {
                "ticker": selected_ticker,
                "model_name": selected_model,
                "use_cache": True,
            })
            if result:
                st.session_state.last_prediction = result

    pred = st.session_state.last_prediction

    if pred:
        label    = pred["prediction_label"]
        prob     = pred["probability"]
        conf     = pred["confidence_label"].upper()
        close    = pred["latest_close"]
        is_bull  = label == "BULLISH"

        # ── Signal + Metrics Row ───────────────────────────────────────────
        c_signal, c_prob, c_conf, c_close = st.columns([2, 1, 1, 1])

        with c_signal:
            card_cls = "signal-bullish" if is_bull else "signal-bearish"
            arrow    = "↑" if is_bull else "↓"
            st.markdown(f"""
            <div class="signal-card {card_cls}">
                <div class="signal-label">{arrow} {label}</div>
                <div class="signal-sublabel">Next-day direction signal</div>
            </div>
            """, unsafe_allow_html=True)

        prob_color = "var(--accent-green)" if is_bull else "var(--accent-red)"
        with c_prob:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:8px;padding:1.2rem 1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.68rem;
                            letter-spacing:0.15em;text-transform:uppercase;
                            color:var(--text-muted);margin-bottom:0.4rem;">Probability</div>
                <div style="font-family:var(--font-mono);font-size:1.6rem;
                            font-weight:500;color:{prob_color};">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        conf_color = {"HIGH": "var(--accent-green)", "MODERATE": "var(--accent-amber)"}.get(conf, "var(--text-secondary)")
        with c_conf:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:8px;padding:1.2rem 1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.68rem;
                            letter-spacing:0.15em;text-transform:uppercase;
                            color:var(--text-muted);margin-bottom:0.4rem;">Confidence</div>
                <div style="font-family:var(--font-mono);font-size:1.6rem;
                            font-weight:500;color:{conf_color};">{conf}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_close:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:8px;padding:1.2rem 1rem;text-align:center;">
                <div style="font-family:var(--font-mono);font-size:0.68rem;
                            letter-spacing:0.15em;text-transform:uppercase;
                            color:var(--text-muted);margin-bottom:0.4rem;">Last Close</div>
                <div style="font-family:var(--font-mono);font-size:1.6rem;
                            font-weight:500;color:var(--text-primary);">${close:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # ── Narrative ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Model Reasoning</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="narrative-block">{pred["narrative"]}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        # ── SHAP Chart ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">SHAP Feature Attribution</div>', unsafe_allow_html=True)

        features = pred.get("top_features", [])
        if features:
            df_shap = pd.DataFrame(features)
            colors = [
                "rgba(0,230,118,0.85)" if v > 0 else "rgba(255,61,87,0.85)"
                for v in df_shap["shap_value"]
            ]
            fig = go.Figure(go.Bar(
                x=df_shap["shap_value"],
                y=df_shap["feature"],
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                ),
                text=[f"{v:+.4f}" for v in df_shap["shap_value"]],
                textfont=dict(family="DM Mono, monospace", size=11, color="#7a8fa8"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                margin=dict(l=10, r=60, t=10, b=10),
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(30,45,61,0.8)",
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor="rgba(58,80,107,0.9)",
                    zerolinewidth=1.5,
                    tickfont=dict(family="DM Mono, monospace", size=10),
                    title=dict(text="SHAP Value  (← bearish  ·  bullish →)",
                               font=dict(size=10, color="#3d5068")),
                ),
                yaxis=dict(
                    autorange="reversed",
                    showgrid=False,
                    tickfont=dict(family="DM Mono, monospace", size=11, color="#a0b4c8"),
                ),
                hoverlabel=dict(
                    bgcolor="#131920",
                    bordercolor="#1e2d3d",
                    font=dict(family="DM Mono, monospace", size=11),
                ),
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
        with st.spinner("Fetching market data..."):
            summary = api_post("/market/summary", {
                "ticker": selected_ticker,
                "period_years": 1,
            })
            if summary:
                st.session_state.market_summary = summary

    mkt = st.session_state.market_summary

    if mkt and mkt.get("ticker") == selected_ticker:
        st.markdown('<div class="section-label">12-Month Summary</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="market-stat-row">
            <div class="market-stat">
                <div class="market-stat-label">Trading Days</div>
                <div class="market-stat-value">{mkt['rows']}</div>
            </div>
            <div class="market-stat">
                <div class="market-stat-label">52W Low</div>
                <div class="market-stat-value" style="color:var(--accent-red);">
                    ${mkt['close_min']:,.2f}
                </div>
            </div>
            <div class="market-stat">
                <div class="market-stat-label">52W High</div>
                <div class="market-stat-value" style="color:var(--accent-green);">
                    ${mkt['close_max']:,.2f}
                </div>
            </div>
            <div class="market-stat">
                <div class="market-stat-label">Mean Close</div>
                <div class="market-stat-value">${mkt['close_mean']:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 52W range bar
        pct = (mkt["close_mean"] - mkt["close_min"]) / max(
            mkt["close_max"] - mkt["close_min"], 0.01
        ) * 100
        st.markdown('<div class="section-label">52-Week Price Position</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:8px;padding:1.2rem 1.5rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.6rem;">
                <span style="font-family:var(--font-mono);font-size:0.75rem;
                             color:var(--accent-red);">${mkt['close_min']:,.2f}</span>
                <span style="font-family:var(--font-mono);font-size:0.75rem;
                             color:var(--text-muted);">Mean ${mkt['close_mean']:,.2f}</span>
                <span style="font-family:var(--font-mono);font-size:0.75rem;
                             color:var(--accent-green);">${mkt['close_max']:,.2f}</span>
            </div>
            <div style="background:var(--bg-elevated);border-radius:4px;height:6px;position:relative;">
                <div style="position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;
                            background:linear-gradient(90deg,var(--accent-red),var(--accent-cyan));
                            border-radius:4px;"></div>
                <div style="position:absolute;left:{pct:.1f}%;top:-3px;
                            width:12px;height:12px;border-radius:50%;
                            background:var(--accent-cyan);
                            box-shadow:0 0 8px var(--accent-cyan);
                            transform:translateX(-50%);"></div>
            </div>
            <div style="font-family:var(--font-mono);font-size:0.7rem;
                        color:var(--text-muted);text-align:right;margin-top:0.5rem;">
                Mean at {pct:.1f}th percentile of 52W range
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);
                    margin-top:1rem;text-align:right;">
            {mkt['start_date']} → {mkt['end_date']}
            &nbsp;·&nbsp; {len(mkt.get('columns', []))} columns
            &nbsp;·&nbsp; {mkt['null_count']} nulls
        </div>
        """, unsafe_allow_html=True)

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

    # Message history
    chat_area = st.container()
    with chat_area:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem 1rem;
                        font-family:var(--font-mono);font-size:0.78rem;
                        color:var(--text-muted);letter-spacing:0.05em;">
                Ask anything about financial markets, indicators, or model predictions.
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-bubble-user">'
                        f'<div class="chat-role">You</div>{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-bubble-ai">'
                        f'<div class="chat-role">FinSight AI</div>{msg["content"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

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
            result = api_post("/rag/chat", {"query": user_input, "use_rag": use_rag})
            if result:
                st.session_state.chat_history.append({
                    "role": "assistant",
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
    st.markdown("""
    <div style="font-family:var(--font-body);font-size:0.875rem;color:var(--text-secondary);
                margin-bottom:1.5rem;line-height:1.6;max-width:680px;">
        The agent autonomously plans, selects, and chains tools — prediction,
        SHAP explanation, sentiment analysis, and knowledge retrieval — to answer
        complex multi-step financial queries.
    </div>
    """, unsafe_allow_html=True)

    EXAMPLES = [
        "Predict AAPL and explain the key drivers behind the signal.",
        "What is the sentiment of this headline: 'Fed signals rate pause as inflation cools'?",
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
        placeholder="Ask a multi-step financial question...",
        height=90,
        label_visibility="collapsed",
    )

    run_agent_col, _ = st.columns([2, 6])
    with run_agent_col:
        agent_clicked = st.button("▶  Run Agent", type="primary", use_container_width=True)

    if agent_clicked:
        if agent_query.strip():
            with st.spinner("Agent planning and executing tools..."):
                result = api_post("/agent/run", {"query": agent_query})

            if result:
                st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Agent Response</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="narrative-block">{result["response"]}</div>',
                    unsafe_allow_html=True,
                )

                if result.get("tools_used"):
                    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Tools Invoked</div>',
                                unsafe_allow_html=True)
                    chips = "".join(
                        f'<span class="tool-chip">{t}</span>'
                        for t in result["tools_used"]
                    )
                    st.markdown(f'<div style="margin-top:0.3rem;">{chips}</div>',
                                unsafe_allow_html=True)

                with st.expander("Raw tool results"):
                    import json
                    st.code(
                        json.dumps(result.get("tool_results", []), indent=2),
                        language="json",
                    )
        else:
            st.warning("Enter a query to run the agent.")