"""
app/frontend/streaming_signal.py
=================================

Drop-in streaming replacement for the "Signal" tab's Analyse button.

Uses raw SSE line parsing instead of sseclient-py so there is no
dependency on a third-party SSE library and no version-mismatch issues.

Usage — in dashboard.py, replace the tab_predict run button block:

    from app.frontend.streaming_signal import render_streaming_signal_tab

    with tab_predict:
        render_streaming_signal_tab(
            selected_ticker=selected_ticker,
            selected_horizon=selected_horizon,
            horizon_label=horizon_label,
            api_base=API_BASE,
            api_key=_API_KEY,
        )

    # The rest of the tab_predict block is UNCHANGED —
    # it still reads st.session_state.last_prediction as before.
    pred = st.session_state.last_prediction
    ...
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def render_streaming_signal_tab(
    selected_ticker: str,
    selected_horizon: str,
    horizon_label: str,
    api_base: str,
    api_key: Optional[str] = None,
) -> None:
    """
    Render the Signal tab run button with streaming progress support.

    Populates ``st.session_state.last_prediction`` exactly as the
    original synchronous button did, so the downstream rendering block
    in dashboard.py requires zero changes.
    """
    run_col, _ = st.columns([2, 6])
    with run_col:
        run_clicked = st.button("▶  Analyse Signal", type="primary", width="stretch")

    if run_clicked:
        st.session_state.last_prediction = None
        _run_streaming_prediction(
            ticker=selected_ticker,
            horizon=selected_horizon,
            api_base=api_base,
            api_key=api_key,
        )

    _render_stale_guard(selected_ticker, selected_horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Stage labels
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_LABELS: dict[str, str] = {
    "ingest": "Fetching market data",
    "features": "Engineering features",
    "training": "Training models",
    "model_select": "Selecting best model",
    "model_load": "Loading model artifact",
    "inference": "Running inference",
    "shap": "Running SHAP analysis",
    "news": "Retrieving news intelligence",
    "fusion": "LLM signal fusion",
    "complete": "Complete",
}


# ─────────────────────────────────────────────────────────────────────────────
# Raw SSE parser — no third-party library required
# ─────────────────────────────────────────────────────────────────────────────


def _iter_sse_events(response) -> Iterator[str]:
    """
    Yield the ``data:`` payload of each SSE event from a streaming
    requests.Response.

    SSE wire format per spec:
        data: <json payload>\n
        \n                        ← blank line terminates the event

    Only ``data:`` lines are yielded; ``id:``, ``event:``, ``retry:``
    and comment lines (``:``) are silently skipped.
    """
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            yield line[len("data:") :].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming fetch — pure requests, no sseclient
# ─────────────────────────────────────────────────────────────────────────────


def _run_streaming_prediction(
    ticker: str,
    horizon: str,
    api_base: str,
    api_key: Optional[str],
) -> None:
    import requests

    url = f"{api_base.rstrip('/')}/predict/stream"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    body = {"ticker": ticker, "horizon": horizon, "use_cache": True}

    # UI placeholders updated in-place as events arrive
    status_ph = st.empty()
    progress_ph = st.empty()

    status_ph.markdown(
        '<div style="font-family:var(--font-mono);font-size:0.8rem;'
        'color:var(--text-secondary);">Connecting to prediction pipeline…</div>',
        unsafe_allow_html=True,
    )

    try:
        with requests.post(
            url,
            json=body,
            headers=headers,
            stream=True,  # keep the socket open for SSE
            timeout=180,
        ) as resp:
            if resp.status_code == 401:
                status_ph.error("Authentication required. Check FINSIGHT_API_KEY.")
                progress_ph.empty()
                return

            if resp.status_code != 200:
                status_ph.error(f"API error {resp.status_code}: {resp.text[:300]}")
                progress_ph.empty()
                return

            # Iterate the raw SSE stream without sseclient
            for data_str in _iter_sse_events(resp):
                # Stream sentinel
                if data_str == "[DONE]":
                    break

                # Skip empty keep-alive lines
                if not data_str:
                    continue

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                data = event.get("data", {})

                if event_type == "progress":
                    pct = max(0, min(100, int(data.get("pct", 0))))
                    stage = data.get("stage", "")
                    message = data.get("message", "")
                    label = _STAGE_LABELS.get(stage, stage)

                    status_ph.markdown(
                        f'<div style="font-family:var(--font-mono);font-size:0.78rem;'
                        f'color:var(--text-secondary);padding:0.35rem 0;">'
                        f'<span style="color:var(--accent-cyan);">◈</span> '
                        f"<strong>{label}</strong>"
                        f"{'  —  ' + message if message else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    progress_ph.progress(pct / 100)

                elif event_type == "result":
                    # Populate the same session-state key the original sync
                    # button used — downstream rendering is unchanged.
                    st.session_state.last_prediction = data
                    status_ph.empty()
                    progress_ph.empty()
                    break

                elif event_type == "error":
                    status_ph.error(
                        f"Prediction failed: {data.get('message', 'Unknown error')}"
                    )
                    progress_ph.empty()
                    return

    except requests.exceptions.ConnectionError:
        status_ph.error(
            f"Cannot reach API at {api_base}. "
            "Check that the server is running and FRONTEND_API_BASE is correct."
        )
        progress_ph.empty()

    except requests.exceptions.Timeout:
        status_ph.error(
            "Request timed out. The server may be training models for the "
            "first time — this can take several minutes. Please try again."
        )
        progress_ph.empty()

    except Exception as exc:
        status_ph.error(f"Streaming error: {exc}")
        progress_ph.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Stale-result guard
# ─────────────────────────────────────────────────────────────────────────────


def _render_stale_guard(selected_ticker: str, selected_horizon: str) -> None:
    """
    Show a placeholder when no prediction exists or the cached result
    belongs to a different ticker/horizon.  The actual prediction result
    rendering lives in dashboard.py's tab_predict block and is untouched.
    """
    pred = st.session_state.get("last_prediction")

    if not pred:
        st.markdown(
            """
            <div class="placeholder-state">
                <div class="placeholder-icon">◈</div>
                <div class="placeholder-text">Select a ticker and horizon,
                then click Analyse Signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if pred.get("ticker") != selected_ticker or pred.get("horizon") != selected_horizon:
        st.markdown(
            """
            <div class="placeholder-state">
                <div class="placeholder-icon">◈</div>
                <div class="placeholder-text">Ticker or horizon changed —
                click Analyse Signal to refresh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
