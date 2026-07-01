"""
Design note — plain-language first, technical detail on demand
-----------------------------------------------------------------
The underlying analysis (mean-variance optimization, Ledoit-Wolf
shrinkage, risk attribution, VaR, the efficient frontier...) is
unavoidably technical, but most users opening this tab just want two
questions answered: "how risky is this?" and "should I change anything?"
This module surfaces those two answers in plain language up front —
a risk level, a risk-adjusted return score, a worst-case-loss estimate,
and a per-ticker buy/trim/hold table — and tucks the correlation matrix,
risk attribution breakdowns, efficient frontier, and raw numbers into a
single "Technical details" expander for the minority of users who want
them. All of the original chart/table content is still here; nothing
was removed, just re-organized and given plain-language framing.

Implements its own lightweight ``requests`` networking (rather than
importing dashboard.py's ``api_post`` helper) to avoid a circular import —
the same pattern already used by ``app/frontend/streaming_signal.py``.
Relies on the CSS classes (``.section-label``, ``.placeholder-state``,
etc.) injected once by dashboard.py's global ``<style>`` block, since
this module is only ever rendered inside a tab of that same page.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

_DEFAULT_TICKERS = "AAPL, MSFT, GOOGL, AMZN, NVDA"

# Action-table threshold: a recommended-vs-current weight gap smaller than
# this is treated as "Hold" rather than nudging the user to make a trade
# over what's likely just optimizer noise.
_HOLD_BAND = 0.03


# ─────────────────────────────────────────────────────────────────────────────
# Networking
# ─────────────────────────────────────────────────────────────────────────────


def _build_headers(api_key: Optional[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _analyze_portfolio(
    api_base: str,
    api_key: Optional[str],
    payload: dict,
    timeout: int = 600,  # raised from 180 s — cold-start training
) -> Optional[dict]:  # for 3+ tickers can easily take 2–3 min
    url = f"{api_base.rstrip('/')}/portfolio/analyze"
    try:
        resp = requests.post(
            url, json=payload, headers=_build_headers(api_key), timeout=timeout
        )
        if resp.status_code == 422:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Invalid portfolio configuration: {detail}")
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {api_base}. Check that the server is running.")
    except requests.exceptions.Timeout:
        payload.get("n_tickers_hint", "?")
        st.error(
            f"The analysis timed out after {timeout // 60} min. "
            f"This usually happens on the first run for a ticker, when the "
            f"model has to be trained from scratch. "
            f"Try enabling **Quick analysis** (no AI signal) in the settings "
            f"above for an instant result, then re-run with AI signals "
            f"once the models are cached."
        )
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"API error {exc.response.status_code}: {detail}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Input parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_positions_table(df: pd.DataFrame) -> list[dict]:
    positions: list[dict] = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue
        pos: dict = {"ticker": ticker}
        shares = row.get("Shares")
        if shares not in (None, "", 0):
            try:
                pos["shares"] = float(shares)
            except (TypeError, ValueError):
                pass
        positions.append(pos)
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language translation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _risk_level(annual_volatility: float) -> tuple[str, str]:
    """Maps annualized volatility to a plain risk label + traffic-light dot."""
    if annual_volatility < 0.15:
        return "Low", "🟢"
    if annual_volatility < 0.30:
        return "Moderate", "🟡"
    return "High", "🔴"


def _sharpe_descriptor(sharpe: float) -> str:
    """
    Plain-language read on the Sharpe ratio (return earned per unit of risk
    taken). Thresholds follow the common rule of thumb: >2 excellent, >1
    good, >0.5 average, otherwise below average/poor.
    """
    if sharpe >= 2.0:
        return "Excellent"
    if sharpe >= 1.0:
        return "Good"
    if sharpe >= 0.5:
        return "Average"
    if sharpe >= 0:
        return "Below average"
    return "Poor"


def _action_for_gap(current: float, optimal: float) -> str:
    gap = optimal - current
    if gap > _HOLD_BAND:
        return "Buy more"
    if gap < -_HOLD_BAND:
        return "Trim"
    return "Hold"


def _style_action_column(val: str) -> str:
    color = {"Buy more": "#00e676", "Trim": "#ff3d57", "Hold": "#7a8fa8"}.get(val, "")
    return f"color: {color}; font-weight: 600;"


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations (technical — used inside the "Technical details" expander)
# ─────────────────────────────────────────────────────────────────────────────


def _render_correlation_heatmap(
    correlation_matrix: dict[str, dict[str, float]],
) -> None:
    tickers = list(correlation_matrix.keys())
    z = [[correlation_matrix[row][col] for col in tickers] for row in tickers]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            colorscale="RdBu",
            reversescale=True,  # high positive -> deep red, negative -> deep blue
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(family="DM Mono, monospace", size=10),
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(
                title="ρ", tickfont=dict(family="DM Mono, monospace", size=10)
            ),
        )
    )
    fig.update_layout(
        height=max(320, 40 * len(tickers)),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_risk_attribution(risk_attribution: list[dict], title: str) -> None:
    if not risk_attribution:
        st.caption("No risk attribution data available.")
        return
    df = pd.DataFrame(risk_attribution).sort_values("pct_of_total_risk", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["pct_of_total_risk"] * 100,
            y=df["ticker"],
            orientation="h",
            marker=dict(color="rgba(0,212,255,0.85)", line=dict(width=0)),
            text=[f"{v * 100:.1f}%" for v in df["pct_of_total_risk"]],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=11, color="#7a8fa8"),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of total risk<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=title, font=dict(size=12, color="#7a8fa8", family="DM Mono, monospace")
        ),
        margin=dict(l=10, r=40, t=40, b=10),
        height=max(220, 36 * len(df)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
        xaxis=dict(
            title="% of total portfolio risk",
            showgrid=True,
            gridcolor="rgba(30,45,61,0.8)",
        ),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_efficient_frontier(
    frontier: list[dict],
    current_point: tuple[float, float],
    optimal_point: tuple[float, float],
) -> None:
    if not frontier:
        st.caption(
            "Efficient frontier unavailable (degenerate expected-return inputs)."
        )
        return
    df = pd.DataFrame(frontier)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["volatility"],
            y=df["expected_return"],
            mode="lines+markers",
            name="Efficient frontier",
            line=dict(color="#00d4ff", width=2),
            marker=dict(size=5, color="#00d4ff"),
            hovertemplate="Vol: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_point[0]],
            y=[current_point[1]],
            mode="markers",
            name="Current portfolio",
            marker=dict(size=14, color="#ffc107", symbol="diamond"),
            hovertemplate="Current<br>Vol: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[optimal_point[0]],
            y=[optimal_point[1]],
            mode="markers",
            name="Optimal portfolio",
            marker=dict(size=14, color="#00e676", symbol="star"),
            hovertemplate="Optimal<br>Vol: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title="Annualized volatility", showgrid=True, gridcolor="rgba(30,45,61,0.8)"
        ),
        yaxis=dict(
            title="Expected return (proxy)",
            showgrid=True,
            gridcolor="rgba(30,45,61,0.8)",
        ),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_sector_pie(sector_exposure: dict[str, float]) -> None:
    if not sector_exposure:
        st.caption("Sector exposure unavailable.")
        return
    labels = list(sector_exposure.keys())
    values = list(sector_exposure.values())

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(family="DM Mono, monospace", size=11),
            marker=dict(
                colors=[
                    "#00d4ff",
                    "#00e676",
                    "#ffc107",
                    "#ff3d57",
                    "#b388ff",
                    "#7a8fa8",
                    "#3d5068",
                    "#0d8bb0",
                    "#0a9e54",
                    "#c98f00",
                ]
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#7a8fa8", size=11),
        showlegend=True,
        legend=dict(orientation="v"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language sections (main view)
# ─────────────────────────────────────────────────────────────────────────────


def _render_summary_metrics(result: dict) -> None:
    risk_label, risk_dot = _risk_level(result["current_portfolio_volatility"])
    sharpe_label = _sharpe_descriptor(result["sharpe_ratio"])
    var = result["var"]

    portfolio_value = result.get("_portfolio_value_for_display")
    if portfolio_value:
        loss_value = f"${var['historical_var_value']:,.0f}"
    else:
        loss_value = f"{var['historical_var_pct']:.1%} of value"

    chance_of_worse = 1 - var["confidence"]

    sharpe_color = (
        "#00e676"
        if result["sharpe_ratio"] >= 1.0
        else "#ffc107"
        if result["sharpe_ratio"] >= 0.5
        else "#ff3d57"
    )
    risk_colors = {"Low": "#00e676", "Moderate": "#ffc107", "High": "#ff3d57"}
    risk_color = risk_colors.get(risk_label, "#7a8fa8")

    st.markdown(
        f"""
        <div class="market-stat-row">
          <div class="market-stat">
            <div class="market-stat-label">Risk Level</div>
            <div class="market-stat-value" style="color:{risk_color};">
              {risk_dot} {risk_label}
            </div>
            <div style="font-size:0.72rem;color:var(--text-secondary);margin-top:0.25rem;">
              {result["current_portfolio_volatility"]:.1%} annualised volatility
            </div>
          </div>
          <div class="market-stat">
            <div class="market-stat-label">Risk-Adjusted Return</div>
            <div class="market-stat-value" style="color:{sharpe_color};">
              {sharpe_label}
            </div>
            <div style="font-size:0.72rem;color:var(--text-secondary);margin-top:0.25rem;">
              Sharpe ratio {result["sharpe_ratio"]:.2f} &nbsp;·&nbsp;
              &gt;1.0 good, &gt;2.0 excellent
            </div>
          </div>
          <div class="market-stat">
            <div class="market-stat-label">Potential Loss (worst case)</div>
            <div class="market-stat-value">{loss_value}</div>
            <div style="font-size:0.72rem;color:var(--text-secondary);margin-top:0.25rem;">
              ~{chance_of_worse:.0%} chance of exceeding this
              over {var["horizon_days"]} day(s)
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_action_table(result: dict, portfolio_value: float) -> None:
    st.markdown(
        '<div class="section-label">What Should You Do?</div>', unsafe_allow_html=True
    )
    st.caption(
        "Comparing what you hold now to the mix that best balances risk and "
        'return, given your settings. Small differences are left as "Hold" '
        "since they're not worth the cost/effort of trading."
    )

    rows = []
    for t in result["tickers"]:
        current = result["current_weights"].get(t, 0.0)
        optimal = result["optimal_weights"].get(t, 0.0)
        row = {
            "Ticker": t,
            "Current %": current,
            "Recommended %": optimal,
            "Action": _action_for_gap(current, optimal),
        }
        if portfolio_value:
            row["Current $"] = current * portfolio_value
            row["Recommended $"] = optimal * portfolio_value
        rows.append(row)

    df = pd.DataFrame(rows)
    fmt = {"Current %": "{:.1%}", "Recommended %": "{:.1%}"}
    if portfolio_value:
        fmt["Current $"] = "${:,.0f}"
        fmt["Recommended $"] = "${:,.0f}"

    styled = df.style.format(fmt).map(_style_action_column, subset=["Action"])
    st.dataframe(styled, width="stretch", hide_index=True)


def _render_predictions_table(predictions: list[dict]) -> None:
    if not predictions:
        return
    label_map = {
        "BULLISH": "📈 Likely to rise",
        "BEARISH": "📉 Likely to fall",
        "UNKNOWN": "❓ No signal",
    }
    rows = [
        {
            "Ticker": p["ticker"],
            "AI Outlook": label_map.get(p["prediction_label"], p["prediction_label"]),
            "Confidence": p["confidence_label"].title(),
        }
        for p in predictions
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "This is a directional signal from the prediction model — a "
        "ranking of which stocks look relatively stronger, not a return "
        "forecast or financial advice."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def render_portfolio_tab(api_base: str, api_key: Optional[str] = None) -> None:
    st.markdown(
        '<div class="section-label">Portfolio Construction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:var(--font-body);font-size:0.875rem;'
        "color:var(--text-secondary);margin-bottom:1.25rem;line-height:1.6;"
        'max-width:760px;">'
        "Add the stocks you hold (or are considering) below. We'll estimate "
        "how risky the mix is, whether it could be better balanced, and "
        "what — if anything — you might want to change.</div>",
        unsafe_allow_html=True,
    )

    if "portfolio_positions_df" not in st.session_state:
        default_tickers = [t.strip() for t in _DEFAULT_TICKERS.split(",")]
        # Explicit dtypes are important: without them Streamlit's internal
        # added_rows / edited_rows diff can infer a third column from dtype
        # metadata mismatches when the base DataFrame columns aren't exactly
        # what the editor expects (e.g. object vs str for Ticker, int vs
        # float for Shares).
        st.session_state.portfolio_positions_df = pd.DataFrame(
            {
                "Ticker": pd.array(default_tickers, dtype="object"),
                "Shares": pd.array([0.0] * len(default_tickers), dtype="float64"),
            }
        ).reset_index(drop=True)  # guarantee a clean 0-based RangeIndex;
        # a non-contiguous index (e.g. [0,2,3] after a delete+rerun cycle)
        # is the direct cause of the intermittent ghost column — Streamlit
        # renders non-standard indices as a visible unnamed column when
        # hide_index is not set.

    # IMPORTANT: do NOT reassign st.session_state.portfolio_positions_df to
    # the returned edited_df here.  st.data_editor with a fixed key tracks its
    # own incremental edits (added_rows / edited_rows / deleted_rows) in
    # st.session_state["portfolio_editor"] internally.  Feeding the merged
    # result back as the new base on every rerun creates a conflict: on the
    # first rerun after an edit the widget sees "base already contains my edit"
    # AND "I still have that edit in my internal state", so it either drops the
    # row or only commits it on the second rerun.  Leaving the base dataframe
    # alone and letting the widget own its incremental state fixes both the
    # missing-ticker and missing-shares symptoms entirely.
    edited_df = st.data_editor(
        st.session_state.portfolio_positions_df,
        num_rows="dynamic",
        width="stretch",
        key="portfolio_editor",
        hide_index=True,  # suppresses the index column unconditionally —
        # belt-and-suspenders guard so even if the backing DataFrame's index
        # ever becomes non-standard (e.g. after an unexpected rerun cycle),
        # it is never exposed to the user as a mysterious extra column.
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", required=True),
            "Shares": st.column_config.NumberColumn(
                "Shares", min_value=0.0, step=1.0, help="Leave 0 for equal-weight"
            ),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        portfolio_value = st.number_input(
            "Total portfolio value ($, optional)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Add this to see dollar amounts instead of just percentages.",
        )
    with col_b:
        horizon_label_map = {"1d": "Next Day", "7d": "Next Week", "1m": "Next Month"}
        horizon = st.selectbox(
            "Time horizon for the AI signal",
            options=list(horizon_label_map.keys()),
            format_func=lambda k: horizon_label_map[k],
        )

    with st.expander("⚙ Advanced settings (optional)"):
        st.caption(
            "Most people can leave these as-is. They only matter if you want "
            "to enforce specific diversification rules."
        )
        include_predictions = st.toggle(
            "Include AI signal (recommended)",
            value=True,
            help=(
                "When on, the AI predicts which stocks look relatively "
                "stronger and uses that to guide the recommended mix. "
                "**First run per ticker trains the model from scratch and "
                "can take 1–2 minutes.** Turn this off for an instant "
                "risk/correlation result without the AI signal — you can "
                "always turn it back on once the models are cached."
            ),
        )
        if include_predictions:
            st.info(
                "⏱ **First run note:** if these are new tickers, the AI "
                "model needs to be trained first. This takes 1–3 minutes "
                "per ticker. Subsequent runs are fast (under 10 s total).",
                icon=None,
            )
        col_d, col_e = st.columns(2)
        with col_d:
            max_position_weight = st.slider(
                "Max % in any single stock",
                min_value=0.05,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Caps how concentrated the recommended mix can be in "
                "one stock. Default allows up to 50%.",
            )
            long_only = st.checkbox(
                "Long-only (no short selling)",
                value=True,
                help="Keep this checked unless you specifically trade short positions.",
            )
        with col_e:
            max_sector_weight = st.slider(
                "Max % in any single industry/sector",
                min_value=0.10,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="No cap by default, since holding several stocks from the "
                "same industry (e.g. a few tech names) is perfectly normal. "
                "Lower this only if you want to force spreading across industries.",
            )
            var_confidence = st.selectbox(
                "Worst-case estimate confidence",
                options=[0.95, 0.99],
                index=0,
                format_func=lambda v: f"{v:.0%}",
                help="95% is the standard choice — a tighter (99%) estimate "
                "looks at a rarer, more extreme scenario.",
            )

    run_col, _ = st.columns([2, 6])
    with run_col:
        run_clicked = st.button("▶  Analyze Portfolio", type="primary", width="stretch")

    if run_clicked:
        positions = _parse_positions_table(edited_df)
        if len(positions) < 2:
            st.warning("Add at least 2 positions to analyze a portfolio.")
        else:
            payload = {
                "positions": positions,
                "horizon": horizon,
                "max_position_weight": max_position_weight,
                "max_sector_weight": max_sector_weight,
                "long_only": long_only,
                "var_confidence": var_confidence,
                "include_predictions": include_predictions,
                "n_tickers_hint": len(positions),  # used only for timeout msg
            }
            if portfolio_value > 0:
                payload["portfolio_value"] = portfolio_value

            if include_predictions:
                spinner_msg = (
                    f"Training AI signals for {len(positions)} ticker(s) and "
                    f"running portfolio optimization…  "
                    f"*(first run per ticker takes 1–3 min — subsequent runs "
                    f"are fast once models are cached)*"
                )
            else:
                spinner_msg = "Running portfolio optimization…"

            with st.spinner(spinner_msg):
                result = _analyze_portfolio(api_base, api_key, payload)

            if result:
                result["_portfolio_value_for_display"] = portfolio_value
                st.session_state.portfolio_result = result

    result = st.session_state.get("portfolio_result")

    if not result:
        st.markdown(
            """
            <div class="placeholder-state">
                <div class="placeholder-icon">◈</div>
                <div class="placeholder-text">Enter positions above and click Analyze Portfolio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)

    portfolio_value_for_display = result.get("_portfolio_value_for_display", 0.0)

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    _render_summary_metrics(result)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    _render_action_table(result, portfolio_value_for_display)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Where Your Money Is Invested</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Spreading investments across industries reduces the chance that "
        "one bad sector day takes your whole portfolio down with it."
    )
    _render_sector_pie(result["sector_exposure"])

    if result.get("predictions"):
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-label">AI Outlook Per Stock</div>',
            unsafe_allow_html=True,
        )
        _render_predictions_table(result["predictions"])

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    with st.expander("📐 Technical details (correlation, risk breakdown, methodology)"):
        st.caption(
            "Everything below is the same analysis shown above, just broken "
            "down into the underlying statistics for anyone who wants to dig "
            "deeper."
        )

        st.markdown("**Correlation matrix**")
        st.caption(
            "How closely each pair of stocks tends to move together. Close "
            "to 1 = they move in lockstep (less diversification benefit); "
            "close to 0 = unrelated; negative = they tend to move in "
            "opposite directions (good for smoothing out swings)."
        )
        _render_correlation_heatmap(result["correlation_matrix"])

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown("**What's driving the risk?**")
        st.caption(
            "Not the same as how much money is in each stock — a smaller, "
            "more volatile or more correlated position can still account "
            "for a large share of overall portfolio risk."
        )
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            _render_risk_attribution(
                result["current_risk_attribution"], "Current allocation"
            )
        with col_risk2:
            _render_risk_attribution(
                result["optimal_risk_attribution"], "Recommended allocation"
            )

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown("**Risk vs. return tradeoff (efficient frontier)**")
        st.caption(
            "Each point on the curve is the best possible return achievable "
            "for a given level of risk, using only these stocks. The diamond "
            "is where you are now; the star is the recommended mix."
        )
        _render_efficient_frontier(
            result["efficient_frontier"],
            current_point=(
                result["current_portfolio_volatility"],
                result["current_expected_return"],
            ),
            optimal_point=(
                result["optimal_portfolio_volatility"],
                result["expected_return"],
            ),
        )

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        st.markdown("**Raw numbers**")
        weights_df = pd.DataFrame(
            {
                "Current weight": result["current_weights"],
                "Optimal weight": result["optimal_weights"],
            }
        ).fillna(0.0)
        st.dataframe(weights_df.style.format("{:.2%}"), width="stretch")
        st.caption(
            f"Covariance estimation method: {result['covariance_method'].replace('_', ' ')} "
            f"· Lookback: {result['lookback_days']} trading days · "
            "Constraints applied: " + ", ".join(result.get("constraints_applied", []))
        )
