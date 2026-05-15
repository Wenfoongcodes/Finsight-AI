"""
FinSight AI — Canonical Output Formatting Contract
===================================================

Single source of truth for every numeric, textual, and structural formatting
decision made anywhere in the pipeline.  All modules that produce user-visible
output (narratives, API responses, agent tool results, training summaries)
MUST import constants and helpers from here instead of inlining literals.

Design rationale
----------------
Scattered ``round(x, 4)`` calls, ``f"{p:.1%}"`` one-liners, and hand-rolled
narrative templates create silent inconsistencies that are invisible in unit
tests but jarring in production:

* The probability shown in the SHAP narrative might read "72.3 %" while the
  API JSON field shows ``0.7234`` and the dashboard badge shows "72%".
* Training result dicts use ``round(..., 4)`` but fold metrics are logged at
  ``:.3f``, so the same AUC appears differently in logs vs the leaderboard.
* Timestamps from different sub-systems use different ISO-8601 variants
  (with/without microseconds, with/without timezone suffix).

Centralising here means:

1. A single PR changes the precision everywhere.
2. ``mypy`` / ``ruff`` catch callers that bypass the contract.
3. New modules default to correct formatting by importing one symbol.

Sections
--------
NUMERIC_PRECISION   — canonical decimal places for every metric class
FLOAT_FORMATS       — ``format()``-compatible strings for f-strings
NARRATIVE_TEMPLATES — parameterised string templates for all narratives
Timestamp helpers   — UTC-aware ISO-8601 generation
Formatter helpers   — thin wrappers that apply the constants
"""

from __future__ import annotations

from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# Numeric precision constants
# ─────────────────────────────────────────────────────────────────────────────

# Probabilities (0–1 floats displayed as percentages)
PROB_DECIMAL_PLACES: int = 1  # "72.3%"  — one decimal in percent form
PROB_RAW_DECIMAL_PLACES: int = 4  # 0.7234   — four decimals in raw form

# Financial metrics (AUC, accuracy, F1, etc.)
METRIC_DECIMAL_PLACES: int = 4  # 0.7234

# Price values (close, VWAP, etc.)
PRICE_DECIMAL_PLACES: int = 2  # $185.50

# SHAP values — signed attribution scores
SHAP_DECIMAL_PLACES: int = 4  # +0.0312

# Feature snapshot values stored at inference time
FEATURE_DECIMAL_PLACES: int = 4  # 67.1234

# Sentiment / fusion scores (−1 … +1)
SENTIMENT_DECIMAL_PLACES: int = 3  # 0.312

# News article weights
WEIGHT_DECIMAL_PLACES: int = 3  # 0.875

# Training duration (seconds)
DURATION_DECIMAL_PLACES: int = 2  # 12.34 s

# ─────────────────────────────────────────────────────────────────────────────
# Float format strings  (used with format(value, FMT) or f"{value:{FMT}}")
# ─────────────────────────────────────────────────────────────────────────────

FMT_PROB_PCT: str = f".{PROB_DECIMAL_PLACES}%"  # "72.3%"
FMT_PROB_RAW: str = f".{PROB_RAW_DECIMAL_PLACES}f"  # "0.7234"
FMT_METRIC: str = f".{METRIC_DECIMAL_PLACES}f"  # "0.7234"
FMT_PRICE: str = f",.{PRICE_DECIMAL_PLACES}f"  # "185.50" (comma-thousands)
FMT_SHAP: str = f"+.{SHAP_DECIMAL_PLACES}f"  # "+0.0312"
FMT_SHAP_ABS: str = f".{SHAP_DECIMAL_PLACES}f"  # "0.0312"
FMT_SENTIMENT: str = f"+.{SENTIMENT_DECIMAL_PLACES}f"  # "+0.312"
FMT_WEIGHT: str = f".{WEIGHT_DECIMAL_PLACES}f"  # "0.875"
FMT_DURATION: str = f".{DURATION_DECIMAL_PLACES}f"  # "12.34"

# ─────────────────────────────────────────────────────────────────────────────
# Confidence label thresholds
# (shared between explainability.py and prediction_service.py)
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_HIGH_DELTA: float = 0.15  # |p_bull − 0.5| > this → HIGH
CONFIDENCE_MODERATE_DELTA: float = 0.05  # |p_bull − 0.5| > this → MODERATE
# else LOW


def confidence_label(p_bullish: float) -> str:
    """
    Return 'high' | 'moderate' | 'low' from a calibrated P(bullish).

    Single implementation — eliminates the duplicated threshold logic that
    previously existed in both ``prediction_service.py`` and
    ``explainability.py``.
    """
    delta = abs(p_bullish - 0.5)
    if delta > CONFIDENCE_HIGH_DELTA:
        return "high"
    if delta > CONFIDENCE_MODERATE_DELTA:
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

# All timestamps produced by FinSight AI use this ISO-8601 format.
# Examples: "2026-05-14T09:41:22+00:00"
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"


def utc_now_iso() -> str:
    """Return the current UTC time as a canonical ISO-8601 string."""
    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


def parse_iso(ts: str) -> datetime:
    """
    Parse a canonical ISO-8601 timestamp string back to an aware datetime.

    Handles both the ``+00:00`` suffix produced by ``utc_now_iso()`` and the
    plain ``Z`` suffix that some external sources use.
    """
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


# ─────────────────────────────────────────────────────────────────────────────
# Numeric formatter helpers
# ─────────────────────────────────────────────────────────────────────────────


def fmt_prob(value: float, as_pct: bool = True) -> str:
    """Format a probability.  ``as_pct=True`` → '72.3%'; ``False`` → '0.7234'."""
    if as_pct:
        return format(value, FMT_PROB_PCT)
    return format(value, FMT_PROB_RAW)


def fmt_metric(value: float) -> str:
    """Format an ML metric (AUC, accuracy, F1 …) to canonical precision."""
    return format(value, FMT_METRIC)


def fmt_price(value: float) -> str:
    """Format a price with dollar sign and thousands separator."""
    return f"${value:{FMT_PRICE}}"


def fmt_shap(value: float) -> str:
    """Format a SHAP value with explicit sign."""
    return format(value, FMT_SHAP)


def fmt_sentiment(value: float) -> str:
    """Format a sentiment score with explicit sign."""
    return format(value, FMT_SENTIMENT)


def fmt_duration(seconds: float) -> str:
    """Format a training/inference duration in seconds."""
    return f"{seconds:{FMT_DURATION}}s"


def round_metric(value: float) -> float:
    """Round a metric float to canonical precision for JSON serialisation."""
    return round(value, METRIC_DECIMAL_PLACES)


def round_prob(value: float) -> float:
    """Round a raw probability to canonical precision for JSON serialisation."""
    return round(value, PROB_RAW_DECIMAL_PLACES)


def round_price(value: float) -> float:
    """Round a price to canonical precision for JSON serialisation."""
    return round(value, PRICE_DECIMAL_PLACES)


def round_shap(value: float) -> float:
    """Round a SHAP value to canonical precision for JSON serialisation."""
    return round(value, SHAP_DECIMAL_PLACES)


def round_sentiment(value: float) -> float:
    """Round a sentiment score to canonical precision for JSON serialisation."""
    return round(value, SENTIMENT_DECIMAL_PLACES)


def round_weight(value: float) -> float:
    """Round a news item composite weight to canonical precision."""
    return round(value, WEIGHT_DECIMAL_PLACES)


# ─────────────────────────────────────────────────────────────────────────────
# Narrative template builders
# ─────────────────────────────────────────────────────────────────────────────

# Separator used between sections in all multi-line narrative strings.
NARRATIVE_SECTION_SEP: str = "  "  # two spaces — keeps narratives on one line

# Bullet character used in all bulleted lists inside narratives.
NARRATIVE_BULLET: str = "•"


def build_prediction_narrative(
    ticker: str,
    direction: str,
    prob: float,
    confidence: str,
    bullish_features: list[dict],
    bearish_features: list[dict],
) -> str:
    """
    Build a canonical plain-English prediction narrative.

    This is the *single* implementation used by both ``SHAPExplainer`` and any
    other caller that needs to describe a prediction.  Format is deterministic:

        [AAPL] Prediction: BULLISH (UP) (Probability: 72.3%, Confidence: high)
          • Bullish: rsi_14 (+0.0312), macd_histogram (+0.0198)
          • Bearish headwinds: realized_vol_21d (-0.0155)

    Args:
        ticker:           Ticker symbol.  Empty string omits the ``[TICKER]`` prefix.
        direction:        'BULLISH (UP)' or 'BEARISH (DOWN)'.
        prob:             Probability of the predicted direction (already mapped to
                          the correct side — pass ``p_bullish`` when BULLISH, else
                          ``1 - p_bullish``).
        confidence:       'high' | 'moderate' | 'low'.
        bullish_features: List of dicts with 'feature' and 'shap_value' keys.
        bearish_features: List of dicts with 'feature' and 'shap_value' keys.

    Returns:
        Single-line narrative string.
    """
    prefix = f"[{ticker}] " if ticker else ""
    header = (
        f"{prefix}Prediction: {direction} "
        f"(Probability: {fmt_prob(prob)}, Confidence: {confidence})"
    )

    parts: list[str] = [header]

    if bullish_features:
        bull_items = ", ".join(
            f"{f['feature']} ({fmt_shap(f['shap_value'])})" for f in bullish_features
        )
        parts.append(f"{NARRATIVE_BULLET} Bullish: {bull_items}.")

    if bearish_features:
        bear_items = ", ".join(
            f"{f['feature']} ({fmt_shap(f['shap_value'])})" for f in bearish_features
        )
        parts.append(f"{NARRATIVE_BULLET} Bearish headwinds: {bear_items}.")

    return NARRATIVE_SECTION_SEP.join(parts)


def build_fusion_rule_narrative(
    ml_dir: str,
    ml_prob: float,
    agg_sentiment: str,
    agg_score: float,
    final_dir: str,
) -> str:
    """
    Build a canonical rule-based fusion narrative (used when LLM is unavailable).

    Format:
        Rule-based fusion (LLM unavailable).
        ML signal: BULLISH (p=0.7234).
        News sentiment: positive (score=+0.312).
        Final: BULLISH.
    """
    return (
        f"Rule-based fusion (LLM unavailable). "
        f"ML signal: {ml_dir} (p={fmt_prob(ml_prob, as_pct=False)}). "
        f"News sentiment: {agg_sentiment} (score={fmt_sentiment(agg_score)}). "
        f"Final: {final_dir}."
    )


def build_tool_context_string(tool_name: str, output: object) -> str:
    """
    Build a canonical tool-result context string for agent prompt injection.

    Format:
        [tool_name] { … JSON up to 2000 chars … }

    The 2 000-char cap is preserved from the original implementation.
    """
    import json

    serialised = json.dumps(output, default=str)
    if len(serialised) > 2000:
        serialised = serialised[:1997] + "…"
    return f"[{tool_name}] {serialised}"
