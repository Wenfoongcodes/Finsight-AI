"""
FinSight AI — Signal Fusion Service (v6)

Changes vs v5
-------------

1.  **Horizon-aware recency filtering (Req 1)**

    ``SignalFusionService.fuse()`` now receives the ``horizon`` from
    ``PredictionService`` and passes it to
    ``FinancialIntelligenceService.get_brief()``.  This propagates the
    per-horizon lookback window through the entire news retrieval chain,
    so a 1-day prediction only considers articles published within the last
    3 days, while a 6-month prediction accepts articles up to 90 days old.

    The ``PredictionResponse`` dataclass gains no new field — ``horizon``
    was already present.  ``SignalFusionService.fuse()`` signature change::

        def fuse(self, ticker, prediction_response) → FusedSignal
        # becomes
        def fuse(self, ticker, prediction_response, horizon="1d") → FusedSignal

2.  **Deterministic output formatting (Req 2)**

    All numeric operations now use helpers from ``app.core.formatting``:
    - ``fusion_probability`` → ``round_prob()``   (4 d.p.)
    - ``sentiment_score``    → ``round_sentiment()`` (3 d.p.)
    - Narrative strings      → ``build_fusion_rule_narrative()``
    - Timestamps             → ``utc_now_iso()``
    - Model resolution log   → consistent ``logger.info`` format

    The ``_rule_based_fusion()`` narrative previously used ad-hoc ``:.3f``
    and ``:.3f`` — now routed through the canonical builder so the format
    is identical regardless of whether the LLM or the rule engine produced
    the narrative.

``IntelligenceBrief`` / ``FusedSignal`` schema is unchanged from v5.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.exceptions import LLMError
from app.core.formatting import (
    build_fusion_rule_narrative,
    round_prob,
    utc_now_iso,
)
from app.core.logging_config import get_logger
from app.services.news_intelligence import (
    FinancialIntelligenceService,
    NewsItem,
)
from configs.settings import settings

logger = get_logger("signal_fusion")

_PROVIDER_MODEL_MAP: dict[str, str] = {
    "groq": "llama3-70b-8192",
    "ollama": "llama3",
    "azure": "gpt-4o-mini",
}

# ── Fusion system prompt ──────────────────────────────────────────────────────

_FUSION_SYSTEM = """You are a senior quantitative analyst at a hedge fund.

You receive:
(A) ML prediction with SHAP feature drivers
(B) A list of recent financial news items, each with a sentiment score and
    source credibility weight

Fusion rules:
- Strong news (high weight, clear sentiment) overrides weak ML signal
- Strong ML (|p_bullish - 0.5| > 0.20) can override weak/neutral news
- Conflicting strong signals → NEUTRAL
- NEUTRAL is a valid and sometimes correct output

Return ONLY valid JSON with no markdown, no code fences, no prose:
{
  "final_direction": "BULLISH|BEARISH|NEUTRAL",
  "final_confidence": "HIGH|MODERATE|LOW",
  "fusion_probability": <float 0..1>,
  "news_sentiment": "positive|negative|neutral",
  "synthesis_narrative": "<2-4 sentences: what the evidence shows and why>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# FusedSignal
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FusedSignal:
    """
    Output of the signal fusion pipeline.

    All float fields use canonical precision from ``app.core.formatting``:
    - ``fusion_probability`` → ``round_prob()``      (4 d.p.)
    - ``news_score``         → ``round_sentiment()``  (3 d.p.)

    New field (v6)
    --------------
    generated_at : UTC ISO-8601 timestamp of fusion.
    recency_note : Forwarded from the ``IntelligenceBrief`` for transparency.
    """

    final_direction: str
    final_confidence: str
    fusion_probability: float  # round_prob() — 4 d.p.
    synthesis_narrative: str
    news_items: list[NewsItem] = field(default_factory=list)
    ml_direction: str = ""
    ml_probability: float = 0.5  # round_prob() — 4 d.p.
    fusion_applied: bool = True
    news_sentiment: str = "neutral"
    generated_at: str = field(default_factory=utc_now_iso)
    recency_note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────


class SignalFusionService:
    """
    Fuses the ML prediction with financial news via LLM synthesis.

    Degradation hierarchy (never raises to caller):
    ┌──────────────────────┬──────────────────────────────────────────────┐
    │ Failure              │ Behaviour                                    │
    ├──────────────────────┼──────────────────────────────────────────────┤
    │ News retrieval fails │ Return ML-only FusedSignal                   │
    │ LLM call (LLMError)  │ Rule-based fusion from news sentiment        │
    │ JSON parse error     │ ML-only FusedSignal with news_items preserved│
    └──────────────────────┴──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        llm_client=None,
        intelligence_service: Optional[FinancialIntelligenceService] = None,
    ) -> None:
        self._llm = llm_client
        self._intel_svc = intelligence_service or FinancialIntelligenceService()

    # ── Public ────────────────────────────────────────────────────────────────

    def fuse(
        self,
        ticker: str,
        prediction_response,
        horizon: str = "1d",
    ) -> FusedSignal:
        """
        Run the full fusion pipeline.  Never raises.

        Parameters
        ----------
        ticker:               Stock ticker symbol.
        prediction_response:  ``PredictionResponse`` from ``PredictionService``.
        horizon:              Prediction horizon — forwarded to the news service
                              so the correct lookback window is applied.

        Returns
        -------
        FusedSignal with ``fusion_applied=True`` on LLM success,
        a rule-based signal on LLM failure, or an ML-only signal if news
        retrieval also fails.
        """
        ml_dir = "BULLISH" if prediction_response.prediction == 1 else "BEARISH"
        ml_prob = round_prob(prediction_response.p_bullish)
        ml_conf = prediction_response.confidence_label.upper()

        ml_only = FusedSignal(
            final_direction=ml_dir,
            final_confidence=ml_conf,
            fusion_probability=ml_prob,
            synthesis_narrative="ML-only signal — news fusion unavailable.",
            news_items=[],
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            fusion_applied=False,
            news_sentiment="neutral",
            generated_at=utc_now_iso(),
            recency_note="",
        )

        # ── Step 1: Retrieve news (with horizon-aware recency filter) ─────────
        brief = self._intel_svc.get_brief(ticker, horizon=horizon)

        if not brief.retrieval_success or not brief.top_news:
            logger.info(
                "[%s/%s] No news retrieved — returning ML-only signal.", ticker, horizon
            )
            return ml_only

        news_items = brief.top_news
        agg_sentiment = brief.aggregate_sentiment
        agg_score = brief.sentiment_score  # already round_sentiment()
        recency_note = brief.recency_note

        # Propagate news metadata to the ML-only fallback in case LLM fails.
        ml_only.news_items = news_items
        ml_only.news_sentiment = agg_sentiment
        ml_only.recency_note = recency_note

        # ── Step 2: LLM synthesis ─────────────────────────────────────────────
        try:
            fused = self._llm_fuse(
                ticker,
                prediction_response,
                ml_dir,
                ml_conf,
                news_items,
                agg_sentiment,
                agg_score,
                recency_note,
            )
            logger.info(
                "[%s/%s] Fusion: %s → %s (applied=%s, recency=%s)",
                ticker,
                horizon,
                ml_dir,
                fused.final_direction,
                fused.fusion_applied,
                recency_note,
            )
            return fused

        except LLMError as exc:
            logger.warning(
                "[%s/%s] LLM fusion failed: %s — rule-based fallback.",
                ticker,
                horizon,
                exc,
            )
        except ValueError as exc:
            logger.warning(
                "[%s/%s] LLM parse failed: %s — rule-based fallback.",
                ticker,
                horizon,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "[%s/%s] LLM fusion error: %s — rule-based fallback.",
                ticker,
                horizon,
                exc,
            )

        # ── Step 3: Rule-based fallback ───────────────────────────────────────
        return self._rule_based_fusion(
            ticker,
            ml_dir,
            ml_conf,
            ml_prob,
            news_items,
            agg_sentiment,
            agg_score,
            recency_note,
        )

    # ── LLM fusion ────────────────────────────────────────────────────────────

    def _llm_fuse(
        self,
        ticker: str,
        prediction_response,
        ml_dir: str,
        ml_conf: str,
        news_items: list[NewsItem],
        agg_sentiment: str,
        agg_score: float,
        recency_note: str,
    ) -> FusedSignal:
        """
        Call the LLM to synthesise the ML signal with the news items.

        Raises:
            LLMError:   From OpenAIClient on API failure.
            ValueError: From _parse() on unparseable JSON.
        """
        llm = self._get_llm()

        shap_text = "\n".join(
            f"  • {f['feature']}: {f['shap_value']:+.4f}"
            for f in prediction_response.shap_explanation.get("top_features", [])[:5]
        )

        news_str = "\n".join(
            f"  [{i + 1}] {n.title} | sentiment={n.sentiment} "
            f"| weight={n.final_weight:.3f} | source={n.domain}"
            f"\n        {n.snippet[:500].strip()}"
            for i, n in enumerate(news_items[:6])
        )

        user_msg = (
            f"=== ML SIGNAL ===\n"
            f"Ticker:       {ticker}\n"
            f"Direction:    {ml_dir}\n"
            f"P(bullish):   {prediction_response.p_bullish:.4f}\n"
            f"Confidence:   {ml_conf}\n\n"
            f"Top SHAP drivers:\n{shap_text}\n\n"
            f"=== NEWS (source-weighted, recency-filtered) ===\n"
            f"Recency note: {recency_note}\n"
            f"{news_str}\n\n"
            f"Aggregate sentiment: {agg_sentiment} (score={agg_score:+.3f})\n"
        )

        messages = [
            {"role": "system", "content": _FUSION_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        model = self._resolve_model()
        logger.info("[%s] LLM fusion model: %s", ticker, model)

        raw, _ = llm.chat(messages, model=model, temperature=0.0, max_tokens=600)

        if not raw or not raw.strip():
            raise ValueError(
                f"LLM returned empty response for {ticker} (model={model})."
            )

        data = self._parse(raw)

        return FusedSignal(
            final_direction=data.get("final_direction", ml_dir),
            final_confidence=data.get("final_confidence", ml_conf),
            fusion_probability=round_prob(
                float(data.get("fusion_probability", prediction_response.p_bullish))
            ),
            synthesis_narrative=data.get("synthesis_narrative", ""),
            news_items=news_items,
            ml_direction=ml_dir,
            ml_probability=round_prob(prediction_response.p_bullish),
            fusion_applied=True,
            news_sentiment=data.get("news_sentiment", agg_sentiment),
            generated_at=utc_now_iso(),
            recency_note=recency_note,
        )

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_fusion(
        self,
        ticker: str,
        ml_dir: str,
        ml_conf: str,
        ml_prob: float,
        news_items: list[NewsItem],
        agg_sentiment: str,
        agg_score: float,
        recency_note: str,
    ) -> FusedSignal:
        """
        Deterministic fusion when LLM is unavailable.

        Rules:
        - Strong news (|score| >= 0.30) may override ML direction.
        - Conflicting strong signals → NEUTRAL.
        - Weak/neutral news → preserve ML direction unchanged.
        """
        strong_threshold = 0.30

        if abs(agg_score) >= strong_threshold:
            news_bullish = agg_sentiment == "positive"
            ml_bullish = ml_dir == "BULLISH"

            if news_bullish == ml_bullish:
                final_dir = ml_dir
                final_prob = round_prob(ml_prob)
                confidence = "MODERATE"
            else:
                final_dir = "NEUTRAL"
                final_prob = round_prob(0.5)
                confidence = "LOW"
        else:
            final_dir = ml_dir
            final_prob = round_prob(ml_prob)
            confidence = ml_conf

        # Use canonical narrative builder — deterministic format regardless
        # of which code path produced it.
        narrative = build_fusion_rule_narrative(
            ml_dir=ml_dir,
            ml_prob=ml_prob,
            agg_sentiment=agg_sentiment,
            agg_score=agg_score,
            final_dir=final_dir,
        )

        return FusedSignal(
            final_direction=final_dir,
            final_confidence=confidence,
            fusion_probability=final_prob,
            synthesis_narrative=narrative,
            news_items=news_items,
            ml_direction=ml_dir,
            ml_probability=round_prob(ml_prob),
            fusion_applied=False,
            news_sentiment=agg_sentiment,
            generated_at=utc_now_iso(),
            recency_note=recency_note,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_model(self) -> str:
        configured = settings.LLM_MODEL
        base_url = (settings.LLM_BASE_URL or "").lower().strip()
        if "gpt" not in configured.lower():
            return configured
        for provider_key, default_model in _PROVIDER_MODEL_MAP.items():
            if provider_key in base_url:
                logger.debug(
                    "Provider '%s' detected — remapping '%s' → '%s'",
                    provider_key,
                    configured,
                    default_model,
                )
                return default_model
        return configured

    @staticmethod
    def _parse(text: str) -> dict:
        """
        Robustly extract a JSON object from an LLM response.

        Raises:
            ValueError: When no valid JSON object is found.
        """
        text = re.sub(r"```(?:json|JSON)?\s*", "", text).strip()
        text = text.replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No JSON object in LLM response. First 300 chars: {text[:300]!r}"
            )

        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON from LLM: {exc}. Candidate: {candidate[:300]!r}"
            ) from exc

    def _get_llm(self):
        from app.rag.llm_chat import OpenAIClient

        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm
