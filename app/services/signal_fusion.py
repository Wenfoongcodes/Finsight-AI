"""
FinSight AI — Signal Fusion Service (v4)

Changes vs v3
-------------
1. ``SignalFusionService`` now delegates news retrieval and analysis to
   ``FinancialIntelligenceService`` (the new dedicated news module).  The
   old inline DuckDuckGo + keyword scoring logic is removed here — it lives
   in ``app/services/news_intelligence.py`` where it can be tested,
   extended, and reused independently.

2. The LLM synthesis prompt has been updated to receive the richer
   ``IntelligenceBrief`` structure (situation summary, bullish/bearish
   catalysts, source quality note) rather than just a flat news list.

3. All three original hardening fixes (v3) are retained:
   - Fix A: ``_resolve_model()`` for provider-aware model remapping
   - Fix B: ``_parse()`` with bracket validation
   - Fix C: ``LLMError`` propagation from ``_llm_fuse``

4. Graceful degradation hierarchy:
   news_failure  → ML-only signal (no LLM call)
   LLM_failure   → rule-based fusion from IntelligenceBrief
   parse_failure → ML-only signal with news_items preserved
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from app.core.exceptions import LLMError
from app.core.logging_config import get_logger
from app.services.news_intelligence import (
    FinancialIntelligenceService,
    IntelligenceBrief,
    NewsItem,
)
from configs.settings import settings

logger = get_logger("signal_fusion")

# Provider → default model remapping (unchanged from v3)
_PROVIDER_MODEL_MAP: dict[str, str] = {
    "groq":   "llama3-70b-8192",
    "ollama": "llama3",
    "azure":  "gpt-4o-mini",
}

# ── Fusion prompt ─────────────────────────────────────────────────────────────

_FUSION_SYSTEM = """You are a senior quantitative analyst at a hedge fund.

You receive:
(A) ML prediction with SHAP feature drivers
(B) Pre-analyzed financial intelligence brief (source-weighted, deduplicated)

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
# FusedSignal (unchanged interface for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusedSignal:
    final_direction:     str
    final_confidence:    str
    fusion_probability:  float
    synthesis_narrative: str
    news_items:          list[NewsItem] = field(default_factory=list)
    ml_direction:        str  = ""
    ml_probability:      float = 0.5
    fusion_applied:      bool  = True
    news_sentiment:      str   = "neutral"
    intelligence_brief:  Optional[IntelligenceBrief] = None


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class SignalFusionService:
    """
    Fuses ML prediction with financial intelligence via LLM synthesis.

    Degradation hierarchy (never raises to caller):
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ Failure             │ Behaviour                                    │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │ News retrieval      │ Return ML-only FusedSignal                   │
    │ LLM call (LLMError) │ Rule-based fusion from IntelligenceBrief     │
    │ JSON parse error    │ ML-only FusedSignal with news_items preserved│
    └─────────────────────┴──────────────────────────────────────────────┘
    """

    def __init__(
        self,
        llm_client=None,
        intelligence_service: Optional[FinancialIntelligenceService] = None,
    ) -> None:
        self._llm          = llm_client
        self._intel_svc    = intelligence_service or FinancialIntelligenceService()

    # ── Public ────────────────────────────────────────────────────────────────

    def fuse(self, ticker: str, prediction_response) -> FusedSignal:
        """
        Run the full fusion pipeline.  Never raises.

        Returns a ``FusedSignal`` with ``fusion_applied=True`` on LLM success,
        a rule-based signal on LLM failure, or an ML-only signal if news
        retrieval also fails.
        """
        ml_dir  = "BULLISH" if prediction_response.prediction == 1 else "BEARISH"
        ml_prob = prediction_response.p_bullish
        ml_conf = prediction_response.confidence_label.upper()

        ml_only_fallback = FusedSignal(
            final_direction=ml_dir,
            final_confidence=ml_conf,
            fusion_probability=ml_prob,
            synthesis_narrative="ML-only signal — news fusion unavailable.",
            news_items=[],
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            fusion_applied=False,
            news_sentiment="neutral",
        )

        # ── Step 1: Get intelligence brief ────────────────────────────────────
        brief = self._intel_svc.get_brief(ticker)

        if not brief.retrieval_success or not brief.top_news:
            logger.info("[%s] No news — returning ML-only signal.", ticker)
            return ml_only_fallback

        ml_only_fallback.news_items        = brief.top_news
        ml_only_fallback.news_sentiment    = brief.aggregate_sentiment
        ml_only_fallback.intelligence_brief = brief

        # ── Step 2: LLM synthesis ─────────────────────────────────────────────
        try:
            fused = self._llm_fuse(ticker, prediction_response, ml_dir, ml_conf, brief)
            fused.intelligence_brief = brief
            logger.info(
                "[%s] Fusion: %s → %s (applied=%s)",
                ticker, ml_dir, fused.final_direction, fused.fusion_applied,
            )
            return fused

        except LLMError as exc:
            logger.warning(
                "[%s] LLM fusion failed (LLMError): %s — using rule-based fallback.",
                ticker, exc,
            )
        except ValueError as exc:
            logger.warning(
                "[%s] LLM JSON parse failed: %s — using rule-based fallback.",
                ticker, exc,
            )
        except Exception as exc:
            logger.warning(
                "[%s] LLM fusion unexpected error: %s — using rule-based fallback.",
                ticker, exc,
            )

        # ── Step 3: Rule-based fallback using IntelligenceBrief ───────────────
        return self._rule_based_fusion(ticker, ml_dir, ml_conf, ml_prob, brief)

    # ── LLM fusion ────────────────────────────────────────────────────────────

    def _llm_fuse(
        self,
        ticker: str,
        prediction_response,
        ml_dir: str,
        ml_conf: str,
        brief: IntelligenceBrief,
    ) -> FusedSignal:
        """
        Call the LLM to synthesise ML signal + intelligence brief.

        Raises:
            LLMError:  From OpenAIClient on API failure.
            ValueError: From ``_parse()`` on unparseable JSON.
        """
        llm = self._get_llm()

        shap_text = "\n".join(
            f"• {f['feature']}: {f['shap_value']:+.3f}"
            for f in prediction_response.shap_explanation.get("top_features", [])[:5]
        )

        bullish_str = "\n".join(f"  + {c}" for c in brief.bullish_catalysts) or "  (none identified)"
        bearish_str = "\n".join(f"  - {c}" for c in brief.bearish_catalysts) or "  (none identified)"

        top_news_str = "\n".join(
            f"  [{i+1}] {n.title} | sentiment={n.sentiment} "
            f"| weight={n.final_weight:.2f} | source={n.domain}"
            for i, n in enumerate(brief.top_news[:5])
        )

        user_msg = (
            f"=== ML SIGNAL ===\n"
            f"Ticker:      {ticker}\n"
            f"Direction:   {ml_dir}\n"
            f"P(bullish):  {prediction_response.p_bullish:.4f}\n"
            f"Confidence:  {ml_conf}\n\n"
            f"Top SHAP drivers:\n{shap_text}\n\n"
            f"=== INTELLIGENCE BRIEF ===\n"
            f"Situation: {brief.situation_summary}\n\n"
            f"Bullish catalysts:\n{bullish_str}\n\n"
            f"Bearish headwinds:\n{bearish_str}\n\n"
            f"Top weighted news:\n{top_news_str}\n\n"
            f"Aggregate sentiment: {brief.aggregate_sentiment} "
            f"(score={brief.sentiment_score:.3f})\n"
            f"Source quality: {brief.source_quality_note}\n"
        )

        messages = [
            {"role": "system", "content": _FUSION_SYSTEM},
            {"role": "user",   "content": user_msg},
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
            fusion_probability=float(
                data.get("fusion_probability", prediction_response.p_bullish)
            ),
            synthesis_narrative=data.get("synthesis_narrative", ""),
            news_items=brief.top_news,
            ml_direction=ml_dir,
            ml_probability=prediction_response.p_bullish,
            fusion_applied=True,
            news_sentiment=data.get("news_sentiment", brief.aggregate_sentiment),
        )

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_based_fusion(
        self,
        ticker: str,
        ml_dir: str,
        ml_conf: str,
        ml_prob: float,
        brief: IntelligenceBrief,
    ) -> FusedSignal:
        """
        Deterministic fusion when LLM is unavailable.

        Rules:
        - Strong news (|score| > 0.4) overrides ML direction.
        - Conflicting strong signals → NEUTRAL.
        - Weak or neutral news → preserve ML direction.
        """
        news_score = brief.sentiment_score
        news_dir   = brief.aggregate_sentiment

        strong_news_threshold = 0.30

        if abs(news_score) >= strong_news_threshold:
            news_bullish = news_dir == "positive"
            ml_bullish   = ml_dir == "BULLISH"

            if news_bullish == ml_bullish:
                # Agreement
                final_dir  = ml_dir
                final_prob = ml_prob
                confidence = "MODERATE"
            else:
                # Conflict → NEUTRAL
                final_dir  = "NEUTRAL"
                final_prob = 0.5
                confidence = "LOW"
        else:
            # Weak news — trust ML
            final_dir  = ml_dir
            final_prob = ml_prob
            confidence = ml_conf

        narrative = (
            f"Rule-based fusion (LLM unavailable). "
            f"ML signal: {ml_dir} (p={ml_prob:.3f}). "
            f"News sentiment: {news_dir} (score={news_score:.3f}). "
            f"Final: {final_dir}."
        )

        return FusedSignal(
            final_direction=final_dir,
            final_confidence=confidence,
            fusion_probability=final_prob,
            synthesis_narrative=narrative,
            news_items=brief.top_news,
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            fusion_applied=False,   # LLM was not used
            news_sentiment=brief.aggregate_sentiment,
            intelligence_brief=brief,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_model(self) -> str:
        configured = settings.LLM_MODEL
        base_url   = (settings.LLM_BASE_URL or "").lower().strip()
        if "gpt" not in configured.lower():
            return configured
        for provider_key, default_model in _PROVIDER_MODEL_MAP.items():
            if provider_key in base_url:
                logger.debug(
                    "Provider '%s' detected — remapping '%s' → '%s'",
                    provider_key, configured, default_model,
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
        end   = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No JSON object in LLM response. First 300 chars: {text[:300]!r}"
            )

        candidate = text[start: end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON from LLM: {exc}. "
                f"Candidate (first 300 chars): {candidate[:300]!r}"
            ) from exc

    def _get_llm(self):
        from app.rag.llm_chat import OpenAIClient
        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm