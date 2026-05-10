"""
FinSight AI — Signal Fusion Service (Upgraded v2)

Key upgrade:
- Introduces deterministic news intelligence layer BEFORE LLM fusion
- Converts raw snippets into structured, weighted financial signals
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
from datetime import datetime

from app.core.exceptions import LLMError, ToolExecutionError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("signal_fusion")

# ── Constants ────────────────────────────────────────────────────────────────

_NEWS_TOP_K: int = 5
_SNIPPET_CHARS: int = 350

HIGH_CREDIBILITY = {
    "reuters.com": 1.0,
    "bloomberg.com": 1.0,
    "wsj.com": 0.95,
    "cnbc.com": 0.9,
    "finance.yahoo.com": 0.8,
}

BULLISH_KEYWORDS = {
    "beats", "surge", "upgrade", "raised", "strong", "record profit",
    "growth", "positive outlook", "exceeds", "bullish"
}

BEARISH_KEYWORDS = {
    "miss", "downgrade", "lawsuit", "investigation", "fraud",
    "decline", "cut", "weak", "loss", "bankruptcy", "warning"
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str

    sentiment: str = "neutral"
    relevance_score: float = 0.5
    severity_score: float = 0.5
    credibility_score: float = 0.5
    final_weight: float = 0.5


@dataclass
class FusedSignal:
    final_direction: str
    final_confidence: str
    fusion_probability: float
    synthesis_narrative: str
    news_items: list[NewsItem] = field(default_factory=list)
    ml_direction: str = ""
    ml_probability: float = 0.5
    fusion_applied: bool = True
    news_sentiment: str = "neutral"


# ── Fusion Prompt ────────────────────────────────────────────────────────────

_FUSION_SYSTEM = """You are a senior quantitative analyst at a hedge fund.

You receive:
(A) ML prediction
(B) PRE-ANALYZED financial news signal

The news is already weighted by:
- sentiment
- severity
- credibility
- recency

Do NOT re-interpret raw headlines. Focus on reasoning fusion.

Rules:
- Strong news overrides weak ML
- Strong ML can override weak news
- If both strong and conflicting → NEUTRAL
- NEUTRAL is valid

Return ONLY JSON:
{
  "final_direction": "BULLISH|BEARISH|NEUTRAL",
  "final_confidence": "HIGH|MODERATE|LOW",
  "fusion_probability": 0..1,
  "news_sentiment": "positive|negative|neutral",
  "synthesis_narrative": "2–4 sentences"
}"""


# ── Service ─────────────────────────────────────────────────────────────────

class SignalFusionService:

    def __init__(self, llm_client=None, news_top_k: int = _NEWS_TOP_K):
        self._llm = llm_client
        self._news_k = news_top_k

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────────────

    def fuse(self, ticker: str, prediction_response) -> FusedSignal:

        ml_dir = "BULLISH" if prediction_response.prediction == 1 else "BEARISH"
        ml_prob = prediction_response.p_bullish
        ml_conf = prediction_response.confidence_label.upper()

        news_items = []
        try:
            news_items = self._fetch_news(ticker)
            news_items = self._analyze_news(news_items)
        except Exception as exc:
            logger.warning("[%s] News fetch failed: %s", ticker, exc)

        fallback = FusedSignal(
            final_direction=ml_dir,
            final_confidence=ml_conf,
            fusion_probability=ml_prob,
            synthesis_narrative=f"ML-only signal due to missing news.",
            news_items=news_items,
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            fusion_applied=False,
            news_sentiment="neutral",
        )

        if not news_items:
            return fallback

        try:
            return self._llm_fuse(
                ticker,
                prediction_response,
                ml_dir,
                ml_conf,
                news_items,
            )
        except Exception as exc:
            logger.warning("[%s] LLM failed: %s", ticker, exc)
            return fallback

    # ────────────────────────────────────────────────────────────────────────
    # NEWS PIPELINE (NEW CORE LOGIC)
    # ────────────────────────────────────────────────────────────────────────

    def _analyze_news(self, items: list[NewsItem]) -> list[NewsItem]:
        seen = set()
        processed = []

        for item in items:
            key = item.title.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            text = (item.title + " " + item.snippet).lower()

            # sentiment scoring
            bull = sum(1 for w in BULLISH_KEYWORDS if w in text)
            bear = sum(1 for w in BEARISH_KEYWORDS if w in text)

            if bull > bear:
                item.sentiment = "positive"
                sentiment_score = 1.0
            elif bear > bull:
                item.sentiment = "negative"
                sentiment_score = -1.0
            else:
                item.sentiment = "neutral"
                sentiment_score = 0.0

            # severity scoring (keyword intensity)
            severe_keywords = {"bankruptcy", "fraud", "lawsuit", "investigation", "earnings"}
            item.severity_score = min(1.0, sum(k in text for k in severe_keywords) / 3)

            # credibility scoring
            domain = urlparse(item.url).netloc.replace("www.", "")
            item.credibility_score = HIGH_CREDIBILITY.get(domain, 0.6)

            # final weight
            item.final_weight = (
                item.credibility_score * 0.4 +
                item.severity_score * 0.3 +
                abs(sentiment_score) * 0.3
            )

            processed.append(item)

        return sorted(processed, key=lambda x: x.final_weight, reverse=True)

    # ────────────────────────────────────────────────────────────────────────
    # NEWS FETCH
    # ────────────────────────────────────────────────────────────────────────

    def _fetch_news(self, ticker: str) -> list[NewsItem]:

        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        query = f"{ticker} stock earnings news"

        items = []
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=self._news_k))

        for r in raw:
            items.append(
                NewsItem(
                    title=r.get("title", ""),
                    snippet=r.get("body", "")[:_SNIPPET_CHARS],
                    url=r.get("href", ""),
                )
            )

        return items

    # ────────────────────────────────────────────────────────────────────────
    # LLM FUSION
    # ────────────────────────────────────────────────────────────────────────

    def _llm_fuse(self, ticker, prediction_response, ml_dir, ml_conf, news_items):

        llm = self._get_llm()

        aggregated_score = sum(n.final_weight for n in news_items) / len(news_items)

        news_summary = "\n".join(
            f"- {n.title} | sentiment={n.sentiment} | weight={n.final_weight:.2f}"
            for n in news_items
        )

        shap = prediction_response.shap_explanation.get("top_features", [])[:5]
        shap_text = "\n".join(
            f"• {f['feature']}: {f['shap_value']:+.3f}"
            for f in shap
        )

        user_msg = f"""
Ticker: {ticker}

ML:
- Direction: {ml_dir}
- Probability: {prediction_response.p_bullish:.3f}

News Aggregate Score: {aggregated_score:.3f}

Top News:
{news_summary}

SHAP:
{shap_text}
"""

        messages = [
            {"role": "system", "content": _FUSION_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        raw, _ = llm.chat(messages, temperature=0.0, max_tokens=500)

        data = self._parse(raw)

        return FusedSignal(
            final_direction=data.get("final_direction", ml_dir),
            final_confidence=data.get("final_confidence", ml_conf),
            fusion_probability=float(data.get("fusion_probability", prediction_response.p_bullish)),
            synthesis_narrative=data.get("synthesis_narrative", ""),
            news_items=news_items,
            ml_direction=ml_dir,
            ml_probability=prediction_response.p_bullish,
            fusion_applied=True,
            news_sentiment=data.get("news_sentiment", "neutral"),
        )

    # ────────────────────────────────────────────────────────────────────────

    def _parse(self, text: str) -> dict:
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])

    def _get_llm(self):
        from app.rag.llm_chat import OpenAIClient
        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm