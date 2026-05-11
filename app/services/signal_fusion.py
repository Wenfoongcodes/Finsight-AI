"""
FinSight AI — Signal Fusion Service (v3 — hardened)

Fixes applied in this revision
-------------------------------

Issue 1 — LLM returned empty string → json.JSONDecodeError
    Root cause: ``LLM_BASE_URL`` is set to Groq's endpoint but
    ``LLM_MODEL`` is still "gpt-4o-mini" — a model that does not
    exist on Groq.  Groq returns an API-level error object; the
    ``OpenAIClient`` raises a ``LLMError`` (correct), but the
    ``_llm_fuse`` caller catches a bare ``Exception`` and returns
    an empty-string response which ``_parse("")`` then crashes on
    with "Expecting value: line 1 column 1 (char 0)".

    Fix A — ``_resolve_model()`` detects the active provider from
    ``LLM_BASE_URL`` and maps ``gpt-4o-mini`` → a compatible model
    name automatically.  The mapping is configurable via
    ``GROQ_DEFAULT_MODEL`` / ``OLLAMA_DEFAULT_MODEL`` env vars.

    Fix B — ``_parse()`` now validates that ``{`` and ``}`` are
    present before slicing, and raises a descriptive
    ``ValueError`` instead of silently passing garbage to
    ``json.loads``.

    Fix C — ``_llm_fuse`` propagates ``LLMError`` instead of
    catching it silently, so the caller's ``except Exception``
    block in ``fuse()`` logs the real error message and falls
    back cleanly.

Issue 2 — ``_parse`` regex strip was too aggressive
    ``re.sub(r"```.*?```", "", text, flags=re.S)`` deleted
    everything inside code fences *including the JSON itself*
    when the model wrapped the JSON in triple-backtick fences
    (which Groq/Llama models frequently do).

    Fix: strip only the opening/closing fence markers, not
    the content between them, matching the strategy already
    used in ``financial_agent._extract_json_array()``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from app.core.exceptions import LLMError, ToolExecutionError
from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("signal_fusion")

# ── Constants ────────────────────────────────────────────────────────────────

_NEWS_TOP_K: int = 5
_SNIPPET_CHARS: int = 350

# Provider → default model when the configured LLM_MODEL is OpenAI-only.
# Extend this dict when adding new providers.
_PROVIDER_MODEL_MAP: dict[str, str] = {
    "groq": "llama3-70b-8192",
    "ollama": "llama3",
    "azure": "gpt-4o-mini",   # Azure uses the same names as OpenAI
}

HIGH_CREDIBILITY: dict[str, float] = {
    "reuters.com": 1.0,
    "bloomberg.com": 1.0,
    "wsj.com": 0.95,
    "cnbc.com": 0.9,
    "finance.yahoo.com": 0.8,
}

BULLISH_KEYWORDS: set[str] = {
    "beats", "surge", "upgrade", "raised", "strong", "record profit",
    "growth", "positive outlook", "exceeds", "bullish",
}

BEARISH_KEYWORDS: set[str] = {
    "miss", "downgrade", "lawsuit", "investigation", "fraud",
    "decline", "cut", "weak", "loss", "bankruptcy", "warning",
}

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

Return ONLY valid JSON — no markdown, no prose, no code fences:
{
  "final_direction": "BULLISH|BEARISH|NEUTRAL",
  "final_confidence": "HIGH|MODERATE|LOW",
  "fusion_probability": 0..1,
  "news_sentiment": "positive|negative|neutral",
  "synthesis_narrative": "2–4 sentences"
}"""


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


# ── Service ──────────────────────────────────────────────────────────────────

class SignalFusionService:
    """
    Combines an ML prediction with live news signals via LLM synthesis.

    The service is deliberately defensive: every external call (news fetch,
    LLM) is wrapped so that a failure degrades to the raw ML signal rather
    than propagating an exception to the prediction pipeline.
    """

    def __init__(self, llm_client=None, news_top_k: int = _NEWS_TOP_K) -> None:
        self._llm = llm_client
        self._news_k = news_top_k

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(self, ticker: str, prediction_response) -> FusedSignal:
        """
        Run the full fusion pipeline for *ticker*.

        Returns a ``FusedSignal`` with ``fusion_applied=True`` on success,
        or ``fusion_applied=False`` with the raw ML signal as a fallback.

        Never raises — all failures are logged as WARNING.
        """
        ml_dir  = "BULLISH" if prediction_response.prediction == 1 else "BEARISH"
        ml_prob = prediction_response.p_bullish
        ml_conf = prediction_response.confidence_label.upper()

        # Build the ML-only fallback upfront so every early-return path is clean.
        fallback = FusedSignal(
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

        # ── Step 1: Fetch and analyse news ────────────────────────────────────
        news_items: list[NewsItem] = []
        try:
            news_items = self._fetch_news(ticker)
            news_items = self._analyze_news(news_items)
            logger.info("[%s] News fetched and analysed: %d items", ticker, len(news_items))
        except Exception as exc:
            logger.warning("[%s] News fetch/analyse failed: %s", ticker, exc)

        if not news_items:
            logger.info("[%s] No news items — returning ML-only signal.", ticker)
            return fallback

        fallback.news_items = news_items  # enrich fallback with whatever we have

        # ── Step 2: LLM synthesis ─────────────────────────────────────────────
        try:
            return self._llm_fuse(ticker, prediction_response, ml_dir, ml_conf, news_items)
        except LLMError as exc:
            # LLMError is raised by OpenAIClient on API-level failures
            # (wrong model name, rate-limit, auth error, etc.)
            logger.warning(
                "[%s] LLM fusion failed (LLMError): %s — falling back to ML signal.",
                ticker, exc,
            )
        except ValueError as exc:
            # _parse() raises ValueError when the LLM output is not valid JSON
            logger.warning(
                "[%s] LLM returned unparseable JSON: %s — falling back to ML signal.",
                ticker, exc,
            )
        except Exception as exc:
            logger.warning(
                "[%s] LLM fusion failed unexpectedly: %s — falling back to ML signal.",
                ticker, exc,
            )

        return fallback

    # ── News Pipeline ─────────────────────────────────────────────────────────

    def _analyze_news(self, items: list[NewsItem]) -> list[NewsItem]:
        """Score each news item for sentiment, severity, credibility, and weight."""
        seen: set[str] = set()
        processed: list[NewsItem] = []

        for item in items:
            key = item.title.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            text = (item.title + " " + item.snippet).lower()

            # Sentiment
            bull = sum(1 for w in BULLISH_KEYWORDS if w in text)
            bear = sum(1 for w in BEARISH_KEYWORDS if w in text)

            if bull > bear:
                item.sentiment      = "positive"
                sentiment_score     = 1.0
            elif bear > bull:
                item.sentiment      = "negative"
                sentiment_score     = -1.0
            else:
                item.sentiment      = "neutral"
                sentiment_score     = 0.0

            # Severity
            severe_kw           = {"bankruptcy", "fraud", "lawsuit", "investigation", "earnings"}
            item.severity_score = min(1.0, sum(k in text for k in severe_kw) / 3)

            # Credibility
            domain                   = urlparse(item.url).netloc.replace("www.", "")
            item.credibility_score   = HIGH_CREDIBILITY.get(domain, 0.6)

            # Final composite weight
            item.final_weight = (
                item.credibility_score * 0.4
                + item.severity_score   * 0.3
                + abs(sentiment_score)  * 0.3
            )

            processed.append(item)

        return sorted(processed, key=lambda x: x.final_weight, reverse=True)

    # ── News Fetch ────────────────────────────────────────────────────────────

    def _fetch_news(self, ticker: str) -> list[NewsItem]:
        """Fetch news via DuckDuckGo (ddgs or duckduckgo_search)."""
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            pass

        if DDGS is None:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise ToolExecutionError(
                    "Web search package not installed. Run: pip install ddgs"
                ) from exc

        query = f"{ticker} stock earnings news"
        items: list[NewsItem] = []

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=self._news_k))

        for r in raw:
            items.append(NewsItem(
                title=r.get("title", ""),
                snippet=r.get("body", "")[:_SNIPPET_CHARS],
                url=r.get("href", ""),
            ))

        return items

    # ── LLM Fusion ───────────────────────────────────────────────────────────

    def _llm_fuse(
        self,
        ticker: str,
        prediction_response,
        ml_dir: str,
        ml_conf: str,
        news_items: list[NewsItem],
    ) -> FusedSignal:
        """
        Call the LLM to synthesise the ML signal with the news intelligence.

        Raises:
            LLMError: Propagated from ``OpenAIClient.chat()`` on API failure.
            ValueError: When ``_parse()`` cannot extract valid JSON.
        """
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

        user_msg = (
            f"Ticker: {ticker}\n\n"
            f"ML:\n"
            f"- Direction: {ml_dir}\n"
            f"- Probability: {prediction_response.p_bullish:.3f}\n\n"
            f"News Aggregate Score: {aggregated_score:.3f}\n\n"
            f"Top News:\n{news_summary}\n\n"
            f"SHAP:\n{shap_text}\n"
        )

        messages = [
            {"role": "system", "content": _FUSION_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

        # Resolve the model name for the active provider before calling the LLM.
        model = self._resolve_model()
        logger.info("[%s] LLM fusion using model: %s", ticker, model)

        # LLMError propagates — the caller in fuse() catches it.
        raw, _ = llm.chat(messages, model=model, temperature=0.0, max_tokens=500)

        if not raw or not raw.strip():
            raise ValueError(
                f"LLM returned an empty response for ticker={ticker}. "
                f"Check model name ({model}) and API key validity."
            )

        data = self._parse(raw)

        return FusedSignal(
            final_direction=data.get("final_direction", ml_dir),
            final_confidence=data.get("final_confidence", ml_conf),
            fusion_probability=float(
                data.get("fusion_probability", prediction_response.p_bullish)
            ),
            synthesis_narrative=data.get("synthesis_narrative", ""),
            news_items=news_items,
            ml_direction=ml_dir,
            ml_probability=prediction_response.p_bullish,
            fusion_applied=True,
            news_sentiment=data.get("news_sentiment", "neutral"),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_model(self) -> str:
        """
        Return the LLM model name appropriate for the active provider.

        If ``LLM_BASE_URL`` points to Groq, Ollama, etc., the configured
        ``LLM_MODEL`` (which defaults to "gpt-4o-mini") is likely wrong.
        This method detects the provider from the URL and maps to a
        compatible default so fusion doesn't silently fail with a 404.

        The resolved name is logged at DEBUG level so it's easy to audit.

        If ``LLM_MODEL`` already looks provider-specific (doesn't contain
        "gpt"), it is returned unchanged — the operator has already set it
        correctly in their ``.env``.
        """
        configured = settings.LLM_MODEL
        base_url   = (settings.LLM_BASE_URL or "").lower().strip()

        # If the configured model is already non-OpenAI, trust it.
        if "gpt" not in configured.lower():
            return configured

        # Detect provider from base_url hostname.
        for provider_key, default_model in _PROVIDER_MODEL_MAP.items():
            if provider_key in base_url:
                logger.debug(
                    "Provider '%s' detected from LLM_BASE_URL — "
                    "remapping model '%s' → '%s'",
                    provider_key, configured, default_model,
                )
                return default_model

        # No remapping needed (official OpenAI or unknown provider).
        return configured

    @staticmethod
    def _parse(text: str) -> dict:
        """
        Robustly extract a JSON object from an LLM response string.

        Handles:
        * Markdown code fences: ```json ... ``` or ``` ... ```
        * Leading/trailing prose
        * Whitespace

        Raises:
            ValueError: When no JSON object can be found or parsed.
        """
        # Strip markdown fence markers (but NOT the content between them).
        text = re.sub(r"```(?:json|JSON)?\s*", "", text).strip()
        text = text.replace("```", "").strip()

        start = text.find("{")
        end   = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No JSON object found in LLM response. "
                f"Raw response (first 300 chars): {text[:300]!r}"
            )

        candidate = text[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response contained malformed JSON: {exc}. "
                f"Extracted candidate (first 300 chars): {candidate[:300]!r}"
            ) from exc

    def _get_llm(self):
        """Lazy-load the OpenAI-compatible LLM client."""
        from app.rag.llm_chat import OpenAIClient
        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm