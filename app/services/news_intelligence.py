"""
FinSight AI — Financial Intelligence Retrieval Service (v3)

Changes vs v2
-------------
1.  **Snippet-based situation summary** — ``IntelligenceSummarizer.summarize()``
    now builds the ``situation_summary`` from the actual article snippets
    (the body text retrieved from each source), not just the headlines.
    Each top article contributes a condensed extract so the summary reflects
    real content rather than keyword-level title strings.

2.  **Catalyst extraction removed** — ``bullish_catalysts`` and
    ``bearish_catalysts`` are no longer populated.  The fields remain on
    ``IntelligenceBrief`` as empty lists for API schema backward compatibility,
    but the frontend no longer renders them (removed in dashboard v4).

All retrieval, scoring, source-quality, deduplication, and retry logic is
unchanged from v2.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.core.logging_config import get_logger

logger = get_logger("news_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# Source quality classification  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

TIER1_SOURCES: dict[str, float] = {
    "reuters.com": 1.00,
    "bloomberg.com": 1.00,
    "wsj.com": 0.98,
    "ft.com": 0.97,
    "sec.gov": 1.00,
    "federalreserve.gov": 1.00,
    "ecb.europa.eu": 0.95,
    "bls.gov": 0.95,
    "bea.gov": 0.95,
    "cnbc.com": 0.90,
    "marketwatch.com": 0.88,
    "finance.yahoo.com": 0.85,
    "barrons.com": 0.92,
    "morningstar.com": 0.90,
    "seekingalpha.com": 0.75,
    "fool.com": 0.70,
    "investopedia.com": 0.72,
    "thestreet.com": 0.78,
    "zacks.com": 0.80,
}

TIER2_SOURCES: dict[str, float] = {
    "apnews.com": 0.82,
    "nytimes.com": 0.80,
    "washingtonpost.com": 0.78,
    "theguardian.com": 0.75,
    "economist.com": 0.85,
    "businessinsider.com": 0.68,
    "fortune.com": 0.72,
    "forbes.com": 0.70,
    "techcrunch.com": 0.65,
}

BLOCKED_DOMAINS: set[str] = {
    "wikipedia.org",
    "en.wikipedia.org",
    "wikidata.org",
    "grokipedia.org",
    "yandex.com",
    "yandex.ru",
    "ask.com",
    "answers.com",
    "quora.com",
    "reddit.com",
    "pinterest.com",
    "tumblr.com",
    "medium.com",
    "substack.com",
}

BULLISH_KEYWORDS: frozenset[str] = frozenset(
    {
        "beats",
        "beat",
        "surge",
        "surged",
        "upgrade",
        "upgraded",
        "raised",
        "raise",
        "strong",
        "record profit",
        "growth",
        "positive outlook",
        "exceeds",
        "bullish",
        "rally",
        "rallied",
        "outperform",
        "record high",
        "buyback",
        "dividend",
        "raised guidance",
        "above expectations",
        "strong demand",
        "expansion",
        "breakout",
    }
)

BEARISH_KEYWORDS: frozenset[str] = frozenset(
    {
        "miss",
        "missed",
        "downgrade",
        "downgraded",
        "lawsuit",
        "investigation",
        "fraud",
        "decline",
        "cut",
        "weak",
        "loss",
        "losses",
        "bankruptcy",
        "warning",
        "sell-off",
        "selloff",
        "below expectations",
        "layoffs",
        "recall",
        "probe",
        "fine",
        "penalty",
        "default",
        "restructuring",
        "missed earnings",
        "lowered guidance",
        "weak demand",
        "contraction",
    }
)

SEVERITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "bankruptcy",
        "fraud",
        "lawsuit",
        "investigation",
        "earnings",
        "federal reserve",
        "fed rate",
        "gdp",
        "inflation",
        "recession",
        "merger",
        "acquisition",
        "ipo",
        "guidance",
        "restatement",
    }
)

_MAX_SNIPPET_CHARS = 400
_FETCH_TIMEOUT_S = 12
_MAX_RETRIES = 2
_RETRY_DELAY_S = 1.5
_DEFAULT_CREDIBILITY = 0.55

# Maximum characters from each article's snippet to include in the situation summary
_SUMMARY_SNIPPET_CHARS = 180


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    severity_score: float = 0.0
    credibility_score: float = _DEFAULT_CREDIBILITY
    final_weight: float = 0.5

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc.replace("www.", "")
        except Exception:
            return ""

    @property
    def fingerprint(self) -> str:
        normalised = re.sub(r"[^a-z0-9]", "", self.title.lower())
        return hashlib.md5(normalised.encode()).hexdigest()[:12]


@dataclass
class IntelligenceBrief:
    """
    Institutional-style market intelligence summary.

    ``bullish_catalysts`` and ``bearish_catalysts`` are retained for API
    schema backward compatibility but are always empty in v3 — the
    frontend no longer renders them.
    """

    ticker: str
    situation_summary: str
    bullish_catalysts: list[str] = field(default_factory=list)  # kept for compat
    bearish_catalysts: list[str] = field(default_factory=list)  # kept for compat
    aggregate_sentiment: str = "neutral"
    sentiment_score: float = 0.0
    top_news: list[NewsItem] = field(default_factory=list)
    source_quality_note: str = ""
    retrieval_success: bool = True
    error_message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Source quality helpers
# ─────────────────────────────────────────────────────────────────────────────


def _credibility_score(domain: str) -> float:
    clean = domain.replace("www.", "").lower()
    if clean in TIER1_SOURCES:
        return TIER1_SOURCES[clean]
    if clean in TIER2_SOURCES:
        return TIER2_SOURCES[clean]
    return _DEFAULT_CREDIBILITY


def _is_blocked(domain: str) -> bool:
    return domain.replace("www.", "").lower() in BLOCKED_DOMAINS


# ─────────────────────────────────────────────────────────────────────────────
# News Retriever
# ─────────────────────────────────────────────────────────────────────────────


class NewsRetriever:
    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k

    def retrieve(self, ticker: str) -> list[NewsItem]:
        queries = [
            f"{ticker} stock earnings results quarterly",
            f"{ticker} analyst rating price target",
            f"{ticker} SEC filing guidance outlook",
        ]
        seen_fps: set[str] = set()
        items: list[NewsItem] = []

        for query in queries:
            try:
                raw = self._fetch_ddgs(query, max_results=self.top_k)
                for r in raw:
                    item = NewsItem(
                        title=r.get("title", ""),
                        snippet=r.get("body", "")[:_MAX_SNIPPET_CHARS],
                        url=r.get("href", ""),
                    )
                    if _is_blocked(item.domain):
                        logger.debug("Blocked domain skipped: %s", item.domain)
                        continue
                    fp = item.fingerprint
                    if fp in seen_fps:
                        continue
                    seen_fps.add(fp)
                    item.credibility_score = _credibility_score(item.domain)
                    items.append(item)
            except Exception as exc:
                logger.warning("News query failed for %r: %s", query, exc)

        items.sort(key=lambda x: x.credibility_score, reverse=True)
        result = items[: self.top_k]
        logger.info(
            "[%s] Retrieved %d news items (after dedup/filter)", ticker, len(result)
        )
        return result

    def _fetch_ddgs(self, query: str, max_results: int) -> list[dict]:
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            pass
        if DDGS is None:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise RuntimeError(
                    "Web search package not installed. Run: pip install ddgs"
                ) from exc

        for attempt in range(_MAX_RETRIES + 1):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))
            except Exception:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY_S * (2**attempt)
                    logger.debug(
                        "DDGS attempt %d failed; retrying in %.1fs…", attempt + 1, wait
                    )
                    time.sleep(wait)
                else:
                    raise


# ─────────────────────────────────────────────────────────────────────────────
# News Analyzer  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────


class NewsAnalyzer:
    """
    Composite weight = credibility×0.40 + severity×0.30 + |sentiment|×0.30
    """

    def analyze(self, items: list[NewsItem]) -> list[NewsItem]:
        for item in items:
            text = (item.title + " " + item.snippet).lower()

            bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
            bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

            if bull > bear:
                item.sentiment = "positive"
                item.sentiment_score = min(1.0, bull / max(bull + bear, 1))
            elif bear > bull:
                item.sentiment = "negative"
                item.sentiment_score = -min(1.0, bear / max(bull + bear, 1))
            else:
                item.sentiment = "neutral"
                item.sentiment_score = 0.0

            item.severity_score = min(
                1.0,
                sum(1 for kw in SEVERITY_KEYWORDS if kw in text) / 3.0,
            )
            item.final_weight = (
                item.credibility_score * 0.40
                + item.severity_score * 0.30
                + abs(item.sentiment_score) * 0.30
            )

        items.sort(key=lambda x: x.final_weight, reverse=True)
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence Summarizer  (v3 — snippet-based summary)
# ─────────────────────────────────────────────────────────────────────────────


class IntelligenceSummarizer:
    """
    Converts scored news items into an institutional-style market brief.

    ``situation_summary`` is now constructed from the article snippets
    (body text), not just headlines.  Each of the top articles contributes
    a trimmed extract so the summary reflects actual reported content.
    Catalyst lists are intentionally left empty (frontend removed them).
    """

    def summarize(self, ticker: str, items: list[NewsItem]) -> IntelligenceBrief:
        if not items:
            return IntelligenceBrief(
                ticker=ticker,
                situation_summary="No reliable financial news retrieved for this ticker.",
                aggregate_sentiment="neutral",
                sentiment_score=0.0,
                top_news=[],
                retrieval_success=False,
                error_message="No items retrieved.",
            )

        # ── Weighted aggregate sentiment ──────────────────────────────────────
        total_weight = sum(i.final_weight for i in items) or 1.0
        agg_score = (
            sum(i.sentiment_score * i.final_weight for i in items) / total_weight
        )

        if agg_score > 0.10:
            agg_sentiment = "positive"
        elif agg_score < -0.10:
            agg_sentiment = "negative"
        else:
            agg_sentiment = "neutral"

        # ── Build situation summary from snippets ─────────────────────────────
        # Use top-3 items by final_weight.  For each, extract the snippet
        # (body text) trimmed to _SUMMARY_SNIPPET_CHARS.  If the snippet is
        # empty or too short, fall back to the title.
        top_items = items[:3]
        extracts: list[str] = []
        for item in top_items:
            body = (item.snippet or "").strip()
            # Strip redundant whitespace and truncate cleanly at a word boundary
            body = re.sub(r"\s+", " ", body)
            if len(body) >= 40:
                trimmed = body[:_SUMMARY_SNIPPET_CHARS].rsplit(" ", 1)[0].rstrip(".,;")
                extracts.append(f"{item.title}: {trimmed}.")
            else:
                # Fallback to headline only when snippet is absent/trivial
                extracts.append(item.title.strip().rstrip(".") + ".")

        summary_body = " ".join(extracts)
        situation = (
            f"Market sentiment for {ticker} is {agg_sentiment} "
            f"based on {len(items)} source-weighted articles. "
            f"{summary_body}"
        )

        # ── Source quality note ───────────────────────────────────────────────
        tier1_count = sum(1 for i in items if i.credibility_score >= 0.90)
        source_note = (
            f"{tier1_count}/{len(items)} articles from Tier-1 institutional sources."
        )

        return IntelligenceBrief(
            ticker=ticker,
            situation_summary=situation,
            # catalysts intentionally empty — frontend removed them
            bullish_catalysts=[],
            bearish_catalysts=[],
            aggregate_sentiment=agg_sentiment,
            sentiment_score=round(agg_score, 4),
            top_news=items,
            source_quality_note=source_note,
            retrieval_success=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Facade
# ─────────────────────────────────────────────────────────────────────────────


class FinancialIntelligenceService:
    """
    End-to-end financial intelligence facade.
    Always returns an ``IntelligenceBrief``; never raises.
    """

    def __init__(self, top_k: int = 8) -> None:
        self._retriever = NewsRetriever(top_k=top_k)
        self._analyzer = NewsAnalyzer()
        self._summarizer = IntelligenceSummarizer()

    def get_brief(self, ticker: str) -> IntelligenceBrief:
        try:
            items = self._retriever.retrieve(ticker)
            items = self._analyzer.analyze(items)
            brief = self._summarizer.summarize(ticker, items)
            logger.info(
                "[%s] Brief: sentiment=%s score=%.3f items=%d",
                ticker,
                brief.aggregate_sentiment,
                brief.sentiment_score,
                len(items),
            )
            return brief
        except Exception as exc:
            logger.warning("[%s] Intelligence retrieval failed: %s", ticker, exc)
            return IntelligenceBrief(
                ticker=ticker,
                situation_summary="Intelligence retrieval unavailable.",
                aggregate_sentiment="neutral",
                sentiment_score=0.0,
                top_news=[],
                retrieval_success=False,
                error_message=str(exc),
            )
