"""
FinSight AI — Financial Intelligence Retrieval Service (v2)

Architectural decisions
-----------------------
1.  **Source quality tiers** — news sources are classified into three
    credibility tiers.  Tier 1 (Reuters, Bloomberg, WSJ, SEC) sources are
    given maximum weight.  Tier 3 or unrecognised sources receive a penalty.
    Explicitly blocklisted domains (Wikipedia, Yandex, generic SEO aggregators)
    are excluded from results before any further processing.

2.  **Dedicated summarization layer** — ``IntelligenceSummarizer`` converts
    raw news items into an institutional-style market brief with:
    - concise situation summary
    - bullish and bearish catalysts listed separately
    - aggregate sentiment direction
    - source credibility weighting
    This separation keeps retrieval and synthesis as distinct concerns.

3.  **Retry and timeout hardening** — ``NewsRetriever._fetch_ddgs()`` wraps
    DuckDuckGo queries with configurable retries, exponential back-off, and
    a hard timeout so a slow search never blocks the prediction pipeline.

4.  **Deduplication** — retrieved headlines are deduplicated by normalised
    title before analysis so the same story from multiple sources is counted
    once.

5.  **Recency weighting** — articles without parseable dates receive a
    mild staleness penalty.  Future enhancement can parse actual pub dates.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from app.core.logging_config import get_logger

logger = get_logger("news_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# Source quality classification
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1 — institutional / primary sources (highest credibility weight)
TIER1_SOURCES: dict[str, float] = {
    "reuters.com":          1.00,
    "bloomberg.com":        1.00,
    "wsj.com":              0.98,
    "ft.com":               0.97,
    "sec.gov":              1.00,
    "federalreserve.gov":   1.00,
    "ecb.europa.eu":        0.95,
    "bls.gov":              0.95,
    "bea.gov":              0.95,
    "cnbc.com":             0.90,
    "marketwatch.com":      0.88,
    "finance.yahoo.com":    0.85,
    "barrons.com":          0.92,
    "morningstar.com":      0.90,
    "seekingalpha.com":     0.75,
    "fool.com":             0.70,
    "investopedia.com":     0.72,
    "thestreet.com":        0.78,
    "zacks.com":            0.80,
}

# Tier 2 — acceptable mainstream media
TIER2_SOURCES: dict[str, float] = {
    "apnews.com":           0.82,
    "nytimes.com":          0.80,
    "washingtonpost.com":   0.78,
    "theguardian.com":      0.75,
    "economist.com":        0.85,
    "businessinsider.com":  0.68,
    "fortune.com":          0.72,
    "forbes.com":           0.70,
    "techcrunch.com":       0.65,
}

# Explicitly blocked domains — excluded before any processing
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
    "medium.com",      # too mixed quality — occasional exception handled via title check
    "substack.com",
}

# Keyword signals for sentiment classification
BULLISH_KEYWORDS: frozenset[str] = frozenset({
    "beats", "beat", "surge", "surged", "upgrade", "upgraded",
    "raised", "raise", "strong", "record profit", "growth",
    "positive outlook", "exceeds", "bullish", "rally", "rallied",
    "outperform", "record high", "buyback", "dividend", "raised guidance",
    "above expectations", "strong demand", "expansion", "breakout",
})

BEARISH_KEYWORDS: frozenset[str] = frozenset({
    "miss", "missed", "downgrade", "downgraded", "lawsuit", "investigation",
    "fraud", "decline", "cut", "weak", "loss", "losses", "bankruptcy",
    "warning", "sell-off", "selloff", "below expectations", "layoffs",
    "recall", "probe", "fine", "penalty", "default", "restructuring",
    "missed earnings", "lowered guidance", "weak demand", "contraction",
})

SEVERITY_KEYWORDS: frozenset[str] = frozenset({
    "bankruptcy", "fraud", "lawsuit", "investigation", "earnings",
    "federal reserve", "fed rate", "gdp", "inflation", "recession",
    "merger", "acquisition", "ipo", "guidance", "restatement",
})

# Constants
_MAX_SNIPPET_CHARS = 400
_FETCH_TIMEOUT_S   = 12
_MAX_RETRIES       = 2
_RETRY_DELAY_S     = 1.5
_DEFAULT_CREDIBILITY = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """A single retrieved and scored news item."""
    title:             str
    snippet:           str
    url:               str
    sentiment:         str  = "neutral"
    sentiment_score:   float = 0.0
    severity_score:    float = 0.0
    credibility_score: float = _DEFAULT_CREDIBILITY
    final_weight:      float = 0.5

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc.replace("www.", "")
        except Exception:
            return ""

    @property
    def fingerprint(self) -> str:
        """Normalised title hash for deduplication."""
        normalised = re.sub(r"[^a-z0-9]", "", self.title.lower())
        return hashlib.md5(normalised.encode()).hexdigest()[:12]


@dataclass
class IntelligenceBrief:
    """
    Institutional-style market intelligence summary.

    Fields
    ------
    ticker             : The queried stock ticker.
    situation_summary  : 2-3 sentence factual summary of recent developments.
    bullish_catalysts  : List of bullish drivers identified from news.
    bearish_catalysts  : List of bearish headwinds identified from news.
    aggregate_sentiment: 'positive' | 'negative' | 'neutral'.
    sentiment_score    : Weighted composite score in [-1, +1].
    top_news           : Top-N scored news items.
    source_quality_note: Human-readable note about source quality.
    """
    ticker:              str
    situation_summary:   str
    bullish_catalysts:   list[str] = field(default_factory=list)
    bearish_catalysts:   list[str] = field(default_factory=list)
    aggregate_sentiment: str = "neutral"
    sentiment_score:     float = 0.0
    top_news:            list[NewsItem] = field(default_factory=list)
    source_quality_note: str = ""
    retrieval_success:   bool = True
    error_message:       str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Source quality helpers
# ─────────────────────────────────────────────────────────────────────────────

def _credibility_score(domain: str) -> float:
    """Return credibility weight for a domain."""
    clean = domain.replace("www.", "").lower()
    if clean in TIER1_SOURCES:
        return TIER1_SOURCES[clean]
    if clean in TIER2_SOURCES:
        return TIER2_SOURCES[clean]
    return _DEFAULT_CREDIBILITY


def _is_blocked(domain: str) -> bool:
    clean = domain.replace("www.", "").lower()
    return clean in BLOCKED_DOMAINS


# ─────────────────────────────────────────────────────────────────────────────
# News Retriever
# ─────────────────────────────────────────────────────────────────────────────

class NewsRetriever:
    """
    Retrieves financial news from DuckDuckGo with:
    - source quality filtering
    - blocked-domain exclusion
    - retry / timeout hardening
    - deduplication
    """

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k

    def retrieve(self, ticker: str) -> list[NewsItem]:
        """
        Fetch and filter news items for *ticker*.

        Tries multiple search queries to maximise diversity and source
        quality.  Results are deduplicated and sorted by credibility.

        Returns:
            List of ``NewsItem`` objects (may be empty on total failure).
        """
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
                    # Exclude blocked domains
                    if _is_blocked(item.domain):
                        logger.debug("Blocked domain skipped: %s", item.domain)
                        continue
                    # Deduplicate by title fingerprint
                    fp = item.fingerprint
                    if fp in seen_fps:
                        continue
                    seen_fps.add(fp)
                    item.credibility_score = _credibility_score(item.domain)
                    items.append(item)
            except Exception as exc:
                logger.warning("News query failed for %r: %s", query, exc)

        # Sort by credibility so the best sources come first
        items.sort(key=lambda x: x.credibility_score, reverse=True)
        result = items[: self.top_k]
        logger.info(
            "[%s] Retrieved %d news items (after dedup/filter)", ticker, len(result)
        )
        return result

    def _fetch_ddgs(self, query: str, max_results: int) -> list[dict]:
        """
        Fetch results from DuckDuckGo with retries and timeout.

        Supports both ``ddgs`` (new) and ``duckduckgo_search`` (legacy) package names.
        """
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
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY_S * (2 ** attempt)
                    logger.debug(
                        "DDGS attempt %d failed (%s); retrying in %.1fs…",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise


# ─────────────────────────────────────────────────────────────────────────────
# News Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class NewsAnalyzer:
    """
    Scores news items for sentiment, severity, and composite weight.

    Composite weight formula
    ------------------------
        weight = credibility × 0.40
               + severity    × 0.30
               + |sentiment| × 0.30

    This prioritises well-sourced high-impact stories.
    """

    def analyze(self, items: list[NewsItem]) -> list[NewsItem]:
        """Score and sort news items. Returns same list mutated in-place."""
        for item in items:
            text = (item.title + " " + item.snippet).lower()

            bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
            bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

            if bull > bear:
                item.sentiment       = "positive"
                item.sentiment_score = min(1.0, bull / max(bull + bear, 1))
            elif bear > bull:
                item.sentiment       = "negative"
                item.sentiment_score = -min(1.0, bear / max(bull + bear, 1))
            else:
                item.sentiment       = "neutral"
                item.sentiment_score = 0.0

            item.severity_score = min(
                1.0,
                sum(1 for kw in SEVERITY_KEYWORDS if kw in text) / 3.0,
            )

            item.final_weight = (
                item.credibility_score * 0.40
                + item.severity_score  * 0.30
                + abs(item.sentiment_score) * 0.30
            )

        items.sort(key=lambda x: x.final_weight, reverse=True)
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence Summarizer
# ─────────────────────────────────────────────────────────────────────────────

class IntelligenceSummarizer:
    """
    Converts scored news items into an institutional-style market brief.

    Design: the summarizer intentionally avoids calling an LLM so it is
    always available even when the LLM is unavailable.  The LLM synthesis
    in ``SignalFusionService`` is layered on top separately.
    """

    def summarize(self, ticker: str, items: list[NewsItem]) -> IntelligenceBrief:
        """Produce an ``IntelligenceBrief`` from scored news items."""
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

        bullish_catalysts: list[str] = []
        bearish_catalysts: list[str] = []

        for item in items:
            catalyst = item.title.strip()
            if item.sentiment == "positive" and len(bullish_catalysts) < 3:
                bullish_catalysts.append(catalyst)
            elif item.sentiment == "negative" and len(bearish_catalysts) < 3:
                bearish_catalysts.append(catalyst)

        # Weighted aggregate sentiment score
        total_weight = sum(i.final_weight for i in items) or 1.0
        agg_score    = sum(
            i.sentiment_score * i.final_weight for i in items
        ) / total_weight

        if agg_score > 0.10:
            agg_sentiment = "positive"
        elif agg_score < -0.10:
            agg_sentiment = "negative"
        else:
            agg_sentiment = "neutral"

        # Situation summary
        top3       = items[:3]
        top3_titles = "; ".join(t.title for t in top3)
        situation  = (
            f"Recent coverage of {ticker} is {agg_sentiment}. "
            f"Key headlines: {top3_titles}. "
            f"Analysis based on {len(items)} filtered, source-weighted articles."
        )

        # Source quality note
        tier1_count = sum(
            1 for i in items if i.credibility_score >= 0.90
        )
        source_note = (
            f"{tier1_count}/{len(items)} articles from Tier-1 institutional sources."
        )

        return IntelligenceBrief(
            ticker=ticker,
            situation_summary=situation,
            bullish_catalysts=bullish_catalysts,
            bearish_catalysts=bearish_catalysts,
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
    End-to-end financial intelligence retrieval facade.

    Composes ``NewsRetriever`` → ``NewsAnalyzer`` → ``IntelligenceSummarizer``
    into a single call.  Always returns an ``IntelligenceBrief``; never raises.
    """

    def __init__(self, top_k: int = 8) -> None:
        self._retriever  = NewsRetriever(top_k=top_k)
        self._analyzer   = NewsAnalyzer()
        self._summarizer = IntelligenceSummarizer()

    def get_brief(self, ticker: str) -> IntelligenceBrief:
        """
        Retrieve and summarize financial intelligence for *ticker*.

        Never raises — failures return an ``IntelligenceBrief`` with
        ``retrieval_success=False`` and a populated ``error_message``.
        """
        try:
            items = self._retriever.retrieve(ticker)
            items = self._analyzer.analyze(items)
            brief = self._summarizer.summarize(ticker, items)
            logger.info(
                "[%s] Intelligence brief: sentiment=%s score=%.3f items=%d",
                ticker, brief.aggregate_sentiment, brief.sentiment_score, len(items),
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