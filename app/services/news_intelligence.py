from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.core.formatting import (
    round_sentiment,
    round_weight,
    utc_now_iso,
)
from app.core.logging_config import get_logger
from app.services.news_recency import NewsRecencyFilter

logger = get_logger("news_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# Source quality classification  (unchanged from v3)
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

_MAX_SNIPPET_CHARS: int = 400
_FETCH_TIMEOUT_S: int = 12
_MAX_RETRIES: int = 2
_RETRY_DELAY_S: float = 1.5
_DEFAULT_CREDIBILITY: float = 0.55

# Characters contributed by each article to the situation summary.
_SUMMARY_SNIPPET_CHARS: int = 180


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str
    sentiment: str = "neutral"
    # All float fields use canonical precision from app.core.formatting
    sentiment_score: float = 0.0  # round_sentiment() — 3 d.p.
    severity_score: float = 0.0  # round_sentiment() — 3 d.p. (same scale)
    credibility_score: float = _DEFAULT_CREDIBILITY  # 3 d.p.
    final_weight: float = 0.5  # round_weight() — 3 d.p.

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

    New fields (v4)
    ---------------
    articles_retrieved : Total articles before recency filtering.
    articles_kept      : Articles that passed the lookback filter.
    recency_note       : Human-readable filter summary included in narratives.
    generated_at       : UTC ISO-8601 timestamp of brief creation.
    """

    ticker: str
    situation_summary: str
    # Kept empty for API schema backward compatibility (removed from frontend v4)
    bullish_catalysts: list[str] = field(default_factory=list)
    bearish_catalysts: list[str] = field(default_factory=list)
    aggregate_sentiment: str = "neutral"
    sentiment_score: float = 0.0  # round_sentiment() — 3 d.p.
    top_news: list[NewsItem] = field(default_factory=list)
    source_quality_note: str = ""
    retrieval_success: bool = True
    error_message: str = ""
    # v4 additions
    articles_retrieved: int = 0
    articles_kept: int = 0
    recency_note: str = ""
    generated_at: str = field(default_factory=utc_now_iso)


# ─────────────────────────────────────────────────────────────────────────────
# Source quality helpers  (unchanged)
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
# News Retriever  (unchanged from v3)
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
            "[%s] Retrieved %d news items (after dedup/domain-filter)",
            ticker,
            len(result),
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
# News Analyzer  — uses canonical formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


class NewsAnalyzer:
    """
    Composite weight = credibility × 0.40 + severity × 0.30 + |sentiment| × 0.30

    All score fields are rounded via canonical helpers before assignment so
    downstream JSON serialisation is deterministic.
    """

    def analyze(self, items: list[NewsItem]) -> list[NewsItem]:
        for item in items:
            text = (item.title + " " + item.snippet).lower()

            bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
            bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

            if bull > bear:
                item.sentiment = "positive"
                raw_score = min(1.0, bull / max(bull + bear, 1))
            elif bear > bull:
                item.sentiment = "negative"
                raw_score = -min(1.0, bear / max(bull + bear, 1))
            else:
                item.sentiment = "neutral"
                raw_score = 0.0

            item.sentiment_score = round_sentiment(raw_score)

            item.severity_score = round_sentiment(
                min(1.0, sum(1 for kw in SEVERITY_KEYWORDS if kw in text) / 3.0)
            )
            item.final_weight = round_weight(
                item.credibility_score * 0.40
                + item.severity_score * 0.30
                + abs(item.sentiment_score) * 0.30
            )

        items.sort(key=lambda x: x.final_weight, reverse=True)
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Intelligence Summarizer  — deterministic output format
# ─────────────────────────────────────────────────────────────────────────────


class IntelligenceSummarizer:
    """
    Converts scored news items into an institutional-style market brief.

    Output format contract
    ----------------------
    ``situation_summary`` always follows this canonical structure::

        Market sentiment for {TICKER} is {sentiment} based on {N}
        source-weighted, recency-filtered articles ({recency_note}).
        {article_1_title}: {trimmed_snippet}. {article_2_title}: ...

    ``sentiment_score`` is rounded via ``round_sentiment()`` (3 d.p.).
    ``generated_at``    is a canonical UTC ISO-8601 string.
    """

    def summarize(
        self,
        ticker: str,
        items: list[NewsItem],
        articles_retrieved: int = 0,
        recency_note: str = "",
    ) -> IntelligenceBrief:
        if not items:
            return IntelligenceBrief(
                ticker=ticker,
                situation_summary=(
                    "No reliable financial news retrieved for this ticker "
                    f"within the recency window. {recency_note}".strip()
                ),
                aggregate_sentiment="neutral",
                sentiment_score=0.0,
                top_news=[],
                articles_retrieved=articles_retrieved,
                articles_kept=0,
                recency_note=recency_note,
                retrieval_success=False,
                error_message="No items passed recency filter.",
                generated_at=utc_now_iso(),
            )

        # ── Weighted aggregate sentiment ──────────────────────────────────────
        total_weight = sum(i.final_weight for i in items) or 1.0
        agg_score_raw = (
            sum(i.sentiment_score * i.final_weight for i in items) / total_weight
        )
        agg_score = round_sentiment(agg_score_raw)

        if agg_score > 0.10:
            agg_sentiment = "positive"
        elif agg_score < -0.10:
            agg_sentiment = "negative"
        else:
            agg_sentiment = "neutral"

        # ── Build situation summary from snippets ─────────────────────────────
        top_items = items[:3]
        extracts: list[str] = []
        for item in top_items:
            body = re.sub(r"\s+", " ", (item.snippet or "").strip())
            if len(body) >= 40:
                trimmed = body[:_SUMMARY_SNIPPET_CHARS].rsplit(" ", 1)[0].rstrip(".,;")
                extracts.append(f"{item.title}: {trimmed}.")
            else:
                extracts.append(item.title.strip().rstrip(".") + ".")

        summary_body = " ".join(extracts)
        recency_clause = f" ({recency_note})" if recency_note else ""
        situation = (
            f"Market sentiment for {ticker} is {agg_sentiment} "
            f"based on {len(items)} source-weighted, recency-filtered "
            f"articles{recency_clause}. "
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
            bullish_catalysts=[],
            bearish_catalysts=[],
            aggregate_sentiment=agg_sentiment,
            sentiment_score=agg_score,
            top_news=items,
            source_quality_note=source_note,
            retrieval_success=True,
            articles_retrieved=articles_retrieved,
            articles_kept=len(items),
            recency_note=recency_note,
            generated_at=utc_now_iso(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Facade  — wires in recency filter
# ─────────────────────────────────────────────────────────────────────────────


class FinancialIntelligenceService:
    """
    End-to-end financial intelligence facade.

    Always returns an ``IntelligenceBrief``; never raises.

    Parameters
    ----------
    top_k:
        Maximum articles to retrieve per query set.
    unknown_date_policy:
        What to do with articles whose publish date cannot be determined.
        ``"accept_with_penalty"`` (default) halves the ``final_weight`` of
        undated articles rather than discarding them outright, striking a
        balance between completeness and staleness risk.
    """

    def __init__(
        self,
        top_k: int = 8,
        unknown_date_policy: str = "accept_with_penalty",
    ) -> None:
        self._retriever = NewsRetriever(top_k=top_k)
        self._analyzer = NewsAnalyzer()
        self._summarizer = IntelligenceSummarizer()
        self._unknown_date_policy = unknown_date_policy

    def get_brief(
        self,
        ticker: str,
        horizon: str = "1d",
    ) -> IntelligenceBrief:
        """
        Retrieve, filter by recency, analyse, and summarise news for *ticker*.

        Parameters
        ----------
        ticker:  Stock ticker symbol.
        horizon: Prediction horizon key — determines the lookback window.
                 Defaults to ``"1d"`` (3-day lookback).
        """
        try:
            # ── 1. Retrieve raw items ─────────────────────────────────────────
            raw_items = self._retriever.retrieve(ticker)
            articles_retrieved = len(raw_items)

            # ── 2. Recency filter ─────────────────────────────────────────────
            recency_filter = NewsRecencyFilter(
                horizon=horizon,
                unknown_date_policy=self._unknown_date_policy,
            )
            items, dropped = recency_filter.apply(raw_items)

            recency_note = (
                f"max age {recency_filter.max_age_days}d; "
                f"{len(dropped)} article(s) discarded as stale"
            )
            logger.info(
                "[%s/%s] Recency filter: %d retrieved, %d kept, %d dropped",
                ticker,
                horizon,
                articles_retrieved,
                len(items),
                len(dropped),
            )

            # ── 3. Analyse remaining items ────────────────────────────────────
            items = self._analyzer.analyze(items)

            # ── 4. Summarise ──────────────────────────────────────────────────
            brief = self._summarizer.summarize(
                ticker,
                items,
                articles_retrieved=articles_retrieved,
                recency_note=recency_note,
            )

            logger.info(
                "[%s/%s] Brief: sentiment=%s score=%.3f items=%d "
                "(retrieved=%d kept=%d)",
                ticker,
                horizon,
                brief.aggregate_sentiment,
                brief.sentiment_score,
                len(items),
                articles_retrieved,
                len(items),
            )
            return brief

        except Exception as exc:
            logger.warning(
                "[%s/%s] Intelligence retrieval failed: %s", ticker, horizon, exc
            )
            return IntelligenceBrief(
                ticker=ticker,
                situation_summary="Intelligence retrieval unavailable.",
                aggregate_sentiment="neutral",
                sentiment_score=0.0,
                top_news=[],
                articles_retrieved=0,
                articles_kept=0,
                recency_note="",
                retrieval_success=False,
                error_message=str(exc),
                generated_at=utc_now_iso(),
            )
