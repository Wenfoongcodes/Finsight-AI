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
from app.services.news_recency import (
    HORIZON_MAX_AGE_DAYS,
    DEFAULT_MAX_AGE_DAYS,
    NewsRecencyFilter,
)

logger = get_logger("news_retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# Source quality classification
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

# Domains blocked regardless of article classification.
# Kept intentionally short — the news endpoint already excludes price-data
# aggregators categorically.  This list only needs to cover sources that DDG
# sometimes mis-classifies as news (e.g. aggregator landing pages, forums).
BLOCKED_DOMAINS: set[str] = {
    "wikipedia.org",
    "en.wikipedia.org",
    "wikidata.org",
    "quora.com",
    "reddit.com",
    "pinterest.com",
    "tumblr.com",
    "medium.com",
    "substack.com",
    "stockanalysis.com",
    "macrotrends.net",
    "tradingview.com",
    "finviz.com",
    "barchart.com",
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
_MAX_RETRIES: int = 2
_RETRY_DELAY_S: float = 1.5
_DEFAULT_CREDIBILITY: float = 0.55
_SUMMARY_SNIPPET_CHARS: int = 180

# Minimum words a body snippet must have.  News articles always produce
# multi-sentence snippets; the few non-article pages that slip through the
# news endpoint tend to return very short fragments.
_MIN_SNIPPET_WORDS: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# DDG timelimit mapping
# ─────────────────────────────────────────────────────────────────────────────

def _horizon_to_ddg_timelimit(horizon: str) -> str:
    """
    Return the tightest DuckDuckGo timelimit string for the given horizon.

    DDG news timelimit values: 'd' = past day, 'w' = past week, 'm' = past month.
    """
    max_age = HORIZON_MAX_AGE_DAYS.get(horizon, DEFAULT_MAX_AGE_DAYS)
    if max_age <= 3:
        return "d"
    if max_age <= 7:
        return "w"
    return "m"


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
    # Raw metadata from the search result.
    # The DDG news endpoint returns a structured "date" string; storing it here
    # lets ArticleDateExtractor find it without falling back to regex parsing.
    metadata: dict = field(default_factory=dict)

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
    """Institutional-style market intelligence summary."""

    ticker: str
    situation_summary: str
    bullish_catalysts: list[str] = field(default_factory=list)
    bearish_catalysts: list[str] = field(default_factory=list)
    aggregate_sentiment: str = "neutral"
    sentiment_score: float = 0.0
    top_news: list[NewsItem] = field(default_factory=list)
    source_quality_note: str = ""
    retrieval_success: bool = True
    error_message: str = ""
    articles_retrieved: int = 0
    articles_kept: int = 0
    recency_note: str = ""
    generated_at: str = field(default_factory=utc_now_iso)


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
    """
    Retrieves financial news articles via the DuckDuckGo *news* endpoint.

    Why .news() instead of .text()
    --------------------------------
    DDG's general text search (.text()) returns any page that ranks for
    the query keywords — price screeners, quote pages, OHLCV history tables,
    and financial data aggregators all rank highly for stock queries because
    they are heavily linked and keyword-dense.  No amount of post-hoc domain
    blocking or URL-pattern filtering can keep up with the open-ended
    population of such sites.

    DDG's news endpoint (.news()) only indexes content that DDG's crawler
    has classified as a news article.  Price-data pages are categorically
    absent — they are never classified as news — so the root cause is
    eliminated at retrieval time rather than patched downstream.

    The .news() result dict has slightly different field names from .text():
        .text()  → {"title", "href",  "body", "date", ...}
        .news()  → {"title", "url",   "body", "date", "source", ...}

    _parse_news_result() normalises both shapes into a common dict so the
    rest of the pipeline is unaffected.

    Remaining post-retrieval filters
    ---------------------------------
    The domain blocklist and snippet word-count check are kept as a
    lightweight safety net for the rare cases where DDG mis-classifies a
    non-article page as news (e.g. aggregator landing pages, press-release
    syndication services with very short copy).
    """

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k

    def retrieve(self, ticker: str, horizon: str = "1d") -> list[NewsItem]:
        """
        Retrieve deduplicated news articles for *ticker*.

        Parameters
        ----------
        ticker:  Stock ticker symbol (e.g. "AAPL").
        horizon: Prediction horizon key — selects DDG timelimit bucket and
                 is forwarded to the recency filter for lookback window.
        """
        timelimit = _horizon_to_ddg_timelimit(horizon)

        # Three complementary queries targeting different editorial angles.
        # Quoting the ticker prevents DDG from broadening to unrelated results.
        # The news endpoint already restricts to article content, so
        # -site: exclusions and keyword anchors like "news" are not needed —
        # concise queries produce better recall from the news index.
        queries = [
            f'"{ticker}" stock',
            f'"{ticker}" earnings analyst',
            f'"{ticker}" guidance SEC filing',
        ]

        seen_fps: set[str] = set()
        items: list[NewsItem] = []

        for query in queries:
            try:
                raw = self._fetch_ddgs_news(
                    query, max_results=self.top_k, timelimit=timelimit
                )
                for r in raw:
                    item = self._parse_news_result(r)
                    if item is None:
                        continue

                    # ── Safety net 1: domain blocklist ────────────────────────
                    if _is_blocked(item.domain):
                        logger.debug("Blocked domain: %s", item.domain)
                        continue

                    # ── Safety net 2: snippet word-count ──────────────────────
                    # News articles always produce multi-sentence snippets.
                    # Very short snippets are a signal of mis-classified pages.
                    if len(item.snippet.split()) < _MIN_SNIPPET_WORDS:
                        logger.debug(
                            "Snippet too short (%d words): %s",
                            len(item.snippet.split()),
                            item.url,
                        )
                        continue

                    # ── Deduplication ─────────────────────────────────────────
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
            "[%s] Retrieved %d news articles via DDG news endpoint "
            "(horizon=%s timelimit=%s)",
            ticker,
            len(result),
            horizon,
            timelimit,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_news_result(r: dict) -> NewsItem | None:
        """
        Normalise a raw DDG news result dict into a NewsItem.

        DDG .news() field map
        ---------------------
        "title"  → item.title
        "url"    → item.url       (note: .text() uses "href" instead)
        "body"   → item.snippet
        "date"   → item.metadata["date"]  (ISO-8601 string, usually present)
        "source" → item.metadata["source"] (publisher name, e.g. "Reuters")

        Returns None when the result is malformed (missing title or url).
        """
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()

        if not title or not url:
            return None

        item = NewsItem(
            title=title,
            snippet=r.get("body", "")[:_MAX_SNIPPET_CHARS],
            url=url,
        )

        # Persist structured date so ArticleDateExtractor finds it immediately
        # on the metadata lookup path rather than falling back to regex parsing.
        raw_date = r.get("date") or r.get("published")
        if raw_date:
            item.metadata["date"] = str(raw_date)

        raw_source = r.get("source")
        if raw_source:
            item.metadata["source"] = str(raw_source)

        return item

    def _fetch_ddgs_news(
        self,
        query: str,
        max_results: int,
        timelimit: str = "w",
    ) -> list[dict]:
        """
        Execute a DuckDuckGo *news* search.

        Uses DDGS.news() rather than DDGS.text().  The news endpoint only
        returns pages DDG has classified as news articles, which eliminates
        price-data pages, screeners, and financial data aggregators
        categorically — they are never present in the news index.

        Parameters
        ----------
        query:       Natural language search query.
        max_results: Maximum results to request.
        timelimit:   DDG recency filter — 'd' (day), 'w' (week), 'm' (month).
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
                    # .news() is the key change from all previous iterations.
                    # It is available on both the ddgs and duckduckgo_search
                    # packages and accepts the same timelimit parameter.
                    return list(
                        ddgs.news(
                            query,
                            max_results=max_results,
                            timelimit=timelimit,
                        )
                    )
            except Exception:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY_S * (2**attempt)
                    logger.debug(
                        "DDGS news attempt %d failed; retrying in %.1fs…",
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise


# ─────────────────────────────────────────────────────────────────────────────
# News Analyzer
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
# Intelligence Summarizer
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
# Facade
# ─────────────────────────────────────────────────────────────────────────────


class FinancialIntelligenceService:
    """
    End-to-end financial intelligence facade.

    Always returns an ``IntelligenceBrief``; never raises.

    Content quality is enforced at the retrieval primitive level — by using
    DDG's news endpoint instead of its text search endpoint — rather than by
    post-hoc filtering.  This is the correct architectural approach: the news
    endpoint only indexes pages DDG has classified as news articles, so
    price-data pages, screeners, and financial data aggregators are
    categorically absent from the result set regardless of query phrasing.

    A lightweight domain blocklist and snippet word-count check are retained
    as a safety net for the rare cases where DDG mis-classifies a non-article
    page as news.

    Recency enforcement
    --------------------
    Two layers, innermost to outermost:
    1. DDG news timelimit ('d'/'w'/'m') — server-side, horizon-aware.
    2. NewsRecencyFilter — local validation against the exact lookback window.
       unknown_date_policy="reject": articles with no parseable date are dropped.
       min_kept=0: the filter never rescues dropped articles.

    Parameters
    ----------
    top_k:
        Maximum articles to retrieve per query set.
    unknown_date_policy:
        Policy for articles whose publish date cannot be parsed.
        Defaults to "reject". The news endpoint reliably returns a structured
        "date" field, so date-less results are genuinely anomalous.
    """

    def __init__(
        self,
        top_k: int = 8,
        unknown_date_policy: str = "reject",
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
        horizon: Prediction horizon key — forwarded to both the retriever
                 (timelimit selection) and the recency filter (lookback window).
        """
        try:
            # ── 1. Retrieve via DDG news endpoint ─────────────────────────────
            raw_items = self._retriever.retrieve(ticker, horizon=horizon)
            articles_retrieved = len(raw_items)

            # ── 2. Recency filter ─────────────────────────────────────────────
            recency_filter = NewsRecencyFilter(
                horizon=horizon,
                unknown_date_policy=self._unknown_date_policy,
                min_kept=0,
            )
            items, dropped = recency_filter.apply(raw_items)

            recency_note = (
                f"max age {recency_filter.max_age_days}d; "
                f"{len(dropped)} article(s) discarded as stale or undated"
            )
            logger.info(
                "[%s/%s] Recency filter: %d retrieved, %d kept, %d dropped",
                ticker,
                horizon,
                articles_retrieved,
                len(items),
                len(dropped),
            )

            # ── 3. Sentiment + weight scoring ─────────────────────────────────
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