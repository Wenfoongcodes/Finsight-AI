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

BLOCKED_DOMAINS: set[str] = {
    # ── General noise ─────────────────────────────────────────────────────────
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
    # ── Price-data aggregators — no editorial content ─────────────────────────
    # These sites return price tables, screeners, and historical OHLCV data.
    # DDG ranks them highly for stock queries because they are heavily linked,
    # but they carry no news narrative for the LLM to reason over.
    "macrotrends.net",
    "stockanalysis.com",
    "tradingview.com",
    "finviz.com",
    "barchart.com",
    "wisesheets.io",
    "simplywall.st",
    "chartmill.com",
    "gurufocus.com",
    "stockscreener.com",
    "alphaquery.com",
    "dividendhistory.org",
    "stockhistory.app",
    "tickertape.in",
    "tickertape.com",
    "statmuse.com",
    "wallstreetzen.com",
    "investing.com",
    "stlouisfed.org",
}

# Domains that publish BOTH editorial content and price/quote pages.
# Items from these domains are checked against PRICE_PAGE_PATH_PATTERNS
# before being admitted — the domain-level credibility score applies only
# to URLs that pass the path check.
MIXED_CONTENT_DOMAINS: set[str] = {
    "finance.yahoo.com",
    "marketwatch.com",
    "cnbc.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "ft.com",
    "barrons.com",
}

# URL path segments that identify price/data pages on mixed-content domains.
# A URL whose path contains any of these strings is rejected regardless of
# the domain's credibility score.
PRICE_PAGE_PATH_PATTERNS: tuple[str, ...] = (
    "/quote/",
    "/quotes/",
    "/symbol/",
    "/stocks/",
    "/price/",
    "/chart/",
    "/history/",
    "/historical",
    "/financials/",
    "/balance-sheet",
    "/income-statement",
    "/cash-flow",
    "/ownership/",
    "/holders/",
    "/statistics/",
    "/key-statistics",
    "/screener",
    "/markets/stocks/",
    "/investing/stock/",
    "/market-data/",
)

# DDG -site: exclusions appended to every query string.
# Eliminates the noisiest price-data domains server-side before DDG transmits
# results, reducing both latency and the post-retrieval filtering burden.
_DDG_SITE_EXCLUSIONS: str = (
    " -site:macrotrends.net"
    " -site:stockanalysis.com"
    " -site:tradingview.com"
    " -site:finviz.com"
    " -site:barchart.com"
    " -site:wisesheets.io"
    " -site:simplywall.st"
    " -site:finance.yahoo.com/quote"
    " -site:marketwatch.com/investing/stock"
    " -site:investing.com"
)

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

# Minimum word count a snippet must have to be considered news prose.
# Price-data pages return fragments like "AAPL 185.20 +1.3% Open: 184.50"
# which are far too terse to carry any editorial content.
_MIN_SNIPPET_WORDS: int = 15

_SUMMARY_SNIPPET_CHARS: int = 180


# ─────────────────────────────────────────────────────────────────────────────
# DDG timelimit mapping
# ─────────────────────────────────────────────────────────────────────────────

def _horizon_to_ddg_timelimit(horizon: str) -> str:
    """Return the tightest DuckDuckGo timelimit string for the given horizon."""
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
    # Raw metadata from the search result (e.g. DDG's "date" field).
    # Stored here so ArticleDateExtractor can find a structured publish date
    # without having to parse free-form snippet text.
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
    """Return True when the domain is on the blocklist."""
    return domain.replace("www.", "").lower() in BLOCKED_DOMAINS


def _is_price_page(domain: str, url: str) -> bool:
    """
    Return True when a URL from a mixed-content domain resolves to a price
    or data page rather than an editorial article.

    Mixed-content domains (e.g. finance.yahoo.com, marketwatch.com) publish
    both editorial articles and price/quote/chart pages.  The domain-level
    credibility score is only meaningful for their editorial content, so URLs
    that match known data-page path patterns are rejected here regardless of
    how high the domain scores.

    Pure news domains (e.g. reuters.com article pages) are not affected
    because they are not in MIXED_CONTENT_DOMAINS.
    """
    clean_domain = domain.replace("www.", "").lower()
    if clean_domain not in MIXED_CONTENT_DOMAINS:
        return False
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in PRICE_PAGE_PATH_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# News Retriever
# ─────────────────────────────────────────────────────────────────────────────


class NewsRetriever:
    """
    Retrieves financial news articles via DuckDuckGo.

    Query strategy
    --------------
    Each query is phrased to signal editorial intent to DDG's ranking model.
    Words like "news", "report", "analysis", "announces", and "says" strongly
    bias DDG toward article pages rather than data/screener pages.
    ``_DDG_SITE_EXCLUSIONS`` is appended to every query to eliminate the
    noisiest price-data aggregators server-side.

    Post-retrieval filtering layers (applied in order)
    ---------------------------------------------------
    1. Domain blocklist      — ``_is_blocked()``  full domains with no
                               editorial content.
    2. Price-page URL filter — ``_is_price_page()``  quote/chart/screener
                               paths on mixed-content domains.
    3. Snippet word-count    — rejects fragments shorter than
                               ``_MIN_SNIPPET_WORDS`` words (price tickers,
                               data tables) that carry no news prose.
    4. Fingerprint dedup     — title-hash deduplication across queries.

    Recency
    -------
    ``horizon`` selects the tightest DDG ``timelimit`` bucket that still
    covers the prediction window, filtering stale results server-side before
    the local ``NewsRecencyFilter`` runs.
    """

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k

    def retrieve(self, ticker: str, horizon: str = "1d") -> list[NewsItem]:
        """
        Retrieve deduplicated news articles for *ticker*.

        Parameters
        ----------
        ticker:  Stock ticker symbol (e.g. "AAPL").
        horizon: Prediction horizon key — used to select the DDG time bucket
                 and communicate context for logging.
        """
        timelimit = _horizon_to_ddg_timelimit(horizon)

        # ── Query design ──────────────────────────────────────────────────────
        # Three complementary queries targeting different editorial content
        # types.  Quoting the ticker ("AAPL") prevents DDG from broadening to
        # unrelated results.  _DDG_SITE_EXCLUSIONS appended to every query
        # eliminates the highest-volume price-data domains server-side.
        queries = [
            # Breaking news and recent corporate events
            f'"{ticker}" stock news{_DDG_SITE_EXCLUSIONS}',
            # Analyst commentary: upgrades, downgrades, price-target changes
            f'"{ticker}" analyst report upgrade downgrade{_DDG_SITE_EXCLUSIONS}',
            # Earnings, guidance revisions, SEC filings, M&A
            f'"{ticker}" earnings guidance announcement{_DDG_SITE_EXCLUSIONS}',
        ]

        seen_fps: set[str] = set()
        items: list[NewsItem] = []

        for query in queries:
            try:
                raw = self._fetch_ddgs(
                    query, max_results=self.top_k, timelimit=timelimit
                )
                for r in raw:
                    url = r.get("href", "")
                    title = r.get("title", "")
                    snippet = r.get("body", "")

                    item = NewsItem(
                        title=title,
                        snippet=snippet[:_MAX_SNIPPET_CHARS],
                        url=url,
                    )

                    # Persist DDG's structured date into metadata so
                    # ArticleDateExtractor finds it without regex fallback.
                    raw_date = r.get("date") or r.get("published")
                    if raw_date:
                        item.metadata["date"] = str(raw_date)

                    # ── Filter layer 1: domain blocklist ──────────────────────
                    if _is_blocked(item.domain):
                        logger.debug("Blocked domain: %s", item.domain)
                        continue

                    # ── Filter layer 2: price-page URL pattern ────────────────
                    # Rejects quote/chart/screener URLs from mixed-content
                    # domains (e.g. finance.yahoo.com/quote/AAPL,
                    # marketwatch.com/investing/stock/aapl/charts).
                    if _is_price_page(item.domain, url):
                        logger.debug("Price page filtered: %s", url)
                        continue

                    # ── Filter layer 3: snippet word-count floor ──────────────
                    # Price-data pages produce terse fragments like
                    # "AAPL 185.20 +1.3% Open: 184.50 Vol: 54M" which are
                    # useless for sentiment analysis and LLM synthesis.
                    if len(snippet.split()) < _MIN_SNIPPET_WORDS:
                        logger.debug(
                            "Snippet too short (%d words), likely price data: %s",
                            len(snippet.split()),
                            url,
                        )
                        continue

                    # ── Filter layer 4: fingerprint deduplication ─────────────
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
            "[%s] Retrieved %d news items (horizon=%s ddg_timelimit=%s)",
            ticker,
            len(result),
            horizon,
            timelimit,
        )
        return result

    def _fetch_ddgs(
        self,
        query: str,
        max_results: int,
        timelimit: str = "w",
    ) -> list[dict]:
        """
        Execute a DuckDuckGo text search.

        Parameters
        ----------
        query:       Natural language search query (with -site: exclusions).
        max_results: Maximum number of results to request.
        timelimit:   DDG recency filter — 'd' (day), 'w' (week), 'm' (month).
                     Applied server-side before results are transmitted.
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
                    return list(
                        ddgs.text(
                            query,
                            max_results=max_results,
                            timelimit=timelimit,
                        )
                    )
            except Exception:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY_S * (2**attempt)
                    logger.debug(
                        "DDGS attempt %d failed; retrying in %.1fs…",
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

    Content-quality enforcement happens at three layers:
    1. **Query design** — queries are phrased with editorial-intent language
       ("news", "analyst report", "announcement") and ``_DDG_SITE_EXCLUSIONS``
       eliminates high-volume price-data domains at DDG's server.
    2. **Post-retrieval filtering** — ``NewsRetriever`` applies domain blocklist,
       price-page URL pattern, and snippet word-count checks before items enter
       the pipeline.
    3. **Recency filter** — ``NewsRecencyFilter`` validates publish dates against
       the horizon-specific lookback window.  ``unknown_date_policy="reject"``
       drops articles whose publish date cannot be determined.

    Parameters
    ----------
    top_k:
        Maximum articles to retrieve per query set.
    unknown_date_policy:
        What to do with articles whose publish date cannot be determined.
        Defaults to ``"reject"`` — an undated result from a time-filtered DDG
        query is more likely a stale aggregator link than fresh news.
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
                 (DDG timelimit + query context) and the recency filter
                 (lookback window).
        """
        try:
            # ── 1. Retrieve ───────────────────────────────────────────────────
            # horizon drives DDG timelimit selection and is embedded in
            # logging context.  Post-retrieval content-quality filters run
            # inside NewsRetriever.retrieve() before items are returned.
            raw_items = self._retriever.retrieve(ticker, horizon=horizon)
            articles_retrieved = len(raw_items)

            # ── 2. Recency filter ─────────────────────────────────────────────
            # unknown_date_policy="reject" and min_kept=0:
            #   - Articles with no parseable publish date are dropped.
            #   - The filter never rescues dropped articles to satisfy a floor.
            # An empty result signals "no verifiable fresh news" and causes
            # SignalFusionService to return an ML-only FusedSignal.
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