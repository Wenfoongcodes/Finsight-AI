"""
FinSight AI — Phase 10: Agentic AI System
Defines individual agent tools and an orchestrator that decides which tool
to invoke based on the user's query using LLM reasoning.

Fixes applied in this revision (from log analysis 2026-05-10)
-------------------------------------------------------------

Issue 1 — RuntimeWarning: duckduckgo_search renamed to ddgs
    The ``duckduckgo-search`` package was renamed to ``ddgs``.  The import
    inside ``search_web`` now tries ``ddgs`` first and falls back to
    ``duckduckgo_search`` for backward compatibility, so the RuntimeWarning
    is eliminated regardless of which version is installed.

    Install:  pip install ddgs

Issue 2 — ToolExecutionError: Knowledge base is empty
    The planner selected ``retrieve_financial_context`` even though the
    vector store had never been populated.  ``retrieve_financial_context``
    now guards against this with an explicit ``_store_initialized`` check
    and returns a graceful empty result dict instead of raising.  The
    planner prompt also receives the current KB status so the LLM can
    avoid selecting this tool when the store is empty.

Issue 3 — DEBUG flood from rustls / h2 / hyper_util / primp / cookie_store
    The ``ddgs`` library uses a Rust HTTP client (primp/reqwest) that
    registers Python loggers at DEBUG level.  These are now suppressed in
    ``_suppress_ddgs_loggers()`` which is called once at module import time.

Issue 4 — analyze_sentiment called with the raw query string
    The planner called ``analyze_sentiment(text="NVDA latest news")`` — the
    literal search query, not a real headline.  Fixed by:
    (a) Rewriting the tool description to explicitly forbid passing query
        strings and require actual news content.
    (b) Adding an ordering rule to ``_PLANNER_USER`` that requires
        ``search_web`` to run before ``analyze_sentiment`` and instructs
        the LLM to use a returned snippet as the ``text`` argument.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.core.exceptions import AgentError, ToolExecutionError
from app.core.logging_config import get_logger
from app.rag.llm_chat import FinancialChatSystem, OpenAIClient
from configs.settings import settings

logger = get_logger("agents")


# ─────────────────────────────────────────────────────────────────────────────
# Suppress noisy third-party loggers (Issue 3)
# ─────────────────────────────────────────────────────────────────────────────


def _suppress_ddgs_loggers() -> None:
    """
    Silence DEBUG-level loggers emitted by the ``ddgs`` Rust HTTP client.

    ``ddgs`` (formerly ``duckduckgo-search``) uses ``primp``/``reqwest``
    internally, which registers Python loggers for every TLS handshake,
    H2 frame, and cookie operation.  Without suppression these flood the
    console at WARNING-level when the root logger is set to DEBUG.

    Called once at module import time — safe to call multiple times.
    """
    noisy_roots = [
        "rustls",
        "h2",
        "hyper_util",
        "reqwest",
        "primp",
        "cookie_store",
        "duckduckgo_search",
        "ddgs",
    ]
    for name in noisy_roots:
        logging.getLogger(name).setLevel(logging.WARNING)


_suppress_ddgs_loggers()


# ─────────────────────────────────────────────────────────────────────────────
# JSON Extraction Utility
# ─────────────────────────────────────────────────────────────────────────────


def _extract_json_array(text: str) -> str:
    """
    Robustly extract a JSON array from an LLM response string.

    Handles: markdown fences, trailing prose, leading prose, whitespace.
    Strategy: strip fences with regex, then slice first ``[`` to last ``]``.
    """
    text = re.sub(r"```(?:json|JSON)?\s*", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")
    return text[start : end + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Standardized output from an agent tool."""

    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None

    def to_context_string(self) -> str:
        if self.success:
            return f"[{self.tool_name}] {json.dumps(self.output, default=str)[:2000]}"
        return f"[{self.tool_name}] FAILED: {self.error}"


@dataclass
class AgentTool:
    """Agent tool descriptor with callable function."""

    name: str
    description: str
    fn: Callable[..., Any]
    required_args: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Individual Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────


class FinancialAgentTools:
    """
    Collection of agent tools for financial analysis.

    ``PredictionService`` is lazy-loaded on first use to avoid paying
    ``ModelTrainer`` + ``FeatureEngineer`` construction cost on every request.
    """

    def __init__(self, prediction_service: Optional[Any] = None) -> None:
        self._prediction_service = prediction_service

    @property
    def _svc(self) -> Any:
        if self._prediction_service is None:
            from app.services.prediction_service import PredictionService

            self._prediction_service = PredictionService()
        return self._prediction_service

    # ── Financial tools ───────────────────────────────────────────────────────

    def predict_stock(self, ticker: str, model_name: str = "xgboost") -> dict:
        """
        Generate a next-day price direction prediction with probability.

        Args:
            ticker:     Stock ticker symbol (e.g. 'AAPL').
            model_name: ML model to use.

        Returns:
            Dict with prediction, probability, and confidence.
        """
        try:
            result = self._svc.predict(ticker, model_name=model_name)
            return {
                "ticker": result.ticker,
                "prediction": "BULLISH" if result.prediction == 1 else "BEARISH",
                "probability": result.probability,
                "confidence": result.confidence_label,
                "latest_close": result.latest_close,
            }
        except Exception as exc:
            raise ToolExecutionError(f"predict_stock failed: {exc}") from exc

    def explain_prediction(self, ticker: str, model_name: str = "xgboost") -> dict:
        """
        Generate a SHAP-based explanation for the latest prediction.

        Args:
            ticker:     Stock ticker symbol.
            model_name: ML model identifier.

        Returns:
            Dict with narrative and top SHAP features.
        """
        try:
            result = self._svc.predict(ticker, model_name=model_name)
            return {
                "ticker": result.ticker,
                "narrative": result.narrative,
                "top_features": result.shap_explanation.get("top_features", [])[:5],
            }
        except Exception as exc:
            raise ToolExecutionError(f"explain_prediction failed: {exc}") from exc

    def retrieve_financial_context(self, query: str, rag_pipeline: Any) -> dict:
        """
        Retrieve relevant context from the RAG knowledge base.

        Gracefully returns an empty result when the knowledge base has not
        been populated yet, instead of raising an exception that silently
        marks the tool as failed.  This prevents the planner from wasting
        a tool slot on an empty store.

        Args:
            query:        Natural language query.
            rag_pipeline: Initialized ``RAGPipeline`` instance.

        Returns:
            Dict with ``query``, ``results``, and optionally a ``note``
            field when the store is empty.
        """
        # Issue 2 fix: guard before calling retrieve() so an empty KB
        # returns a clear message instead of raising ToolExecutionError.
        if not getattr(rag_pipeline, "_store_initialized", False):
            logger.info(
                "retrieve_financial_context: knowledge base is empty — skipping."
            )
            return {
                "query": query,
                "results": [],
                "note": (
                    "Knowledge base is empty. "
                    "Ingest documents or article URLs via the sidebar first."
                ),
            }

        try:
            results = rag_pipeline.retrieve(query, top_k=settings.RAG_TOP_K)
            return {
                "query": query,
                "results": [
                    {"content": r["content"][:300], "score": r["score"]}
                    for r in results
                ],
            }
        except Exception as exc:
            raise ToolExecutionError(
                f"retrieve_financial_context failed: {exc}"
            ) from exc

    def analyze_sentiment(self, text: str) -> dict:
        """
        Perform rule-based sentiment analysis on actual financial news text.

        IMPORTANT: ``text`` must be a real news headline or article excerpt —
        NOT a search query string.  Pass a snippet returned by ``search_web``,
        not the original user query.

        Args:
            text: A real financial news headline or excerpt (min ~10 words).

        Returns:
            Dict with sentiment label (positive/negative/neutral) and score.
        """
        try:
            bullish_kw = {
                "surge",
                "rally",
                "beat",
                "strong",
                "growth",
                "bullish",
                "up",
                "gain",
                "profit",
                "outperform",
                "record",
                "high",
                "positive",
                "rises",
                "jumped",
                "soared",
                "lifted",
                "upgraded",
                "raises",
            }
            bearish_kw = {
                "drop",
                "fall",
                "miss",
                "weak",
                "loss",
                "bearish",
                "down",
                "decline",
                "risk",
                "concern",
                "low",
                "negative",
                "crash",
                "fell",
                "tumbled",
                "slumped",
                "cut",
                "downgraded",
                "warning",
            }
            words = set(text.lower().split())
            bull_count = len(words & bullish_kw)
            bear_count = len(words & bearish_kw)
            total = bull_count + bear_count or 1
            score = (bull_count - bear_count) / total

            label = (
                "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            )
            return {"text": text[:300], "sentiment": label, "score": round(score, 3)}
        except Exception as exc:
            raise ToolExecutionError(f"analyze_sentiment failed: {exc}") from exc

    def get_market_summary(self, ticker: str) -> dict:
        """
        Retrieve OHLCV summary statistics for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with OHLCV summary statistics.
        """
        try:
            from app.ml.data_ingestion import get_data_summary, ingest_market_data

            df = ingest_market_data(ticker, period_years=1)
            return get_data_summary(df, ticker)
        except Exception as exc:
            raise ToolExecutionError(f"get_market_summary failed: {exc}") from exc

    def search_web(self, query: str, max_results: int = 5) -> dict:
        """
        Search the web for current financial news and information.

        Uses DuckDuckGo (no API key required).  Tries the new ``ddgs`` package
        name first and falls back to the legacy ``duckduckgo_search`` import
        so both ``pip install ddgs`` and ``pip install duckduckgo-search``
        work without code changes.

        Use this tool for:
        - Recent news, earnings, Fed decisions, M&A announcements
        - Current stock prices or market conditions
        - Anything that may postdate the LLM training cutoff

        Install:  pip install ddgs

        Args:
            query:       Natural language search query.
            max_results: Number of results to return (1–10).

        Returns:
            Dict with ``query``, ``results`` list, and a ``summary`` string::

                {
                    "query": "NVDA latest news",
                    "results": [
                        {"title": "Nvidia hits record...", "url": "...", "snippet": "..."},
                        ...
                    ],
                    "summary": "[1] Nvidia hits record...\\n    URL: ...\\n    ..."
                }
        """
        # Issue 1 fix: try new package name first, fall back to old name.
        DDGS = None
        try:
            from ddgs import DDGS  # pip install ddgs (new name)
        except ImportError:
            pass

        if DDGS is None:
            try:
                from duckduckgo_search import (
                    DDGS,
                )  # pip install duckduckgo-search (old name)
            except ImportError:
                raise ToolExecutionError(
                    "Web search package not installed. Run: pip install ddgs"
                )

        try:
            max_results = max(1, min(int(max_results), 10))

            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))

            if not raw:
                return {
                    "query": query,
                    "results": [],
                    "summary": "No results found.",
                }

            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:400],
                }
                for r in raw
            ]

            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}"
                )
            summary = "\n\n".join(lines)

            logger.info("Web search: query=%r | results=%d", query, len(results))
            return {"query": query, "results": results, "summary": summary}

        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"search_web failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = (
    "You are a JSON-only tool planner for a financial AI agent. "
    "You must respond with a valid JSON array and absolutely nothing else — "
    "no markdown, no explanation, no prose before or after the array. "
    "Your entire response must be parseable by json.loads()."
)

# Issue 4 fix: added explicit ordering rules that prevent the planner from:
#   (a) passing the raw user query string to analyze_sentiment
#   (b) calling retrieve_financial_context when the KB is empty
#   (c) calling analyze_sentiment before search_web has run

_PLANNER_USER = """Available tools:
{tool_descriptions}

Knowledge base status: {kb_status}

User query: {query}

Return a JSON array of tool calls:
[
  {{"tool": "tool_name", "args": {{"arg1": "value1"}}}},
  ...
]

IMPORTANT ordering and usage rules:
1. Use search_web for ANY question about recent events, current prices, or news.
2. If you include analyze_sentiment, you MUST call search_web first and use
   a snippet from its results as the text argument — NEVER pass the user's
   query string to analyze_sentiment.
3. Only include retrieve_financial_context if knowledge base status is POPULATED.
4. Order tools so dependencies run first (e.g. search_web before analyze_sentiment).
5. Use at most 3 tool calls total.
6. If no tools are needed, return exactly: []"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class FinancialAgent:
    """
    Agentic AI orchestrator that selects and executes tools to answer
    complex financial queries.

    All four issues from the 2026-05-10 log are addressed here:

    1. ``search_web`` uses a dual-import strategy (ddgs → duckduckgo_search)
       eliminating the RuntimeWarning from the renamed package.
    2. ``retrieve_financial_context`` guards against an empty knowledge base
       and the planner receives the current KB status so it can avoid
       selecting this tool when nothing has been ingested.
    3. Noisy loggers from the Rust HTTP client are suppressed at module
       import time via ``_suppress_ddgs_loggers()``.
    4. The planner prompt explicitly forbids passing query strings to
       ``analyze_sentiment`` and requires ``search_web`` to run first.
    """

    def __init__(
        self,
        tools_instance: Optional[FinancialAgentTools] = None,
        chat_system: Optional[FinancialChatSystem] = None,
        rag_pipeline: Optional[Any] = None,
    ) -> None:
        self.tools_instance = tools_instance or FinancialAgentTools()
        self.chat_system = chat_system
        self.rag_pipeline = rag_pipeline
        self._llm: Optional[OpenAIClient] = None
        self._tool_registry: dict[str, AgentTool] = self._build_tool_registry()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_llm(self) -> OpenAIClient:
        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm

    def _kb_status(self) -> str:
        """
        Return a one-word KB status string injected into the planner prompt.

        The planner uses this to decide whether ``retrieve_financial_context``
        is worth calling.  This prevents Issue 2 at the planning stage rather
        than only at execution time.
        """
        if self.rag_pipeline is None:
            return "UNAVAILABLE"
        if getattr(self.rag_pipeline, "_store_initialized", False):
            count = getattr(self.rag_pipeline, "document_count", "?")
            return f"POPULATED ({count} chunks)"
        return "EMPTY — do not call retrieve_financial_context"

    def _build_tool_registry(self) -> dict[str, AgentTool]:
        """Register all available agent tools."""
        ti = self.tools_instance
        tools = [
            AgentTool(
                name="predict_stock",
                description=(
                    "Predict next-day price direction (bullish/bearish) for a stock ticker."
                ),
                fn=ti.predict_stock,
                required_args=["ticker"],
            ),
            AgentTool(
                name="explain_prediction",
                description=(
                    "Generate a SHAP-based explanation for why a prediction was made."
                ),
                fn=ti.explain_prediction,
                required_args=["ticker"],
            ),
            AgentTool(
                name="analyze_sentiment",
                description=(
                    "Analyze the sentiment of an actual financial news headline or "
                    "article excerpt. The 'text' argument MUST be real news content "
                    "from search results — never a search query string. "
                    "Always call search_web first and pass one of its result snippets "
                    "as the text argument."
                ),
                fn=ti.analyze_sentiment,
                required_args=["text"],
            ),
            AgentTool(
                name="get_market_summary",
                description="Retrieve OHLCV summary statistics for a stock ticker.",
                fn=ti.get_market_summary,
                required_args=["ticker"],
            ),
            AgentTool(
                name="search_web",
                description=(
                    "Search the web for current financial news, recent earnings, "
                    "Fed decisions, or any time-sensitive information. "
                    "Use this for ANY question about events that may have happened "
                    "recently or that the LLM may not know about."
                ),
                fn=ti.search_web,
                required_args=["query"],
            ),
        ]

        # Only register retrieve_financial_context when a rag_pipeline is wired in.
        # The tool itself still guards against empty stores at execution time.
        if self.rag_pipeline:
            tools.append(
                AgentTool(
                    name="retrieve_financial_context",
                    description=(
                        "Search the ingested financial knowledge base for relevant context. "
                        "Only useful when the knowledge base status is POPULATED."
                    ),
                    fn=lambda query: ti.retrieve_financial_context(
                        query, self.rag_pipeline
                    ),
                    required_args=["query"],
                )
            )

        return {t.name: t for t in tools}

    def _validate_step(self, step: dict) -> tuple[bool, str]:
        """Validate a planned tool call against the registry."""
        tool_name = step.get("tool", "")
        args = step.get("args", {})

        if not isinstance(tool_name, str) or not tool_name.strip():
            return False, "Missing or empty tool name"
        if tool_name not in self._tool_registry:
            return False, f"Unknown tool: {tool_name!r}"
        if not isinstance(args, dict):
            return False, f"'args' must be a dict, got {type(args).__name__}"

        missing = [
            a for a in self._tool_registry[tool_name].required_args if a not in args
        ]
        if missing:
            return False, f"Missing required args for {tool_name!r}: {missing}"

        return True, "ok"

    def _plan_tool_calls(self, query: str) -> list[dict]:
        """
        Use the LLM to plan which tools to invoke.

        Injects the current KB status into the prompt so the LLM can make
        an informed decision about whether to call retrieve_financial_context.

        Returns ``[]`` on any failure so the agent degrades gracefully.
        """
        try:
            tool_descriptions = "\n".join(
                f"- {t.name}: {t.description}" for t in self._tool_registry.values()
            )
            user_message = _PLANNER_USER.format(
                tool_descriptions=tool_descriptions,
                kb_status=self._kb_status(),
                query=query,
            )

            raw_response, _ = self._get_llm().chat(
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            array_str = _extract_json_array(raw_response)
            plan = json.loads(array_str)

            if not isinstance(plan, list):
                logger.warning(
                    "Planner returned non-list JSON (%s); ignoring.",
                    type(plan).__name__,
                )
                return []

            validated: list[dict] = []
            for step in plan:
                ok, reason = self._validate_step(step)
                if ok:
                    validated.append(step)
                else:
                    logger.warning("Dropping invalid tool call %s — %s", step, reason)

            logger.info("Agent plan for %r: %s", query[:60], validated)
            return validated

        except Exception as exc:
            logger.warning(
                "Tool planning failed: %s. Falling back to no-tool response.", exc
            )
            return []

    def _execute_tool(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a single validated tool call."""
        tool = self._tool_registry[tool_name]
        try:
            output = tool.fn(**args)
            return ToolResult(tool_name=tool_name, success=True, output=output)
        except ToolExecutionError as exc:
            logger.warning("Tool %r execution error: %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name, success=False, output=None, error=str(exc)
            )
        except Exception as exc:
            logger.warning("Tool %r unexpected error: %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name, success=False, output=None, error=str(exc)
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Execute the full agentic loop: plan → validate → execute → respond.

        Args:
            query: User's natural language query.

        Returns:
            Dict with ``'response'``, ``'tools_used'``, and ``'tool_results'``.

        Raises:
            AgentError: On unrecoverable failure.
        """
        try:
            plan = self._plan_tool_calls(query)
            tool_results: list[ToolResult] = []

            for step in plan[:3]:
                result = self._execute_tool(step["tool"], step.get("args", {}))
                tool_results.append(result)

            tool_context = "\n".join(r.to_context_string() for r in tool_results)

            if self.chat_system:
                chat_resp = self.chat_system.chat(
                    user_query=query,
                    use_rag=True,
                    prediction_context=tool_context if tool_context else None,
                )
                final_response = chat_resp.content
            else:
                final_response = (
                    f"Tool results:\n{tool_context}\n\nQuery: {query}\n"
                    "(LLM not available — returning raw tool output)"
                )

            return {
                "query": query,
                "response": final_response,
                "tools_used": [r.tool_name for r in tool_results if r.success],
                "tool_results": [
                    {"tool": r.tool_name, "success": r.success, "output": r.output}
                    for r in tool_results
                ],
            }

        except Exception as exc:
            raise AgentError(f"Agent run failed: {exc}") from exc
