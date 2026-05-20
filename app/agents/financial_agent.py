from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.core.exceptions import AgentError, ToolExecutionError
from app.core.formatting import build_tool_context_string
from app.core.logging_config import get_logger
from app.rag.llm_chat import FinancialChatSystem, OpenAIClient
from configs.settings import settings

logger = get_logger("agents")


# ─────────────────────────────────────────────────────────────────────────────
# Suppress noisy third-party loggers (original Issue 3)
# ─────────────────────────────────────────────────────────────────────────────


def _suppress_ddgs_loggers() -> None:
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
    Handles markdown fences, trailing prose, leading prose, whitespace.
    """
    text = re.sub(r"```(?:json|JSON)?\s*", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")
    return text[start : end + 1]


# ─────────────────────────────────────────────────────────────────────────────
# Response Sanitiser
# ─────────────────────────────────────────────────────────────────────────────


def _sanitise_response(text: str) -> str:
    """
    Convert LLM Markdown output to clean HTML that integrates with the
    FinSight design system.

    Rules (applied in order):
    1. ``### Heading`` / ``## Heading`` / ``# Heading``
       → ``<span class="agent-heading">Heading</span>``
    2. ``**bold text**``
       → ``<span class="agent-bold">bold text</span>``
    3. ``*italic text*`` or ``_italic text_``
       → plain text  (italic is removed — font consistency)
    4. ``- bullet item`` or ``* bullet item`` (line-leading)
       → ``<span class="agent-bullet">•</span> item``
    5. Blank lines → ``<br>`` paragraph breaks
    6. Remaining ``\n`` inside a paragraph → single space
       (avoids raw newline artefacts inside the HTML div)

    No external Markdown library is required.
    All span class names are defined in ``dashboard.py`` CSS.
    """
    if not text:
        return ""

    # ── 1. ATX headings (### / ## / #) ───────────────────────────────────────
    text = re.sub(
        r"^#{1,3}\s+(.+)$",
        lambda m: f'<span class="agent-heading">{m.group(1).strip()}</span>',
        text,
        flags=re.MULTILINE,
    )

    # ── 2. Bold (**text** or __text__) ───────────────────────────────────────
    text = re.sub(
        r"\*{2}(.+?)\*{2}",
        lambda m: f'<span class="agent-bold">{m.group(1)}</span>',
        text,
    )
    text = re.sub(
        r"_{2}(.+?)_{2}",
        lambda m: f'<span class="agent-bold">{m.group(1)}</span>',
        text,
    )

    # ── 3. Italic (*text* or _text_) — strip to plain text ───────────────────
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # ── 4. Bullet lists (- item / * item at line start) ──────────────────────
    text = re.sub(
        r"^[\-\*]\s+(.+)$",
        lambda m: f'<span class="agent-bullet">•</span> {m.group(1).rstrip()}',
        text,
        flags=re.MULTILINE,
    )

    # ── 5. Blank lines → paragraph break ─────────────────────────────────────
    text = re.sub(r"\n{2,}", "<br><br>", text)

    # ── 6. Remaining single newlines → space ─────────────────────────────────
    text = text.replace("\n", " ")

    return text.strip()


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
        """
        Canonical tool-result string for LLM prompt injection.
        Delegates to ``build_tool_context_string()`` from
        ``app.core.formatting`` (2 000-char cap, deterministic format).
        """
        if self.success:
            return build_tool_context_string(self.tool_name, self.output)
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

    def predict_stock(self, ticker: str, model_name: Optional[str] = None) -> dict:
        """
        Generate a next-day price direction prediction with probability.

        ``model_name`` is accepted for backward-compatibility with planner
        prompts that include it, but is silently ignored — ``PredictionService``
        auto-selects the best trained model via the leaderboard.  The actual
        model used is reported in the output dict.

        Args:
            ticker:     Stock ticker symbol (e.g. 'AAPL').
            model_name: Ignored.  Kept so old planner JSON does not cause a
                        ``ToolExecutionError`` on unexpected-argument validation.

        Returns:
            Dict with prediction, probability, confidence, and model used.
        """
        if model_name is not None:
            logger.debug(
                "predict_stock: model_name=%r was passed but is ignored "
                "(PredictionService uses auto-selection).",
                model_name,
            )
        try:
            result = self._svc.predict(ticker)
            return {
                "ticker": result.ticker,
                "prediction": "BULLISH" if result.prediction == 1 else "BEARISH",
                "probability": result.probability,
                "p_bullish": result.p_bullish,
                "p_bearish": result.p_bearish,
                "confidence": result.confidence_label,
                "latest_close": result.latest_close,
                "model_used": result.model_name,  # report actual auto-selected model
                "horizon": result.horizon,
            }
        except Exception as exc:
            raise ToolExecutionError(f"predict_stock failed: {exc}") from exc

    def explain_prediction(self, ticker: str, model_name: Optional[str] = None) -> dict:
        """
        Generate a SHAP-based explanation for the latest prediction.

        ``model_name`` is accepted but ignored — see ``predict_stock()``
        for the same rationale.

        Args:
            ticker:     Stock ticker symbol.
            model_name: Ignored.

        Returns:
            Dict with narrative and top SHAP features.
        """
        if model_name is not None:
            logger.debug(
                "explain_prediction: model_name=%r was passed but is ignored.",
                model_name,
            )
        try:
            result = self._svc.predict(ticker)
            return {
                "ticker": result.ticker,
                "narrative": result.narrative,
                "top_features": result.shap_explanation.get("top_features", [])[:5],
                "model_used": result.model_name,
                "horizon": result.horizon,
            }
        except Exception as exc:
            raise ToolExecutionError(f"explain_prediction failed: {exc}") from exc

    def retrieve_financial_context(self, query: str, rag_pipeline: Any) -> dict:
        """
        Retrieve relevant context from the RAG knowledge base.

        Gracefully returns an empty result when the knowledge base has not
        been populated yet (original Issue 2 fix preserved).
        """
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
        not the original user query (original Issue 4 fix preserved).

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
            return {
                "text": text[:300],
                "sentiment": label,
                "score": round(score, 3),
            }
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
        name first and falls back to ``duckduckgo_search`` (original Issue 1
        fix preserved).

        Args:
            query:       Natural language search query.
            max_results: Number of results to return (1–10).

        Returns:
            Dict with ``query``, ``results`` list, and a ``summary`` string.
        """
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            pass

        if DDGS is None:
            try:
                from duckduckgo_search import DDGS
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
6. Do NOT include model_name in predict_stock or explain_prediction args —
   model selection is automatic.
7. If no tools are needed, return exactly: []"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class FinancialAgent:
    """
    Agentic AI orchestrator that selects and executes tools to answer
    complex financial queries.

    Fixes applied in this revision
    --------------------------------
    1. ``predict_stock`` / ``explain_prediction`` no longer pass ``model_name``
       to ``PredictionService.predict()`` — the argument is accepted but
       ignored with a debug log.
    2. Planner prompt explicitly forbids ``model_name`` in tool args.
    3. LLM response is passed through ``_sanitise_response()`` before being
       returned so Markdown artefacts (bold, italic, bullets, headings) are
       converted to design-system HTML span classes.
    4. ``ToolResult.to_context_string()`` delegates to the canonical formatter.
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
        if self.rag_pipeline is None:
            return "UNAVAILABLE"
        if getattr(self.rag_pipeline, "_store_initialized", False):
            count = getattr(self.rag_pipeline, "document_count", "?")
            return f"POPULATED ({count} chunks)"
        return "EMPTY — do not call retrieve_financial_context"

    def _build_tool_registry(self) -> dict[str, AgentTool]:
        ti = self.tools_instance
        tools = [
            AgentTool(
                name="predict_stock",
                description=(
                    "Predict next-day price direction (bullish/bearish) for a stock "
                    "ticker. Required args: ticker. Do NOT pass model_name — "
                    "model selection is automatic."
                ),
                fn=ti.predict_stock,
                required_args=["ticker"],
            ),
            AgentTool(
                name="explain_prediction",
                description=(
                    "Generate a SHAP-based explanation for why a prediction was made. "
                    "Required args: ticker. Do NOT pass model_name."
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

        if self.rag_pipeline:
            tools.append(
                AgentTool(
                    name="retrieve_financial_context",
                    description=(
                        "Search the ingested financial knowledge base for relevant "
                        "context. Only useful when the knowledge base status is "
                        "POPULATED."
                    ),
                    fn=lambda query: ti.retrieve_financial_context(
                        query, self.rag_pipeline
                    ),
                    required_args=["query"],
                )
            )

        return {t.name: t for t in tools}

    def _validate_step(self, step: dict) -> tuple[bool, str]:
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

        The LLM response is sanitised through ``_sanitise_response()`` before
        being returned so Markdown formatting is converted to design-system
        HTML classes instead of being rendered as raw asterisks or triggering
        browser-default italic/bold styles.

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
                raw_response = chat_resp.content
            else:
                raw_response = (
                    f"Tool results:\n{tool_context}\n\nQuery: {query}\n"
                    "(LLM not available — returning raw tool output)"
                )

            # Sanitise: convert Markdown artefacts to design-system HTML
            final_response = _sanitise_response(raw_response)

            return {
                "query": query,
                "response": final_response,
                "tools_used": [r.tool_name for r in tool_results if r.success],
                "tool_results": [
                    {
                        "tool": r.tool_name,
                        "success": r.success,
                        "output": r.output,
                    }
                    for r in tool_results
                ],
            }

        except Exception as exc:
            raise AgentError(f"Agent run failed: {exc}") from exc
