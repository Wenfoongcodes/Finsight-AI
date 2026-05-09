"""
FinSight AI — Phase 10: Agentic AI System
Defines individual agent tools and an orchestrator that decides which tool
to invoke based on the user's query using LLM reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from app.core.exceptions import AgentError, ToolExecutionError
from app.core.logging_config import get_logger
from app.rag.llm_chat import FinancialChatSystem, OpenAIClient
from app.services.prediction_service import PredictionService
from configs.settings import settings

logger = get_logger("agents")


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definition
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
    """

    def __init__(self) -> None:
        self.prediction_service = PredictionService()

    def predict_stock(self, ticker: str, model_name: str = "xgboost") -> dict:
        """
        Generate a stock direction prediction with probability.

        Args:
            ticker: Stock ticker symbol.
            model_name: ML model to use.

        Returns:
            Dict with prediction, probability, and confidence.
        """
        try:
            result = self.prediction_service.predict(ticker, model_name=model_name)
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
        Generate SHAP-based explanation for the latest prediction.

        Args:
            ticker: Stock ticker symbol.
            model_name: ML model identifier.

        Returns:
            Dict with narrative and top SHAP features.
        """
        try:
            result = self.prediction_service.predict(ticker, model_name=model_name)
            return {
                "ticker": result.ticker,
                "narrative": result.narrative,
                "top_features": result.shap_explanation.get("top_features", [])[:5],
            }
        except Exception as exc:
            raise ToolExecutionError(f"explain_prediction failed: {exc}") from exc

    def retrieve_financial_context(self, query: str, rag_pipeline: Any) -> dict:
        """
        Retrieve relevant financial context from the knowledge base.

        Args:
            query: Natural language query.
            rag_pipeline: Initialized RAGPipeline instance.

        Returns:
            Dict with top retrieved context snippets.
        """
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
            raise ToolExecutionError(f"retrieve_financial_context failed: {exc}") from exc

    def analyze_sentiment(self, text: str) -> dict:
        """
        Perform rule-based sentiment analysis on financial text.

        Args:
            text: Financial news headline or text snippet.

        Returns:
            Dict with sentiment label and score.
        """
        try:
            # Lightweight keyword-based sentiment (no external API required)
            bullish_kw = {
                "surge", "rally", "beat", "strong", "growth", "bullish", "up",
                "gain", "profit", "outperform", "record", "high", "positive",
            }
            bearish_kw = {
                "drop", "fall", "miss", "weak", "loss", "bearish", "down",
                "decline", "risk", "concern", "low", "negative", "crash",
            }
            words = set(text.lower().split())
            bull_count = len(words & bullish_kw)
            bear_count = len(words & bearish_kw)
            total = bull_count + bear_count or 1
            score = (bull_count - bear_count) / total  # [-1, 1]

            if score > 0.1:
                label = "positive"
            elif score < -0.1:
                label = "negative"
            else:
                label = "neutral"

            return {"text": text[:200], "sentiment": label, "score": round(score, 3)}
        except Exception as exc:
            raise ToolExecutionError(f"analyze_sentiment failed: {exc}") from exc

    def get_market_summary(self, ticker: str) -> dict:
        """
        Retrieve basic market data summary for a ticker.

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


# ─────────────────────────────────────────────────────────────────────────────
# Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SELECTION_PROMPT = """You are a financial AI agent orchestrator.
Given a user query and a list of available tools, determine which tools to call
and in what order to best answer the query.

Available tools:
{tool_descriptions}

User query: {query}

Respond ONLY with a JSON array of tool calls in this format:
[
  {{"tool": "tool_name", "args": {{"arg1": "value1"}}}},
  ...
]

Rules:
- Include only tools that are directly relevant.
- Order tools so dependencies are called first.
- Use at most 3 tool calls per query.
- If no tools are needed, return: []"""


class FinancialAgent:
    """
    Agentic AI orchestrator that selects and executes tools
    to answer complex financial queries.
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

        self._tool_registry: dict[str, AgentTool] = self._build_tool_registry()

    def _build_tool_registry(self) -> dict[str, AgentTool]:
        """Register all available agent tools."""
        ti = self.tools_instance
        tools = [
            AgentTool(
                name="predict_stock",
                description="Predict next-day price direction (bullish/bearish) for a stock ticker.",
                fn=ti.predict_stock,
                required_args=["ticker"],
            ),
            AgentTool(
                name="explain_prediction",
                description="Generate a SHAP-based explanation for why a prediction was made.",
                fn=ti.explain_prediction,
                required_args=["ticker"],
            ),
            AgentTool(
                name="analyze_sentiment",
                description="Analyze the sentiment of a financial news headline or text.",
                fn=ti.analyze_sentiment,
                required_args=["text"],
            ),
            AgentTool(
                name="get_market_summary",
                description="Retrieve OHLCV summary statistics for a stock ticker.",
                fn=ti.get_market_summary,
                required_args=["ticker"],
            ),
        ]
        if self.rag_pipeline:
            tools.append(AgentTool(
                name="retrieve_financial_context",
                description="Search the financial knowledge base for relevant information.",
                fn=lambda query: ti.retrieve_financial_context(query, self.rag_pipeline),
                required_args=["query"],
            ))
        return {t.name: t for t in tools}

    def _plan_tool_calls(self, query: str) -> list[dict]:
        """
        Use LLM to plan which tools to call.

        Args:
            query: User query string.

        Returns:
            List of {'tool': name, 'args': dict} dicts.
        """
        try:
            tool_descriptions = "\n".join(
                f"- {t.name}: {t.description}" for t in self._tool_registry.values()
            )
            prompt = TOOL_SELECTION_PROMPT.format(
                tool_descriptions=tool_descriptions, query=query
            )

            llm = OpenAIClient()
            response, _ = llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )

            # Strip markdown code fences if present
            cleaned = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            plan = json.loads(cleaned)
            if not isinstance(plan, list):
                return []

            logger.info("Agent plan for query '%s': %s", query[:60], plan)
            return plan

        except Exception as exc:
            logger.warning("Tool planning failed: %s. Falling back to no-tool response.", exc)
            return []

    def _execute_tool(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a single tool call."""
        if tool_name not in self._tool_registry:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )
        tool = self._tool_registry[tool_name]
        try:
            output = tool.fn(**args)
            return ToolResult(tool_name=tool_name, success=True, output=output)
        except ToolExecutionError as exc:
            return ToolResult(tool_name=tool_name, success=False, output=None, error=str(exc))
        except Exception as exc:
            return ToolResult(tool_name=tool_name, success=False, output=None, error=str(exc))

    def run(self, query: str) -> dict:
        """
        Execute the full agentic loop: plan → execute tools → generate response.

        Args:
            query: User's natural language query.

        Returns:
            Dict with 'response', 'tools_used', and 'tool_results'.

        Raises:
            AgentError: On unrecoverable failure.
        """
        try:
            # 1. Plan
            plan = self._plan_tool_calls(query)

            # 2. Execute
            tool_results: list[ToolResult] = []
            for step in plan[:3]:  # Max 3 tool calls
                result = self._execute_tool(step.get("tool", ""), step.get("args", {}))
                tool_results.append(result)

            # 3. Build context from tool results
            tool_context = "\n".join(r.to_context_string() for r in tool_results)

            # 4. Generate final response
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
