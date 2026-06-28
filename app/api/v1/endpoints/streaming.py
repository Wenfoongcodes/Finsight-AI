"""
app/api/v1/endpoints/streaming.py
==================================
Server-Sent Events (SSE) streaming endpoints for FinSight AI.

Provides two streaming endpoints:

    POST /api/v1/predict/stream
        Streams prediction pipeline progress as SSE events.
        Emits `progress` events at each major stage, followed by a single
        `result` event carrying the complete PredictionResult payload.

    POST /api/v1/agent/stream
        Streams agent plan-execute-synthesize loop progress.
        Emits `plan`, `tool_start`, `tool_result`, and `result` events.

Event schema
------------
Each SSE event is a JSON-encoded dict with at minimum:

    {
        "type": "progress" | "result" | "error" | "plan" | "tool_start" | "tool_result",
        "data": { ... }
    }

The `data` field is type-specific:

    progress  → {"stage": str, "message": str, "pct": int (0-100)}
    result    → the full PredictionResult or AgentResponse payload
    error     → {"message": str, "detail": str}
    plan      → {"tools": [str, ...]}
    tool_start→ {"tool": str, "args": dict}
    tool_result→ {"tool": str, "success": bool, "output": any}

SSE wire format (per the spec):
    data: <json>\n\n

A final `data: [DONE]\n\n` sentinel closes the stream so clients that
do not read Content-Length can detect end-of-stream reliably.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import AgentRequest, PredictionRequest
from app.core.logging_config import get_logger

logger = get_logger("api.streaming")

streaming_router = APIRouter(tags=["Streaming"])

# ── SSE helpers ───────────────────────────────────────────────────────────────


def _sse(event_type: str, data: Any) -> str:
    """
    Encode a single SSE event.

    Returns a string of the form::

        data: {"type": "...", "data": {...}}\n\n
    """
    payload = json.dumps({"type": event_type, "data": data}, default=str)
    return f"data: {payload}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _sse_error(message: str, detail: str = "") -> str:
    return _sse("error", {"message": message, "detail": detail})


# ── Streaming prediction endpoint ─────────────────────────────────────────────


@streaming_router.post("/predict/stream")
async def predict_stream(request: PredictionRequest) -> StreamingResponse:
    """
    Stream prediction pipeline progress as Server-Sent Events.

    Connect with ``Accept: text/event-stream`` or any SSE-capable client.
    Each event is a JSON object; the final event has ``type == "result"``
    and carries the complete prediction payload.

    Example curl::

        curl -N -X POST http://localhost:8000/api/v1/predict/stream \\
             -H "Content-Type: application/json" \\
             -d '{"ticker":"AAPL","horizon":"1d"}'
    """
    return StreamingResponse(
        _prediction_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


async def _prediction_event_generator(
    request: PredictionRequest,
) -> AsyncIterator[str]:
    """
    Async generator that drives the prediction pipeline and yields SSE events.

    Runs the synchronous PredictionService in a thread pool via
    ``asyncio.get_event_loop().run_in_executor`` so the event loop is not
    blocked during computation.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def _progress_callback(stage: str, message: str, pct: int = 0) -> None:
        """
        Thread-safe callback invoked by PredictionService at each stage.
        Enqueues an SSE progress event onto the asyncio queue.
        """
        event = _sse("progress", {"stage": stage, "message": message, "pct": pct})
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run_prediction():
        """Runs in a thread pool; returns the PredictionResponse or raises."""
        # Import here to avoid circular-import at module load time.
        from app.services.prediction_service import PredictionService

        svc = PredictionService()
        return svc.predict(
            ticker=request.ticker,
            horizon=request.horizon,
            use_cache=request.use_cache,
            progress_callback=_progress_callback,
        )

    # Sentinel value placed on the queue when the thread completes.
    _DONE = object()

    async def _run_in_thread():
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                result = await loop.run_in_executor(pool, _run_prediction)
                loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("error", exc),
                )

    task = asyncio.ensure_future(_run_in_thread())

    # Drain the queue and yield events until a terminal item arrives.
    try:
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=120.0)

            if isinstance(item, str):
                # Progress SSE event string — yield directly.
                yield item

            elif isinstance(item, tuple) and item[0] == "result":
                # Final result — serialise to PredictionResult schema and yield.
                prediction_response = item[1]
                payload = _serialise_prediction_response(
                    prediction_response, request.ticker, request.horizon
                )
                yield _sse("result", payload)
                yield _sse_done()
                break

            elif isinstance(item, tuple) and item[0] == "error":
                exc = item[1]
                logger.error(
                    "Streaming prediction error for %s: %s",
                    request.ticker,
                    exc,
                    exc_info=exc,
                )
                yield _sse_error(str(exc), traceback.format_exc())
                yield _sse_done()
                break

    except asyncio.TimeoutError:
        yield _sse_error(
            "Prediction timed out",
            "The server did not complete the prediction within 120 seconds.",
        )
        yield _sse_done()
    except Exception as exc:
        logger.error("Unexpected streaming error: %s", exc, exc_info=True)
        yield _sse_error("Internal server error", str(exc))
        yield _sse_done()
    finally:
        task.cancel()


def _serialise_prediction_response(resp, ticker: str, horizon: str) -> dict:
    """
    Convert a PredictionResponse dataclass to a JSON-safe dict that matches
    the PredictionResult Pydantic schema.
    """

    fused = resp.fused_signal
    if fused:
        fused_direction = fused.final_direction
        fused_confidence = fused.final_confidence
        fused_probability = fused.fusion_probability
        fusion_narrative = fused.synthesis_narrative
        fusion_applied = fused.fusion_applied
        news_sentiment = fused.news_sentiment
        news_items = [
            {"title": n.title, "snippet": n.snippet, "url": n.url}
            for n in fused.news_items
        ]
    else:
        fused_direction = "BULLISH" if resp.prediction == 1 else "BEARISH"
        fused_confidence = resp.confidence_label.upper()
        fused_probability = resp.p_bullish
        fusion_narrative = resp.narrative
        fusion_applied = False
        news_sentiment = "neutral"
        news_items = []

    return {
        "ticker": resp.ticker,
        "model_name": resp.model_name,
        "horizon": resp.horizon,
        "prediction": resp.prediction,
        "prediction_label": "BULLISH" if resp.prediction == 1 else "BEARISH",
        "probability": resp.probability,
        "p_bullish": resp.p_bullish,
        "p_bearish": resp.p_bearish,
        "confidence_label": resp.confidence_label,
        "confidence_degraded": resp.confidence_degraded,
        "selection_reason": resp.selection_reason,
        "latest_close": resp.latest_close,
        "narrative": resp.narrative,
        "top_features": resp.shap_explanation.get("top_features", []),
        "auto_trained": resp.auto_trained,
        "feature_selection_meta": resp.feature_selection_meta,
        "fused_direction": fused_direction,
        "fused_confidence": fused_confidence,
        "fused_probability": fused_probability,
        "fusion_narrative": fusion_narrative,
        "fusion_applied": fusion_applied,
        "news_sentiment": news_sentiment,
        "news_items": news_items,
    }


# ── Streaming agent endpoint ──────────────────────────────────────────────────


@streaming_router.post("/agent/stream")
async def agent_stream(request: AgentRequest) -> StreamingResponse:
    """
    Stream agent plan-execute-synthesize progress as Server-Sent Events.

    Event sequence:

    1. ``plan``        — tools the planner decided to call
    2. ``tool_start``  — emitted before each tool executes
    3. ``tool_result`` — emitted after each tool completes
    4. ``progress``    — "Synthesising response..."
    5. ``result``      — final agent response
    6. ``[DONE]``      — stream sentinel

    Example curl::

        curl -N -X POST http://localhost:8000/api/v1/agent/stream \\
             -H "Content-Type: application/json" \\
             -d '{"query":"Predict AAPL and explain the drivers"}'
    """
    return StreamingResponse(
        _agent_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _agent_event_generator(request: AgentRequest) -> AsyncIterator[str]:
    """
    Drives the FinancialAgent with streaming callbacks and yields SSE events.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _enqueue(event_type: str, data: Any) -> None:
        """Thread-safe enqueue from the agent worker thread."""
        event = _sse(event_type, data)
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run_agent():
        from app.rag.llm_chat import FinancialChatSystem
        from app.rag.rag_pipeline import RAGPipeline

        rag = RAGPipeline()
        chat = FinancialChatSystem(rag_pipeline=rag)

        agent = StreamingFinancialAgent(
            chat_system=chat,
            rag_pipeline=rag,
            event_callback=_enqueue,
        )
        return agent.run(request.query)

    async def _run_in_thread():
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                result = await loop.run_in_executor(pool, _run_agent)
                loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    task = asyncio.ensure_future(_run_in_thread())

    try:
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=120.0)

            if isinstance(item, str):
                yield item
            elif isinstance(item, tuple) and item[0] == "result":
                yield _sse("result", item[1])
                yield _sse_done()
                break
            elif isinstance(item, tuple) and item[0] == "error":
                exc = item[1]
                logger.error("Streaming agent error: %s", exc, exc_info=exc)
                yield _sse_error(str(exc))
                yield _sse_done()
                break

    except asyncio.TimeoutError:
        yield _sse_error("Agent timed out after 120 seconds.")
        yield _sse_done()
    except Exception as exc:
        yield _sse_error("Internal server error", str(exc))
        yield _sse_done()
    finally:
        task.cancel()


# ── Streaming-aware agent wrapper ─────────────────────────────────────────────


class StreamingFinancialAgent:
    """
    Thin wrapper around ``FinancialAgent`` that emits SSE events at each
    step of the plan-execute-synthesize loop.

    This approach avoids duplicating agent logic — it inherits the full
    ``FinancialAgent`` implementation and overrides only the execution loop
    to inject streaming callbacks at the right checkpoints.
    """

    def __init__(self, chat_system, rag_pipeline, event_callback):
        from app.agents.financial_agent import FinancialAgent

        self._agent = FinancialAgent(
            chat_system=chat_system,
            rag_pipeline=rag_pipeline,
        )
        self._cb = event_callback  # (event_type: str, data: Any) -> None

    def run(self, query: str) -> dict:
        from app.agents.financial_agent import _sanitise_response
        from app.core.exceptions import AgentError

        try:
            # ── 1. Plan ───────────────────────────────────────────────────────
            plan = self._agent._plan_tool_calls(query)
            self._cb("plan", {"tools": [s["tool"] for s in plan]})

            # ── 2. Execute each tool with before/after events ─────────────────
            tool_results = []
            for step in plan[:3]:
                tool_name = step["tool"]
                args = step.get("args", {})
                self._cb("tool_start", {"tool": tool_name, "args": args})

                result = self._agent._execute_tool(tool_name, args)
                tool_results.append(result)

                self._cb(
                    "tool_result",
                    {
                        "tool": result.tool_name,
                        "success": result.success,
                        "output": result.output,
                    },
                )

            # ── 3. Synthesise ─────────────────────────────────────────────────
            self._cb(
                "progress",
                {
                    "stage": "synthesis",
                    "message": "Synthesising response from tool results…",
                    "pct": 90,
                },
            )

            tool_context = "\n".join(r.to_context_string() for r in tool_results)

            if self._agent.chat_system:
                chat_resp = self._agent.chat_system.chat(
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
            raise AgentError(f"Streaming agent run failed: {exc}") from exc
