"""
Unit tests for ``app.agents.financial_agent``.

Covers:
- ``_extract_json_array``       — robust JSON extraction from LLM prose.
- ``_sanitise_response``        — Markdown -> design-system HTML conversion.
- ``_convert_markdown_tables``  — GFM pipe-table -> <table> conversion.
- ``FinancialAgent._validate_step`` — tool-call schema validation.
- ``FinancialAgent._plan_tool_calls`` — LLM planning with graceful
  degradation on malformed output.
- ``FinancialAgent.run``        — end-to-end orchestration with mocked
  tools, LLM, and chat system.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.agents.financial_agent import (
    FinancialAgent,
    ToolResult,
    _convert_markdown_tables,
    _extract_json_array,
    _sanitise_response,
)
from app.core.exceptions import ToolExecutionError

# ─────────────────────────────────────────────────────────────────────────────
# _extract_json_array
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractJsonArray:
    def test_extracts_clean_json_array(self):
        text = '[{"tool": "predict_stock", "args": {"ticker": "AAPL"}}]'
        assert json.loads(_extract_json_array(text)) == [
            {"tool": "predict_stock", "args": {"ticker": "AAPL"}}
        ]

    def test_strips_markdown_code_fences(self):
        text = '```json\n[{"tool": "search_web", "args": {"query": "Fed"}}]\n```'
        result = _extract_json_array(text)
        assert json.loads(result) == [{"tool": "search_web", "args": {"query": "Fed"}}]

    def test_ignores_leading_and_trailing_prose(self):
        text = 'Sure, here is the plan:\n[{"tool": "search_web", "args": {}}]\nLet me know!'
        result = _extract_json_array(text)
        assert json.loads(result) == [{"tool": "search_web", "args": {}}]

    def test_raises_value_error_when_no_array_present(self):
        with pytest.raises(ValueError):
            _extract_json_array("I don't think any tools are needed here.")

    def test_raises_when_brackets_out_of_order(self):
        with pytest.raises(ValueError):
            _extract_json_array("] this is backwards [")


# ─────────────────────────────────────────────────────────────────────────────
# _sanitise_response
# ─────────────────────────────────────────────────────────────────────────────


class TestSanitiseResponse:
    def test_empty_string_returns_empty_string(self):
        assert _sanitise_response("") == ""

    def test_heading_converted_to_span(self):
        result = _sanitise_response("### Summary")
        assert '<span class="agent-heading">Summary</span>' in result

    def test_bold_converted_to_span(self):
        result = _sanitise_response("This is **important** news.")
        assert '<span class="agent-bold">important</span>' in result

    def test_italic_is_stripped_not_converted(self):
        result = _sanitise_response("This is *subtle* emphasis.")
        assert "subtle" in result
        assert "*subtle*" not in result
        assert (
            "<span" not in result or "agent-bold" not in result.split("subtle")[0][-20:]
        )

    def test_bullet_list_converted_to_bullet_spans(self):
        result = _sanitise_response("- First point\n- Second point")
        assert result.count('<span class="agent-bullet">•</span>') == 2

    def test_blank_lines_become_double_br(self):
        result = _sanitise_response("Paragraph one.\n\nParagraph two.")
        assert "<br><br>" in result

    def test_single_newlines_collapsed_to_space(self):
        result = _sanitise_response("Line one\nLine two")
        assert "\n" not in result
        assert "Line one Line two" in result

    def test_combined_markdown_elements(self):
        text = "## Outlook\n**AAPL** looks strong.\n- Bullish RSI\n- Positive news"
        result = _sanitise_response(text)
        assert '<span class="agent-heading">Outlook</span>' in result
        assert '<span class="agent-bold">AAPL</span>' in result
        assert result.count("agent-bullet") == 2


class TestConvertMarkdownTables:
    def test_converts_simple_pipe_table_to_html(self):
        text = (
            "| Model | AUC |\n|-------|-----|\n| xgboost | 0.71 |\n| lightgbm | 0.68 |"
        )
        result = _convert_markdown_tables(text)
        assert "<table>" in result
        assert "<th>Model</th>" in result
        assert "<th>AUC</th>" in result
        assert "<td>xgboost</td>" in result
        assert "<td>0.71</td>" in result

    def test_non_table_text_passed_through_unchanged(self):
        text = "Just a normal paragraph with no pipes at all."
        assert _convert_markdown_tables(text) == text

    def test_lone_pipe_row_without_separator_is_not_converted(self):
        text = "| not | a | table |"
        result = _convert_markdown_tables(text)
        assert "<table>" not in result


# ─────────────────────────────────────────────────────────────────────────────
# ToolResult
# ─────────────────────────────────────────────────────────────────────────────


class TestToolResult:
    def test_successful_result_uses_context_string_builder(self):
        result = ToolResult(
            tool_name="predict_stock", success=True, output={"p_bullish": 0.7}
        )
        ctx = result.to_context_string()
        assert "predict_stock" in ctx

    def test_failed_result_reports_error(self):
        result = ToolResult(
            tool_name="predict_stock",
            success=False,
            output=None,
            error="model not found",
        )
        ctx = result.to_context_string()
        assert "FAILED" in ctx
        assert "model not found" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# FinancialAgent — tool registry & step validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def agent():
    tools_instance = MagicMock()
    tools_instance.predict_stock = MagicMock(return_value={"prediction": 1})
    tools_instance.explain_prediction = MagicMock(return_value={"shap": []})
    tools_instance.analyze_sentiment = MagicMock(return_value={"sentiment": "positive"})
    tools_instance.get_market_summary = MagicMock(return_value={"close": 190.0})
    tools_instance.search_web = MagicMock(return_value=[{"title": "news"}])
    return FinancialAgent(
        tools_instance=tools_instance, chat_system=None, rag_pipeline=None
    )


class TestValidateStep:
    def test_valid_step_passes(self, agent):
        ok, reason = agent._validate_step(
            {"tool": "predict_stock", "args": {"ticker": "AAPL"}}
        )
        assert ok is True
        assert reason == "ok"

    def test_unknown_tool_rejected(self, agent):
        ok, reason = agent._validate_step({"tool": "launch_missiles", "args": {}})
        assert ok is False
        assert "Unknown tool" in reason

    def test_missing_tool_name_rejected(self, agent):
        ok, reason = agent._validate_step({"args": {}})
        assert ok is False

    def test_empty_tool_name_rejected(self, agent):
        ok, reason = agent._validate_step({"tool": "  ", "args": {}})
        assert ok is False

    def test_non_dict_args_rejected(self, agent):
        ok, reason = agent._validate_step({"tool": "predict_stock", "args": "AAPL"})
        assert ok is False
        assert "must be a dict" in reason

    def test_missing_required_args_rejected(self, agent):
        ok, reason = agent._validate_step({"tool": "predict_stock", "args": {}})
        assert ok is False
        assert "ticker" in reason

    def test_kb_status_unavailable_without_rag_pipeline(self, agent):
        assert agent._kb_status() == "UNAVAILABLE"

    def test_kb_status_reflects_populated_rag_pipeline(self):
        rag = MagicMock()
        rag._store_initialized = True
        rag.document_count = 42
        agent = FinancialAgent(tools_instance=MagicMock(), rag_pipeline=rag)
        assert "POPULATED" in agent._kb_status()
        assert "42" in agent._kb_status()


# ─────────────────────────────────────────────────────────────────────────────
# FinancialAgent — planning
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanToolCalls:
    def test_valid_plan_is_parsed_and_validated(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = (
            json.dumps([{"tool": "predict_stock", "args": {"ticker": "AAPL"}}]),
            10,
        )
        plan = agent._plan_tool_calls("What's the outlook for AAPL?")
        assert plan == [{"tool": "predict_stock", "args": {"ticker": "AAPL"}}]

    def test_invalid_steps_are_dropped_but_valid_ones_kept(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = (
            json.dumps(
                [
                    {"tool": "predict_stock", "args": {"ticker": "AAPL"}},
                    {"tool": "nonexistent_tool", "args": {}},
                ]
            ),
            10,
        )
        plan = agent._plan_tool_calls("query")
        assert len(plan) == 1
        assert plan[0]["tool"] == "predict_stock"

    def test_non_list_json_returns_empty_plan(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = (json.dumps({"tool": "predict_stock"}), 10)
        plan = agent._plan_tool_calls("query")
        assert plan == []

    def test_llm_failure_returns_empty_plan_not_exception(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.side_effect = RuntimeError("LLM down")
        plan = agent._plan_tool_calls("query")
        assert plan == []

    def test_malformed_json_returns_empty_plan(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = ("no json here at all", 5)
        plan = agent._plan_tool_calls("query")
        assert plan == []


# ─────────────────────────────────────────────────────────────────────────────
# FinancialAgent — execute_tool
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteTool:
    def test_successful_execution_returns_success_result(self, agent):
        result = agent._execute_tool("predict_stock", {"ticker": "AAPL"})
        assert result.success is True
        assert result.output == {"prediction": 1}

    def test_tool_execution_error_is_captured_not_raised(self, agent):
        agent.tools_instance.predict_stock.side_effect = ToolExecutionError(
            "model missing"
        )
        result = agent._execute_tool("predict_stock", {"ticker": "AAPL"})
        assert result.success is False
        assert "model missing" in result.error

    def test_unexpected_exception_is_captured_not_raised(self, agent):
        agent.tools_instance.predict_stock.side_effect = RuntimeError("boom")
        result = agent._execute_tool("predict_stock", {"ticker": "AAPL"})
        assert result.success is False
        assert "boom" in result.error


# ─────────────────────────────────────────────────────────────────────────────
# FinancialAgent — run() end-to-end
# ─────────────────────────────────────────────────────────────────────────────


class TestRun:
    def test_run_executes_planned_tools_and_returns_response(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = (
            json.dumps([{"tool": "predict_stock", "args": {"ticker": "AAPL"}}]),
            10,
        )
        chat_system = MagicMock()
        chat_system.chat.return_value = MagicMock(content="AAPL looks **bullish**.")
        agent.chat_system = chat_system

        result = agent.run("What's the outlook for AAPL?")

        assert result["query"] == "What's the outlook for AAPL?"
        assert "predict_stock" in result["tools_used"]
        assert '<span class="agent-bold">bullish</span>' in result["response"]
        assert len(result["tool_results"]) == 1

    def test_run_caps_tool_execution_at_three_steps(self, agent):
        agent._llm = MagicMock()
        plan = [
            {"tool": "predict_stock", "args": {"ticker": "AAPL"}},
            {"tool": "get_market_summary", "args": {"ticker": "AAPL"}},
            {"tool": "search_web", "args": {"query": "AAPL news"}},
            {"tool": "explain_prediction", "args": {"ticker": "AAPL"}},
        ]
        agent._llm.chat.return_value = (json.dumps(plan), 10)
        agent.chat_system = MagicMock(
            chat=MagicMock(return_value=MagicMock(content="ok"))
        )

        result = agent.run("Give me everything on AAPL")

        assert len(result["tool_results"]) == 3  # capped, 4th step ignored

    def test_run_falls_back_to_raw_tool_output_without_chat_system(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = (
            json.dumps([{"tool": "predict_stock", "args": {"ticker": "AAPL"}}]),
            10,
        )
        agent.chat_system = None

        result = agent.run("Predict AAPL")

        assert (
            "LLM not available" in result["response"]
            or "predict_stock" in result["response"]
        )

    def test_run_only_lists_successful_tools_in_tools_used(self, agent):
        agent._llm = MagicMock()
        agent._llm.chat.return_value = (
            json.dumps([{"tool": "predict_stock", "args": {"ticker": "AAPL"}}]),
            10,
        )
        agent.tools_instance.predict_stock.side_effect = RuntimeError("failed")
        agent.chat_system = MagicMock(
            chat=MagicMock(return_value=MagicMock(content="ok"))
        )

        result = agent.run("Predict AAPL")

        assert result["tools_used"] == []
        assert result["tool_results"][0]["success"] is False
