"""
Unit tests for ``app.rag.llm_chat``.

Covers:
- ``ConversationMemory``       — turn tracking, truncation, clearing.
- ``OpenAIClient``             — init guards, error wrapping, reasoning_effort
                                   passthrough (mocking the ``openai`` SDK).
- ``FinancialChatSystem``      — RAG context injection, graceful RAG failure,
                                   memory persistence across turns.

The real ``openai`` package is mocked out entirely so these tests run
without network access or an API key.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import LLMError
from app.rag.llm_chat import ChatMessage, ConversationMemory, FinancialChatSystem

# ─────────────────────────────────────────────────────────────────────────────
# ConversationMemory
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationMemory:
    def test_add_and_get_history_round_trips(self):
        mem = ConversationMemory(max_turns=10)
        mem.add("user", "What is RSI?")
        mem.add("assistant", "Relative Strength Index...")

        history = mem.get_history()
        assert history == [
            {"role": "user", "content": "What is RSI?"},
            {"role": "assistant", "content": "Relative Strength Index..."},
        ]

    def test_turn_count_counts_pairs_not_messages(self):
        mem = ConversationMemory(max_turns=10)
        mem.add("user", "Q1")
        mem.add("assistant", "A1")
        mem.add("user", "Q2")
        mem.add("assistant", "A2")
        assert mem.turn_count == 2

    def test_truncates_to_max_turns_keeping_most_recent(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(5):
            mem.add("user", f"Q{i}")
            mem.add("assistant", f"A{i}")

        history = mem.get_history()
        # max_turns=2 -> at most 4 messages retained, most recent ones.
        assert len(history) == 4
        assert history[0]["content"] == "Q3"
        assert history[-1]["content"] == "A4"

    def test_clear_empties_history(self):
        mem = ConversationMemory()
        mem.add("user", "hello")
        mem.clear()
        assert mem.get_history() == []
        assert mem.turn_count == 0

    def test_chat_message_to_dict(self):
        msg = ChatMessage(role="user", content="hi")
        assert msg.to_dict() == {"role": "user", "content": "hi"}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIClient
# ─────────────────────────────────────────────────────────────────────────────


def _make_fake_openai_module():
    """Build a fake ``openai`` module with a controllable ``OpenAI`` class."""
    fake_module = MagicMock()
    return fake_module


@pytest.fixture
def fake_openai_sdk(monkeypatch):
    """
    Install a fake ``openai`` module in ``sys.modules`` so ``OpenAIClient``
    can be imported/instantiated without the real SDK, and return the
    mock client instance ``OpenAI(...)`` will produce.
    """
    fake_client_instance = MagicMock()
    fake_openai_cls = MagicMock(return_value=fake_client_instance)
    fake_module = SimpleNamespace(OpenAI=fake_openai_cls)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    yield fake_client_instance, fake_openai_cls


class TestOpenAIClientInit:
    def test_raises_llm_error_when_api_key_missing(self, fake_openai_sdk, monkeypatch):
        from app.rag.llm_chat import OpenAIClient

        monkeypatch.setattr("app.rag.llm_chat.settings.OPENAI_API_KEY", None)
        with pytest.raises(LLMError):
            OpenAIClient()

    def test_uses_default_endpoint_when_base_url_unset(
        self, fake_openai_sdk, monkeypatch
    ):
        from app.rag.llm_chat import OpenAIClient

        _, fake_cls = fake_openai_sdk
        monkeypatch.setattr("app.rag.llm_chat.settings.OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("app.rag.llm_chat.settings.LLM_BASE_URL", None)

        client = OpenAIClient()
        _, kwargs = fake_cls.call_args
        assert "base_url" not in kwargs
        assert client._base_url is None

    def test_empty_string_base_url_normalised_to_none(
        self, fake_openai_sdk, monkeypatch
    ):
        from app.rag.llm_chat import OpenAIClient

        _, fake_cls = fake_openai_sdk
        monkeypatch.setattr("app.rag.llm_chat.settings.OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("app.rag.llm_chat.settings.LLM_BASE_URL", "   ")

        OpenAIClient()
        _, kwargs = fake_cls.call_args
        assert "base_url" not in kwargs

    def test_custom_base_url_passed_through(self, fake_openai_sdk, monkeypatch):
        from app.rag.llm_chat import OpenAIClient

        _, fake_cls = fake_openai_sdk
        monkeypatch.setattr("app.rag.llm_chat.settings.OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            "app.rag.llm_chat.settings.LLM_BASE_URL", "https://api.groq.com/openai/v1"
        )

        client = OpenAIClient()
        _, kwargs = fake_cls.call_args
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert client._base_url == "https://api.groq.com/openai/v1"


class TestOpenAIClientChat:
    def _build_client(self, fake_openai_sdk, monkeypatch):
        from app.rag.llm_chat import OpenAIClient

        monkeypatch.setattr("app.rag.llm_chat.settings.OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("app.rag.llm_chat.settings.LLM_BASE_URL", None)
        return OpenAIClient()

    def test_chat_returns_content_and_token_count(self, fake_openai_sdk, monkeypatch):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.usage = MagicMock(total_tokens=42)
        fake_client_instance.chat.completions.create.return_value = mock_response

        content, tokens = client.chat([{"role": "user", "content": "hi"}])

        assert content == "Hello!"
        assert tokens == 42

    def test_chat_wraps_provider_errors_in_llm_error(
        self, fake_openai_sdk, monkeypatch
    ):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)

        api_error = Exception("rate limited")
        api_error.status_code = 429
        api_error.message = "Too many requests"
        fake_client_instance.chat.completions.create.side_effect = api_error

        with pytest.raises(LLMError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])

        assert "429" in str(exc_info.value)

    def test_chat_passes_reasoning_effort_as_extra_body(
        self, fake_openai_sdk, monkeypatch
    ):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage = None
        fake_client_instance.chat.completions.create.return_value = mock_response

        client.chat([{"role": "user", "content": "hi"}], reasoning_effort="low")

        _, kwargs = fake_client_instance.chat.completions.create.call_args
        assert kwargs["extra_body"] == {"reasoning_effort": "low"}

    def test_chat_handles_none_usage_gracefully(self, fake_openai_sdk, monkeypatch):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_response.usage = None
        fake_client_instance.chat.completions.create.return_value = mock_response

        _, tokens = client.chat([{"role": "user", "content": "hi"}])
        assert tokens == 0

    def test_chat_handles_empty_content_as_empty_string(
        self, fake_openai_sdk, monkeypatch
    ):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_response.usage = MagicMock(total_tokens=5)
        fake_client_instance.chat.completions.create.return_value = mock_response

        content, _ = client.chat([{"role": "user", "content": "hi"}])
        assert content == ""

    def test_chat_stream_yields_deltas(self, fake_openai_sdk, monkeypatch):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)

        chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Hel"))])
        chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content="lo"))])
        fake_client_instance.chat.completions.create.return_value = iter(
            [chunk1, chunk2]
        )

        deltas = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert deltas == ["Hel", "lo"]

    def test_chat_stream_wraps_errors_in_llm_error(self, fake_openai_sdk, monkeypatch):
        fake_client_instance, _ = fake_openai_sdk
        client = self._build_client(fake_openai_sdk, monkeypatch)
        fake_client_instance.chat.completions.create.side_effect = RuntimeError("boom")

        with pytest.raises(LLMError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))


# ─────────────────────────────────────────────────────────────────────────────
# FinancialChatSystem
# ─────────────────────────────────────────────────────────────────────────────


class TestFinancialChatSystem:
    def _mock_llm(self, content="Here's the analysis.", tokens=100):
        llm = MagicMock()
        llm.chat.return_value = (content, tokens)
        return llm

    def test_chat_without_rag_returns_response_and_updates_memory(self):
        system = FinancialChatSystem(rag_pipeline=None)
        system._llm = self._mock_llm()

        response = system.chat("What is the outlook for AAPL?", use_rag=True)

        assert response.content == "Here's the analysis."
        assert response.used_rag is False
        assert response.tokens_used == 100

        memory = system._get_memory("default")
        assert memory.turn_count == 1

    def test_chat_uses_rag_context_when_store_initialized(self):
        rag = MagicMock()
        rag._store_initialized = True
        rag.build_context.return_value = "Relevant financial context:\n[1] ..."
        rag.retrieve.return_value = [{"content": "AAPL beat earnings" * 5}]

        system = FinancialChatSystem(rag_pipeline=rag)
        system._llm = self._mock_llm()

        response = system.chat("Tell me about AAPL earnings")

        assert response.used_rag is True
        assert len(response.context_snippets) == 1
        rag.build_context.assert_called_once()

    def test_chat_skips_rag_when_store_not_initialized(self):
        rag = MagicMock()
        rag._store_initialized = False

        system = FinancialChatSystem(rag_pipeline=rag)
        system._llm = self._mock_llm()

        response = system.chat("Tell me about AAPL")

        assert response.used_rag is False
        rag.build_context.assert_not_called()

    def test_chat_degrades_gracefully_when_rag_retrieval_raises(self):
        rag = MagicMock()
        rag._store_initialized = True
        rag.build_context.side_effect = RuntimeError("FAISS index corrupted")

        system = FinancialChatSystem(rag_pipeline=rag)
        system._llm = self._mock_llm()

        response = system.chat("Tell me about AAPL")

        # Should not raise — falls back to no RAG context.
        assert response.used_rag is False
        assert response.content == "Here's the analysis."

    def test_chat_skips_rag_context_when_no_relevant_context_found(self):
        rag = MagicMock()
        rag._store_initialized = True
        rag.build_context.return_value = "No relevant context found."

        system = FinancialChatSystem(rag_pipeline=rag)
        system._llm = self._mock_llm()

        response = system.chat("Tell me about AAPL")
        assert response.used_rag is False

    def test_prediction_context_is_prepended(self):
        system = FinancialChatSystem(rag_pipeline=None)
        llm = self._mock_llm()
        system._llm = llm

        system.chat(
            "Explain this prediction", prediction_context="ML says BULLISH, p=0.7"
        )

        messages = llm.chat.call_args[0][0]
        system_msg = messages[0]["content"]
        assert "ML says BULLISH" in system_msg

    def test_memory_persists_across_multiple_turns_same_session(self):
        system = FinancialChatSystem(rag_pipeline=None)
        system._llm = self._mock_llm()

        system.chat("First question", session_id="session-1")
        system.chat("Second question", session_id="session-1")

        memory = system._get_memory("session-1")
        assert memory.turn_count == 2

    def test_separate_sessions_have_isolated_memory(self):
        system = FinancialChatSystem(rag_pipeline=None)
        system._llm = self._mock_llm()

        system.chat("Hi from session A", session_id="a")
        system.chat("Hi from session B", session_id="b")

        assert system._get_memory("a").turn_count == 1
        assert system._get_memory("b").turn_count == 1

    def test_reset_memory_clears_specific_session(self):
        system = FinancialChatSystem(rag_pipeline=None)
        system._llm = self._mock_llm()
        system.chat("hello", session_id="a")

        system.reset_memory(session_id="a")

        assert "a" not in system._memory_store

    def test_reset_memory_clears_all_sessions_when_no_id_given(self):
        system = FinancialChatSystem(rag_pipeline=None)
        system._llm = self._mock_llm()
        system.chat("hello", session_id="a")
        system.chat("hello", session_id="b")

        system.reset_memory()

        assert system._memory_store == {}
