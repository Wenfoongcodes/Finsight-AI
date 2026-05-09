"""
FinSight AI — Phase 9: LLM Chat System
Provides a conversational financial assistant powered by OpenAI GPT
with RAG context injection and multi-turn memory management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.exceptions import LLMError
from app.core.logging_config import get_logger
from app.rag.rag_pipeline import RAGPipeline
from configs.settings import settings

logger = get_logger("llm_chat")

SYSTEM_PROMPT = """You are FinSight AI, an expert financial analyst and AI assistant.
You have access to real-time market prediction models, SHAP-based explainability,
and a financial knowledge base. Your role is to:

1. Answer financial questions with clarity and precision.
2. Explain ML model predictions in plain English.
3. Ground your answers in retrieved financial context when available.
4. Always acknowledge uncertainty where it exists.
5. Never give direct investment advice — always recommend consulting a qualified financial advisor.

Be concise, factual, and data-driven. When SHAP explanations are provided, interpret them clearly."""


# ─────────────────────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """A single message in the conversation history."""
    role: str   # 'user' | 'assistant' | 'system'
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """LLM response with metadata."""
    content: str
    used_rag: bool
    context_snippets: list[str] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Manages multi-turn conversation history with a configurable window.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._messages: list[ChatMessage] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(ChatMessage(role=role, content=content))
        # Trim to max_turns pairs (user + assistant = 2 messages each)
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def get_history(self) -> list[dict]:
        return [m.to_dict() for m in self._messages]

    def clear(self) -> None:
        self._messages = []

    @property
    def turn_count(self) -> int:
        return len(self._messages) // 2


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient:
    """
    Thin wrapper around the OpenAI chat completion API.

    Raises:
        LLMError: On API failure.
        ImportError: If openai package is not installed.
    """

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required: pip install openai") from exc

        if not settings.OPENAI_API_KEY:
            raise LLMError(
                "OPENAI_API_KEY is not set.",
                detail="Set it in .env or environment variables.",
            )

        from openai import OpenAI
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url="https://api.groq.com/openai/v1",)

    def chat(
        self,
        messages: list[dict],
        model: str = settings.LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ) -> tuple[str, int]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: OpenAI model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            (response_text, tokens_used)

        Raises:
            LLMError: On API error.
        """
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return content, tokens
        except Exception as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Financial Chat System
# ─────────────────────────────────────────────────────────────────────────────

class FinancialChatSystem:
    """
    RAG-augmented conversational AI for financial analysis.

    Combines:
    - Multi-turn conversation memory
    - RAG context retrieval
    - OpenAI LLM completion
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_turns: int = 10,
    ) -> None:
        self.rag = rag_pipeline
        self.memory = ConversationMemory(max_turns=memory_turns)
        self._llm: Optional[OpenAIClient] = None

    def _get_llm(self) -> OpenAIClient:
        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm

    def _build_messages(self, user_query: str, context: Optional[str] = None) -> list[dict]:
        """Assemble full message list including system prompt, RAG context, and history."""
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\nContext from financial knowledge base:\n{context}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.memory.get_history())
        messages.append({"role": "user", "content": user_query})
        return messages

    def chat(
        self,
        user_query: str,
        use_rag: bool = True,
        prediction_context: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a user message and return an LLM-generated response.

        Args:
            user_query: Natural language user input.
            use_rag: Whether to inject RAG context.
            prediction_context: Optional SHAP/prediction narrative to include.

        Returns:
            ChatResponse with generated text and metadata.

        Raises:
            LLMError: On API failure.
        """
        context = ""
        snippets: list[str] = []
        used_rag = False

        if use_rag and self.rag and self.rag._store_initialized:
            try:
                rag_context = self.rag.build_context(user_query)
                if rag_context and "No relevant context" not in rag_context:
                    context = rag_context
                    snippets = [r["content"][:200] for r in self.rag.retrieve(user_query)]
                    used_rag = True
            except Exception as exc:
                logger.warning("RAG retrieval failed: %s", exc)

        if prediction_context:
            context = f"{prediction_context}\n\n{context}".strip()

        full_query = user_query
        messages = self._build_messages(full_query, context=context if context else None)

        llm = self._get_llm()
        response_text, tokens = llm.chat(messages)

        self.memory.add("user", user_query)
        self.memory.add("assistant", response_text)

        logger.info(
            "Chat response generated: tokens=%d, rag=%s, turns=%d",
            tokens, used_rag, self.memory.turn_count,
        )

        return ChatResponse(
            content=response_text,
            used_rag=used_rag,
            context_snippets=snippets,
            model=settings.LLM_MODEL,
            tokens_used=tokens,
        )

    def ask_about_prediction(
        self,
        ticker: str,
        prediction_narrative: str,
        user_question: str,
    ) -> ChatResponse:
        """
        Ask a specific question about a recent prediction with context injection.

        Args:
            ticker: Ticker symbol.
            prediction_narrative: SHAP narrative from PredictionService.
            user_question: User's follow-up question.

        Returns:
            ChatResponse.
        """
        enriched_query = (
            f"Regarding {ticker}: {user_question}\n\n"
            f"Model Prediction Context: {prediction_narrative}"
        )
        return self.chat(enriched_query, use_rag=True)

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared.")
