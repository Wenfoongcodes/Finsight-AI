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

Be concise, factual, and data-driven. When SHAP explanations are provided, interpret them clearly.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str   # 'user' | 'assistant' | 'system'
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    content: str
    used_rag: bool
    context_snippets: list[str] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._messages: list[ChatMessage] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(ChatMessage(role=role, content=content))

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
    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required: pip install openai") from exc

        if not settings.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set.")

        self._client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

    def chat(self, messages, model=settings.LLM_MODEL,
             temperature=settings.LLM_TEMPERATURE,
             max_tokens=settings.LLM_MAX_TOKENS):

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
# Financial Chat System (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

class FinancialChatSystem:
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_turns: int = 10,
    ) -> None:
        self.rag = rag_pipeline
        self.memory_turns = memory_turns

        # ✅ session-based memory store
        self._memory_store: dict[str, ConversationMemory] = {}

        self._llm: Optional[OpenAIClient] = None

    def _get_llm(self) -> OpenAIClient:
        if self._llm is None:
            self._llm = OpenAIClient()
        return self._llm

    def _get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memory_store:
            self._memory_store[session_id] = ConversationMemory(
                max_turns=self.memory_turns
            )
        return self._memory_store[session_id]

    def _build_messages(self, user_query: str, context: Optional[str] = None,
                        memory: ConversationMemory = None) -> list[dict]:

        system_content = SYSTEM_PROMPT

        if context:
            system_content += f"\n\nContext from financial knowledge base:\n{context}"

        messages = [{"role": "system", "content": system_content}]

        if memory:
            messages.extend(memory.get_history())

        messages.append({"role": "user", "content": user_query})
        return messages

    # ───────────────────────────────────────────────────────────────
    # FIXED CHAT METHOD
    # ───────────────────────────────────────────────────────────────

    def chat(
        self,
        user_query: str,
        use_rag: bool = True,
        session_id: str | None = None,
        prediction_context: Optional[str] = None,
    ) -> ChatResponse:

        if session_id is None:
            session_id = "default"

        memory = self._get_memory(session_id)

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

        messages = self._build_messages(user_query, context, memory)

        llm = self._get_llm()
        response_text, tokens = llm.chat(messages)

        memory.add("user", user_query)
        memory.add("assistant", response_text)

        logger.info(
            "Chat response generated: tokens=%d, rag=%s, turns=%d",
            tokens, used_rag, memory.turn_count,
        )

        return ChatResponse(
            content=response_text,
            used_rag=used_rag,
            context_snippets=snippets,
            model=settings.LLM_MODEL,
            tokens_used=tokens,
        )

    def reset_memory(self, session_id: str | None = None) -> None:
        if session_id:
            self._memory_store.pop(session_id, None)
        else:
            self._memory_store.clear()

        logger.info("Conversation memory cleared.")