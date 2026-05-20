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
    role: str  # 'user' | 'assistant' | 'system'
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
            self._messages = self._messages[-(self.max_turns * 2) :]

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
    Thin wrapper around the OpenAI Python SDK.

    Respects ``settings.LLM_BASE_URL`` for alternative providers (Groq,
    Azure OpenAI, Ollama, etc.).  When ``LLM_BASE_URL`` is not set (or is
    an empty string), the official OpenAI endpoint is used.

    Provider examples::

        # Official OpenAI (default — leave LLM_BASE_URL unset)
        LLM_BASE_URL=

        # Groq  (remember to set LLM_MODEL to a Groq-supported model)
        LLM_BASE_URL=https://api.groq.com/openai/v1
        LLM_MODEL=llama3-70b-8192

        # Ollama (local)
        LLM_BASE_URL=http://localhost:11434/v1
        LLM_MODEL=llama3
    """

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required: pip install openai") from exc

        if not settings.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set.")

        # Normalise empty string → None so the SDK uses its own default endpoint.
        base_url = settings.LLM_BASE_URL or None
        if isinstance(base_url, str) and not base_url.strip():
            base_url = None

        client_kwargs: dict = {"api_key": settings.OPENAI_API_KEY}
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info("LLM base URL override: %s", base_url)
        else:
            logger.info("LLM using default OpenAI endpoint")

        self._client = OpenAI(**client_kwargs)
        self._base_url = base_url  # stored for logging / diagnostics

    def chat(
        self,
        messages,
        model: str = settings.LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ) -> tuple[str, int]:
        """
        Send a chat completion request.

        Args:
            messages:    List of message dicts (role + content).
            model:       Model name.  When using a non-OpenAI provider via
                         ``LLM_BASE_URL``, make sure this matches a model
                         that provider actually supports.
            temperature: Sampling temperature.
            max_tokens:  Maximum completion tokens.

        Returns:
            ``(content_str, total_tokens)``

        Raises:
            LLMError: On any API error, including invalid model names,
                      auth failures, and rate limits.
        """
        # Always log the model being used — essential for diagnosing
        # provider/model-name mismatches like "gpt-4o-mini" sent to Groq.
        logger.debug(
            "LLM chat request | model=%s | endpoint=%s | temperature=%.2f | max_tokens=%d",
            model,
            self._base_url or "https://api.openai.com/v1",
            temperature,
            max_tokens,
        )

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            logger.debug(
                "LLM chat response | model=%s | tokens=%d | chars=%d",
                model,
                tokens,
                len(content),
            )
            return content, tokens

        except Exception as exc:
            # Extract the most informative error message available.
            # The OpenAI SDK wraps errors in ``openai.APIError`` subclasses
            # which carry a ``message`` attribute with the provider's body.
            detail = getattr(exc, "message", None) or str(exc)
            status = getattr(exc, "status_code", None)
            if status:
                detail = f"HTTP {status}: {detail}"

            raise LLMError(
                f"LLM API call failed (model={model}, endpoint={self._base_url or 'openai'}): "
                f"{detail}",
                detail=str(exc),
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Financial Chat System
# ─────────────────────────────────────────────────────────────────────────────


class FinancialChatSystem:
    """
    Conversational financial assistant with RAG context injection
    and per-session multi-turn memory.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_turns: int = 10,
    ) -> None:
        self.rag = rag_pipeline
        self.memory_turns = memory_turns
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

    def _build_messages(
        self,
        user_query: str,
        context: Optional[str] = None,
        memory: Optional[ConversationMemory] = None,
    ) -> list[dict]:
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\nContext from financial knowledge base:\n{context}"

        messages = [{"role": "system", "content": system_content}]
        if memory:
            messages.extend(memory.get_history())
        messages.append({"role": "user", "content": user_query})
        return messages

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
                    snippets = [
                        r["content"][:200] for r in self.rag.retrieve(user_query)
                    ]
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
            tokens,
            used_rag,
            memory.turn_count,
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
