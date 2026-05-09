"""
FinSight AI — Custom Exception Hierarchy
Provides domain-specific exceptions for precise error handling.
"""


class FinSightBaseError(Exception):
    """Base exception for all FinSight AI errors."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, detail={self.detail!r})"


# ── Data Layer ────────────────────────────────────────────────────────────────

class DataIngestionError(FinSightBaseError):
    """Raised when market data cannot be fetched or parsed."""


class DataValidationError(FinSightBaseError):
    """Raised when ingested data fails schema/quality checks."""


class InsufficientDataError(FinSightBaseError):
    """Raised when there is not enough data for feature engineering or training."""


# ── Feature Engineering ───────────────────────────────────────────────────────

class FeatureEngineeringError(FinSightBaseError):
    """Raised when feature computation fails."""


# ── ML Layer ─────────────────────────────────────────────────────────────────

class ModelNotFoundError(FinSightBaseError):
    """Raised when a requested model artifact does not exist on disk."""


class ModelTrainingError(FinSightBaseError):
    """Raised when model training encounters an unrecoverable error."""


class PredictionError(FinSightBaseError):
    """Raised when inference fails."""


# ── Explainability Layer ──────────────────────────────────────────────────────

class ExplainabilityError(FinSightBaseError):
    """Raised when SHAP/LIME explanation generation fails."""


# ── RAG / LLM Layer ──────────────────────────────────────────────────────────

class EmbeddingError(FinSightBaseError):
    """Raised when embedding generation fails."""


class VectorStoreError(FinSightBaseError):
    """Raised when the vector store cannot be read or written."""


class LLMError(FinSightBaseError):
    """Raised when the LLM API returns an error or unexpected response."""


class RAGError(FinSightBaseError):
    """Raised when the RAG pipeline fails to retrieve or generate."""


# ── Agent Layer ───────────────────────────────────────────────────────────────

class AgentError(FinSightBaseError):
    """Raised when the AI agent encounters an unrecoverable error."""


class ToolExecutionError(AgentError):
    """Raised when an agent tool call fails."""


# ── API Layer ─────────────────────────────────────────────────────────────────

class APIError(FinSightBaseError):
    """Base class for API-level errors."""


class AuthenticationError(APIError):
    """Raised on invalid or missing API credentials."""


class RateLimitError(APIError):
    """Raised when an external API rate limit is hit."""
