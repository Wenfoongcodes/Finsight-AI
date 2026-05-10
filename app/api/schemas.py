"""
FinSight AI — Phase 7: API Schemas
Pydantic v2 request and response models for all FastAPI endpoints.

Changes in this revision
------------------------
* ``IngestRequest`` gains a ``source_type`` discriminator field
  (``"text"`` | ``"url"``) and an optional ``url`` field.  A
  ``model_validator`` enforces that exactly one of ``texts`` or ``url`` is
  supplied and rejects the other to prevent ambiguous requests.
* ``IngestResponse`` gains ``source_type``, ``char_count``, and ``title``
  fields so callers can see what was actually ingested.
* Pydantic ``protected_namespaces`` silenced on all models that carry a
  ``model_name`` field to eliminate the ``UserWarning`` on startup.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    environment: str


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    model_name: str = Field(default="xgboost", examples=["xgboost", "lightgbm"])
    use_cache: bool = Field(default=True)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str
    feature_value: float


class PredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str
    model_name: str
    prediction: int             # 0 = bearish, 1 = bullish
    prediction_label: str       # 'BULLISH' | 'BEARISH'
    probability: float          # Directional: P(predicted direction)
    p_bullish: float            # Raw P(bullish) — always 0..1
    p_bearish: float            # Raw P(bearish) = 1 - p_bullish
    confidence_label: str
    latest_close: float
    narrative: str
    top_features: list[SHAPFeature]

    @model_validator(mode="after")
    def _validate_probabilities(self) -> "PredictionResult":
        """
        Assert that ``p_bullish + p_bearish`` rounds to 1.0.

        Tolerance of 1e-4 accommodates ``round(..., 4)`` floating-point
        rounding without producing false positives.
        """
        total = self.p_bullish + self.p_bearish
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"p_bullish ({self.p_bullish}) + p_bearish ({self.p_bearish}) "
                f"must sum to 1.0, got {total:.6f}"
            )
        return self


class BatchPredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    tickers: list[str] = Field(..., min_length=1, max_length=20)
    model_name: str = "xgboost"

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, v: list[str]) -> list[str]:
        return [t.upper().strip() for t in v]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str = Field(..., min_length=1, max_length=10)
    model_name: str = Field(default="xgboost")
    run_hpo: bool = Field(default=False)
    hpo_trials: int = Field(default=20, ge=5, le=100)
    period_years: int = Field(default=5, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class TrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str
    model_name: str
    mean_accuracy: float
    mean_f1: float
    mean_roc_auc: float
    mean_mae: float
    mean_rmse: float
    n_folds: int
    n_features: int
    trained_at: str
    best_params: dict


# ─────────────────────────────────────────────────────────────────────────────
# RAG / Chat
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Unified ingestion request that accepts either raw text or a URL.

    Exactly one of ``texts`` or ``url`` must be supplied:

    * **Text ingestion** — set ``source_type="text"`` and provide ``texts``.
    * **URL ingestion**  — set ``source_type="url"`` and provide ``url``.

    The ``model_validator`` enforces this constraint and provides a clear
    error message when neither or both are supplied.
    """

    source_type: Literal["text", "url"] = Field(
        default="text",
        description="Ingestion mode: 'text' for raw strings, 'url' for a web article.",
    )

    # Text-mode fields
    texts: Optional[list[str]] = Field(
        default=None,
        description="List of text strings to ingest (required when source_type='text').",
    )
    source: str = Field(
        default="api_upload",
        description="Arbitrary source label attached to text chunks as metadata.",
    )

    # URL-mode fields
    url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Article URL to fetch and ingest (required when source_type='url').",
    )

    @model_validator(mode="after")
    def _check_exclusive_source(self) -> "IngestRequest":
        """Enforce that exactly one of texts / url is provided."""
        if self.source_type == "text":
            if not self.texts:
                raise ValueError(
                    "texts must be a non-empty list when source_type='text'."
                )
            if self.url is not None:
                raise ValueError(
                    "url must not be set when source_type='text'. "
                    "Use source_type='url' to ingest a web article."
                )
        elif self.source_type == "url":
            if self.url is None:
                raise ValueError(
                    "url is required when source_type='url'."
                )
            if self.texts is not None:
                raise ValueError(
                    "texts must not be set when source_type='url'. "
                    "Use source_type='text' to ingest raw text."
                )
        return self


class IngestResponse(BaseModel):
    """
    Response returned after a successful ingestion.

    Fields
    ------
    ingested_count : Number of source documents (texts) or URLs processed.
    chunks_added   : Number of embedding chunks added to the vector store.
    source_type    : Echo of the request's source_type.
    title          : Article title (URL ingestion only; empty for text mode).
    char_count     : Total characters extracted (URL mode) or 0 (text mode).
    duplicate      : True when a URL was already in the index (URL mode only).
    message        : Human-readable summary.
    """

    ingested_count: int
    chunks_added: int = 0
    source_type: str = "text"
    title: str = ""
    char_count: int = 0
    duplicate: bool = False
    message: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = Field(default=True)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    used_rag: bool
    model: str
    tokens_used: int
    session_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    query: str
    response: str
    tools_used: list[str]
    tool_results: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Market Data
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataRequest(BaseModel):
    ticker: str
    period_years: int = Field(default=1, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class MarketDataSummary(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    rows: int
    columns: list[str]
    close_min: float
    close_max: float
    close_mean: float
    null_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int