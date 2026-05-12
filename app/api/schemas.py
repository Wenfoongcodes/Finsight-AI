"""
FinSight AI — API Schemas (v3)

Changes vs v2
-------------
* ``PredictionRequest`` gains ``horizon`` field ('1d' | '7d' | '1m' | '6m').
* ``PredictionResult`` gains ``horizon``, ``auto_trained`` fields.
* ``BatchPredictionRequest`` gains ``horizon``.
* ``TrainRequest`` gains ``horizon``.
* ``TrainResponse`` gains ``horizon``.
* ``LeaderboardResponse`` gains ``horizon``.
* ``IntelligenceBriefSchema`` added for API consumers.
* All other contracts unchanged.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator

VALID_HORIZONS = ("1d", "7d", "1m", "6m")


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
    """
    Single-ticker prediction request.

    ``model_name`` is intentionally absent — the system selects the
    best-performing model per ticker/horizon automatically.
    """

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    horizon: str = Field(default="1d", examples=["1d", "7d", "1m", "6m"])
    use_cache: bool = Field(default=True)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: str) -> str:
        if v not in VALID_HORIZONS:
            raise ValueError(f"horizon must be one of {VALID_HORIZONS}")
        return v


class NewsItemSchema(BaseModel):
    title: str
    snippet: str
    url: str


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str
    feature_value: float


class IntelligenceBriefSchema(BaseModel):
    """Serialized form of IntelligenceBrief for API consumers."""

    ticker: str
    situation_summary: str
    bullish_catalysts: list[str] = Field(default_factory=list)
    bearish_catalysts: list[str] = Field(default_factory=list)
    aggregate_sentiment: str = "neutral"
    sentiment_score: float = 0.0
    source_quality_note: str = ""
    retrieval_success: bool = True


class PredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    # ML signal
    ticker: str
    model_name: str
    horizon: str = "1d"
    prediction: int
    prediction_label: str
    probability: float
    p_bullish: float
    p_bearish: float
    confidence_label: str
    latest_close: float
    narrative: str
    top_features: list[SHAPFeature]
    auto_trained: bool = False

    # Fused signal
    fused_direction: str = "UNKNOWN"
    fused_confidence: str = "LOW"
    fused_probability: float = 0.5
    fusion_narrative: str = ""
    fusion_applied: bool = False
    news_sentiment: str = "neutral"
    news_items: list[NewsItemSchema] = Field(default_factory=list)

    # Intelligence brief (optional)
    intelligence_brief: Optional[IntelligenceBriefSchema] = None

    @model_validator(mode="after")
    def _validate_probabilities(self) -> "PredictionResult":
        total = self.p_bullish + self.p_bearish
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"p_bullish ({self.p_bullish}) + p_bearish ({self.p_bearish}) "
                f"must sum to 1.0, got {total:.6f}"
            )
        return self


class BatchPredictionRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=20)
    horizon: str = Field(default="1d")

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, v: list[str]) -> list[str]:
        return [t.upper().strip() for t in v]

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: str) -> str:
        if v not in VALID_HORIZONS:
            raise ValueError(f"horizon must be one of {VALID_HORIZONS}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str = Field(..., min_length=1, max_length=10)
    model_name: str = Field(default="xgboost")
    horizon: str = Field(default="1d")
    run_hpo: bool = Field(default=False)
    hpo_trials: int = Field(default=20, ge=5, le=100)
    period_years: int = Field(default=5, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: str) -> str:
        if v not in VALID_HORIZONS:
            raise ValueError(f"horizon must be one of {VALID_HORIZONS}")
        return v


class TrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str
    model_name: str
    horizon: str = "1d"
    mean_accuracy: float
    mean_f1: float
    mean_roc_auc: float
    mean_mae: float
    mean_rmse: float
    n_folds: int
    n_features: int
    trained_at: str
    training_duration_s: float = 0.0
    trigger_reason: str = "manual_request"
    best_params: dict


# ─────────────────────────────────────────────────────────────────────────────
# RAG / Chat
# ─────────────────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    source_type: Literal["text", "url"] = Field(default="text")
    texts: Optional[list[str]] = None
    source: str = Field(default="api_upload")
    url: Optional[AnyHttpUrl] = None

    @model_validator(mode="after")
    def _check_exclusive_source(self) -> "IngestRequest":
        if self.source_type == "text":
            if not self.texts:
                raise ValueError("texts must be non-empty when source_type='text'.")
            if self.url is not None:
                raise ValueError("url must not be set when source_type='text'.")
        elif self.source_type == "url":
            if self.url is None:
                raise ValueError("url is required when source_type='url'.")
            if self.texts is not None:
                raise ValueError("texts must not be set when source_type='url'.")
        return self


class IngestResponse(BaseModel):
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
# Leaderboard
# ─────────────────────────────────────────────────────────────────────────────


class LeaderboardEntry(BaseModel):
    model_config = {"protected_namespaces": ()}

    model: str
    horizon: str = "1d"
    auc: float
    accuracy: float
    f1: float
    trained_at: str


class LeaderboardResponse(BaseModel):
    ticker: str
    horizon: str = "1d"
    entries: list[LeaderboardEntry]
    selected_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
