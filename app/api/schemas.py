"""
FinSight AI — Phase 7: API Schemas
Pydantic v2 request and response models for all FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


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


class BatchPredictionRequest(BaseModel):
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
    texts: list[str] = Field(..., min_length=1)
    source: str = Field(default="api_upload")


class IngestResponse(BaseModel):
    ingested_count: int
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