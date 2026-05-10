"""
FinSight AI — Phase 7: API Schemas
Pydantic v2 request and response models for all FastAPI endpoints.

Changes in this revision
------------------------
* ``PredictionRequest`` no longer accepts ``model_name``.  The system
  selects the best available model automatically via ``ModelSelector``.
  Removing this field eliminates a source of user confusion and enforces
  the invariant that model selection is system-owned, not user-driven.

* ``PredictionResult`` gains ``model_name`` (read-only — which model was
  selected), ``fused_direction``, ``fused_confidence``,
  ``fused_probability``, ``fusion_narrative``, ``fusion_applied``,
  ``news_sentiment``, and ``news_items`` so the API surface exposes the
  full fusion output without requiring callers to parse nested objects.

* ``IngestRequest`` retains the ``source_type`` discriminator introduced
  in the previous revision.

* ``BatchPredictionRequest`` no longer accepts ``model_name`` for the
  same reason as ``PredictionRequest``.

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
    """
    Request body for a single-ticker prediction.

    ``model_name`` has been deliberately removed.  The system selects the
    best available model automatically, ensuring:

    * Novice users are not burdened with a choice they cannot meaningfully make.
    * The selection is deterministic, auditable, and leaderboard-driven.
    * The API contract is simpler and less likely to be misused.
    """

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    use_cache: bool = Field(default=True)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class NewsItemSchema(BaseModel):
    """A single news item used in signal fusion."""
    title:   str
    snippet: str
    url:     str


class SHAPFeature(BaseModel):
    feature:       str
    shap_value:    float
    direction:     str
    feature_value: float


class PredictionResult(BaseModel):
    """
    Full prediction response including ML signal and fused signal.

    ML signal fields
    ----------------
    model_name      : Auto-selected model (e.g. "xgboost").
    prediction      : 0 = bearish, 1 = bullish (raw ML output).
    prediction_label: "BULLISH" | "BEARISH".
    probability     : Directional probability of the predicted class.
    p_bullish       : Calibrated P(bullish) in [0, 1].
    p_bearish       : Calibrated P(bearish) = 1 − p_bullish.
    confidence_label: "high" | "moderate" | "low" (ML-only).
    narrative       : SHAP-based plain-English reasoning (ML-only).
    top_features    : Top SHAP contributors.
    latest_close    : Most recent closing price.

    Fused signal fields (populated when signal fusion succeeds)
    -----------------------------------------------------------
    fused_direction  : "BULLISH" | "BEARISH" | "NEUTRAL".
    fused_confidence : "HIGH" | "MODERATE" | "LOW".
    fused_probability: P(bullish) after fusion (0..1).
    fusion_narrative : LLM synthesis reasoning (2–4 sentences).
    fusion_applied   : False when web search or LLM failed.
    news_sentiment   : Aggregate news sentiment: positive/negative/neutral.
    news_items       : News articles used in fusion (may be empty).
    """

    model_config = {"protected_namespaces": ()}

    # ── ML signal ─────────────────────────────────────────────────────────────
    ticker:           str
    model_name:       str           # which model was auto-selected
    prediction:       int
    prediction_label: str
    probability:      float
    p_bullish:        float
    p_bearish:        float
    confidence_label: str
    latest_close:     float
    narrative:        str
    top_features:     list[SHAPFeature]

    # ── Fused signal ──────────────────────────────────────────────────────────
    fused_direction:   str = "UNKNOWN"
    fused_confidence:  str = "LOW"
    fused_probability: float = 0.5
    fusion_narrative:  str = ""
    fusion_applied:    bool = False
    news_sentiment:    str = "neutral"
    news_items:        list[NewsItemSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_probabilities(self) -> "PredictionResult":
        """
        Assert that p_bullish + p_bearish rounds to 1.0.
        Tolerance of 1e-4 accommodates round(..., 4) floating-point rounding.
        """
        total = self.p_bullish + self.p_bearish
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"p_bullish ({self.p_bullish}) + p_bearish ({self.p_bearish}) "
                f"must sum to 1.0, got {total:.6f}"
            )
        return self


class BatchPredictionRequest(BaseModel):
    """
    Request body for multi-ticker batch predictions.

    ``model_name`` has been removed — same rationale as ``PredictionRequest``.
    The system selects the best available model per ticker independently.
    """

    tickers: list[str] = Field(..., min_length=1, max_length=20)

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, v: list[str]) -> list[str]:
        return [t.upper().strip() for t in v]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    """
    Training requests retain an explicit ``model_name`` because training is
    an operator-level action — not a user-facing one.  The operator must
    choose which model to train and register in the leaderboard.
    """
    model_config = {"protected_namespaces": ()}

    ticker:      str = Field(..., min_length=1, max_length=10)
    model_name:  str = Field(default="xgboost")
    run_hpo:     bool = Field(default=False)
    hpo_trials:  int = Field(default=20, ge=5, le=100)
    period_years: int = Field(default=5, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class TrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker:        str
    model_name:    str
    mean_accuracy: float
    mean_f1:       float
    mean_roc_auc:  float
    mean_mae:      float
    mean_rmse:     float
    n_folds:       int
    n_features:    int
    trained_at:    str
    best_params:   dict


# ─────────────────────────────────────────────────────────────────────────────
# RAG / Chat
# ─────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Unified ingestion request that accepts either raw text or a URL.

    Exactly one of ``texts`` or ``url`` must be supplied:

    * **Text ingestion** — set ``source_type="text"`` and provide ``texts``.
    * **URL ingestion**  — set ``source_type="url"`` and provide ``url``.
    """

    source_type: Literal["text", "url"] = Field(
        default="text",
        description="Ingestion mode: 'text' for raw strings, 'url' for a web article.",
    )
    texts: Optional[list[str]] = Field(
        default=None,
        description="List of text strings to ingest (required when source_type='text').",
    )
    source: str = Field(
        default="api_upload",
        description="Arbitrary source label attached to text chunks as metadata.",
    )
    url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Article URL to fetch and ingest (required when source_type='url').",
    )

    @model_validator(mode="after")
    def _check_exclusive_source(self) -> "IngestRequest":
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
                raise ValueError("url is required when source_type='url'.")
            if self.texts is not None:
                raise ValueError(
                    "texts must not be set when source_type='url'. "
                    "Use source_type='text' to ingest raw text."
                )
        return self


class IngestResponse(BaseModel):
    ingested_count: int
    chunks_added:   int = 0
    source_type:    str = "text"
    title:          str = ""
    char_count:     int = 0
    duplicate:      bool = False
    message:        str


class ChatRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=2000)
    use_rag:    bool = Field(default=True)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response:    str
    used_rag:    bool
    model:       str
    tokens_used: int
    session_id:  Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    query:        str
    response:     str
    tools_used:   list[str]
    tool_results: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Market Data
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataRequest(BaseModel):
    ticker:       str
    period_years: int = Field(default=1, ge=1, le=20)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class MarketDataSummary(BaseModel):
    ticker:     str
    start_date: str
    end_date:   str
    rows:       int
    columns:    list[str]
    close_min:  float
    close_max:  float
    close_mean: float
    null_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Model Leaderboard (new — introspection endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class LeaderboardEntry(BaseModel):
    """Single entry in the model leaderboard for a ticker."""
    model_config = {"protected_namespaces": ()}

    model:      str
    auc:        float
    accuracy:   float
    f1:         float
    trained_at: str


class LeaderboardResponse(BaseModel):
    ticker:  str
    entries: list[LeaderboardEntry]
    selected_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:       str
    detail:      Optional[str] = None
    status_code: int