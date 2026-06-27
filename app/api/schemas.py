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

    ``model_name`` is absent — the system auto-selects the best model.
    ``horizon`` selects the prediction window: '1d' (default), '7d', '1m', '6m'.
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


class PredictionResult(BaseModel):
    """
    Full prediction API response.

    ML signal
    ---------
    model_name, horizon, prediction, prediction_label, probability,
    p_bullish, p_bearish, confidence_label, confidence_degraded,
    selection_reason, latest_close, narrative, top_features, auto_trained.

    Fused signal
    ------------
    fused_direction, fused_confidence, fused_probability, fusion_narrative,
    fusion_applied, news_sentiment, news_items.
    """

    model_config = {"protected_namespaces": ()}

    # ── ML signal ─────────────────────────────────────────────────────────────
    ticker: str
    model_name: str
    horizon: str = "1d"
    prediction: int
    prediction_label: str
    probability: float
    p_bullish: float
    p_bearish: float
    confidence_label: str
    confidence_degraded: bool = False
    selection_reason: str = "leaderboard"
    latest_close: float
    narrative: str
    top_features: list[SHAPFeature]
    auto_trained: bool = False
    feature_selection_meta: Optional[dict] = None

    # ── Fused signal ──────────────────────────────────────────────────────────
    fused_direction: str = "UNKNOWN"
    fused_confidence: str = "LOW"
    fused_probability: float = 0.5
    fusion_narrative: str = ""
    fusion_applied: bool = False
    news_sentiment: str = "neutral"
    news_items: list[NewsItemSchema] = Field(default_factory=list)

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


class PositionInput(BaseModel):
    """
    A single portfolio holding.
    Provide at most one of ``shares``, ``market_value``, or ``weight`` to
    size the position (priority: weight > market_value > shares > none).
    If none of the positions in a request specify any sizing field, the
    portfolio falls back to an equal-weight allocation.
    """

    ticker: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    shares: Optional[float] = Field(default=None, ge=0)
    market_value: Optional[float] = Field(default=None, ge=0)
    weight: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _uppercase_ticker(self) -> "PositionInput":
        self.ticker = self.ticker.upper().strip()
        return self


class PortfolioAnalyzeRequest(BaseModel):
    """
    Request body for ``POST /api/v1/portfolio/analyze``.
    ``max_position_weight`` and ``max_sector_weight`` must each be
    feasible given the number of positions / distinct sectors respectively
    — e.g. 5 positions capped at 10% each can sum to at most 50%, and 2
    sectors capped at 30% each can sum to at most 60%; either is
    infeasible (the service raises a 422 in that case rather than silently
    returning a portfolio that doesn't actually sum to 100%). The defaults
    here (50% per position, sector cap disabled) are chosen specifically
    to never trigger this out of the box; tighten them explicitly once you
    know your portfolio's size and sector spread.
    """

    positions: list[PositionInput] = Field(..., min_length=2, max_length=30)
    portfolio_value: Optional[float] = Field(default=None, ge=0)
    horizon: str = Field(default="1d", examples=["1d", "7d", "1m", "6m"])
    include_predictions: bool = Field(default=True)
    max_position_weight: float = Field(
        default=0.50,
        gt=0,
        le=1.0,
        description=(
            "Cap on weight in any single position. Default 0.50 stays "
            "feasible down to the minimum supported portfolio size (2 "
            "positions); tighten it (e.g. 0.10-0.15) for larger, more "
            "diversified portfolios."
        ),
    )
    max_sector_weight: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description=(
            "Cap on total weight per sector. Default 1.0 = disabled, since "
            "a legitimate request can span just one sector (e.g. an "
            "all-tech watchlist) and a binding default would reject it. "
            "Set a tighter value (e.g. 0.30) to enforce diversification "
            "for portfolios that genuinely span multiple sectors."
        ),
    )
    long_only: bool = Field(default=True)
    turnover_limit: Optional[float] = Field(default=None, ge=0, le=2.0)
    var_confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    var_horizon_days: int = Field(default=1, ge=1, le=30)
    use_ewma_covariance: bool = Field(default=False)
    lookback_days: int = Field(default=252, ge=60, le=1260)
    return_scale_factor: float = Field(default=0.50, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _check_unique_tickers(self) -> "PortfolioAnalyzeRequest":
        tickers = [p.ticker for p in self.positions]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers are not allowed in positions.")
        return self


class TickerPredictionSummary(BaseModel):
    model_config = {"protected_namespaces": ()}

    ticker: str
    prediction_label: str
    p_bullish: float
    confidence_label: str
    model_name: str
    error: Optional[str] = None


class RiskContributionSchema(BaseModel):
    ticker: str
    weight: float
    marginal_contribution: float
    contribution_to_variance: float
    pct_of_total_risk: float


class EfficientFrontierPoint(BaseModel):
    expected_return: float
    volatility: float


class VaRSchema(BaseModel):
    confidence: float
    horizon_days: int
    parametric_var_pct: float
    parametric_var_value: Optional[float] = None
    historical_var_pct: float
    historical_var_value: Optional[float] = None
    method_note: str


class PortfolioAnalyzeResponse(BaseModel):
    tickers: list[str]
    dropped_tickers: list[str]
    predictions: list[TickerPredictionSummary] = Field(default_factory=list)
    expected_returns: dict[str, float]
    correlation_matrix: dict[str, dict[str, float]]
    covariance_method: str
    lookback_days: int
    current_weights: dict[str, float]
    optimal_weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    constraints_applied: list[str]
    optimizer_converged: bool
    current_risk_attribution: list[RiskContributionSchema]
    optimal_risk_attribution: list[RiskContributionSchema]
    current_portfolio_volatility: float
    current_expected_return: float
    optimal_portfolio_volatility: float
    efficient_frontier: list[EfficientFrontierPoint]
    var: VaRSchema
    sector_exposure: dict[str, float]
    generated_at: str
    warnings: list[str] = Field(default_factory=list)
