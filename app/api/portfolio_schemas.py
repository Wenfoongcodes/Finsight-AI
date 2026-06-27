# ── Portfolio analysis schemas — append these to app/api/schemas.py ──────────
#
# Mirrors the convention used by app/api/versioning_schemas.py: kept in a
# separate file here for clarity / reviewability. In the actual repo these
# classes should be merged into app/api/schemas.py alongside the existing
# ones (or left as a standalone import — both work since routes.py-style
# imports already pull from multiple schema modules).

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


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
    include_predictions: bool = Field(
        default=True,
        description=(
            "Set to false to skip per-ticker ML predictions entirely and "
            "run the optimizer on neutral expected returns only. Makes the "
            "request ~10× faster — useful when you just want the "
            "correlation/risk/VaR output without waiting for ML inference."
        ),
    )
    prediction_timeout_s: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Per-ticker prediction timeout in seconds. Tickers that don't "
            "finish within this window are demoted to a neutral proxy "
            "(p_bullish=0.5) with a warning rather than failing the whole "
            "request. 120 s covers first-run model training; warm calls "
            "are typically under 5 s."
        ),
    )
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
