"""
POST /api/v1/portfolio/analyze
    Accepts a list of positions (ticker + sizing), runs the existing
    per-ticker prediction pipeline for each (unless ``include_predictions``
    is False), then layers portfolio-level analysis on top: correlation
    matrix estimation, mean-variance optimization, efficient frontier,
    risk attribution, sector exposure, and Value-at-Risk.

Performance notes
-----------------
Per-ticker predictions are the dominant latency contributor, especially
on cold-start (first run per ticker) where the full feature-engineering
+ model-training pipeline runs. Three mitigations are applied here:

1. Concurrent predictions — all tickers are dispatched simultaneously
   into a ``ThreadPoolExecutor`` (bounded at ``MAX_PRED_WORKERS`` threads)
   rather than running sequentially. For N uncached tickers, wall-clock
   latency drops from ~N × t_single to roughly t_slowest.

2. Per-ticker timeout — each future is given ``prediction_timeout_s``
   seconds (request-configurable, default 120 s). A ticker that exceeds
   its timeout is demoted to a "neutral" expected-return proxy (p=0.5)
   with a warning, rather than failing the whole request.

3. include_predictions=False fast path — if the caller sets this flag,
   the prediction stage is skipped entirely and the optimizer runs on
   neutral expected returns only. The UI exposes this as a "Quick
   analysis" toggle for users who just want the risk/correlation output
   without waiting for ML inference.

A future ``/portfolio/analyze/stream`` SSE variant (mirroring
``app/api/v1/endpoints/streaming.py``) would yield each ticker's
prediction as it completes rather than batching them all — not
implemented here, but the PortfolioAnalysisService / PredictionService
split is designed so that wiring is a thin addition.
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.api.portfolio_schemas import (
    EfficientFrontierPoint,
    PortfolioAnalyzeRequest,
    PortfolioAnalyzeResponse,
    RiskContributionSchema,
    TickerPredictionSummary,
    VaRSchema,
)
from app.core.exceptions import PortfolioAnalysisError
from app.core.logging_config import get_logger
from app.services.portfolio_analysis import (
    CovarianceEstimator,
    PortfolioAnalysisService,
    PortfolioPosition,
    ReturnSeriesBuilder,
)
from app.services.prediction_service import PredictionService

logger = get_logger("api.portfolio")

portfolio_router = APIRouter(prefix="/portfolio", tags=["Portfolio Analysis"])

# Maximum number of concurrent prediction threads.  Kept conservative so a
# burst of large portfolio requests doesn't exhaust the process thread pool.
MAX_PRED_WORKERS = 8

# ── Shared singleton — PredictionService caches trainers/engineers/selector;
#    PortfolioAnalysisService is lightweight and built fresh per request
#    below so per-request lookback/EWMA overrides never leak across
#    concurrent requests. ────────────────────────────────────────────────────

_prediction_service: Optional[PredictionService] = None


def _get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def _predict_one(
    ticker: str, horizon: str, svc: PredictionService
) -> tuple[str, Optional[object], Optional[str]]:
    """
    Run a single-ticker prediction and return ``(ticker, response, error)``.
    Designed to be called from a thread pool; exceptions are caught and
    returned as an error string so the pool can keep going for other tickers.
    """
    try:
        resp = svc.predict(ticker, horizon=horizon, run_fusion=False)
        return ticker, resp, None
    except Exception as exc:
        return ticker, None, str(exc)


@portfolio_router.post("/analyze", response_model=PortfolioAnalyzeResponse)
async def analyze_portfolio(
    request: PortfolioAnalyzeRequest,
) -> PortfolioAnalyzeResponse:
    """
    Run full portfolio-level analysis for a set of positions.

    Pipeline:
    1. (Optional) concurrent per-ticker ML predictions via the existing
       ``PredictionService`` — news/LLM signal fusion is skipped
       (``run_fusion=False``) since only the calibrated ``p_bullish`` is
       needed as the mean-variance expected-return proxy.
    2. Correlation / covariance matrix estimation (sample, EWMA, or
       Ledoit-Wolf shrinkage depending on portfolio size).
    3. Constrained mean-variance optimization + efficient frontier sweep.
    4. Risk attribution (current and optimal allocations).
    5. Portfolio volatility and parametric/historical VaR.
    6. Sector exposure aggregation.
    """
    tickers = [p.ticker for p in request.positions]

    prediction_summaries: list[TickerPredictionSummary] = []
    prediction_probs: dict[str, float] = {}

    if request.include_predictions:
        svc = _get_prediction_service()
        timeout_s = request.prediction_timeout_s

        # Dispatch all tickers concurrently into a thread pool.
        with cf.ThreadPoolExecutor(
            max_workers=min(len(tickers), MAX_PRED_WORKERS),
            thread_name_prefix="portfolio_pred",
        ) as pool:
            future_map: dict[cf.Future, str] = {
                pool.submit(_predict_one, t, request.horizon, svc): t for t in tickers
            }

            for future in cf.as_completed(future_map, timeout=timeout_s + 5):
                submitted_ticker = future_map[future]
                try:
                    ticker, resp, error = future.result(timeout=timeout_s)
                except cf.TimeoutError:
                    error = (
                        f"prediction timed out after {timeout_s}s "
                        f"(first-run model training is the likely cause; "
                        f"subsequent requests for this ticker will be faster)"
                    )
                    ticker, resp = submitted_ticker, None
                except Exception as exc:
                    ticker, resp, error = submitted_ticker, None, str(exc)

                if resp is not None:
                    prediction_probs[ticker] = resp.p_bullish
                    prediction_summaries.append(
                        TickerPredictionSummary(
                            ticker=ticker,
                            prediction_label=(
                                "BULLISH" if resp.prediction == 1 else "BEARISH"
                            ),
                            p_bullish=resp.p_bullish,
                            confidence_label=resp.confidence_label,
                            model_name=resp.model_name,
                        )
                    )
                else:
                    logger.warning(
                        "[portfolio] Prediction skipped for %s: %s", ticker, error
                    )
                    prediction_summaries.append(
                        TickerPredictionSummary(
                            ticker=ticker,
                            prediction_label="UNKNOWN",
                            p_bullish=0.5,
                            confidence_label="low",
                            model_name="none",
                            error=error,
                        )
                    )

        # Preserve original ticker order for the response.
        prediction_summaries.sort(key=lambda s: tickers.index(s.ticker))

    try:
        portfolio_svc = PortfolioAnalysisService(
            return_builder=ReturnSeriesBuilder(
                period_years=max(2, (request.lookback_days // 252) + 1)
            ),
            covariance_estimator=CovarianceEstimator(
                lookback_days=request.lookback_days,
                use_ewma=request.use_ewma_covariance,
            ),
        )

        positions = [
            PortfolioPosition(
                ticker=p.ticker,
                shares=p.shares or 0.0,
                market_value=p.market_value,
                current_weight=p.weight,
            )
            for p in request.positions
        ]

        result = portfolio_svc.analyze(
            positions=positions,
            prediction_probs=prediction_probs,
            portfolio_value=request.portfolio_value,
            max_position_weight=request.max_position_weight,
            max_sector_weight=request.max_sector_weight,
            long_only=request.long_only,
            turnover_limit=request.turnover_limit,
            var_confidence=request.var_confidence,
            var_horizon_days=request.var_horizon_days,
            return_scale_factor=request.return_scale_factor,
        )

        return PortfolioAnalyzeResponse(
            tickers=result.tickers,
            dropped_tickers=result.dropped_tickers,
            predictions=prediction_summaries,
            expected_returns=result.expected_returns,
            correlation_matrix=result.correlation_matrix,
            covariance_method=result.covariance_method,
            lookback_days=result.lookback_days,
            current_weights=result.current_weights,
            optimal_weights=result.optimal_weights,
            expected_return=result.expected_return,
            expected_volatility=result.expected_volatility,
            sharpe_ratio=result.sharpe_ratio,
            constraints_applied=result.constraints_applied,
            optimizer_converged=result.optimizer_converged,
            current_risk_attribution=[
                RiskContributionSchema(**r) for r in result.current_risk_attribution
            ],
            optimal_risk_attribution=[
                RiskContributionSchema(**r) for r in result.optimal_risk_attribution
            ],
            current_portfolio_volatility=result.current_portfolio_volatility,
            current_expected_return=result.current_expected_return,
            optimal_portfolio_volatility=result.optimal_portfolio_volatility,
            efficient_frontier=[
                EfficientFrontierPoint(**p) for p in result.efficient_frontier
            ],
            var=VaRSchema(**result.var),
            sector_exposure=result.sector_exposure,
            generated_at=result.generated_at,
            warnings=result.warnings,
        )

    except PortfolioAnalysisError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except Exception as e:
        logger.exception("Portfolio analysis failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
