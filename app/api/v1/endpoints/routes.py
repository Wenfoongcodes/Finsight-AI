"""
FinSight AI — FastAPI Route Handlers (v4)

Changes vs v3
-------------
* ``IntelligenceBriefSchema`` import removed — class no longer exists in schemas.
* The block that built ``intelligence_brief_schema`` from ``resp.intelligence_brief``
  is removed entirely.
* ``intelligence_brief=intelligence_brief_schema`` kwarg removed from the
  ``PredictionResult(...)`` constructor call.

All other route logic unchanged.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.api.schemas import (
    AgentRequest,
    AgentResponse,
    BatchPredictionRequest,
    ChatRequest,
    ChatResponse as ChatResponseSchema,
    IngestRequest,
    IngestResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    MarketDataRequest,
    MarketDataSummary,
    NewsItemSchema,
    PredictionRequest,
    PredictionResult,
    SHAPFeature,
    TrainRequest,
    TrainResponse,
)
from app.core.exceptions import (
    DataIngestionError,
    DataValidationError,
    InsufficientDataError,
    ModelNotFoundError,
    ModelTrainingError,
    PredictionError,
    RAGError,
    LLMError,
    AgentError,
)
from app.core.logging_config import get_logger
from app.ml.data_ingestion import MIN_ROWS_SUMMARY, get_data_summary, ingest_market_data
from app.ml.feature_engineering import FeatureEngineer
from app.ml.training.trainer import ModelTrainer
from app.rag.llm_chat import FinancialChatSystem
from app.rag.rag_pipeline import RAGPipeline
from app.services.model_selector import ModelSelector
from app.services.prediction_service import PredictionService

logger = get_logger("api.routes")

# ── Shared singletons ─────────────────────────────────────────────────────────

_prediction_service: PredictionService | None = None
_rag_pipeline: RAGPipeline | None = None
_chat_system: FinancialChatSystem | None = None
_trainer: ModelTrainer | None = None
_selector: ModelSelector | None = None


def _get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def _get_rag() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def _get_chat() -> FinancialChatSystem:
    global _chat_system
    if _chat_system is None:
        _chat_system = FinancialChatSystem(rag_pipeline=_get_rag())
    return _chat_system


def _get_trainer() -> ModelTrainer:
    global _trainer
    if _trainer is None:
        _trainer = ModelTrainer()
    return _trainer


def _get_selector() -> ModelSelector:
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Router
# ─────────────────────────────────────────────────────────────────────────────

prediction_router = APIRouter(prefix="/predict", tags=["Predictions"])


@prediction_router.post("/", response_model=PredictionResult)
async def predict(request: PredictionRequest) -> PredictionResult:
    """
    Generate a price-direction prediction for the requested ticker and horizon.

    The system selects the best trained model automatically.  When no model
    exists it trains all candidates and picks the highest AUC before returning.
    """
    try:
        svc = _get_prediction_service()
        resp = svc.predict(
            ticker=request.ticker,
            horizon=request.horizon,
            use_cache=request.use_cache,
        )

        # ── Fused signal fields ───────────────────────────────────────────────
        fused = resp.fused_signal
        if fused:
            fused_direction = fused.final_direction
            fused_confidence = fused.final_confidence
            fused_probability = fused.fusion_probability
            fusion_narrative = fused.synthesis_narrative
            fusion_applied = fused.fusion_applied
            news_sentiment = fused.news_sentiment
            news_items = [
                NewsItemSchema(title=n.title, snippet=n.snippet, url=n.url)
                for n in fused.news_items
            ]
        else:
            fused_direction = "BULLISH" if resp.prediction == 1 else "BEARISH"
            fused_confidence = resp.confidence_label.upper()
            fused_probability = resp.p_bullish
            fusion_narrative = resp.narrative
            fusion_applied = False
            news_sentiment = "neutral"
            news_items = []

        return PredictionResult(
            # ML signal
            ticker=resp.ticker,
            model_name=resp.model_name,
            horizon=resp.horizon,
            prediction=resp.prediction,
            prediction_label="BULLISH" if resp.prediction == 1 else "BEARISH",
            probability=resp.probability,
            p_bullish=resp.p_bullish,
            p_bearish=resp.p_bearish,
            confidence_label=resp.confidence_label,
            confidence_degraded=resp.confidence_degraded,
            selection_reason=resp.selection_reason,
            latest_close=resp.latest_close,
            narrative=resp.narrative,
            top_features=[
                SHAPFeature(**f) for f in resp.shap_explanation.get("top_features", [])
            ],
            auto_trained=resp.auto_trained,
            # Fused signal
            fused_direction=fused_direction,
            fused_confidence=fused_confidence,
            fused_probability=fused_probability,
            fusion_narrative=fusion_narrative,
            fusion_applied=fusion_applied,
            news_sentiment=news_sentiment,
            news_items=news_items,
        )

    except ModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except (DataIngestionError, DataValidationError, InsufficientDataError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@prediction_router.post("/batch", response_model=dict)
async def batch_predict(request: BatchPredictionRequest) -> dict:
    """Batch predictions for multiple tickers at the same horizon."""
    svc = _get_prediction_service()
    results = svc.batch_predict(tickers=request.tickers, horizon=request.horizon)
    return {
        ticker: (
            {
                "prediction": "BULLISH" if r.prediction == 1 else "BEARISH",
                "probability": r.probability,
                "p_bullish": r.p_bullish,
                "p_bearish": r.p_bearish,
                "confidence": r.confidence_label,
                "confidence_degraded": r.confidence_degraded,
                "selection_reason": r.selection_reason,
                "model_selected": r.model_name,
                "horizon": r.horizon,
                "auto_trained": r.auto_trained,
                "fused_direction": r.fused_signal.final_direction
                if r.fused_signal
                else "N/A",
                "fusion_applied": r.fused_signal.fusion_applied
                if r.fused_signal
                else False,
            }
            if not isinstance(r, str)
            else {"error": r}
        )
        for ticker, r in results.items()
    }


@prediction_router.get("/leaderboard/{ticker}", response_model=LeaderboardResponse)
async def model_leaderboard(
    ticker: str,
    horizon: str = Query(default="1d", description="Prediction horizon"),
) -> LeaderboardResponse:
    """Return the model performance leaderboard for a ticker and horizon."""
    ticker = ticker.upper().strip()
    selector = _get_selector()
    board = selector.leaderboard(ticker, horizon=horizon)
    sel = selector.select(ticker, horizon=horizon)

    return LeaderboardResponse(
        ticker=ticker,
        horizon=horizon,
        entries=[LeaderboardEntry(**e) for e in board],
        selected_model=sel.model_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training Router
# ─────────────────────────────────────────────────────────────────────────────

training_router = APIRouter(prefix="/train", tags=["Training"])


@training_router.post("/", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """Train a model for a given ticker / model / horizon and persist the artifact."""
    try:
        raw_df = ingest_market_data(request.ticker, period_years=request.period_years)
        engineer = FeatureEngineer()
        feature_df = engineer.build_features(raw_df)
        X, y = engineer.split_X_y(feature_df, horizon=request.horizon)

        trainer = _get_trainer()
        _, result = trainer.train(
            model_name=request.model_name,
            X=X,
            y=y,
            ticker=request.ticker,
            horizon=request.horizon,
            run_hpo=request.run_hpo,
            hpo_trials=request.hpo_trials,
        )
        return TrainResponse(**result.to_dict())

    except (DataIngestionError, DataValidationError, InsufficientDataError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except ModelTrainingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Market Data Router
# ─────────────────────────────────────────────────────────────────────────────

market_router = APIRouter(prefix="/market", tags=["Market Data"])


@market_router.post("/summary", response_model=MarketDataSummary)
async def market_summary(request: MarketDataRequest) -> MarketDataSummary:
    """Retrieve OHLCV summary statistics for a ticker."""
    try:
        df = ingest_market_data(
            request.ticker,
            period_years=request.period_years,
            min_rows=MIN_ROWS_SUMMARY,
        )
        summary = get_data_summary(df, request.ticker)
        return MarketDataSummary(**summary)
    except (DataIngestionError, DataValidationError, InsufficientDataError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )


# ─────────────────────────────────────────────────────────────────────────────
# RAG / Chat Router
# ─────────────────────────────────────────────────────────────────────────────

rag_router = APIRouter(prefix="/rag", tags=["RAG & Chat"])


@rag_router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """Ingest financial content (raw text or a URL) into the RAG knowledge base."""
    rag = _get_rag()

    if request.source_type == "text":
        try:
            rag.ingest_texts(request.texts, source=request.source)
            return IngestResponse(
                ingested_count=len(request.texts),
                chunks_added=0,
                source_type="text",
                message=f"Successfully ingested {len(request.texts)} document(s).",
            )
        except RAGError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    url_str = str(request.url)
    try:
        result = rag.ingest_url(url_str)
    except RAGError as e:
        msg = str(e)
        is_external = any(kw in msg for kw in ("HTTP ", "timed out", "fetch"))
        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
                if is_external
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=msg,
        )

    message = (
        f"URL already in knowledge base (first ingested {result['fetched_at']})."
        if result["duplicate"]
        else (
            f"Ingested '{result['title']}' "
            f"({result['char_count']:,} chars, {result['chunks']} chunks)."
        )
    )

    return IngestResponse(
        ingested_count=1,
        chunks_added=result["chunks"],
        source_type="url",
        title=result["title"],
        char_count=result["char_count"],
        duplicate=result["duplicate"],
        message=message,
    )


@rag_router.post("/chat", response_model=ChatResponseSchema)
async def chat(request: ChatRequest) -> ChatResponseSchema:
    """Send a message to the financial AI assistant."""
    try:
        chat_sys = _get_chat()
        resp = chat_sys.chat(
            user_query=request.query,
            use_rag=request.use_rag,
            session_id=request.session_id,
        )
        return ChatResponseSchema(
            response=resp.content,
            used_rag=resp.used_rag,
            model=resp.model,
            tokens_used=resp.tokens_used,
            session_id=request.session_id,
        )
    except LLMError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Router
# ─────────────────────────────────────────────────────────────────────────────

agent_router = APIRouter(prefix="/agent", tags=["AI Agent"])


@agent_router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest) -> AgentResponse:
    """Run the agentic AI to answer complex financial queries via tool orchestration."""
    try:
        from app.agents.financial_agent import FinancialAgent

        agent = FinancialAgent(
            chat_system=_get_chat(),
            rag_pipeline=_get_rag(),
        )
        result = agent.run(request.query)
        return AgentResponse(**result)
    except AgentError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    except LLMError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
