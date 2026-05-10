"""
FinSight AI — Phase 7: FastAPI Route Handlers
All HTTP endpoints for prediction, training, chat, and agent services.

Changes in this revision
------------------------
* ``ingest_documents`` handles both ``source_type="text"`` and
  ``source_type="url"`` branches, delegating to the appropriate
  ``RAGPipeline`` method.  URL fetch errors surface as 502 Bad Gateway
  (external dependency failure) rather than 500 Internal Server Error.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    AgentRequest, AgentResponse,
    BatchPredictionRequest,
    ChatRequest, ChatResponse as ChatResponseSchema,
    IngestRequest, IngestResponse,
    MarketDataRequest, MarketDataSummary,
    PredictionRequest, PredictionResult, SHAPFeature,
    TrainRequest, TrainResponse,
)
from app.core.exceptions import (
    DataIngestionError, DataValidationError,
    InsufficientDataError,
    ModelNotFoundError, ModelTrainingError, PredictionError,
    RAGError, LLMError, AgentError,
)
from app.core.logging_config import get_logger
from app.ml.data_ingestion import (
    MIN_ROWS_SUMMARY,
    get_data_summary,
    ingest_market_data,
)
from app.ml.feature_engineering import FeatureEngineer
from app.ml.training.trainer import ModelTrainer
from app.rag.llm_chat import FinancialChatSystem
from app.rag.rag_pipeline import RAGPipeline
from app.services.prediction_service import PredictionService
from configs.settings import settings

logger = get_logger("api.routes")

# ── Shared service singletons ─────────────────────────────────────────────────

_prediction_service: PredictionService | None = None
_rag_pipeline: RAGPipeline | None = None
_chat_system: FinancialChatSystem | None = None
_trainer: ModelTrainer | None = None


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


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Router
# ─────────────────────────────────────────────────────────────────────────────

prediction_router = APIRouter(prefix="/predict", tags=["Predictions"])


@prediction_router.post("/", response_model=PredictionResult)
async def predict(request: PredictionRequest) -> PredictionResult:
    """
    Generate a next-day price direction prediction for a stock.
    Trains a model on demand if no artifact exists for the ticker/model pair.
    """
    try:
        svc  = _get_prediction_service()
        resp = svc.predict(
            request.ticker,
            model_name=request.model_name,
            use_cache=request.use_cache,
        )
        return PredictionResult(
            ticker=resp.ticker,
            model_name=resp.model_name,
            prediction=resp.prediction,
            prediction_label="BULLISH" if resp.prediction == 1 else "BEARISH",
            probability=resp.probability,
            p_bullish=resp.p_bullish,
            p_bearish=resp.p_bearish,
            confidence_label=resp.confidence_label,
            latest_close=resp.latest_close,
            narrative=resp.narrative,
            top_features=[
                SHAPFeature(**f)
                for f in resp.shap_explanation.get("top_features", [])
            ],
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
    """Generate predictions for multiple tickers at once."""
    svc     = _get_prediction_service()
    results = svc.batch_predict(request.tickers, model_name=request.model_name)
    return {
        ticker: (
            {
                "prediction": "BULLISH" if r.prediction == 1 else "BEARISH",
                "probability": r.probability,
                "p_bullish":   r.p_bullish,
                "p_bearish":   r.p_bearish,
                "confidence":  r.confidence_label,
            }
            if not isinstance(r, str)
            else {"error": r}
        )
        for ticker, r in results.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training Router
# ─────────────────────────────────────────────────────────────────────────────

training_router = APIRouter(prefix="/train", tags=["Training"])


@training_router.post("/", response_model=TrainResponse)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Train a new model for a given ticker and persist the artifact.
    This may take several minutes for HPO runs.
    """
    try:
        raw_df     = ingest_market_data(request.ticker, period_years=request.period_years)
        engineer   = FeatureEngineer()
        feature_df = engineer.build_features(raw_df)
        X, y       = engineer.split_X_y(feature_df)

        trainer    = _get_trainer()
        _, result  = trainer.train(
            model_name=request.model_name,
            X=X,
            y=y,
            ticker=request.ticker,
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
        df      = ingest_market_data(
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
    """
    Ingest financial content into the RAG knowledge base.

    Supports two modes controlled by ``source_type``:

    * ``"text"`` — ingest one or more raw text strings directly.
    * ``"url"``  — fetch a web article by URL and ingest its content.

    URL ingestion errors from the remote server (timeouts, HTTP 4xx/5xx)
    are returned as **502 Bad Gateway** to distinguish them from internal
    server errors (500).  Duplicate URL ingestion returns 200 with
    ``duplicate: true`` in the response body.
    """
    rag = _get_rag()

    # ── Text ingestion ────────────────────────────────────────────────────────
    if request.source_type == "text":
        try:
            rag.ingest_texts(request.texts, source=request.source)
            return IngestResponse(
                ingested_count=len(request.texts),
                chunks_added=0,          # chunk count not tracked per-call in text mode
                source_type="text",
                message=(
                    f"Successfully ingested {len(request.texts)} "
                    f"document(s) from text input."
                ),
            )
        except RAGError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )

    # ── URL ingestion ─────────────────────────────────────────────────────────
    # source_type == "url" is guaranteed by the schema validator at this point.
    url_str = str(request.url)
    try:
        result = rag.ingest_url(url_str)
    except RAGError as e:
        # Distinguish external fetch failures (502) from internal errors (500).
        # A RAGError whose message contains "HTTP" or "timed out" originates
        # from the remote server, not from our pipeline.
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

    if result["duplicate"]:
        message = f"URL already in knowledge base (first ingested {result['fetched_at']})."
    else:
        message = (
            f"Successfully ingested article '{result['title']}' "
            f"({result['char_count']:,} chars, {result['chunks']} chunks)."
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
        resp     = chat_sys.chat(
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
    """
    Run the agentic AI to answer complex financial queries using tool orchestration.
    """
    try:
        from app.agents.financial_agent import FinancialAgent
        agent  = FinancialAgent(
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