"""
FinSight AI — Phase 7: FastAPI Application Entry Point
Configures middleware, exception handlers, and mounts all routers.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse
from app.api.v1.endpoints.routes import (
    agent_router,
    market_router,
    prediction_router,
    rag_router,
    training_router,
)
from app.core.exceptions import (
    DataIngestionError,
    FinSightBaseError,
    LLMError,
    ModelNotFoundError,
)
from app.core.logging_config import get_logger, setup_logging
from configs.settings import settings

logger = get_logger("app")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO", log_file="finsight.log")
    logger.info(
        "FinSight AI starting up | env=%s | debug=%s",
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    yield
    logger.info("FinSight AI shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Explainable Financial Decision Support System — "
            "ML predictions, SHAP explanations, RAG-grounded chat, and agentic AI."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Timing Middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

    # ── Exception Handlers ────────────────────────────────────────────────────
    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                error=exc.message, detail=exc.detail, status_code=404
            ).model_dump(),
        )

    @app.exception_handler(DataIngestionError)
    async def data_ingestion_handler(request: Request, exc: DataIngestionError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=exc.message, detail=exc.detail, status_code=422
            ).model_dump(),
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error=exc.message, detail=exc.detail, status_code=503
            ).model_dump(),
        )

    @app.exception_handler(FinSightBaseError)
    async def generic_finsight_handler(request: Request, exc: FinSightBaseError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=exc.message, detail=exc.detail, status_code=500
            ).model_dump(),
        )

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "ok",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
        }

    # ── Routers ───────────────────────────────────────────────────────────────
    api_prefix = "/api/v1"
    app.include_router(prediction_router, prefix=api_prefix)
    app.include_router(training_router, prefix=api_prefix)
    app.include_router(market_router, prefix=api_prefix)
    app.include_router(rag_router, prefix=api_prefix)
    app.include_router(agent_router, prefix=api_prefix)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
