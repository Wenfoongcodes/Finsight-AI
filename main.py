from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorResponse
from app.api.v1.endpoints.portfolio import portfolio_router
from app.api.v1.endpoints.routes import (
    agent_router,
    market_router,
    prediction_router,
    rag_router,
    training_router,
)
from app.api.v1.endpoints.streaming import streaming_router
from app.api.v1.endpoints.versioning import versioning_router
from app.core.cache_cleanup import cleanup_on_startup
from app.core.exceptions import (
    DataIngestionError,
    FinSightBaseError,
    LLMError,
    ModelNotFoundError,
)
from app.core.logging_config import get_logger, setup_logging
from app.core.security import SecurityMiddleware
from configs.settings import settings

logger = get_logger("app")


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_startup_config() -> None:
    """
    Emit structured warnings for missing or insecure configuration at boot.

    Does NOT raise — the application can start in a degraded state (e.g.
    LLM features unavailable) and return informative errors at request time.
    """
    warnings: list[str] = []

    if not settings.OPENAI_API_KEY:
        warnings.append(
            "OPENAI_API_KEY is not set — RAG chat, agent, and signal fusion "
            "features will be unavailable."
        )

    if settings.API_KEY_ENABLED and not settings.API_SECRET_KEY:
        warnings.append(
            "API_KEY_ENABLED=true but API_SECRET_KEY is not set — "
            "all authenticated requests will be rejected with 401."
        )

    if settings.ENVIRONMENT == "production":
        if not settings.API_KEY_ENABLED:
            warnings.append(
                "Running in PRODUCTION without API key auth. "
                "Set API_KEY_ENABLED=true and API_SECRET_KEY to secure the API."
            )
        if settings.DEBUG:
            warnings.append(
                "DEBUG=true in PRODUCTION — disable before deploying publicly."
            )

    for w in warnings:
        logger.warning("[startup] %s", w)

    if not warnings:
        logger.info("[startup] Configuration validation passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO", log_file="finsight.log")

    logger.info(
        "FinSight AI starting | env=%s | debug=%s | auth=%s | rate_limit=%s | "
        "cors_origins=%s",
        settings.ENVIRONMENT,
        settings.DEBUG,
        settings.API_KEY_ENABLED,
        settings.RATE_LIMIT_ENABLED,
        settings.ALLOWED_ORIGINS,
    )

    _validate_startup_config()

    # ── Cache cleanup: raw parquet on every boot ──────────────────────────────
    cleanup_on_startup(dry_run=False)

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
        # In production, disable interactive docs if you want to lock down the API
        # docs_url=None if settings.ENVIRONMENT == "production" else "/docs",
    )

    # ── Security middleware (first — before CORS so auth runs on all requests) ──
    app.add_middleware(SecurityMiddleware)

    # ── CORS ──────────────────────────────────────────────────────────────────
    # ``settings.ALLOWED_ORIGINS`` is a Python list parsed from the
    # ``ALLOWED_ORIGINS`` env var (comma-separated, or "*").
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # ── Request timing + structured access log ────────────────────────────────
    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Attach timing header
        elapsed_str = f"{elapsed:.4f}s"
        response.headers["X-Process-Time"] = elapsed_str

        # Structured access log — useful for log aggregation (Loki, CloudWatch)
        request_id = getattr(request.state, "request_id", "-")
        logger.info(
            "%s %s %s | dur=%s | rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_str,
            request_id,
        )
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
    async def health_check(request: Request):
        return {
            "status": "ok",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "request_id": getattr(request.state, "request_id", None),
            "features": {
                "llm": bool(settings.OPENAI_API_KEY),
                "auth": settings.API_KEY_ENABLED,
                "rate_limiting": settings.RATE_LIMIT_ENABLED,
            },
        }

    # ── Routers ───────────────────────────────────────────────────────────────
    api_prefix = "/api/v1"
    app.include_router(prediction_router, prefix=api_prefix)
    app.include_router(training_router, prefix=api_prefix)
    app.include_router(market_router, prefix=api_prefix)
    app.include_router(rag_router, prefix=api_prefix)
    app.include_router(agent_router, prefix=api_prefix)
    app.include_router(streaming_router, prefix=api_prefix)
    app.include_router(versioning_router, prefix=api_prefix)
    app.include_router(portfolio_router, prefix=api_prefix)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        # These are important for production: set workers > 1 via env/CLI
        # workers=1 here because reload=True is incompatible with workers>1
    )
