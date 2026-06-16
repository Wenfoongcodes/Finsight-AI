"""
app/api/v1/endpoints/versioning.py
====================================
Model versioning and rollback API endpoints.

Endpoints
---------
GET  /api/v1/versions/{ticker}/{model_name}/{horizon}
     Return full version history for a ticker/model/horizon slot.

POST /api/v1/versions/promote
     Promote a specific version to active (rollback mechanism).

POST /api/v1/versions/rollback
     Roll back to the version that was active before the current one.

DELETE /api/v1/versions/prune
     Delete old version directories, keeping the last N.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.versioning_schemas import (
    PromoteVersionRequest,
    PromoteVersionResponse,
    PruneVersionsRequest,
    PruneVersionsResponse,
    RollbackRequest,
    VersionEntry,
    VersionHistoryResponse,
)
from app.core.exceptions import ModelNotFoundError
from app.core.logging_config import get_logger
from app.ml.training.trainer import ModelTrainer

logger = get_logger("api.versioning")

versioning_router = APIRouter(prefix="/versions", tags=["Model Versioning"])

# ── Shared trainer singleton ───────────────────────────────────────────────────

_trainer: ModelTrainer | None = None


def _get_trainer() -> ModelTrainer:
    global _trainer
    if _trainer is None:
        _trainer = ModelTrainer()
    return _trainer


# ─────────────────────────────────────────────────────────────────────────────
# GET /versions/{ticker}/{model_name}/{horizon}
# ─────────────────────────────────────────────────────────────────────────────


@versioning_router.get(
    "/{ticker}/{model_name}/{horizon}",
    response_model=VersionHistoryResponse,
    summary="Get version history",
    description=(
        "Return the full version history for a ticker/model/horizon slot, "
        "including metrics for each version and which is currently active."
    ),
)
async def get_version_history(
    ticker: str,
    model_name: str,
    horizon: str,
) -> VersionHistoryResponse:
    ticker = ticker.upper().strip()
    trainer = _get_trainer()

    entries = trainer.list_versions(ticker, model_name, horizon)
    active_id = trainer._store.get_active_version_id(ticker, model_name, horizon)

    version_entries = [
        VersionEntry(
            version_id=e.get("version_id", ""),
            trained_at=e.get("trained_at", ""),
            trigger_reason=e.get("trigger_reason", ""),
            mean_roc_auc=float(e.get("mean_roc_auc", 0.0)),
            mean_accuracy=float(e.get("mean_accuracy", 0.0)),
            mean_f1=float(e.get("mean_f1", 0.0)),
            n_features=int(e.get("n_features", 0)),
            feature_hash=e.get("feature_hash", ""),
            best_params=e.get("best_params", {}),
            is_active=e.get("is_active", False),
            exists_on_disk=e.get("exists_on_disk", True),
            training_duration_s=float(e.get("training_duration_s", 0.0)),
            n_folds=int(e.get("n_folds", 0)),
        )
        for e in entries
    ]

    return VersionHistoryResponse(
        ticker=ticker,
        model_name=model_name,
        horizon=horizon,
        active_version_id=active_id,
        versions=version_entries,
        total_versions=len(version_entries),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /versions/promote
# ─────────────────────────────────────────────────────────────────────────────


@versioning_router.post(
    "/promote",
    response_model=PromoteVersionResponse,
    summary="Promote a version to active",
    description=(
        "Explicitly promote a specific version to active. "
        "This is the primary mechanism for both upgrading to a new model "
        "and rolling back to a previous one — pass any version_id from the "
        "version history."
    ),
)
async def promote_version(request: PromoteVersionRequest) -> PromoteVersionResponse:
    ticker = request.ticker.upper().strip()
    trainer = _get_trainer()

    previous_active = trainer._store.get_active_version_id(
        ticker, request.model_name, request.horizon
    )

    try:
        trainer.promote_version(
            ticker, request.model_name, request.horizon, request.version_id
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    return PromoteVersionResponse(
        ticker=ticker,
        model_name=request.model_name,
        horizon=request.horizon,
        promoted_version_id=request.version_id,
        previous_version_id=previous_active,
        message=(
            f"Version {request.version_id} is now active for "
            f"{ticker}/{request.model_name}/{request.horizon}."
            + (
                f" Previous active version was {previous_active}."
                if previous_active and previous_active != request.version_id
                else ""
            )
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /versions/rollback
# ─────────────────────────────────────────────────────────────────────────────


@versioning_router.post(
    "/rollback",
    response_model=PromoteVersionResponse,
    summary="Roll back to the previous version",
    description=(
        "Roll back to the version that was active before the current one. "
        "The rolled-back version becomes the new active version immediately."
    ),
)
async def rollback_version(request: RollbackRequest) -> PromoteVersionResponse:
    ticker = request.ticker.upper().strip()
    trainer = _get_trainer()

    previous_active = trainer._store.get_active_version_id(
        ticker, request.model_name, request.horizon
    )

    try:
        restored_version_id = trainer.rollback(
            ticker, request.model_name, request.horizon
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    return PromoteVersionResponse(
        ticker=ticker,
        model_name=request.model_name,
        horizon=request.horizon,
        promoted_version_id=restored_version_id,
        previous_version_id=previous_active,
        message=(
            f"Rolled back to version {restored_version_id} for "
            f"{ticker}/{request.model_name}/{request.horizon}. "
            f"Previous active version {previous_active} is no longer active."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /versions/prune
# ─────────────────────────────────────────────────────────────────────────────


@versioning_router.delete(
    "/prune",
    response_model=PruneVersionsResponse,
    summary="Prune old versions",
    description=(
        "Delete old version directories, keeping the ``keep_last`` most recent "
        "plus the currently active version (which is always preserved). "
        "Useful for managing storage on HuggingFace Spaces."
    ),
)
async def prune_versions(request: PruneVersionsRequest) -> PruneVersionsResponse:
    ticker = request.ticker.upper().strip()
    trainer = _get_trainer()

    try:
        deleted = trainer.prune_versions(
            ticker, request.model_name, request.horizon, keep_last=request.keep_last
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    return PruneVersionsResponse(
        ticker=ticker,
        model_name=request.model_name,
        horizon=request.horizon,
        deleted_version_ids=deleted,
        deleted_count=len(deleted),
        message=(
            f"Pruned {len(deleted)} version(s) for "
            f"{ticker}/{request.model_name}/{request.horizon}. "
            f"Kept last {request.keep_last} versions plus the active version."
            if deleted
            else f"Nothing to prune — all versions within keep_last={request.keep_last}."
        ),
    )
