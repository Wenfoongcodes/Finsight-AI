# ── Versioning schemas — append these to app/api/schemas.py ──────────────────
#
# Add the following classes to app/api/schemas.py.
# They are kept in a separate file here for clarity; in the actual repo
# they should be merged into app/api/schemas.py alongside the existing classes.

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class VersionEntry(BaseModel):
    """Single entry in the version history."""

    model_config = {"protected_namespaces": ()}

    version_id: str
    trained_at: str
    trigger_reason: str = ""
    mean_roc_auc: float
    mean_accuracy: float
    mean_f1: float
    n_features: int
    feature_hash: str = ""
    best_params: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    exists_on_disk: bool = True
    training_duration_s: float = 0.0
    n_folds: int = 0


class VersionHistoryResponse(BaseModel):
    """Full version history for a ticker/model/horizon slot."""

    model_config = {"protected_namespaces": ()}

    ticker: str
    model_name: str
    horizon: str
    active_version_id: Optional[str]
    versions: list[VersionEntry]
    total_versions: int


class PromoteVersionRequest(BaseModel):
    """Request body for promoting a specific version to active."""

    model_config = {"protected_namespaces": ()}

    ticker: str = Field(..., min_length=1, max_length=10)
    model_name: str
    horizon: str
    version_id: str = Field(..., description="Version ID to promote (or roll back to)")


class PromoteVersionResponse(BaseModel):
    """Response after a successful promotion or rollback."""

    model_config = {"protected_namespaces": ()}

    ticker: str
    model_name: str
    horizon: str
    promoted_version_id: str
    previous_version_id: Optional[str]
    message: str


class RollbackRequest(BaseModel):
    """Request body for rolling back to the previous version."""

    model_config = {"protected_namespaces": ()}

    ticker: str = Field(..., min_length=1, max_length=10)
    model_name: str
    horizon: str


class PruneVersionsRequest(BaseModel):
    """Request body for pruning old versions."""

    model_config = {"protected_namespaces": ()}

    ticker: str = Field(..., min_length=1, max_length=10)
    model_name: str
    horizon: str
    keep_last: int = Field(
        default=5, ge=1, le=50, description="Number of versions to keep"
    )


class PruneVersionsResponse(BaseModel):
    """Response after pruning."""

    model_config = {"protected_namespaces": ()}

    ticker: str
    model_name: str
    horizon: str
    deleted_version_ids: list[str]
    deleted_count: int
    message: str
