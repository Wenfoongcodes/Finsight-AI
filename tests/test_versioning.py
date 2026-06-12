"""
tests/test_versioning.py
=========================
Unit and integration tests for the model versioning, promotion, and rollback
system introduced in app/ml/training/versioning.py and the updated trainer.py.

All tests are self-contained — no network access, no real ML training
(the trainer is exercised via the VersionedArtifactStore directly, or with
a tiny synthetic dataset and a fast random_forest config).
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import ModelNotFoundError
from app.ml.training.versioning import (
    DEFAULT_KEEP_LAST,
    VersionedArtifactStore,
    _build_registry_entry,
    _feature_hash,
    make_version_id,
    parse_version_timestamp,
)
from app.ml.training.trainer import ModelTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> VersionedArtifactStore:
    return VersionedArtifactStore(tmp_path)


@pytest.fixture
def trainer(tmp_path: Path) -> ModelTrainer:
    return ModelTrainer(model_dir=tmp_path)


def _dummy_artifact(feature_columns: list[str]) -> dict:
    """Minimal pkl artifact dict."""
    return {
        "model": object(),  # placeholder
        "feature_columns": feature_columns,
        "horizon": "1d",
        "trained_at": "2026-05-14T09:41:22+00:00",
        "version_id": "20260514T094122_a3f7c2d1",
    }


def _dummy_meta(
    version_id: str = "20260514T094122_a3f7c2d1",
    auc: float = 0.62,
    feature_columns: list[str] | None = None,
) -> dict:
    cols = feature_columns or ["rsi_14", "macd", "momentum_5d"]
    return {
        "model_name": "xgboost",
        "ticker": "AAPL",
        "horizon": "1d",
        "trained_at": "2026-05-14T09:41:22+00:00",
        "trigger_reason": "manual_request",
        "mean_roc_auc": auc,
        "mean_accuracy": 0.55,
        "mean_f1": 0.53,
        "n_features": len(cols),
        "best_params": {},
        "training_duration_s": 12.34,
        "n_folds": 5,
        "feature_columns": cols,
        "version_id": version_id,
    }


def _write_version(
    store: VersionedArtifactStore,
    ticker: str,
    model: str,
    horizon: str,
    version_id: str,
    feature_columns: list[str] | None = None,
    auc: float = 0.62,
) -> None:
    """Write a fake versioned artifact + meta + registry entry."""
    cols = feature_columns or ["rsi_14", "macd"]
    vdir = store.version_dir(ticker, model, horizon, version_id)
    vdir.mkdir(parents=True, exist_ok=True)

    artifact = _dummy_artifact(cols)
    artifact["version_id"] = version_id
    with open(store.model_path(ticker, model, horizon, version_id), "wb") as f:
        pickle.dump(artifact, f)

    meta = _dummy_meta(version_id, auc, cols)
    store._atomic_write(store.meta_path(ticker, model, horizon, version_id), meta)

    entry = _build_registry_entry(version_id, meta, cols, is_active=False)
    store.add_registry_entry(ticker, model, horizon, entry)


# ─────────────────────────────────────────────────────────────────────────────
# Version ID helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestVersionId:
    def test_format_is_sortable(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        id1 = make_version_id(["a"], {}, t1)
        id2 = make_version_id(["a"], {}, t2)
        assert id1 < id2, "Version IDs must sort chronologically"

    def test_different_features_give_different_hash(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        id1 = make_version_id(["rsi_14"], {}, t)
        id2 = make_version_id(["macd"], {}, t)
        assert id1 != id2

    def test_same_config_same_hash_different_timestamp(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
        id1 = make_version_id(["a", "b"], {"n": 100}, t1)
        id2 = make_version_id(["a", "b"], {"n": 100}, t2)
        assert id1 != id2  # different timestamps
        assert id1.split("_")[1] == id2.split("_")[1]  # same hash

    def test_roundtrip_timestamp_parsing(self):
        ts = datetime(2026, 5, 14, 9, 41, 22, tzinfo=timezone.utc)
        vid = make_version_id([], {}, ts)
        recovered = parse_version_timestamp(vid)
        assert recovered == ts

    def test_feature_hash_is_deterministic(self):
        h1 = _feature_hash(["b", "a"], {"x": 1})
        h2 = _feature_hash(["a", "b"], {"x": 1})
        assert h1 == h2, "Feature hash must be order-insensitive"

    def test_version_id_length(self):
        vid = make_version_id(["rsi_14"], {})
        # Format: YYYYmmddTHHMMSS_xxxxxxxx  (15 + 1 + 8 = 24 chars)
        parts = vid.split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 15
        assert len(parts[1]) == 8


# ─────────────────────────────────────────────────────────────────────────────
# VersionedArtifactStore — path helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestArtifactStorePaths:
    def test_version_dir_structure(self, store: VersionedArtifactStore, tmp_path: Path):
        vdir = store.version_dir("AAPL", "xgboost", "1d", "v1")
        assert vdir == tmp_path / "AAPL" / "xgboost" / "1d" / "versions" / "v1"

    def test_model_path_is_under_version_dir(self, store: VersionedArtifactStore):
        mp = store.model_path("AAPL", "xgboost", "1d", "v1")
        assert mp.name == "model.pkl"
        assert "versions" in str(mp)

    def test_meta_path_is_under_version_dir(self, store: VersionedArtifactStore):
        mp = store.meta_path("AAPL", "xgboost", "1d", "v1")
        assert mp.name == "meta.json"

    def test_active_path_at_slot_level(self, store: VersionedArtifactStore, tmp_path: Path):
        ap = store._active_path("AAPL", "xgboost", "1d")
        assert ap == tmp_path / "AAPL" / "xgboost" / "1d" / "active.json"

    def test_registry_path_at_slot_level(self, store: VersionedArtifactStore, tmp_path: Path):
        rp = store._registry_path("AAPL", "xgboost", "1d")
        assert rp == tmp_path / "AAPL" / "xgboost" / "1d" / "versions.json"


# ─────────────────────────────────────────────────────────────────────────────
# Active pointer CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestActivePointer:
    def test_get_active_returns_none_when_absent(self, store: VersionedArtifactStore):
        assert store.get_active_version_id("AAPL", "xgboost", "1d") is None

    def test_set_and_get_active(self, store: VersionedArtifactStore):
        store.set_active_version_id("AAPL", "xgboost", "1d", "v42")
        assert store.get_active_version_id("AAPL", "xgboost", "1d") == "v42"

    def test_set_active_is_atomic(self, store: VersionedArtifactStore):
        """Multiple writes — last one wins, file is always valid JSON."""
        for i in range(10):
            store.set_active_version_id("AAPL", "xgboost", "1d", f"v{i}")
        assert store.get_active_version_id("AAPL", "xgboost", "1d") == "v9"

    def test_active_pointer_is_independent_per_slot(self, store: VersionedArtifactStore):
        store.set_active_version_id("AAPL", "xgboost", "1d", "v1")
        store.set_active_version_id("MSFT", "xgboost", "1d", "v2")
        store.set_active_version_id("AAPL", "lightgbm", "1d", "v3")
        assert store.get_active_version_id("AAPL", "xgboost", "1d") == "v1"
        assert store.get_active_version_id("MSFT", "xgboost", "1d") == "v2"
        assert store.get_active_version_id("AAPL", "lightgbm", "1d") == "v3"


# ─────────────────────────────────────────────────────────────────────────────
# Version registry
# ─────────────────────────────────────────────────────────────────────────────


class TestVersionRegistry:
    def test_load_empty_registry(self, store: VersionedArtifactStore):
        assert store.load_registry("AAPL", "xgboost", "1d") == []

    def test_add_entry(self, store: VersionedArtifactStore):
        entry = {"version_id": "v1", "mean_roc_auc": 0.62}
        store.add_registry_entry("AAPL", "xgboost", "1d", entry)
        entries = store.load_registry("AAPL", "xgboost", "1d")
        assert len(entries) == 1
        assert entries[0]["version_id"] == "v1"

    def test_entries_sorted_chronologically(self, store: VersionedArtifactStore):
        for vid in ["v3", "v1", "v2"]:
            store.add_registry_entry("AAPL", "xgboost", "1d", {"version_id": vid})
        entries = store.load_registry("AAPL", "xgboost", "1d")
        ids = [e["version_id"] for e in entries]
        assert ids == sorted(ids)

    def test_idempotent_add(self, store: VersionedArtifactStore):
        entry = {"version_id": "v1", "mean_roc_auc": 0.60}
        store.add_registry_entry("AAPL", "xgboost", "1d", entry)
        updated = {"version_id": "v1", "mean_roc_auc": 0.65}
        store.add_registry_entry("AAPL", "xgboost", "1d", updated)
        entries = store.load_registry("AAPL", "xgboost", "1d")
        assert len(entries) == 1
        assert entries[0]["mean_roc_auc"] == 0.65

    def test_update_active_flags(self, store: VersionedArtifactStore):
        for vid in ["v1", "v2", "v3"]:
            store.add_registry_entry(
                "AAPL", "xgboost", "1d", {"version_id": vid, "is_active": False}
            )
        store.update_registry_active_flags("AAPL", "xgboost", "1d", "v2")
        entries = {e["version_id"]: e for e in store.load_registry("AAPL", "xgboost", "1d")}
        assert entries["v1"]["is_active"] is False
        assert entries["v2"]["is_active"] is True
        assert entries["v3"]["is_active"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Pruning
# ─────────────────────────────────────────────────────────────────────────────


class TestPruning:
    def test_prune_deletes_old_versions(self, store: VersionedArtifactStore):
        for i in range(6):
            _write_version(store, "AAPL", "xgboost", "1d", f"v{i:02d}")
        store.set_active_version_id("AAPL", "xgboost", "1d", "v05")

        deleted = store.prune("AAPL", "xgboost", "1d", keep_last=3)
        assert len(deleted) == 3  # v00, v01, v02 deleted

    def test_prune_never_deletes_active(self, store: VersionedArtifactStore):
        for i in range(5):
            _write_version(store, "AAPL", "xgboost", "1d", f"v{i:02d}")
        # Mark the oldest as active
        store.set_active_version_id("AAPL", "xgboost", "1d", "v00")

        deleted = store.prune("AAPL", "xgboost", "1d", keep_last=2)
        # v00 (active) must NOT be deleted; v01, v02 may be
        assert "v00" not in deleted
        assert store.version_exists("AAPL", "xgboost", "1d", "v00")

    def test_prune_removes_from_registry(self, store: VersionedArtifactStore):
        for i in range(5):
            _write_version(store, "AAPL", "xgboost", "1d", f"v{i:02d}")
        store.set_active_version_id("AAPL", "xgboost", "1d", "v04")

        store.prune("AAPL", "xgboost", "1d", keep_last=2)
        remaining = {e["version_id"] for e in store.load_registry("AAPL", "xgboost", "1d")}
        # v03 and v04 should remain; v04 is active
        assert "v04" in remaining
        assert "v03" in remaining

    def test_prune_no_op_when_within_limit(self, store: VersionedArtifactStore):
        for i in range(3):
            _write_version(store, "AAPL", "xgboost", "1d", f"v{i:02d}")
        store.set_active_version_id("AAPL", "xgboost", "1d", "v02")
        deleted = store.prune("AAPL", "xgboost", "1d", keep_last=5)
        assert deleted == []


# ─────────────────────────────────────────────────────────────────────────────
# Legacy migration
# ─────────────────────────────────────────────────────────────────────────────


class TestLegacyMigration:
    def test_migrate_creates_versioned_layout(self, tmp_path: Path):
        store = VersionedArtifactStore(tmp_path)
        feature_columns = ["rsi_14", "macd", "volume_ratio"]

        # Write a legacy flat artifact
        legacy_pkl = tmp_path / "AAPL_xgboost_1d.pkl"
        with open(legacy_pkl, "wb") as f:
            pickle.dump({"model": None, "feature_columns": feature_columns}, f)

        meta = _dummy_meta(auc=0.64, feature_columns=feature_columns)

        version_id = store.migrate_legacy_artifact(
            "AAPL", "xgboost", "1d", feature_columns, meta
        )
        assert version_id is not None
        assert store.version_exists("AAPL", "xgboost", "1d", version_id)
        assert store.get_active_version_id("AAPL", "xgboost", "1d") == version_id
        assert not legacy_pkl.exists(), "Legacy file should be removed after migration"

    def test_has_legacy_artifact_detection(self, tmp_path: Path):
        store = VersionedArtifactStore(tmp_path)
        assert not store.has_legacy_artifact("AAPL", "xgboost", "1d")

        (tmp_path / "AAPL_xgboost_1d.pkl").write_bytes(b"fake")
        assert store.has_legacy_artifact("AAPL", "xgboost", "1d")


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer — versioning integration
# ─────────────────────────────────────────────────────────────────────────────


class TestModelTrainerVersioning:
    """
    Uses a tiny synthetic dataset and a very fast random_forest config
    to exercise the full trainer → versioning pathway without long runtimes.
    """

    @pytest.fixture
    def tiny_X_y(self):
        np.random.seed(0)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((np.random.rand(n) > 0.5).astype(int))
        return X, y

    def test_train_creates_versioned_artifact(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        _, result = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d",
            hyperparams={"n_estimators": 5, "max_depth": 2},
        )
        assert result.version_id != ""
        assert trainer._store.version_exists("TEST", "random_forest", "1d", result.version_id)

    def test_train_sets_active_version(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        _, result = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d",
            hyperparams={"n_estimators": 5, "max_depth": 2},
        )
        active = trainer._store.get_active_version_id("TEST", "random_forest", "1d")
        assert active == result.version_id

    def test_two_trains_create_two_versions(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        params = {"n_estimators": 5, "max_depth": 2}
        _, r1 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        import time; time.sleep(1.1)  # ensure different timestamps
        _, r2 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        assert r1.version_id != r2.version_id
        versions = trainer.list_versions("TEST", "random_forest", "1d")
        assert len(versions) == 2

    def test_load_model_loads_active_version(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        _, result = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d",
            hyperparams={"n_estimators": 5, "max_depth": 2},
        )
        model, feature_columns = trainer.load_model("TEST", "random_forest", "1d")
        assert model is not None
        assert feature_columns == list(X.columns)

    def test_promote_version_changes_active(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        params = {"n_estimators": 5, "max_depth": 2}
        _, r1 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        import time; time.sleep(1.1)
        _, r2 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        # r2 is active now; promote r1
        trainer.promote_version("TEST", "random_forest", "1d", r1.version_id)
        active = trainer._store.get_active_version_id("TEST", "random_forest", "1d")
        assert active == r1.version_id

    def test_rollback_restores_previous_version(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        params = {"n_estimators": 5, "max_depth": 2}
        _, r1 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        import time; time.sleep(1.1)
        _, r2 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        # r2 is active
        assert trainer._store.get_active_version_id("TEST", "random_forest", "1d") == r2.version_id
        # Rollback → r1
        restored = trainer.rollback("TEST", "random_forest", "1d")
        assert restored == r1.version_id
        assert trainer._store.get_active_version_id("TEST", "random_forest", "1d") == r1.version_id

    def test_rollback_raises_when_no_previous(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d",
            hyperparams={"n_estimators": 5, "max_depth": 2},
        )
        with pytest.raises(ModelNotFoundError):
            trainer.rollback("TEST", "random_forest", "1d")

    def test_promote_nonexistent_version_raises(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d",
            hyperparams={"n_estimators": 5, "max_depth": 2},
        )
        with pytest.raises(ModelNotFoundError):
            trainer.promote_version("TEST", "random_forest", "1d", "nonexistent_version")

    def test_prune_versions_via_trainer(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        params = {"n_estimators": 5, "max_depth": 2}
        for _ in range(4):
            import time; time.sleep(1.1)
            trainer.train(
                "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
            )
        deleted = trainer.prune_versions("TEST", "random_forest", "1d", keep_last=2)
        remaining = trainer.list_versions("TEST", "random_forest", "1d")
        active_id = trainer._store.get_active_version_id("TEST", "random_forest", "1d")
        assert any(v["is_active"] for v in remaining)
        assert all(v["version_id"] != d for v in remaining for d in deleted)
        # Active version must still be present
        assert any(v["version_id"] == active_id for v in remaining)

    def test_list_versions_returns_enriched_entries(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d",
            hyperparams={"n_estimators": 5, "max_depth": 2},
        )
        versions = trainer.list_versions("TEST", "random_forest", "1d")
        assert len(versions) == 1
        v = versions[0]
        assert "version_id" in v
        assert "is_active" in v
        assert "exists_on_disk" in v
        assert v["is_active"] is True
        assert v["exists_on_disk"] is True

    def test_load_specific_version_by_id(self, trainer: ModelTrainer, tiny_X_y):
        X, y = tiny_X_y
        params = {"n_estimators": 5, "max_depth": 2}
        _, r1 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        import time; time.sleep(1.1)
        _, r2 = trainer.train(
            "random_forest", X, y, ticker="TEST", horizon="1d", hyperparams=params,
        )
        # Load r1 explicitly even though r2 is active
        model, cols = trainer.load_model(
            "TEST", "random_forest", "1d", version_id=r1.version_id
        )
        assert model is not None

    def test_legacy_migration_via_load_or_train(self, tmp_path: Path, tiny_X_y):
        """
        A legacy flat artifact should be transparently migrated on the first
        call to load_or_train, returning the model without retraining.
        """
        from sklearn.ensemble import RandomForestClassifier

        X, y = tiny_X_y
        feature_columns = list(X.columns)

        # Write a legacy flat artifact manually
        model = RandomForestClassifier(n_estimators=3).fit(X, y)
        legacy_pkl = tmp_path / "TEST_random_forest_1d.pkl"
        with open(legacy_pkl, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "horizon": "1d",
                    "trained_at": "2026-01-01T00:00:00+00:00",
                },
                f,
            )

        meta = {
            "model_name": "random_forest",
            "ticker": "TEST",
            "horizon": "1d",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "mean_roc_auc": 0.58,
            "mean_accuracy": 0.54,
            "mean_f1": 0.52,
            "best_params": {},
            "n_features": len(feature_columns),
            "trigger_reason": "manual_request",
        }
        (tmp_path / "TEST_random_forest_1d_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        trainer = ModelTrainer(model_dir=tmp_path)
        loaded_model, loaded_cols, train_result = trainer.load_or_train(
            "TEST", "random_forest", X, y, horizon="1d"
        )
        assert loaded_model is not None
        assert loaded_cols == feature_columns
        assert train_result is None  # migration, not retraining
        assert not legacy_pkl.exists()  # legacy file removed


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVersioningEndpoints:
    """
    Smoke-test the FastAPI versioning endpoints via TestClient.
    Actual model training is mocked so tests are fast.
    """

    @pytest.fixture
    def client_with_versions(self, tmp_path: Path):
        """
        Patch ModelTrainer to use tmp_path and pre-populate two versions.
        """
        from fastapi.testclient import TestClient
        from main import app

        store = VersionedArtifactStore(tmp_path)
        _write_version(store, "AAPL", "xgboost", "1d", "v_old", auc=0.60)
        _write_version(store, "AAPL", "xgboost", "1d", "v_new", auc=0.65)
        store.set_active_version_id("AAPL", "xgboost", "1d", "v_new")
        store.update_registry_active_flags("AAPL", "xgboost", "1d", "v_new")

        # Patch the flat meta for ModelSelector
        (tmp_path / "AAPL_xgboost_1d_meta.json").write_text(
            json.dumps(_dummy_meta("v_new", 0.65)), encoding="utf-8"
        )

        import app.api.v1.endpoints.versioning as vmod
        original_trainer = vmod._trainer

        trainer = ModelTrainer(model_dir=tmp_path)
        vmod._trainer = trainer

        with TestClient(app) as client:
            yield client

        vmod._trainer = original_trainer

    def test_get_version_history(self, client_with_versions):
        resp = client_with_versions.get("/api/v1/versions/AAPL/xgboost/1d")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert data["active_version_id"] == "v_new"
        assert data["total_versions"] == 2

    def test_promote_version(self, client_with_versions):
        resp = client_with_versions.post(
            "/api/v1/versions/promote",
            json={
                "ticker": "AAPL",
                "model_name": "xgboost",
                "horizon": "1d",
                "version_id": "v_old",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["promoted_version_id"] == "v_old"
        assert data["previous_version_id"] == "v_new"

    def test_rollback(self, client_with_versions):
        resp = client_with_versions.post(
            "/api/v1/versions/rollback",
            json={"ticker": "AAPL", "model_name": "xgboost", "horizon": "1d"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["promoted_version_id"] == "v_old"

    def test_promote_nonexistent_returns_404(self, client_with_versions):
        resp = client_with_versions.post(
            "/api/v1/versions/promote",
            json={
                "ticker": "AAPL",
                "model_name": "xgboost",
                "horizon": "1d",
                "version_id": "nonexistent_id",
            },
        )
        assert resp.status_code == 404

    def test_prune_versions(self, client_with_versions):
        resp = client_with_versions.request(
            "DELETE",
            "/api/v1/versions/prune",
            json={
                "ticker": "AAPL",
                "model_name": "xgboost",
                "horizon": "1d",
                "keep_last": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # One of the two versions should be deleted (not the active one)
        assert data["deleted_count"] <= 1