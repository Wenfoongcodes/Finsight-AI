from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from app.core.exceptions import ModelNotFoundError, ModelTrainingError
from app.core.formatting import (
    DURATION_DECIMAL_PLACES,
    round_metric,
    utc_now_iso,
)
from app.core.logging_config import get_logger
from app.ml.models.model_factory import get_model
from app.ml.training.versioning import (
    DEFAULT_AUC_IMPROVEMENT_THRESHOLD,
    DEFAULT_KEEP_LAST,
    VersionedArtifactStore,
    _build_registry_entry,
    make_version_id,
)
from configs.settings import settings

logger = get_logger("training")

# ── Trigger reasons for audit log ─────────────────────────────────────────────
TRIGGER_MISSING = "artifact_not_found"
TRIGGER_CORRUPT = "artifact_corrupt"
TRIGGER_MISMATCH = "feature_mismatch"
TRIGGER_MANUAL = "manual_request"
TRIGGER_STALE = "artifact_stale"


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    f1: float
    roc_auc: float
    mae: float
    rmse: float


@dataclass
class TrainingResult:
    """
    All float metric fields use ``round_metric()`` (4 d.p.) for consistency
    with leaderboard JSON and log output.
    """

    model_name: str
    ticker: str
    horizon: str
    trained_at: str
    n_features: int
    trigger_reason: str = TRIGGER_MANUAL
    fold_results: list[FoldResult] = field(default_factory=list)
    mean_accuracy: float = 0.0
    mean_f1: float = 0.0
    mean_roc_auc: float = 0.0
    mean_mae: float = 0.0
    mean_rmse: float = 0.0
    best_params: dict = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    training_duration_s: float = 0.0
    # Version identifier assigned after artifact save
    version_id: str = ""

    def compute_aggregates(self) -> None:
        """Compute mean metrics with canonical precision."""
        if not self.fold_results:
            return
        self.mean_accuracy = round_metric(
            float(np.mean([f.accuracy for f in self.fold_results]))
        )
        self.mean_f1 = round_metric(float(np.mean([f.f1 for f in self.fold_results])))
        self.mean_roc_auc = round_metric(
            float(np.mean([f.roc_auc for f in self.fold_results]))
        )
        self.mean_mae = round_metric(float(np.mean([f.mae for f in self.fold_results])))
        self.mean_rmse = round_metric(
            float(np.mean([f.rmse for f in self.fold_results]))
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "ticker": self.ticker,
            "horizon": self.horizon,
            "trained_at": self.trained_at,
            "n_features": self.n_features,
            "trigger_reason": self.trigger_reason,
            "mean_accuracy": round_metric(self.mean_accuracy),
            "mean_f1": round_metric(self.mean_f1),
            "mean_roc_auc": round_metric(self.mean_roc_auc),
            "mean_mae": round_metric(self.mean_mae),
            "mean_rmse": round_metric(self.mean_rmse),
            "training_duration_s": round(
                self.training_duration_s, DURATION_DECIMAL_PLACES
            ),
            "best_params": self.best_params,
            "n_folds": len(self.fold_results),
            "feature_columns": self.feature_columns,
            "version_id": self.version_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Splitter
# ─────────────────────────────────────────────────────────────────────────────


class WalkForwardSplitter:
    """
    Time-series-safe expanding window splitter.

    Each fold expands the training window by one fold-size chunk.
    No future data ever appears in any training window — enforces
    zero look-ahead bias.
    """

    def __init__(
        self,
        n_folds: int = settings.WALK_FORWARD_FOLDS,
        min_train_pct: float = 0.4,
    ) -> None:
        self.n_folds = n_folds
        self.min_train_pct = min_train_pct

    def split(self, X: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        min_train = int(n * self.min_train_pct)
        fold_size = max(1, (n - min_train) // self.n_folds)
        splits = []
        for fold in range(self.n_folds):
            train_end = min_train + fold * fold_size
            test_end = min(train_end + fold_size, n)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            if len(test_idx) == 0:
                break
            splits.append((train_idx, test_idx))
        logger.debug("Walk-forward: %d folds generated", len(splits))
        return splits


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """All returned values are rounded via ``round_metric()``."""
    return {
        "accuracy": round_metric(float(accuracy_score(y_true, y_pred))),
        "f1": round_metric(float(f1_score(y_true, y_pred, zero_division=0))),
        "roc_auc": round_metric(
            float(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5)
        ),
        "mae": round_metric(float(mean_absolute_error(y_true, y_prob))),
        "rmse": round_metric(float(np.sqrt(mean_squared_error(y_true, y_prob)))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Optimization
# ─────────────────────────────────────────────────────────────────────────────


def optimize_hyperparameters(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 30,
) -> dict:
    """Run Optuna HPO for a given model."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("optuna is required: pip install optuna") from exc

    splitter = WalkForwardSplitter(n_folds=3)
    splits = splitter.split(X)

    search_spaces: dict[str, Any] = {
        "xgboost": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        },
        "lightgbm": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        },
        "random_forest": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 50),
        },
        "logistic_regression": lambda trial: {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        },
    }

    if model_name not in search_spaces:
        logger.warning("No search space for %s; using defaults.", model_name)
        return {}

    def objective(trial) -> float:
        params = search_spaces[model_name](trial)
        aucs: list[float] = []
        for train_idx, test_idx in splits:
            model = get_model(model_name, **params)
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_te)[:, 1]
            if len(np.unique(y_te)) > 1:
                aucs.append(roc_auc_score(y_te, prob))
        return float(np.mean(aucs)) if aucs else 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(
        "Optuna best for %s: AUC=%.4f | %s",
        model_name,
        study.best_value,
        study.best_params,
    )
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Model Trainer
# ─────────────────────────────────────────────────────────────────────────────


class ModelTrainer:
    """
    Trains a model with walk-forward validation and persists versioned artifacts.

    Versioning behaviour
    --------------------
    Each call to ``train()`` creates a new version directory under
    ``{models_dir}/{TICKER}/{model}/{horizon}/versions/{version_id}/``.
    Old versions are never overwritten.

    ``promote=True`` (default) immediately makes the new version active.
    ``auto_promote_if_better=True`` only promotes if the new AUC exceeds
    the current active version by at least ``auc_threshold``.

    Rollback
    --------
    ``rollback()`` promotes the previous active version.  The full history
    is available via ``list_versions()``.

    Legacy compatibility
    --------------------
    On first access for a ticker/model/horizon that has a legacy flat
    artifact (``{TICKER}_{model}_{horizon}.pkl``), the artifact is
    transparently migrated into the versioned layout and the flat file
    is removed.  All existing call sites continue to work unchanged.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir or settings.MODELS_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._store = VersionedArtifactStore(self.model_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_or_train(
        self,
        ticker: str,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: str = "1d",
        hyperparams: Optional[dict] = None,
        run_hpo: bool = False,
        hpo_trials: int = 30,
        calibrate: bool = True,
        fold_callback: Optional[Callable[[FoldResult], None]] = None,
    ) -> tuple[Any, list[str], "TrainingResult | None"]:
        trigger = self._detect_trigger(ticker, model_name, horizon, list(X.columns))

        if trigger is None:
            model, feature_columns = self.load_model(ticker, model_name, horizon)
            return model, feature_columns, None

        logger.info(
            "[%s/%s/%s] Training triggered: %s",
            ticker,
            model_name,
            horizon,
            trigger,
        )
        model, result = self.train(
            model_name=model_name,
            X=X,
            y=y,
            ticker=ticker,
            horizon=horizon,
            hyperparams=hyperparams,
            run_hpo=run_hpo,
            hpo_trials=hpo_trials,
            calibrate=calibrate,
            trigger_reason=trigger,
            fold_callback=fold_callback,
        )
        return model, result.feature_columns, result

    def train(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        ticker: str = "UNKNOWN",
        horizon: str = "1d",
        hyperparams: Optional[dict] = None,
        run_hpo: bool = False,
        hpo_trials: int = 30,
        calibrate: bool = True,
        trigger_reason: str = TRIGGER_MANUAL,
        fold_callback: Optional[Callable[[FoldResult], None]] = None,
        promote: bool = True,
        auto_promote_if_better: bool = False,
        auc_threshold: float = DEFAULT_AUC_IMPROVEMENT_THRESHOLD,
    ) -> tuple[Any, "TrainingResult"]:
        """
        Train with walk-forward validation and persist a versioned artifact.

        Parameters
        ----------
        promote:
            If True (default), immediately promote the new version to active.
        auto_promote_if_better:
            If True, only promote when the new AUC exceeds the current active
            version's AUC by at least ``auc_threshold``.  Overrides ``promote``
            when set to True.
        auc_threshold:
            Minimum AUC improvement required for auto-promotion.
        fold_callback:
            Optional per-fold callback ``(FoldResult) -> None``.
        """
        try:
            t_start = time.perf_counter()
            best_params = hyperparams or {}

            if run_hpo and not hyperparams:
                logger.info(
                    "[%s/%s/%s] Running HPO (%d trials)…",
                    ticker,
                    model_name,
                    horizon,
                    hpo_trials,
                )
                best_params = optimize_hyperparameters(
                    model_name, X, y, n_trials=hpo_trials
                )

            splitter = WalkForwardSplitter()
            splits = splitter.split(X)

            result = TrainingResult(
                model_name=model_name,
                ticker=ticker,
                horizon=horizon,
                trained_at=utc_now_iso(),
                n_features=X.shape[1],
                trigger_reason=trigger_reason,
                best_params=best_params,
                feature_columns=list(X.columns),
            )

            # ── Walk-forward evaluation ────────────────────────────────────
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                fold_model = get_model(model_name, **best_params)
                fold_model.fit(X_tr, y_tr)

                y_pred = fold_model.predict(X_te)
                y_prob = fold_model.predict_proba(X_te)[:, 1]
                metrics = compute_metrics(y_te.values, y_pred, y_prob)

                fold_result = FoldResult(
                    fold=fold_idx + 1,
                    train_size=len(X_tr),
                    test_size=len(X_te),
                    **metrics,
                )
                result.fold_results.append(fold_result)

                if fold_callback is not None:
                    try:
                        fold_callback(fold_result)
                    except Exception:
                        pass

                logger.info(
                    "[%s/%s/%s] Fold %d/%d — Acc=%.4f F1=%.4f AUC=%.4f",
                    ticker,
                    model_name,
                    horizon,
                    fold_idx + 1,
                    len(splits),
                    metrics["accuracy"],
                    metrics["f1"],
                    metrics["roc_auc"],
                )

            result.compute_aggregates()

            # ── Final model ────────────────────────────────────────────────
            should_calibrate = calibrate and model_name != "logistic_regression"

            if should_calibrate:
                logger.info(
                    "[%s/%s/%s] Applying Platt scaling (cv=3)…",
                    ticker,
                    model_name,
                    horizon,
                )
                base_estimator = get_model(model_name, **best_params)
                final_model = CalibratedClassifierCV(
                    estimator=base_estimator, cv=3, method="sigmoid"
                )
                final_model.fit(X, y)
            else:
                final_model = get_model(model_name, **best_params)
                final_model.fit(X, y)

            result.training_duration_s = round(
                time.perf_counter() - t_start, DURATION_DECIMAL_PLACES
            )

            # ── Generate version identifier ────────────────────────────────
            from datetime import datetime, timezone

            ts = datetime.strptime(result.trained_at[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            version_id = make_version_id(result.feature_columns, best_params, ts)
            result.version_id = version_id

            # ── Persist versioned artifacts ────────────────────────────────
            self._save_versioned_artifacts(
                final_model, result, ticker, model_name, horizon
            )

            # ── Promotion logic ────────────────────────────────────────────
            should_promote = promote
            if auto_promote_if_better:
                should_promote = self._should_promote(
                    ticker, model_name, horizon, result.mean_roc_auc, auc_threshold
                )

            if should_promote:
                self._store.set_active_version_id(
                    ticker, model_name, horizon, version_id
                )
                self._store.update_registry_active_flags(
                    ticker, model_name, horizon, version_id
                )
                # Also write legacy-compatible flat meta.json for ModelSelector
                self._write_legacy_meta(result, ticker, model_name, horizon)
            else:
                logger.info(
                    "[%s/%s/%s] New version %s NOT promoted "
                    "(auto_promote_if_better=True, AUC %.4f not better enough)",
                    ticker,
                    model_name,
                    horizon,
                    version_id,
                    result.mean_roc_auc,
                )

            logger.info(
                "[%s/%s/%s] Training complete in %.2fs | "
                "Acc=%.4f F1=%.4f AUC=%.4f | trigger=%s | version=%s",
                ticker,
                model_name,
                horizon,
                result.training_duration_s,
                result.mean_accuracy,
                result.mean_f1,
                result.mean_roc_auc,
                trigger_reason,
                version_id,
            )
            return final_model, result

        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(
                f"Training failed for {model_name}/{horizon}: {exc}"
            ) from exc

    def load_model(
        self,
        ticker: str,
        model_name: str,
        horizon: str = "1d",
        version_id: Optional[str] = None,
    ) -> tuple[Any, list[str]]:
        """
        Load a model artifact.

        Parameters
        ----------
        version_id:
            Specific version to load.  If None, loads the active version.
            Pass ``version_id`` explicitly for point-in-time loading or
            rollback-preview without changing the active pointer.

        Raises:
            ModelNotFoundError: If no artifact exists.
        """
        # ── Legacy migration ───────────────────────────────────────────────
        if self._store.has_legacy_artifact(ticker, model_name, horizon):
            self._migrate_legacy(ticker, model_name, horizon)

        # ── Resolve version ────────────────────────────────────────────────
        if version_id is None:
            version_id = self._store.get_active_version_id(ticker, model_name, horizon)

        if version_id is None:
            raise ModelNotFoundError(
                f"No active model version for {ticker}/{model_name}/{horizon}",
                detail="Train first or check the version registry.",
            )

        path = self._store.model_path(ticker, model_name, horizon, version_id)
        if not path.exists():
            raise ModelNotFoundError(
                f"Model artifact not found: {path}",
                detail=f"Version {version_id} may have been pruned.",
            )

        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Artifact corrupt at {path}: {exc}",
                detail="Delete the version and retrain.",
            ) from exc

        logger.info(
            "[%s/%s/%s] Loaded version %s", ticker, model_name, horizon, version_id
        )
        return artifact["model"], artifact["feature_columns"]

    def promote_version(
        self, ticker: str, model_name: str, horizon: str, version_id: str
    ) -> None:
        """
        Explicitly promote a specific version to active.

        This is the primary rollback mechanism: call with the desired
        previous version ID to revert to it.

        Raises:
            ModelNotFoundError: If the version does not exist on disk.
        """
        if not self._store.version_exists(ticker, model_name, horizon, version_id):
            raise ModelNotFoundError(
                f"Version {version_id} not found for {ticker}/{model_name}/{horizon}",
                detail="Check list_versions() for available version IDs.",
            )

        previous_active = self._store.get_active_version_id(ticker, model_name, horizon)
        self._store.set_active_version_id(ticker, model_name, horizon, version_id)
        self._store.update_registry_active_flags(
            ticker, model_name, horizon, version_id
        )

        # Update the legacy-compatible flat meta so ModelSelector picks it up
        meta_path = self._store.meta_path(ticker, model_name, horizon, version_id)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._write_legacy_meta_from_dict(meta, ticker, model_name, horizon)
            except Exception as exc:
                logger.warning("Could not update legacy meta after promotion: %s", exc)

        logger.info(
            "[%s/%s/%s] Version promoted: %s → %s",
            ticker,
            model_name,
            horizon,
            previous_active,
            version_id,
        )

    def rollback(self, ticker: str, model_name: str, horizon: str) -> str:
        """
        Roll back to the version that was active before the current one.

        Returns the version ID that was restored.

        Raises:
            ModelNotFoundError: If there is no previous version to roll back to.
        """
        entries = self._store.load_registry(ticker, model_name, horizon)
        if not entries:
            raise ModelNotFoundError(
                f"No version history for {ticker}/{model_name}/{horizon}",
                detail="No rollback target available.",
            )

        current_active = self._store.get_active_version_id(ticker, model_name, horizon)

        # Find the most recent version that is NOT the current active one
        # and still exists on disk
        candidates = [
            e
            for e in reversed(entries)
            if e.get("version_id") != current_active
            and self._store.version_exists(ticker, model_name, horizon, e["version_id"])
        ]

        if not candidates:
            raise ModelNotFoundError(
                f"No previous version available for rollback "
                f"({ticker}/{model_name}/{horizon})",
                detail=f"Current active: {current_active}. "
                "Only one version exists or all others have been pruned.",
            )

        target_version = candidates[0]["version_id"]
        self.promote_version(ticker, model_name, horizon, target_version)

        logger.info(
            "[%s/%s/%s] Rolled back: %s → %s",
            ticker,
            model_name,
            horizon,
            current_active,
            target_version,
        )
        return target_version

    def list_versions(self, ticker: str, model_name: str, horizon: str) -> list[dict]:
        """
        Return full version history from the registry, enriched with
        on-disk existence status.

        Entries are sorted chronologically (oldest first).
        """
        # Migrate legacy artifact first so it shows up in the registry
        if self._store.has_legacy_artifact(ticker, model_name, horizon):
            self._migrate_legacy(ticker, model_name, horizon)

        entries = self._store.load_registry(ticker, model_name, horizon)
        active_id = self._store.get_active_version_id(ticker, model_name, horizon)

        result = []
        for e in entries:
            vid = e.get("version_id", "")
            enriched = dict(e)
            enriched["is_active"] = vid == active_id
            enriched["exists_on_disk"] = self._store.version_exists(
                ticker, model_name, horizon, vid
            )
            result.append(enriched)
        return result

    def prune_versions(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        keep_last: int = DEFAULT_KEEP_LAST,
    ) -> list[str]:
        """
        Delete old version directories, keeping the ``keep_last`` most recent
        and always preserving the active version.

        Returns the list of deleted version IDs.
        """
        deleted = self._store.prune(ticker, model_name, horizon, keep_last)
        logger.info(
            "[%s/%s/%s] Pruned %d versions (keep_last=%d)",
            ticker,
            model_name,
            horizon,
            len(deleted),
            keep_last,
        )
        return deleted

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_trigger(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        current_feature_cols: list[str],
    ) -> Optional[str]:
        """
        Determine whether retraining is needed.  Returns a trigger reason string
        or None (meaning: load the existing active version).

        Checks both the versioned layout and legacy flat files.
        """
        # ── Check versioned active pointer first ───────────────────────────
        active_id = self._store.get_active_version_id(ticker, model_name, horizon)

        if active_id is None:
            # Check for legacy flat artifact
            if self._store.has_legacy_artifact(ticker, model_name, horizon):
                # Migrate and consider it a valid artifact
                self._migrate_legacy(ticker, model_name, horizon)
                active_id = self._store.get_active_version_id(
                    ticker, model_name, horizon
                )
                if active_id is None:
                    return TRIGGER_MISSING
            else:
                return TRIGGER_MISSING

        # ── Verify the active version's artifact exists ────────────────────
        if not self._store.version_exists(ticker, model_name, horizon, active_id):
            logger.warning(
                "[%s/%s/%s] Active version %s missing from disk — retraining.",
                ticker,
                model_name,
                horizon,
                active_id,
            )
            return TRIGGER_CORRUPT

        # ── Load the active version's pkl to check feature columns ─────────
        path = self._store.model_path(ticker, model_name, horizon, active_id)
        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        except Exception as exc:
            logger.warning(
                "[%s/%s/%s] Artifact corrupt (%s) — will retrain.",
                ticker,
                model_name,
                horizon,
                exc,
            )
            return TRIGGER_CORRUPT

        saved_cols = set(artifact.get("feature_columns", []))
        current_cols = set(current_feature_cols)
        if saved_cols != current_cols:
            added = current_cols - saved_cols
            removed = saved_cols - current_cols
            logger.warning(
                "[%s/%s/%s] Feature mismatch — added=%d removed=%d — retraining.",
                ticker,
                model_name,
                horizon,
                len(added),
                len(removed),
            )
            return TRIGGER_MISMATCH

        return None

    def _save_versioned_artifacts(
        self,
        model: Any,
        result: TrainingResult,
        ticker: str,
        model_name: str,
        horizon: str,
    ) -> None:
        version_id = result.version_id
        vdir = self._store.version_dir(ticker, model_name, horizon, version_id)
        vdir.mkdir(parents=True, exist_ok=True)

        model_path = self._store.model_path(ticker, model_name, horizon, version_id)
        meta_path = self._store.meta_path(ticker, model_name, horizon, version_id)

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": result.feature_columns,
                    "horizon": horizon,
                    "trained_at": result.trained_at,
                    "version_id": version_id,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        result_dict = result.to_dict()
        with open(meta_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        # Add to registry
        entry = _build_registry_entry(
            version_id, result_dict, result.feature_columns, is_active=False
        )
        self._store.add_registry_entry(ticker, model_name, horizon, entry)

        logger.info(
            "[%s/%s/%s] Version artifact saved: %s",
            ticker,
            model_name,
            horizon,
            version_id,
        )

    def _write_legacy_meta(
        self, result: TrainingResult, ticker: str, model_name: str, horizon: str
    ) -> None:
        """Write a legacy-compatible flat meta.json so ModelSelector still works."""
        meta_path = (
            self.model_dir / f"{ticker.upper()}_{model_name}_{horizon}_meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _write_legacy_meta_from_dict(
        self, meta: dict, ticker: str, model_name: str, horizon: str
    ) -> None:
        meta_path = (
            self.model_dir / f"{ticker.upper()}_{model_name}_{horizon}_meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _should_promote(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        new_auc: float,
        threshold: float,
    ) -> bool:
        """Return True if new_auc is sufficiently better than the current active AUC."""
        active_id = self._store.get_active_version_id(ticker, model_name, horizon)
        if active_id is None:
            return True  # No current active version — always promote

        entries = self._store.load_registry(ticker, model_name, horizon)
        current_entry = next(
            (e for e in entries if e.get("version_id") == active_id), None
        )
        if current_entry is None:
            return True

        current_auc = float(current_entry.get("mean_roc_auc", 0.0))
        improvement = new_auc - current_auc
        if improvement >= threshold:
            logger.info(
                "[%s/%s/%s] Auto-promote: new AUC %.4f > current %.4f + threshold %.4f",
                ticker,
                model_name,
                horizon,
                new_auc,
                current_auc,
                threshold,
            )
            return True
        logger.info(
            "[%s/%s/%s] No auto-promote: improvement %.4f < threshold %.4f",
            ticker,
            model_name,
            horizon,
            improvement,
            threshold,
        )
        return False

    def _migrate_legacy(self, ticker: str, model_name: str, horizon: str) -> None:
        """Transparently migrate a legacy flat artifact into the versioned layout."""
        legacy_pkl = self._store.legacy_model_path(ticker, model_name, horizon)
        legacy_meta_path = self._store.legacy_meta_path(ticker, model_name, horizon)

        if not legacy_pkl.exists():
            return

        # Load feature_columns from the pkl
        try:
            with open(legacy_pkl, "rb") as f:
                artifact = pickle.load(f)
            feature_columns = artifact.get("feature_columns", [])
        except Exception as exc:
            logger.warning(
                "[%s/%s/%s] Cannot migrate legacy artifact (corrupt): %s",
                ticker,
                model_name,
                horizon,
                exc,
            )
            return

        # Load meta if it exists
        meta: dict = {}
        if legacy_meta_path.exists():
            try:
                meta = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        version_id = self._store.migrate_legacy_artifact(
            ticker, model_name, horizon, feature_columns, meta
        )
        if version_id:
            logger.info(
                "[%s/%s/%s] Legacy artifact migrated → version %s",
                ticker,
                model_name,
                horizon,
                version_id,
            )
