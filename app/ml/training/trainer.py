"""
FinSight AI — Enhanced Training Pipeline (v2)

Key improvements over v1
------------------------
1.  **Multi-horizon support** — ``ModelTrainer.train()`` accepts a
    ``horizon`` parameter ('1d', '7d', '1m', '6m').  Artifacts are keyed
    on ``{TICKER}_{model}_{horizon}`` so each horizon has independent
    model files, metadata, and leaderboard entries.

2.  **Automatic training trigger detection** — ``ModelTrainer.load_or_train()``
    detects missing, corrupted, and feature-incompatible artifacts and
    triggers a fresh training run automatically.  The caller never needs
    to handle ``ModelNotFoundError`` manually.

3.  **Improved calibration** — ``CalibratedClassifierCV`` wraps an *unfitted*
    estimator (no redundant pre-fit call), matching sklearn's intended API.

4.  **Structured training audit log** — every training run logs trigger
    reason, features used, duration, and fold metrics to a structured JSON
    audit entry alongside the model artifact.

5.  **Walk-forward splitter** — unchanged but documented more clearly.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

from app.core.exceptions import ModelNotFoundError, ModelTrainingError
from app.core.logging_config import get_logger
from app.ml.models.model_factory import get_model
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

    def compute_aggregates(self) -> None:
        if not self.fold_results:
            return
        self.mean_accuracy = float(np.mean([f.accuracy for f in self.fold_results]))
        self.mean_f1 = float(np.mean([f.f1 for f in self.fold_results]))
        self.mean_roc_auc = float(np.mean([f.roc_auc for f in self.fold_results]))
        self.mean_mae = float(np.mean([f.mae for f in self.fold_results]))
        self.mean_rmse = float(np.mean([f.rmse for f in self.fold_results]))

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "ticker": self.ticker,
            "horizon": self.horizon,
            "trained_at": self.trained_at,
            "n_features": self.n_features,
            "trigger_reason": self.trigger_reason,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_roc_auc": round(self.mean_roc_auc, 4),
            "mean_mae": round(self.mean_mae, 4),
            "mean_rmse": round(self.mean_rmse, 4),
            "training_duration_s": round(self.training_duration_s, 2),
            "best_params": self.best_params,
            "n_folds": len(self.fold_results),
            "feature_columns": self.feature_columns,
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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        ),
        "mae": float(mean_absolute_error(y_true, y_prob)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_prob))),
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
    Trains a model with walk-forward validation and persists artifacts.

    Multi-horizon support
    ---------------------
    Artifacts are named ``{TICKER}_{model}_{horizon}.pkl`` so each
    prediction horizon has fully independent parameters, features, and
    metrics.  The leaderboard (ModelSelector) reads per-horizon metadata.

    Automatic recovery
    ------------------
    ``load_or_train()`` checks:
    1. Does the artifact file exist?
    2. Is it unpicklable without error?
    3. Do its feature columns match the current feature set?

    If any check fails, it logs the trigger reason and trains a fresh model.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir or settings.MODELS_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ── Artifact paths ────────────────────────────────────────────────────────

    def _artifact_slug(self, ticker: str, model_name: str, horizon: str) -> str:
        return f"{ticker.upper()}_{model_name}_{horizon}"

    def _model_path(self, ticker: str, model_name: str, horizon: str) -> Path:
        return (
            self.model_dir / f"{self._artifact_slug(ticker, model_name, horizon)}.pkl"
        )

    def _meta_path(self, ticker: str, model_name: str, horizon: str) -> Path:
        return (
            self.model_dir
            / f"{self._artifact_slug(ticker, model_name, horizon)}_meta.json"
        )

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
    ) -> tuple[Any, list[str], TrainingResult | None]:
        """
        Load an existing model or train a new one automatically.

        Returns:
            (model, feature_columns, training_result_or_None)

        ``training_result`` is None when an existing artifact is reused.
        """
        trigger = self._detect_trigger(ticker, model_name, horizon, list(X.columns))

        if trigger is None:
            # Happy path — load existing artifact
            model, feature_columns = self.load_model(ticker, model_name, horizon)
            return model, feature_columns, None

        # Trigger detected — train fresh model
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
        )
        return model, result.feature_columns, result

    def _detect_trigger(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        current_feature_cols: list[str],
    ) -> Optional[str]:
        """
        Inspect the artifact and return a trigger reason string, or None
        if the artifact is valid and compatible.
        """
        path = self._model_path(ticker, model_name, horizon)

        if not path.exists():
            return TRIGGER_MISSING

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

        return None  # artifact is valid

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
    ) -> tuple[Any, TrainingResult]:
        """
        Train with walk-forward validation and persist artifacts.

        Args:
            model_name:     Registry model key.
            X:              Feature matrix (time-sorted).
            y:              Binary target series.
            ticker:         Ticker symbol.
            horizon:        Prediction horizon ('1d', '7d', '1m', '6m').
            hyperparams:    Fixed hyperparameters; takes priority over HPO.
            run_hpo:        Run Optuna when no fixed params supplied.
            hpo_trials:     HPO trial count.
            calibrate:      Apply cross-validated Platt scaling.
            trigger_reason: Audit log label for why training was triggered.

        Returns:
            (final_model, TrainingResult)
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
                trained_at=datetime.utcnow().isoformat(),
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

                result.fold_results.append(
                    FoldResult(
                        fold=fold_idx + 1,
                        train_size=len(X_tr),
                        test_size=len(X_te),
                        **metrics,
                    )
                )

                logger.info(
                    "[%s/%s/%s] Fold %d/%d — Acc=%.3f F1=%.3f AUC=%.3f",
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
                    base_estimator, cv=3, method="sigmoid"
                )
                final_model.fit(X, y)
            else:
                final_model = get_model(model_name, **best_params)
                final_model.fit(X, y)

            result.training_duration_s = time.perf_counter() - t_start
            self._save_artifacts(final_model, result, ticker, model_name, horizon)

            logger.info(
                "[%s/%s/%s] Training complete in %.1fs | "
                "Acc=%.3f F1=%.3f AUC=%.3f | trigger=%s",
                ticker,
                model_name,
                horizon,
                result.training_duration_s,
                result.mean_accuracy,
                result.mean_f1,
                result.mean_roc_auc,
                trigger_reason,
            )
            return final_model, result

        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(
                f"Training failed for {model_name}/{horizon}: {exc}"
            ) from exc

    def _save_artifacts(
        self,
        model: Any,
        result: TrainingResult,
        ticker: str,
        model_name: str,
        horizon: str,
    ) -> None:
        slug = self._artifact_slug(ticker, model_name, horizon)
        model_path = self.model_dir / f"{slug}.pkl"
        meta_path = self.model_dir / f"{slug}_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_columns": result.feature_columns,
                    "horizon": horizon,
                    "trained_at": result.trained_at,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(meta_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Artifacts saved: %s", model_path)

    def load_model(
        self,
        ticker: str,
        model_name: str,
        horizon: str = "1d",
    ) -> tuple[Any, list[str]]:
        """
        Load a persisted model artifact.

        Raises:
            ModelNotFoundError: If no artifact exists.
        """
        path = self._model_path(ticker, model_name, horizon)

        if not path.exists():
            raise ModelNotFoundError(
                f"No model artifact at {path}",
                detail=f"Train first: ticker={ticker}, model={model_name}, horizon={horizon}",
            )

        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Artifact corrupt at {path}: {exc}",
                detail="Delete the file and retrain.",
            ) from exc

        logger.info("Model loaded: %s", path)
        return artifact["model"], artifact["feature_columns"]
