"""
FinSight AI — Phase 4/5: Training Pipeline
Implements time-series-aware training with walk-forward validation,
Optuna hyperparameter optimization, and model artifact persistence.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
)

from app.core.exceptions import ModelTrainingError
from app.core.logging_config import get_logger
from app.ml.models.model_factory import get_model
from configs.settings import settings

logger = get_logger("training")


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""
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
    """Aggregated training result including walk-forward metrics."""
    model_name: str
    ticker: str
    trained_at: str
    n_features: int
    fold_results: list[FoldResult] = field(default_factory=list)
    mean_accuracy: float = 0.0
    mean_f1: float = 0.0
    mean_roc_auc: float = 0.0
    mean_mae: float = 0.0
    mean_rmse: float = 0.0
    best_params: dict = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def compute_aggregates(self) -> None:
        """Compute mean metrics across all folds."""
        if not self.fold_results:
            return
        self.mean_accuracy = np.mean([f.accuracy for f in self.fold_results])
        self.mean_f1 = np.mean([f.f1 for f in self.fold_results])
        self.mean_roc_auc = np.mean([f.roc_auc for f in self.fold_results])
        self.mean_mae = np.mean([f.mae for f in self.fold_results])
        self.mean_rmse = np.mean([f.rmse for f in self.fold_results])

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "ticker": self.ticker,
            "trained_at": self.trained_at,
            "n_features": self.n_features,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_roc_auc": round(self.mean_roc_auc, 4),
            "mean_mae": round(self.mean_mae, 4),
            "mean_rmse": round(self.mean_rmse, 4),
            "best_params": self.best_params,
            "n_folds": len(self.fold_results),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Splitter
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardSplitter:
    """
    Time-series-safe expanding window splitter.
    Produces (train_idx, test_idx) pairs with no future leakage.
    """

    def __init__(self, n_folds: int = settings.WALK_FORWARD_FOLDS, min_train_pct: float = 0.4) -> None:
        self.n_folds = n_folds
        self.min_train_pct = min_train_pct

    def split(self, X: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.

        Args:
            X: Feature DataFrame (must be sorted by time).

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(X)
        min_train = int(n * self.min_train_pct)
        fold_size = (n - min_train) // self.n_folds

        splits = []
        for fold in range(self.n_folds):
            train_end = min_train + fold * fold_size
            test_end = train_end + fold_size
            if test_end > n:
                test_end = n
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            if len(test_idx) == 0:
                break
            splits.append((train_idx, test_idx))

        logger.debug("Walk-forward splits: %d folds generated", len(splits))
        return splits


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute all classification and regression proxy metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dict of metric name -> value.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "mae": mean_absolute_error(y_true, y_prob),
        "rmse": np.sqrt(mean_squared_error(y_true, y_prob)),
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
    """
    Run Optuna hyperparameter search for a given model.

    Args:
        model_name: Name of model in registry.
        X: Feature matrix.
        y: Target series.
        n_trials: Number of Optuna trials.

    Returns:
        Best hyperparameter dict.

    Raises:
        ImportError: If optuna is not installed.
    """
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
        aucs = []
        for train_idx, test_idx in splits:
            model = get_model(model_name, **params)
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_te)[:, 1]
            if len(np.unique(y_te)) > 1:
                aucs.append(roc_auc_score(y_te, prob))
        return np.mean(aucs) if aucs else 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna best trial for %s: AUC=%.4f | params=%s",
        model_name, study.best_value, study.best_params,
    )
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Training Engine
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains a model using walk-forward validation and persists artifacts.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or settings.MODELS_DIR
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        ticker: str = "UNKNOWN",
        hyperparams: Optional[dict] = None,
        run_hpo: bool = False,
        hpo_trials: int = 30,
        calibrate: bool = True,
    ) -> tuple[Any, TrainingResult]:
        """
        Train a model with walk-forward validation.

        Args:
            model_name: Registry model name.
            X: Feature matrix (time-sorted).
            y: Binary target series.
            ticker: Ticker symbol for artifact naming.
            hyperparams: Fixed hyperparameter dict (overrides HPO).
            run_hpo: Whether to run Optuna HPO first.
            hpo_trials: Number of HPO trials.
            calibrate: Whether to apply probability calibration (Platt scaling)
                       to the final model before saving.

                       Tree ensembles (RF, XGBoost, LightGBM) produce
                       systematically extreme probabilities on near-50/50 targets
                       because leaf vote counts are not proper probability estimates.

                       We use CalibratedClassifierCV with cv=3 (cross-validated
                       calibration) rather than cv="prefit".  cv="prefit" fits the
                       sigmoid on the same data used to train the model, causing
                       severe overfitting of the calibration curve and pushing
                       all probabilities back to 0/1 — the opposite of the intent.
                       Cross-validated calibration uses held-out folds so the
                       sigmoid mapping is fitted on unseen scores.

                       method="sigmoid" (Platt scaling) is used for all tree models.
                       LogisticRegression is already a proper probability model so
                       calibration is automatically skipped for it.

        Returns:
            (final_calibrated_model, TrainingResult)

        Raises:
            ModelTrainingError: On training failure.
        """
        try:
            best_params = hyperparams or {}

            if run_hpo and not hyperparams:
                logger.info("Running HPO for %s on %s...", model_name, ticker)
                best_params = optimize_hyperparameters(model_name, X, y, n_trials=hpo_trials)

            splitter = WalkForwardSplitter()
            splits = splitter.split(X)

            result = TrainingResult(
                model_name=model_name,
                ticker=ticker,
                trained_at=datetime.utcnow().isoformat(),
                n_features=X.shape[1],
                best_params=best_params,
                feature_columns=list(X.columns),
            )

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                model = get_model(model_name, **best_params)
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)[:, 1]
                metrics = compute_metrics(y_te.values, y_pred, y_prob)

                fold_result = FoldResult(
                    fold=fold_idx + 1,
                    train_size=len(X_tr),
                    test_size=len(X_te),
                    **metrics,
                )
                result.fold_results.append(fold_result)

                logger.info(
                    "Fold %d/%d — Acc=%.3f F1=%.3f AUC=%.3f",
                    fold_idx + 1, len(splits),
                    metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
                )

            result.compute_aggregates()

            # ── Final model + probability calibration ──────────────────────
            # Step 1: Train the base model on the full dataset.
            final_model = get_model(model_name, **best_params)
            final_model.fit(X, y)

            # Step 2: Wrap in cross-validated Platt scaling when applicable.
            # cv=3 uses 3-fold cross-validation to fit the sigmoid mapping on
            # held-out scores, preventing the overfitting that cv="prefit" causes.
            # The saved artifact is the CalibratedClassifierCV wrapper so that
            # model.predict_proba() already returns calibrated probabilities.
            # SHAP unwraps this via calibrated_classifiers_[0].estimator.
            should_calibrate = calibrate and model_name != "logistic_regression"
            if should_calibrate:
                logger.info(
                    "Applying cross-validated Platt scaling (cv=3) to %s...", model_name
                )
                calibrated_model = CalibratedClassifierCV(
                    final_model, cv=3, method="sigmoid"
                )
                calibrated_model.fit(X, y)
                final_model = calibrated_model
                logger.info("Calibration complete.")

            self._save_artifacts(final_model, result, ticker, model_name)
            return final_model, result

        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(f"Training failed for {model_name}: {exc}") from exc

    def _save_artifacts(
        self,
        model: Any,
        result: TrainingResult,
        ticker: str,
        model_name: str,
    ) -> None:
        """Persist model pickle and training metadata JSON."""
        slug = f"{ticker}_{model_name}"
        model_path = Path(self.model_dir) / f"{slug}.pkl"
        meta_path = Path(self.model_dir) / f"{slug}_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump(
                {"model": model, "feature_columns": result.feature_columns},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(meta_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info("Model artifact saved: %s", model_path)

    def load_model(self, ticker: str, model_name: str) -> tuple[Any, list[str]]:
        """
        Load a persisted model and its feature column list.

        Args:
            ticker: Ticker symbol.
            model_name: Model name.

        Returns:
            (model, feature_columns)

        Raises:
            ModelNotFoundError: If artifact not found.
        """
        from app.core.exceptions import ModelNotFoundError

        slug = f"{ticker}_{model_name}"
        model_path = Path(self.model_dir) / f"{slug}.pkl"

        if not model_path.exists():
            raise ModelNotFoundError(
                f"No model artifact at {model_path}",
                detail=f"Train a model first: ticker={ticker}, model={model_name}",
            )

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        logger.info("Model loaded: %s", model_path)
        return artifact["model"], artifact["feature_columns"]