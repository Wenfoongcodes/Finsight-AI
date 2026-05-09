"""
FinSight AI — Phase 5: Model Evaluation & Validation
Comprehensive evaluation suite: classification report, calibration,
feature importance, and cross-model comparison utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from app.core.logging_config import get_logger

logger = get_logger("evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> dict:
    """
    Produce a comprehensive evaluation report for a classifier.

    Args:
        model: Trained sklearn-compatible classifier.
        X_test: Test feature matrix.
        y_test: True binary labels.
        model_name: Identifier for logging.

    Returns:
        Dict containing all evaluation metrics and curve data.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    # ── Core metrics ──────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    ll = log_loss(y_true, y_prob)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ── ROC Curve ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # ── Precision-Recall Curve ───────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # ── Calibration Curve ─────────────────────────────────────────────────────
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    # ── Classification Report ────────────────────────────────────────────────
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    report = {
        "model_name": model_name,
        "n_samples": len(y_true),
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "log_loss": round(ll, 4),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
        "calibration": {
            "fraction_positives": fraction_of_positives.tolist(),
            "mean_predicted": mean_predicted_value.tolist(),
        },
        "classification_report": clf_report,
    }

    logger.info(
        "[%s] Acc=%.3f | F1=%.3f | AUC=%.3f | LogLoss=%.3f",
        model_name, acc, f1, auc, ll,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importance(
    model: Any,
    feature_columns: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based or linear model.

    Supports: RandomForest, XGBoost, LightGBM, LogisticRegression (via coef_).

    Args:
        model: Trained model (may be wrapped in sklearn Pipeline).
        feature_columns: Ordered list of feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted descending.
    """
    # Unwrap Pipeline
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    importances: np.ndarray | None = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    if importances is None:
        logger.warning("Model %s does not expose feature importances.", type(clf).__name__)
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Report
# ─────────────────────────────────────────────────────────────────────────────

def summarize_walk_forward(fold_results: list) -> pd.DataFrame:
    """
    Convert a list of FoldResult objects into a readable DataFrame.

    Args:
        fold_results: List of FoldResult dataclass instances.

    Returns:
        Summary DataFrame indexed by fold.
    """
    rows = []
    for fr in fold_results:
        rows.append({
            "fold": fr.fold,
            "train_size": fr.train_size,
            "test_size": fr.test_size,
            "accuracy": round(fr.accuracy, 4),
            "f1": round(fr.f1, 4),
            "roc_auc": round(fr.roc_auc, 4),
            "mae": round(fr.mae, 4),
            "rmse": round(fr.rmse, 4),
        })
    return pd.DataFrame(rows).set_index("fold")


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate multiple models and return a side-by-side comparison table.

    Args:
        models: Dict of model_name → trained model.
        X_test: Shared test feature matrix.
        y_test: True binary labels.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    records = []
    for name, model in models.items():
        report = evaluate_model(model, X_test, y_test, model_name=name)
        row = {"model": name}
        row.update(report["metrics"])
        records.append(row)

    comparison = pd.DataFrame(records).set_index("model")
    comparison.sort_values("roc_auc", ascending=False, inplace=True)
    return comparison
