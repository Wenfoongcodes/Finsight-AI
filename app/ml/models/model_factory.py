"""
FinSight AI — Phase 4: ML Model Definitions
Defines baseline and advanced model factories with consistent sklearn-compatible API.
Supports: Logistic Regression, Random Forest, XGBoost, LightGBM, LSTM (Keras optional).
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("models")


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, Any] = {}


def register_model(name: str):
    """Decorator to register a model factory function."""

    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Models
# ─────────────────────────────────────────────────────────────────────────────


@register_model("logistic_regression")
def build_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str = "balanced",
    **kwargs,
) -> Pipeline:
    """
    Logistic Regression with StandardScaler preprocessing.

    Logistic regression is a proper probability model so class_weight="balanced"
    is kept here — it adjusts the decision boundary without distorting the
    probability output the way tree ensembles do. Platt scaling is skipped
    for this model at training time.

    Args:
        C: Inverse regularization strength.
        max_iter: Solver iteration limit.
        class_weight: Class weighting strategy.

    Returns:
        sklearn Pipeline.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    random_state=settings.RANDOM_SEED,
                    solver="lbfgs",
                    **kwargs,
                ),
            ),
        ]
    )


@register_model("random_forest")
def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 10,
    **kwargs,
) -> RandomForestClassifier:
    """
    Random Forest Classifier.

    class_weight is intentionally omitted. On near-50/50 targets like
    next-day stock direction, class_weight="balanced" reweights the loss
    to equalise classes which systematically pushes predict_proba output
    toward 0 and 1, producing miscalibrated and misleading confidence scores.
    Probability calibration via Platt scaling (CalibratedClassifierCV) is
    applied at training time in ModelTrainer instead.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (None = fully grown).
        min_samples_split: Minimum samples required to split a node.

    Returns:
        RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Models
# ─────────────────────────────────────────────────────────────────────────────


@register_model("xgboost")
def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    use_label_encoder: bool = False,
    **kwargs,
) -> Any:
    """
    XGBoost Classifier.

    Args:
        n_estimators: Boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        subsample: Row subsampling ratio.
        colsample_bytree: Feature subsampling ratio per tree.

    Returns:
        XGBClassifier instance.

    Raises:
        ImportError: If xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required: pip install xgboost") from exc

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=settings.RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
        **kwargs,
    )


@register_model("lightgbm")
def build_lightgbm(
    n_estimators: int = 300,
    max_depth: int = -1,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    **kwargs,
) -> Any:
    """
    LightGBM Classifier.

    Args:
        n_estimators: Boosting rounds.
        max_depth: Maximum tree depth (-1 = no limit).
        learning_rate: Step size shrinkage.
        num_leaves: Maximum number of leaves per tree.
        subsample: Row subsampling ratio.
        colsample_bytree: Feature subsampling ratio per tree.

    Returns:
        LGBMClassifier instance.

    Raises:
        ImportError: If lightgbm is not installed.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("lightgbm is required: pip install lightgbm") from exc

    return LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional Deep Learning — LSTM
# ─────────────────────────────────────────────────────────────────────────────


def build_lstm(
    input_shape: tuple[int, int],
    units: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> Any:
    """
    LSTM Classifier using Keras (TensorFlow backend).

    Args:
        input_shape: (sequence_length, n_features).
        units: LSTM hidden units.
        dropout: Dropout rate.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Sequential model.

    Raises:
        ImportError: If tensorflow/keras is not installed.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import (
            LSTM,
            BatchNormalization,
            Dense,
            Dropout,
            Input,
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise ImportError("tensorflow is required: pip install tensorflow") from exc

    tf.random.set_seed(settings.RANDOM_SEED)

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout),
            LSTM(units // 2),
            BatchNormalization(),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dropout(dropout / 2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    logger.info(
        "LSTM model built: input_shape=%s, params=%d", input_shape, model.count_params()
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────


def get_model(name: str, **hyperparams) -> Any:
    """
    Retrieve a model from the registry by name.

    Args:
        name: Model name key (e.g. 'xgboost', 'lightgbm').
        **hyperparams: Keyword arguments forwarded to the builder function.

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If model name is not in registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    logger.info("Building model: %s | params: %s", name, hyperparams)
    return MODEL_REGISTRY[name](**hyperparams)


def list_models() -> list[str]:
    """Return list of all registered model names."""
    return list(MODEL_REGISTRY.keys())
