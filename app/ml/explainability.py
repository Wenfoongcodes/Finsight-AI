"""
FinSight AI — Phase 6: Explainable AI Layer
Provides SHAP (global + local) and LIME explanations with natural language summaries.

Bug fixed in this revision
--------------------------
The previous ``generate_narrative()`` computed direction and probability from
``local_exp['prediction_probability']`` and ``local_exp['predicted_class']``,
which come from ``base_value + shap_row.sum()`` inside ``local_explanation()``.

This is correct for the *SHAP explanation* but **wrong for the narrative** when
the model is wrapped in ``CalibratedClassifierCV``:

* ``prediction_service.predict()`` calls ``calibrated_model.predict_proba()``
  which returns Platt-scaled probabilities.
* ``SHAPExplainer`` unwraps the calibration wrapper and runs ``TreeExplainer``
  on the raw base estimator — so ``expected_value`` and the SHAP values are in
  the **uncalibrated** probability space.
* ``base_value + shap_row.sum()`` can therefore be very different from the
  calibrated output, and even point in the opposite direction — producing the
  exact symptom seen in the log:

      Signal card:  BULLISH  (from calibrated model.predict_proba → 0.72)
      Narrative:    BEARISH  (from SHAP base_value + sum → 0.0 after clipping)

Fix: ``generate_narrative()`` now accepts two optional override parameters —
``authoritative_prediction`` (int: 0 or 1 from ``model.predict()``) and
``authoritative_p_bullish`` (float from ``model.predict_proba()``).  When
supplied (always the case from ``PredictionService``), these authoritative
calibrated values are used instead of the SHAP-internal ones, so the narrative
is always consistent with the signal card and probability display.

Threshold consistency
---------------------
``generate_narrative`` uses the same ``_CONFIDENCE_HIGH_DELTA = 0.15`` and
``_CONFIDENCE_MODERATE_DELTA = 0.05`` constants as ``prediction_service.py``
so the confidence label in the narrative matches the badge shown in the UI.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.exceptions import ExplainabilityError
from app.core.logging_config import get_logger

logger = get_logger("explainability")

# ── Shared confidence thresholds (must match prediction_service._confidence_label) ──
_CONFIDENCE_HIGH_DELTA     = 0.15
_CONFIDENCE_MODERATE_DELTA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    SHAP-based global and local explainability for tree and linear models.
    """

    def __init__(self, model: Any, feature_columns: list[str]) -> None:
        """
        Args:
            model: Trained sklearn-compatible model (may be Pipeline or
                   CalibratedClassifierCV wrapper).
            feature_columns: Ordered list of feature names.

        Raises:
            ImportError: If shap is not installed.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        self.model           = model
        self.feature_columns = feature_columns
        self._shap           = shap
        self._explainer: Optional[Any] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_underlying_model(self) -> Any:
        """
        Unwrap composite estimators to reach the raw base model.

        Priority:
        1. sklearn Pipeline  → unwrap via ``named_steps['clf']``
        2. CalibratedClassifierCV → unwrap via
           ``.calibrated_classifiers_[0].estimator``
        """
        clf = self.model

        if hasattr(clf, "named_steps"):
            clf = clf.named_steps.get("clf", clf)

        if hasattr(clf, "calibrated_classifiers_"):
            try:
                clf = clf.calibrated_classifiers_[0].estimator
                logger.debug(
                    "Unwrapped CalibratedClassifierCV → %s", type(clf).__name__
                )
            except (IndexError, AttributeError):
                clf = getattr(clf, "estimator", clf)

        return clf

    def _build_explainer(self, X_background: pd.DataFrame) -> None:
        """Lazy-initialize the SHAP explainer."""
        if self._explainer is not None:
            return

        clf        = self._get_underlying_model()
        model_type = type(clf).__name__.lower()

        try:
            if any(kw in model_type for kw in
                   ["xgb", "lgbm", "randomforest", "gradientboosting"]):
                self._explainer = self._shap.TreeExplainer(clf)
                logger.info("Using TreeExplainer for %s", type(clf).__name__)
            else:
                background  = self._shap.sample(
                    X_background, min(100, len(X_background))
                )
                predict_fn  = (
                    (lambda x: self.model.predict_proba(x)[:, 1])
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict
                )
                self._explainer = self._shap.KernelExplainer(predict_fn, background)
                logger.info("Using KernelExplainer for %s", type(clf).__name__)
        except Exception as exc:
            raise ExplainabilityError(
                f"Failed to build SHAP explainer: {exc}"
            ) from exc

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        Normalises across all SHAP version output formats:
        * Legacy list  ``[class0_arr, class1_arr]`` → take index [1].
        * Modern 3-D ndarray ``(n, features, classes)`` → take ``[…, 1]``.
        * 2-D ndarray ``(n, features)`` → use as-is.

        Args:
            X: Feature matrix.
            max_samples: Row cap for KernelExplainer tractability.

        Returns:
            2-D SHAP values array of shape ``(n_samples, n_features)``.
        """
        try:
            self._build_explainer(X)
            X_subset    = X.iloc[:max_samples] if len(X) > max_samples else X
            shap_values = self._explainer.shap_values(X_subset)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = np.array(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            if shap_values.ndim != 2:
                raise ExplainabilityError(
                    f"Unexpected SHAP values shape: {shap_values.shape}. "
                    "Expected 2-D (n_samples, n_features)."
                )
            return shap_values

        except ExplainabilityError:
            raise
        except Exception as exc:
            raise ExplainabilityError(
                f"SHAP value computation failed: {exc}"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def global_feature_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Global SHAP feature importance (mean |SHAP value|).

        Args:
            X: Feature matrix.
            top_n: Number of top features to return.

        Returns:
            DataFrame with ``'feature'`` and ``'mean_abs_shap'`` columns.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def local_explanation(
        self,
        X_instance: pd.DataFrame,
        top_n: int = 10,
    ) -> dict:
        """
        Local SHAP explanation for a single prediction instance.

        Important: the ``prediction_probability`` and ``predicted_class``
        fields returned here are computed from the raw (uncalibrated) SHAP
        values.  They are **not** used in ``generate_narrative()`` — the
        authoritative calibrated values from ``PredictionService`` are used
        instead to avoid the BULLISH/BEARISH contradiction bug.

        Args:
            X_instance: Single-row DataFrame (1 × n_features).
            top_n: Number of top contributing features to return.

        Returns:
            Dict with SHAP-internal base_value, prediction_probability,
            predicted_class, and top feature contributions.
        """
        shap_values = self.compute_shap_values(X_instance)

        # Resolve base value for the positive class
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr     = np.asarray(ev).ravel()
            base_value = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        else:
            base_value = float(ev)

        shap_row  = shap_values[0]
        pred_prob = float(np.clip(base_value + shap_row.sum(), 0.0, 1.0))

        feature_shap = sorted(
            zip(self.feature_columns, shap_row.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "base_value":             round(base_value, 4),
            "prediction_probability": round(pred_prob, 4),   # SHAP-internal, uncalibrated
            "predicted_class":        int(pred_prob >= 0.5), # SHAP-internal, do NOT use for display
            "top_features": [
                {
                    "feature":       f,
                    "shap_value":    round(v, 4),
                    "direction":     "bullish" if v > 0 else "bearish",
                    "feature_value": round(float(X_instance.iloc[0][f]), 4),
                }
                for f, v in feature_shap[:top_n]
            ],
        }

    def generate_narrative(
        self,
        local_exp: dict,
        ticker: str = "",
        authoritative_prediction: Optional[int] = None,
        authoritative_p_bullish: Optional[float] = None,
    ) -> str:
        """
        Convert a local SHAP explanation to a plain-English narrative.

        The ``direction`` and ``probability`` displayed in the narrative are
        derived from the **authoritative calibrated model outputs** supplied
        by ``PredictionService``, not from the SHAP-internal
        ``prediction_probability`` / ``predicted_class`` fields.

        This is essential because ``SHAPExplainer`` operates on the raw
        (unwrapped, uncalibrated) base estimator.  Its ``base_value +
        shap_row.sum()`` lives in uncalibrated probability space and can
        contradict the calibrated ``model.predict_proba()`` output — producing
        the bug where the signal card shows BULLISH but the narrative reads
        BEARISH at 0.0% probability.

        Args:
            local_exp: Output of ``local_explanation()``.
            ticker: Optional ticker symbol for context prefix.
            authoritative_prediction: ``int`` (0=bearish, 1=bullish) from
                ``model.predict()`` in ``PredictionService``.  When ``None``,
                falls back to the SHAP-internal ``predicted_class`` (legacy
                behaviour, kept for backward compatibility).
            authoritative_p_bullish: ``float`` P(bullish) from calibrated
                ``model.predict_proba()`` in ``PredictionService``.  When
                ``None``, falls back to SHAP-internal ``prediction_probability``.

        Returns:
            Human-readable prediction reasoning string.
        """
        # ── Use authoritative calibrated values when available ─────────────
        if authoritative_prediction is not None:
            pred   = authoritative_prediction
        else:
            pred   = local_exp["predicted_class"]

        if authoritative_p_bullish is not None:
            p_bull = authoritative_p_bullish
        else:
            # Fallback: SHAP internal value (may differ from calibrated output)
            p_bull = local_exp["prediction_probability"]

        direction = "BULLISH (UP)" if pred == 1 else "BEARISH (DOWN)"
        # Directional probability: confidence of the predicted direction
        prob      = p_bull if pred == 1 else (1.0 - p_bull)

        # Confidence label using the same thresholds as prediction_service
        delta = abs(p_bull - 0.5)
        if delta > _CONFIDENCE_HIGH_DELTA:
            confidence = "high"
        elif delta > _CONFIDENCE_MODERATE_DELTA:
            confidence = "moderate"
        else:
            confidence = "low"

        bullish_features = [
            f for f in local_exp["top_features"] if f["direction"] == "bullish"
        ][:3]
        bearish_features = [
            f for f in local_exp["top_features"] if f["direction"] == "bearish"
        ][:3]

        parts = [
            f"{'[' + ticker + '] ' if ticker else ''}"
            f"Prediction: {direction} "
            f"(Probability: {prob:.1%}, Confidence: {confidence})\n"
        ]

        if bullish_features:
            bull_str = ", ".join(
                f"{f['feature']} (+{f['shap_value']:.3f})"
                for f in bullish_features
            )
            parts.append(f"Bullish drivers: {bull_str}.")

        if bearish_features:
            bear_str = ", ".join(
                f"{f['feature']} ({f['shap_value']:.3f})"
                for f in bearish_features
            )
            parts.append(f"Bearish headwinds: {bear_str}.")

        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LIME Explainer
# ─────────────────────────────────────────────────────────────────────────────

class LIMEExplainer:
    """
    LIME local explainability for any sklearn-compatible classifier.
    Useful as a validation complement to SHAP.
    """

    def __init__(self, model: Any, feature_columns: list[str]) -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:
            raise ImportError("lime is required: pip install lime") from exc

        self.model                  = model
        self.feature_columns        = feature_columns
        self._LimeTabularExplainer  = LimeTabularExplainer
        self._explainer: Optional[Any] = None

    def fit(self, X_train: pd.DataFrame) -> "LIMEExplainer":
        """Initialise the LIME explainer from training data."""
        self._explainer = self._LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_columns,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        return self

    def explain_instance(
        self,
        X_instance: pd.DataFrame,
        num_features: int = 10,
    ) -> dict:
        """
        Generate LIME local explanation for a single instance.

        Args:
            X_instance: Single-row feature DataFrame.
            num_features: Number of features to include.

        Returns:
            Dict with feature_contributions and prediction probabilities.

        Raises:
            ExplainabilityError: If explainer not initialized or fails.
        """
        if self._explainer is None:
            raise ExplainabilityError(
                "Call .fit(X_train) before explain_instance()."
            )
        try:
            exp = self._explainer.explain_instance(
                X_instance.values[0],
                self.model.predict_proba,
                num_features=num_features,
            )
            probs = exp.predict_proba
            return {
                "prediction_probabilities": {
                    "bearish": round(float(probs[0]), 4),
                    "bullish": round(float(probs[1]), 4),
                },
                "feature_contributions": [
                    {"feature": feat, "weight": round(weight, 4)}
                    for feat, weight in exp.as_list()
                ],
            }
        except Exception as exc:
            raise ExplainabilityError(f"LIME explanation failed: {exc}") from exc