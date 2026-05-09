"""
FinSight AI — Phase 6: Explainable AI Layer
Provides SHAP (global + local) and LIME explanations with natural language summaries.
Supports tree-based and linear sklearn-compatible models.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.exceptions import ExplainabilityError
from app.core.logging_config import get_logger

logger = get_logger("explainability")


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

        self.model = model
        self.feature_columns = feature_columns
        self._shap = shap
        self._explainer: Optional[Any] = None

    def _get_underlying_model(self) -> Any:
        """
        Unwrap composite estimators to reach the raw base model.

        Handles two common wrappers in order:
        1. sklearn Pipeline  — unwrap via named_steps['clf']
        2. CalibratedClassifierCV — unwrap via .estimator (the base estimator
           before calibration).  SHAP TreeExplainer requires the original tree
           model, not the Platt-scaling wrapper around it.
        """
        clf = self.model

        # Unwrap sklearn Pipeline first
        if hasattr(clf, "named_steps"):
            clf = clf.named_steps.get("clf", clf)

        # Unwrap CalibratedClassifierCV to get the inner base estimator.
        # CalibratedClassifierCV stores the original unfitted estimator in
        # .estimator, and the fitted calibrated estimators in .calibrated_classifiers_.
        # For SHAP we need the fitted base model, which lives at:
        #   .calibrated_classifiers_[0].estimator
        if hasattr(clf, "calibrated_classifiers_"):
            try:
                clf = clf.calibrated_classifiers_[0].estimator
                logger.debug(
                    "Unwrapped CalibratedClassifierCV -> %s", type(clf).__name__
                )
            except (IndexError, AttributeError):
                # Fallback: use .estimator (the unfitted template) — less ideal
                clf = getattr(clf, "estimator", clf)

        return clf

    def _build_explainer(self, X_background: pd.DataFrame) -> None:
        """Lazy-initialize the SHAP explainer."""
        if self._explainer is not None:
            return

        clf = self._get_underlying_model()
        model_type = type(clf).__name__.lower()

        try:
            if any(kw in model_type for kw in ["xgb", "lgbm", "randomforest", "gradientboosting"]):
                self._explainer = self._shap.TreeExplainer(clf)
                logger.info("Using TreeExplainer for %s", type(clf).__name__)
            else:
                background = self._shap.sample(X_background, min(100, len(X_background)))
                predict_fn = (
                    lambda x: self.model.predict_proba(x)[:, 1]  # noqa: E731
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict(x)
                )
                self._explainer = self._shap.KernelExplainer(predict_fn, background)
                logger.info("Using KernelExplainer for %s", type(clf).__name__)
        except Exception as exc:
            raise ExplainabilityError(f"Failed to build SHAP explainer: {exc}") from exc

    def compute_shap_values(
        self, X: pd.DataFrame, max_samples: int = 500
    ) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        Handles three output shapes produced by different SHAP versions and model types:

        * Legacy SHAP + tree ensembles  -> list of 2-D arrays [class0, class1]
          -> take index [1] for positive class.
        * Modern SHAP >=0.40 + tree ensembles -> 3-D ndarray (n_samples, n_features, n_classes)
          -> take [..., 1] slice for positive class.
        * KernelExplainer / linear models  -> 2-D ndarray (n_samples, n_features)
          -> use as-is.

        Args:
            X: Feature matrix.
            max_samples: Row cap to keep KernelExplainer tractable.

        Returns:
            2-D SHAP values array of shape (n_samples, n_features).

        Raises:
            ExplainabilityError: On SHAP computation failure.
        """
        try:
            self._build_explainer(X)
            X_subset = X.iloc[:max_samples] if len(X) > max_samples else X
            shap_values = self._explainer.shap_values(X_subset)

            # ── Normalise to 2-D (n_samples, n_features) ──────────────────
            # Case 1: legacy list output [class0_arr, class1_arr]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = np.array(shap_values)

            # Case 2: modern 3-D ndarray (n_samples, n_features, n_classes)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            # Sanity-check — should now be exactly 2-D
            if shap_values.ndim != 2:
                raise ExplainabilityError(
                    f"Unexpected SHAP values shape after normalisation: {shap_values.shape}. "
                    "Expected 2-D (n_samples, n_features)."
                )

            return shap_values

        except ExplainabilityError:
            raise
        except Exception as exc:
            raise ExplainabilityError(f"SHAP value computation failed: {exc}") from exc

    def global_feature_importance(
        self, X: pd.DataFrame, top_n: int = 15
    ) -> pd.DataFrame:
        """
        Global SHAP feature importance (mean |SHAP value|).

        Args:
            X: Feature matrix.
            top_n: Number of top features to return.

        Returns:
            DataFrame with 'feature' and 'mean_abs_shap' columns.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_columns,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
        return df

    def local_explanation(
        self, X_instance: pd.DataFrame, top_n: int = 10
    ) -> dict:
        """
        Local SHAP explanation for a single prediction instance.

        Args:
            X_instance: Single-row DataFrame (1 x n_features).
            top_n: Number of top contributing features.

        Returns:
            Dict with base_value, prediction_probability, and top feature contributions.
        """
        shap_values = self.compute_shap_values(X_instance)

        # ── Resolve base value for positive class ──────────────────────────
        # TreeExplainer.expected_value shapes across SHAP versions:
        #   - scalar float         : binary KernelExplainer / single output
        #   - list [ev0, ev1]      : legacy tree multi-output
        #   - ndarray of shape (2,): modern tree multi-output (SHAP >=0.40)
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.asarray(ev).ravel()
            base_value = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        else:
            base_value = float(ev)

        # shap_values is guaranteed 2-D (n_samples, n_features) by compute_shap_values
        shap_row = shap_values[0]
        pred_prob = float(base_value + shap_row.sum())
        pred_prob = max(0.0, min(1.0, pred_prob))

        feature_shap = list(zip(self.feature_columns, shap_row.tolist()))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "base_value": round(base_value, 4),
            "prediction_probability": round(pred_prob, 4),
            "predicted_class": int(pred_prob >= 0.5),
            "top_features": [
                {
                    "feature": f,
                    "shap_value": round(v, 4),
                    "direction": "bullish" if v > 0 else "bearish",
                    "feature_value": round(float(X_instance.iloc[0][f]), 4),
                }
                for f, v in feature_shap[:top_n]
            ],
        }

    def generate_narrative(self, local_exp: dict, ticker: str = "") -> str:
        """
        Convert a local SHAP explanation to a plain-English narrative.

        Args:
            local_exp: Output of local_explanation().
            ticker: Optional ticker symbol for context.

        Returns:
            Human-readable prediction reasoning string.
        """
        prob = local_exp["prediction_probability"]
        direction = "BULLISH (UP)" if local_exp["predicted_class"] == 1 else "BEARISH (DOWN)"
        confidence = "high" if abs(prob - 0.5) > 0.2 else "moderate"

        bullish = [f for f in local_exp["top_features"] if f["direction"] == "bullish"][:3]
        bearish = [f for f in local_exp["top_features"] if f["direction"] == "bearish"][:3]

        parts = [
            f"{'[' + ticker + '] ' if ticker else ''}Prediction: {direction} "
            f"(Probability: {prob:.1%}, Confidence: {confidence})\n"
        ]

        if bullish:
            bull_str = ", ".join(
                f"{f['feature']} (+{f['shap_value']:.3f})" for f in bullish
            )
            parts.append(f"Bullish drivers: {bull_str}.")

        if bearish:
            bear_str = ", ".join(
                f"{f['feature']} ({f['shap_value']:.3f})" for f in bearish
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
        """
        Args:
            model: Trained sklearn-compatible model.
            feature_columns: Ordered list of feature names.

        Raises:
            ImportError: If lime is not installed.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:
            raise ImportError("lime is required: pip install lime") from exc

        self.model = model
        self.feature_columns = feature_columns
        self._LimeTabularExplainer = LimeTabularExplainer
        self._explainer: Optional[Any] = None

    def fit(self, X_train: pd.DataFrame) -> "LIMEExplainer":
        """
        Initialize the LIME explainer from training data.

        Args:
            X_train: Training feature matrix.

        Returns:
            Self (for chaining).
        """
        self._explainer = self._LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_columns,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        return self

    def explain_instance(
        self, X_instance: pd.DataFrame, num_features: int = 10
    ) -> dict:
        """
        Generate LIME local explanation for a single instance.

        Args:
            X_instance: Single-row feature DataFrame.
            num_features: Number of features to include in explanation.

        Returns:
            Dict with feature_contributions list and prediction probabilities.

        Raises:
            ExplainabilityError: If explainer not initialized or explanation fails.
        """
        if self._explainer is None:
            raise ExplainabilityError("Call .fit(X_train) before explain_instance().")

        try:
            predict_fn = self.model.predict_proba
            exp = self._explainer.explain_instance(
                X_instance.values[0],
                predict_fn,
                num_features=num_features,
            )

            contributions = [
                {"feature": feat, "weight": round(weight, 4)}
                for feat, weight in exp.as_list()
            ]

            probs = exp.predict_proba
            return {
                "prediction_probabilities": {
                    "bearish": round(float(probs[0]), 4),
                    "bullish": round(float(probs[1]), 4),
                },
                "feature_contributions": contributions,
            }
        except Exception as exc:
            raise ExplainabilityError(f"LIME explanation failed: {exc}") from exc