from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.exceptions import ExplainabilityError
from app.core.formatting import (
    build_prediction_narrative,
    confidence_label,
    round_prob,
    round_shap,
)
from app.core.logging_config import get_logger

logger = get_logger("explainability")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer
# ─────────────────────────────────────────────────────────────────────────────


class SHAPExplainer:
    """
    SHAP-based global and local explainability for tree and linear models.
    """

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        X_background: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Args:
            model:           Trained sklearn-compatible model (may be Pipeline or
                             CalibratedClassifierCV wrapper).
            feature_columns: Ordered list of feature names.
            X_background:    Optional full feature matrix used as background for
                             explainer initialisation.  When supplied, KernelExplainer
                             samples from this distribution and TreeExplainer uses it
                             to anchor the expected_value baseline.  When ``None``,
                             the data passed to ``compute_shap_values`` is used as
                             the fallback background (legacy behaviour).
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        self.model = model
        self.feature_columns = feature_columns
        self.X_background = X_background
        self._shap = shap
        self._explainer: Optional[Any] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_underlying_model(self) -> Any:
        """
        Unwrap composite estimators to reach the raw base model.

        Priority:
        1. sklearn Pipeline  → ``named_steps['clf']``
        2. CalibratedClassifierCV → ``.calibrated_classifiers_[0].estimator``
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

    def _build_explainer(self, X_fallback: pd.DataFrame) -> None:
        """
        Lazy-initialise the SHAP explainer.

        Background priority:
        1. ``self.X_background`` (full matrix supplied at construction).
        2. ``X_fallback``        (data passed to ``compute_shap_values``).
        """
        if self._explainer is not None:
            return

        background = self.X_background if self.X_background is not None else X_fallback
        clf = self._get_underlying_model()
        model_type = type(clf).__name__.lower()

        try:
            if any(
                kw in model_type
                for kw in ["xgb", "lgbm", "randomforest", "gradientboosting"]
            ):
                self._explainer = self._shap.TreeExplainer(clf, data=background)
                logger.info("Using TreeExplainer for %s", type(clf).__name__)
            else:
                bg_sample = self._shap.sample(background, min(100, len(background)))
                predict_fn = (
                    (lambda x: self.model.predict_proba(x)[:, 1])
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict
                )
                self._explainer = self._shap.KernelExplainer(predict_fn, bg_sample)
                logger.info("Using KernelExplainer for %s", type(clf).__name__)
        except Exception as exc:
            raise ExplainabilityError(f"Failed to build SHAP explainer: {exc}") from exc

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
            X:           Feature matrix.
            max_samples: Row cap for KernelExplainer tractability.

        Returns:
            2-D SHAP values array of shape ``(n_samples, n_features)``.
        """
        try:
            self._build_explainer(X)
            X_subset = X.iloc[:max_samples] if len(X) > max_samples else X
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
            raise ExplainabilityError(f"SHAP value computation failed: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def global_feature_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Global SHAP feature importance (mean |SHAP value|).

        Args:
            X:     Feature matrix.
            top_n: Number of top features to return.

        Returns:
            DataFrame with ``'feature'`` and ``'mean_abs_shap'`` columns.
            ``mean_abs_shap`` values are rounded via ``round_shap()``.
        """
        shap_values = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        df = (
            pd.DataFrame({"feature": self.feature_columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        # Apply canonical rounding
        df["mean_abs_shap"] = df["mean_abs_shap"].map(round_shap)
        return df

    def local_explanation(
        self,
        X_instance: pd.DataFrame,
        top_n: int = 10,
    ) -> dict:
        """
        Local SHAP explanation for a single prediction instance.

        All numeric values use canonical rounding:
        - ``base_value``              → ``round_prob()``  (4 d.p.)
        - ``prediction_probability``  → ``round_prob()``  (4 d.p.)
        - ``shap_value``              → ``round_shap()``  (4 d.p.)
        - ``feature_value``           → ``round_shap()``  (4 d.p.)

        The ``prediction_probability`` and ``predicted_class`` fields are
        SHAP-internal (uncalibrated) and are NOT used in
        ``generate_narrative()`` — the authoritative calibrated values from
        ``PredictionService`` override them.

        Args:
            X_instance: Single-row DataFrame (1 × n_features).
            top_n:      Number of top contributing features to return.

        Returns:
            Dict with SHAP-internal base_value, prediction_probability,
            predicted_class, and top feature contributions.
        """
        shap_values = self.compute_shap_values(X_instance)

        # Resolve base value for the positive class
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.asarray(ev).ravel()
            base_value = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        else:
            base_value = float(ev)

        shap_row = shap_values[0]
        pred_prob_raw = float(np.clip(base_value + shap_row.sum(), 0.0, 1.0))

        feature_shap = sorted(
            zip(self.feature_columns, shap_row.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "base_value": round_prob(base_value),
            "prediction_probability": round_prob(
                pred_prob_raw
            ),  # SHAP-internal, uncalibrated
            "predicted_class": int(
                pred_prob_raw >= 0.5
            ),  # SHAP-internal, do NOT use for display
            "top_features": [
                {
                    "feature": f,
                    "shap_value": round_shap(v),
                    "direction": "bullish" if v > 0 else "bearish",
                    "feature_value": round_shap(float(X_instance.iloc[0][f])),
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

        Delegates entirely to ``build_prediction_narrative()`` from
        ``app.core.formatting`` — the canonical, single implementation.

        The ``direction`` and ``probability`` in the narrative are derived from
        the **authoritative calibrated model outputs** supplied by
        ``PredictionService``, not from the SHAP-internal values.

        Args:
            local_exp:                Output of ``local_explanation()``.
            ticker:                   Optional ticker symbol for context prefix.
            authoritative_prediction: ``int`` (0=bearish, 1=bullish) from
                                      ``model.predict()`` in ``PredictionService``.
            authoritative_p_bullish:  ``float`` P(bullish) from calibrated
                                      ``model.predict_proba()`` in ``PredictionService``.

        Returns:
            Human-readable prediction reasoning string.
        """
        pred = (
            authoritative_prediction
            if authoritative_prediction is not None
            else local_exp["predicted_class"]
        )
        p_bull = (
            authoritative_p_bullish
            if authoritative_p_bullish is not None
            else local_exp["prediction_probability"]
        )

        direction = "BULLISH (UP)" if pred == 1 else "BEARISH (DOWN)"
        # Probability shown is the probability of the predicted direction
        prob = p_bull if pred == 1 else (1.0 - p_bull)
        conf = confidence_label(p_bull)  # single shared implementation

        bullish_features = [
            f for f in local_exp["top_features"] if f["direction"] == "bullish"
        ][:3]
        bearish_features = [
            f for f in local_exp["top_features"] if f["direction"] == "bearish"
        ][:3]

        return build_prediction_narrative(
            ticker=ticker,
            direction=direction,
            prob=prob,
            confidence=conf,
            bullish_features=bullish_features,
            bearish_features=bearish_features,
        )


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

        self.model = model
        self.feature_columns = feature_columns
        self._LimeTabularExplainer = LimeTabularExplainer
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

        Probabilities are rounded via ``round_prob()`` for consistency
        with the SHAP explainer output schema.

        Args:
            X_instance:   Single-row feature DataFrame.
            num_features: Number of features to include.

        Returns:
            Dict with feature_contributions and prediction probabilities.
        """
        if self._explainer is None:
            raise ExplainabilityError("Call .fit(X_train) before explain_instance().")
        try:
            exp = self._explainer.explain_instance(
                X_instance.values[0],
                self.model.predict_proba,
                num_features=num_features,
            )
            probs = exp.predict_proba
            return {
                "prediction_probabilities": {
                    "bearish": round_prob(float(probs[0])),
                    "bullish": round_prob(float(probs[1])),
                },
                "feature_contributions": [
                    {"feature": feat, "weight": round_shap(weight)}
                    for feat, weight in exp.as_list()
                ],
            }
        except Exception as exc:
            raise ExplainabilityError(f"LIME explanation failed: {exc}") from exc
