"""
FinSight AI — Prediction Service (v3 — fixed)

Bug fixed vs v3
---------------
``selector.select()`` now returns a ``SelectionResult`` named-tuple, not a
bare string.  This version consumes it correctly:

  - Extracts ``.model_name`` to pass to the trainer.
  - Inspects ``.reason`` to log at the right level and set
    ``confidence_degraded`` on the response when the model was selected
    despite being below MIN_AUC or because no artifacts existed.

No other logic changes vs v3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.exceptions import PredictionError
from app.core.logging_config import get_logger
from app.ml.data_ingestion import ingest_market_data
from app.ml.explainability import SHAPExplainer
from app.ml.feature_engineering import FeatureEngineer, FeatureSelector, HORIZONS
from app.ml.training.trainer import ModelTrainer
from app.services.model_selector import (
    ModelSelector,
    REASON_NO_ARTIFACTS,
    REASON_BELOW_THRESHOLD,
    REASON_LEADERBOARD,
)
from app.services.signal_fusion import FusedSignal, SignalFusionService
from app.services.news_intelligence import IntelligenceBrief
from configs.settings import settings

logger = get_logger("prediction_service")


# ─────────────────────────────────────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResponse:
    """
    Full prediction response.

    Fields
    ------
    ticker              : Stock ticker symbol.
    model_name          : Auto-selected model name.
    horizon             : Prediction horizon ('1d', '7d', '1m', '6m').
    selection_reason    : Why this model was selected (REASON_* constant).
    confidence_degraded : True when model was below MIN_AUC or untrained.
    prediction          : 0 = bearish, 1 = bullish.
    probability         : P(predicted direction).
    p_bullish           : Calibrated P(bullish).
    p_bearish           : 1 - p_bullish.
    confidence_label    : 'high' | 'moderate' | 'low' (ML-only).
    shap_explanation    : SHAP local explanation dict.
    narrative           : Plain-English SHAP narrative.
    latest_close        : Most recent closing price.
    feature_snapshot    : Top-20 feature values at inference time.
    fused_signal        : FusedSignal from news + LLM (or None).
    intelligence_brief  : Raw IntelligenceBrief from news retrieval (or None).
    auto_trained        : True if the model was trained automatically this call.
    """
    ticker:              str
    model_name:          str
    horizon:             str
    selection_reason:    str
    confidence_degraded: bool
    prediction:          int
    probability:         float
    p_bullish:           float
    p_bearish:           float
    confidence_label:    str
    shap_explanation:    dict
    narrative:           str
    latest_close:        float
    feature_snapshot:    dict
    fused_signal:        Optional[FusedSignal] = None
    intelligence_brief:  Optional[IntelligenceBrief] = None
    auto_trained:        bool = False


def _confidence_label(p_bullish: float) -> str:
    delta = abs(p_bullish - 0.5)
    if delta > 0.15:
        return "high"
    if delta > 0.05:
        return "moderate"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class PredictionService:
    """
    End-to-end prediction pipeline:
      ModelSelector → ModelTrainer.load_or_train → SHAP → SignalFusion
    """

    def __init__(self) -> None:
        self.trainer        = ModelTrainer()
        self.engineer       = FeatureEngineer()
        self.selector       = ModelSelector()
        self.fusion_service = SignalFusionService()

    # ── Main entry point ──────────────────────────────────────────────────────

    def predict(
        self,
        ticker: str,
        horizon: str = "1d",
        use_cache: bool = True,
        run_fusion: bool = True,
        apply_feature_selection: bool = False,
    ) -> PredictionResponse:
        """
        Run the full prediction pipeline for ticker/horizon.

        Args:
            ticker:                  Stock ticker symbol (case-insensitive).
            horizon:                 '1d' | '7d' | '1m' | '6m'.
            use_cache:               Use cached market data parquet.
            run_fusion:              Run news + LLM signal fusion.
            apply_feature_selection: Run FeatureSelector before inference.

        Returns:
            PredictionResponse.

        Raises:
            PredictionError: On any unrecoverable pipeline failure.
        """
        ticker  = ticker.upper().strip()
        horizon = horizon.strip()

        if horizon not in HORIZONS:
            raise PredictionError(
                f"Invalid horizon '{horizon}'. Valid options: {list(HORIZONS.keys())}"
            )

        try:
            logger.info(
                "Prediction pipeline started: ticker=%s horizon=%s", ticker, horizon
            )

            # ── 1. Ingest ─────────────────────────────────────────────────────
            raw_df = ingest_market_data(ticker, use_cache=use_cache)

            # ── 2. Feature engineering ────────────────────────────────────────
            feature_df = self.engineer.build_features(raw_df)
            X, y       = self.engineer.split_X_y(feature_df, horizon=horizon)

            # ── 3. Optional feature selection ─────────────────────────────────
            if apply_feature_selection:
                fs = FeatureSelector()
                X  = fs.fit_transform(X, y)
                logger.info(
                    "[%s/%s] FeatureSelector: %d features retained",
                    ticker, horizon, X.shape[1],
                )

            # ── 4. Model selection — explicit reason handling ─────────────────
            sel = self.selector.select(ticker, horizon=horizon)

            # Log at the right level per selection case
            if sel.reason == REASON_NO_ARTIFACTS:
                logger.warning(
                    "[%s/%s] No existing artifacts — will auto-train '%s'.",
                    ticker, horizon, sel.model_name,
                )
            elif sel.reason == REASON_BELOW_THRESHOLD:
                logger.warning(
                    "[%s/%s] Using '%s' despite AUC=%.4f < MIN_AUC. "
                    "Confidence degraded.",
                    ticker, horizon, sel.model_name, sel.auc,
                )
            else:
                logger.info(
                    "[%s/%s] Auto-selected: '%s' (AUC=%.4f).",
                    ticker, horizon, sel.model_name, sel.auc,
                )

            confidence_degraded = sel.reason != REASON_LEADERBOARD

            # ── 5. Load or auto-train ─────────────────────────────────────────
            model, feature_columns, train_result = self.trainer.load_or_train(
                ticker=ticker,
                model_name=sel.model_name,
                X=X,
                y=y,
                horizon=horizon,
            )
            auto_trained = train_result is not None

            if auto_trained:
                logger.info(
                    "[%s/%s] Auto-training complete in %.1fs: "
                    "AUC=%.3f trigger='%s'",
                    ticker, horizon,
                    train_result.training_duration_s,
                    train_result.mean_roc_auc,
                    train_result.trigger_reason,
                )

            # ── 6. Align feature columns ──────────────────────────────────────
            missing = set(feature_columns) - set(X.columns)
            if missing:
                raise PredictionError(
                    f"Feature mismatch for {ticker}/{sel.model_name}/{horizon}",
                    detail=f"Missing columns: {sorted(missing)}",
                )
            X_aligned = X[feature_columns]

            # ── 7. Inference ──────────────────────────────────────────────────
            X_latest  = X_aligned.iloc[[-1]]
            pred      = int(model.predict(X_latest)[0])
            p_bullish = round(float(model.predict_proba(X_latest)[0, 1]), 4)
            p_bearish = round(1.0 - p_bullish, 4)
            prob      = p_bullish if pred == 1 else p_bearish

            # ── 8. SHAP explanation ───────────────────────────────────────────
            explainer = SHAPExplainer(model, feature_columns)
            shap_exp  = explainer.local_explanation(X_latest)

            # ── 9. Narrative ──────────────────────────────────────────────────
            narrative = explainer.generate_narrative(
                shap_exp,
                ticker=ticker,
                authoritative_prediction=pred,
                authoritative_p_bullish=p_bullish,
            )

            # ── 10. Feature snapshot (top 20 for display) ─────────────────────
            snapshot = {
                col: round(float(X_latest.iloc[0][col]), 4)
                for col in feature_columns[:20]
            }

            latest_close = float(raw_df["Close"].iloc[-1])

            response = PredictionResponse(
                ticker=ticker,
                model_name=sel.model_name,
                horizon=horizon,
                selection_reason=sel.reason,
                confidence_degraded=confidence_degraded,
                prediction=pred,
                probability=round(prob, 4),
                p_bullish=p_bullish,
                p_bearish=p_bearish,
                confidence_label=_confidence_label(p_bullish),
                shap_explanation=shap_exp,
                narrative=narrative,
                latest_close=round(latest_close, 4),
                feature_snapshot=snapshot,
                fused_signal=None,
                intelligence_brief=None,
                auto_trained=auto_trained,
            )

            # ── 11. Signal fusion (best-effort — never blocks ML result) ──────
            if run_fusion and settings.OPENAI_API_KEY:
                try:
                    fused = self.fusion_service.fuse(ticker, response)
                    response.fused_signal       = fused
                    response.intelligence_brief = getattr(
                        fused, "intelligence_brief", None
                    )
                    logger.info(
                        "[%s/%s] Fusion: ML=%s → fused=%s (applied=%s)",
                        ticker, horizon,
                        "BULLISH" if pred else "BEARISH",
                        fused.final_direction,
                        fused.fusion_applied,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s/%s] Signal fusion skipped (non-fatal): %s",
                        ticker, horizon, exc,
                    )
            else:
                logger.info(
                    "[%s/%s] Fusion skipped (run_fusion=%s, api_key_set=%s)",
                    ticker, horizon, run_fusion, bool(settings.OPENAI_API_KEY),
                )

            logger.info(
                "Prediction complete: %s/%s → %s "
                "(p_bull=%.3f conf=%s model=%s reason=%s auto_trained=%s)",
                ticker, horizon,
                "BULLISH" if pred else "BEARISH",
                p_bullish,
                response.confidence_label,
                sel.model_name,
                sel.reason,
                auto_trained,
            )
            return response

        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(
                f"Prediction pipeline failed for {ticker}/{horizon}: {exc}"
            ) from exc

    # ── Batch ─────────────────────────────────────────────────────────────────

    def batch_predict(
        self,
        tickers: list[str],
        horizon: str = "1d",
        run_fusion: bool = True,
    ) -> dict[str, PredictionResponse | str]:
        """Predict for multiple tickers; capture per-ticker failures."""
        results: dict[str, PredictionResponse | str] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.predict(
                    ticker, horizon=horizon, run_fusion=run_fusion
                )
            except Exception as exc:
                logger.warning("Batch prediction failed for %s: %s", ticker, exc)
                results[ticker] = str(exc)
        return results