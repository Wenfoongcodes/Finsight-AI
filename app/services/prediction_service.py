"""
FinSight AI — Prediction Service (v6)

Changes vs v5
-------------
* SHAP explainer now receives the full aligned feature matrix (``X_aligned``)
  as background data, not just the single inference row (``X_latest``).

  Root cause of the "SHAP chart empty for custom tickers" bug:
  ``SHAPExplainer._build_explainer()`` received ``X_instance`` (a 1-row
  DataFrame) as its background dataset.  For TreeExplainer this is harmless
  on pre-trained tickers whose model was already cached, but for custom
  tickers where ``load_or_train`` triggers a fresh training run the model
  may be a ``CalibratedClassifierCV`` wrapper.  When the explainer correctly
  unwraps to the base estimator and falls through to ``KernelExplainer``,
  ``shap.sample(X_background, min(100, 1))`` returns a 1-row background.
  SHAP's KernelExplainer with a 1-row background produces all-zero SHAP
  values (no variance to attribute) — so ``top_features`` comes back empty
  and the chart renders nothing.

  Fix: ``SHAPExplainer`` is constructed with ``X_aligned`` (all rows,
  capped internally by ``max_samples=500`` in ``compute_shap_values``).
  ``local_explanation`` still receives only ``X_latest`` for inference.

* All other logic (multi-model training on no-artifacts, narrative,
  batch predict, signal fusion) is unchanged from v5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.exceptions import PredictionError
from app.core.logging_config import get_logger
from app.ml.data_ingestion import ingest_market_data
from app.ml.explainability import SHAPExplainer
from app.ml.feature_engineering import FeatureEngineer, FeatureSelector, HORIZONS
from app.ml.training.trainer import ModelTrainer
from app.services.model_selector import (
    ModelSelector,
    ALL_TRAINING_MODELS,
    REASON_NO_ARTIFACTS,
    REASON_BELOW_THRESHOLD,
    REASON_LEADERBOARD,
)
from app.services.signal_fusion import FusedSignal, SignalFusionService
from configs.settings import settings

logger = get_logger("prediction_service")


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PredictionResponse:
    """
    Full prediction response.

    Fields
    ------
    ticker              : Stock ticker symbol.
    model_name          : Auto-selected model name.
    horizon             : '1d' | '7d' | '1m' | '6m'.
    selection_reason    : REASON_* constant from ModelSelector.
    confidence_degraded : True when no artifact existed or AUC < MIN_AUC.
    prediction          : 0 = bearish, 1 = bullish.
    probability         : P(predicted direction).
    p_bullish           : Calibrated P(bullish).
    p_bearish           : 1 - p_bullish.
    confidence_label    : 'high' | 'moderate' | 'low'.
    shap_explanation    : SHAP local explanation dict.
    narrative           : Plain-English summary.
    latest_close        : Most recent close price.
    feature_snapshot    : Top-20 feature values at inference time.
    fused_signal        : FusedSignal (None when fusion was skipped).
    auto_trained        : True when any training occurred this call.
    """

    ticker: str
    model_name: str
    horizon: str
    selection_reason: str
    confidence_degraded: bool
    prediction: int
    probability: float
    p_bullish: float
    p_bearish: float
    confidence_label: str
    shap_explanation: dict
    narrative: str
    latest_close: float
    feature_snapshot: dict
    fused_signal: Optional[FusedSignal] = None
    auto_trained: bool = False


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
      ModelSelector → (train all models if needed) → best model → SHAP → SignalFusion
    """

    def __init__(self) -> None:
        self.trainer = ModelTrainer()
        self.engineer = FeatureEngineer()
        self.selector = ModelSelector()
        self.fusion_service = SignalFusionService()

    def predict(
        self,
        ticker: str,
        horizon: str = "1d",
        use_cache: bool = True,
        run_fusion: bool = True,
        apply_feature_selection: bool = False,
    ) -> PredictionResponse:
        """
        Run the full prediction pipeline for *ticker* / *horizon*.

        When no artifacts exist the service trains all models in
        ALL_TRAINING_MODELS before selecting the best one by AUC.

        Raises:
            PredictionError: On any unrecoverable failure.
        """
        ticker = ticker.upper().strip()
        horizon = horizon.strip()

        if horizon not in HORIZONS:
            raise PredictionError(
                f"Invalid horizon '{horizon}'. Valid: {list(HORIZONS.keys())}"
            )

        try:
            logger.info("Prediction pipeline: ticker=%s horizon=%s", ticker, horizon)

            # ── 1. Ingest ─────────────────────────────────────────────────────
            raw_df = ingest_market_data(ticker, use_cache=use_cache)

            # ── 2. Feature engineering ────────────────────────────────────────
            feature_df = self.engineer.build_features(raw_df)
            X, y = self.engineer.split_X_y(feature_df, horizon=horizon)

            # ── 3. Optional feature selection ─────────────────────────────────
            if apply_feature_selection:
                fs = FeatureSelector()
                X = fs.fit_transform(X, y)
                logger.info(
                    "[%s/%s] FeatureSelector: %d features retained",
                    ticker,
                    horizon,
                    X.shape[1],
                )

            # ── 4. Model selection ────────────────────────────────────────────
            sel = self.selector.select(ticker, horizon=horizon)
            auto_trained = False

            if sel.reason == REASON_NO_ARTIFACTS:
                logger.warning(
                    "[%s/%s] No artifacts — training all %d models: %s",
                    ticker,
                    horizon,
                    len(ALL_TRAINING_MODELS),
                    ALL_TRAINING_MODELS,
                )
                best_result = None
                for model_name in ALL_TRAINING_MODELS:
                    try:
                        _, train_result = self.trainer.train(
                            model_name=model_name,
                            X=X,
                            y=y,
                            ticker=ticker,
                            horizon=horizon,
                            trigger_reason="no_artifacts_train_all",
                        )
                        logger.info(
                            "[%s/%s] Trained %s: AUC=%.4f",
                            ticker,
                            horizon,
                            model_name,
                            train_result.mean_roc_auc,
                        )
                        if (
                            best_result is None
                            or train_result.mean_roc_auc > best_result.mean_roc_auc
                        ):
                            best_result = train_result
                    except Exception as exc:
                        logger.warning(
                            "[%s/%s] Training %s failed (skipping): %s",
                            ticker,
                            horizon,
                            model_name,
                            exc,
                        )

                if best_result is None:
                    raise PredictionError(
                        f"All model training attempts failed for {ticker}/{horizon}."
                    )

                auto_trained = True
                sel = self.selector.select(ticker, horizon=horizon)
                logger.info(
                    "[%s/%s] Post-training selection: %s (AUC=%.4f, reason=%s)",
                    ticker,
                    horizon,
                    sel.model_name,
                    sel.auc,
                    sel.reason,
                )

            elif sel.reason == REASON_BELOW_THRESHOLD:
                logger.warning(
                    "[%s/%s] Using '%s' (AUC=%.4f < MIN_AUC). Confidence degraded.",
                    ticker,
                    horizon,
                    sel.model_name,
                    sel.auc,
                )
            else:
                logger.info(
                    "[%s/%s] Auto-selected: '%s' (AUC=%.4f).",
                    ticker,
                    horizon,
                    sel.model_name,
                    sel.auc,
                )

            confidence_degraded = sel.reason != REASON_LEADERBOARD

            # ── 5. Load or retrain on artifact issues ─────────────────────────
            model, feature_columns, train_result_incr = self.trainer.load_or_train(
                ticker=ticker,
                model_name=sel.model_name,
                X=X,
                y=y,
                horizon=horizon,
            )
            if train_result_incr is not None:
                auto_trained = True
                logger.info(
                    "[%s/%s] Incremental retrain: AUC=%.3f trigger='%s'",
                    ticker,
                    horizon,
                    train_result_incr.mean_roc_auc,
                    train_result_incr.trigger_reason,
                )

            # ── 6. Align features ─────────────────────────────────────────────
            missing = set(feature_columns) - set(X.columns)
            if missing:
                raise PredictionError(
                    f"Feature mismatch for {ticker}/{sel.model_name}/{horizon}",
                    detail=f"Missing columns: {sorted(missing)}",
                )
            X_aligned = X[feature_columns]

            # ── 7. Inference ──────────────────────────────────────────────────
            X_latest = X_aligned.iloc[[-1]]
            pred = int(model.predict(X_latest)[0])
            p_bullish = round(float(model.predict_proba(X_latest)[0, 1]), 4)
            p_bearish = round(1.0 - p_bullish, 4)
            prob = p_bullish if pred == 1 else p_bearish

            # ── 8. SHAP ───────────────────────────────────────────────────────
            # Pass the full X_aligned as background so the explainer has a
            # meaningful distribution for KernelExplainer (or TreeExplainer's
            # expected_value baseline).  local_explanation() still receives
            # only the single inference row.
            explainer = SHAPExplainer(model, feature_columns, X_background=X_aligned)
            shap_exp = explainer.local_explanation(X_latest)

            # ── 9. Narrative ──────────────────────────────────────────────────
            narrative = explainer.generate_narrative(
                shap_exp,
                ticker=ticker,
                authoritative_prediction=pred,
                authoritative_p_bullish=p_bullish,
            )

            # ── 10. Feature snapshot ──────────────────────────────────────────
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
                auto_trained=auto_trained,
            )

            # ── 11. Signal fusion (best-effort — never blocks) ────────────────
            if run_fusion and settings.OPENAI_API_KEY:
                try:
                    fused = self.fusion_service.fuse(ticker, response)
                    response.fused_signal = fused
                    logger.info(
                        "[%s/%s] Fusion: %s → %s (applied=%s)",
                        ticker,
                        horizon,
                        "BULLISH" if pred else "BEARISH",
                        fused.final_direction,
                        fused.fusion_applied,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s/%s] Signal fusion skipped: %s", ticker, horizon, exc
                    )
            else:
                logger.info(
                    "[%s/%s] Fusion skipped (run_fusion=%s, api_key=%s)",
                    ticker,
                    horizon,
                    run_fusion,
                    bool(settings.OPENAI_API_KEY),
                )

            logger.info(
                "Prediction complete: %s/%s → %s "
                "(p_bull=%.3f conf=%s model=%s reason=%s auto_trained=%s)",
                ticker,
                horizon,
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
