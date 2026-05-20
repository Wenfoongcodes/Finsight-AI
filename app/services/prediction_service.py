from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.exceptions import PredictionError
from app.core.formatting import (
    confidence_label,
    round_price,
    round_prob,
    round_shap,
)
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

    All float fields use canonical precision from ``app.core.formatting``:
    - ``p_bullish``, ``p_bearish``, ``probability`` → ``round_prob()`` (4 d.p.)
    - ``latest_close``                              → ``round_price()`` (2 d.p.)
    - Feature snapshot values                       → ``round_shap()``  (4 d.p.)
    """

    ticker: str
    model_name: str
    horizon: str
    selection_reason: str
    confidence_degraded: bool
    prediction: int
    probability: float  # round_prob()
    p_bullish: float  # round_prob()
    p_bearish: float  # round_prob()
    confidence_label: str
    shap_explanation: dict
    narrative: str
    latest_close: float  # round_price()
    feature_snapshot: dict
    fused_signal: Optional[FusedSignal] = None
    auto_trained: bool = False


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
                    "[%s/%s] Incremental retrain: AUC=%.4f trigger='%s'",
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

            p_bullish = round_prob(float(model.predict_proba(X_latest)[0, 1]))
            p_bearish = round_prob(1.0 - p_bullish)
            prob = round_prob(p_bullish if pred == 1 else p_bearish)

            # ── 8. SHAP ───────────────────────────────────────────────────────
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
                col: round_shap(float(X_latest.iloc[0][col]))
                for col in feature_columns[:20]
            }

            latest_close = round_price(float(raw_df["Close"].iloc[-1]))

            response = PredictionResponse(
                ticker=ticker,
                model_name=sel.model_name,
                horizon=horizon,
                selection_reason=sel.reason,
                confidence_degraded=confidence_degraded,
                prediction=pred,
                probability=prob,
                p_bullish=p_bullish,
                p_bearish=p_bearish,
                confidence_label=confidence_label(p_bullish),  # shared implementation
                shap_explanation=shap_exp,
                narrative=narrative,
                latest_close=latest_close,
                feature_snapshot=snapshot,
                fused_signal=None,
                auto_trained=auto_trained,
            )

            # ── 11. Signal fusion (best-effort — horizon-aware) ───────────────
            if run_fusion and settings.OPENAI_API_KEY:
                try:
                    fused = self.fusion_service.fuse(
                        ticker,
                        response,
                        horizon=horizon,  # ← propagated in v7
                    )
                    response.fused_signal = fused
                    logger.info(
                        "[%s/%s] Fusion: %s → %s (applied=%s, recency=%s)",
                        ticker,
                        horizon,
                        "BULLISH" if pred else "BEARISH",
                        fused.final_direction,
                        fused.fusion_applied,
                        fused.recency_note,
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
                "(p_bull=%.4f conf=%s model=%s reason=%s auto_trained=%s)",
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
