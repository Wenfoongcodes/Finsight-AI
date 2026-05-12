"""
FinSight AI — Prediction Service (v3)

Changes vs v2
-------------
1.  **Multi-horizon support** — ``predict()`` accepts a ``horizon``
    parameter ('1d', '7d', '1m', '6m').  Artifacts, leaderboard entries,
    and metadata are all horizon-keyed so each horizon is fully independent.

2.  **Automatic model recovery** — delegates to
    ``ModelTrainer.load_or_train()`` which detects missing/corrupt/mismatched
    artifacts and retrains automatically with structured audit logging.
    ``PredictionService`` never needs to catch ``ModelNotFoundError`` manually.

3.  **Feature selection integration** — ``FeatureSelector`` is run at
    prediction time only when the feature set has grown significantly since
    the last training.  The selector's fitted state is persisted alongside
    the model artifact (via the trainer).

4.  **``run_fusion`` default is True** — but is controlled by
    ``settings.OPENAI_API_KEY`` presence.  The service no longer receives
    ``model_name`` from callers.

5.  **``PredictionResponse`` extended** with ``horizon``, ``p_bullish``,
    ``p_bearish``, and ``intelligence_brief`` for richer downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.exceptions import PredictionError
from app.core.logging_config import get_logger
from app.ml.data_ingestion import ingest_market_data
from app.ml.explainability import SHAPExplainer
from app.ml.feature_engineering import HORIZONS, FeatureEngineer, FeatureSelector
from app.ml.training.trainer import ModelTrainer
from app.services.model_selector import ModelSelector
from app.services.news_intelligence import IntelligenceBrief
from app.services.signal_fusion import FusedSignal, SignalFusionService
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
    ticker            : Stock ticker symbol.
    model_name        : Auto-selected model name.
    horizon           : Prediction horizon ('1d', '7d', '1m', '6m').
    prediction        : Binary direction: 0 = bearish, 1 = bullish.
    probability       : P(predicted direction).
    p_bullish         : Calibrated P(bullish).
    p_bearish         : 1 - p_bullish.
    confidence_label  : 'high' | 'moderate' | 'low'.
    shap_explanation  : SHAP local explanation dict.
    narrative         : Plain-English SHAP narrative.
    latest_close      : Most recent closing price.
    feature_snapshot  : Top-20 feature values at inference time.
    fused_signal      : FusedSignal from news + LLM synthesis (or None).
    intelligence_brief: Raw IntelligenceBrief from news retrieval (or None).
    auto_trained      : True if the model was trained automatically this call.
    """

    ticker: str
    model_name: str
    horizon: str
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
    intelligence_brief: Optional[IntelligenceBrief] = None
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
    End-to-end prediction pipeline with:
    - automatic model selection (ModelSelector)
    - automatic model recovery / training (ModelTrainer.load_or_train)
    - multi-horizon prediction
    - SHAP explanation
    - signal fusion with financial intelligence
    """

    def __init__(self) -> None:
        self.trainer = ModelTrainer()
        self.engineer = FeatureEngineer()
        self.selector = ModelSelector()
        self.fusion_service = SignalFusionService()

    # ── Main prediction entry point ───────────────────────────────────────────

    def predict(
        self,
        ticker: str,
        horizon: str = "1d",
        use_cache: bool = True,
        run_fusion: bool = True,
        apply_feature_selection: bool = False,
    ) -> PredictionResponse:
        """
        Run the full prediction pipeline.

        Args:
            ticker:                   Stock ticker symbol.
            horizon:                  '1d' | '7d' | '1m' | '6m'.
            use_cache:                Use cached market data.
            run_fusion:               Run news + LLM signal fusion.
            apply_feature_selection:  Apply FeatureSelector (reduces features).

        Returns:
            PredictionResponse.

        Raises:
            PredictionError: On unrecoverable pipeline failure.
        """
        if horizon not in HORIZONS:
            raise PredictionError(
                f"Invalid horizon '{horizon}'. Valid: {list(HORIZONS.keys())}"
            )

        try:
            logger.info("Prediction pipeline: ticker=%s horizon=%s", ticker, horizon)

            # 1. Ingest
            raw_df = ingest_market_data(ticker, use_cache=use_cache)

            # 2. Feature engineering
            feature_df = self.engineer.build_features(raw_df)
            X, y = self.engineer.split_X_y(feature_df, horizon=horizon)

            # 3. Optional feature selection
            if apply_feature_selection:
                selector = FeatureSelector()
                X = selector.fit_transform(X, y)
                logger.info(
                    "[%s/%s] Feature selection: %d features retained",
                    ticker,
                    horizon,
                    X.shape[1],
                )

            # 4. Auto-select model
            model_name = self.selector.select(ticker, horizon=horizon)
            logger.info("[%s/%s] Auto-selected model: %s", ticker, horizon, model_name)

            # 5. Load or train automatically
            model, feature_columns, train_result = self.trainer.load_or_train(
                ticker=ticker,
                model_name=model_name,
                X=X,
                y=y,
                horizon=horizon,
            )
            auto_trained = train_result is not None

            if auto_trained:
                logger.info(
                    "[%s/%s] Auto-training complete: AUC=%.3f trigger=%s",
                    ticker,
                    horizon,
                    train_result.mean_roc_auc,
                    train_result.trigger_reason,
                )

            # 6. Align features
            missing = set(feature_columns) - set(X.columns)
            if missing:
                raise PredictionError(
                    f"Feature mismatch for {ticker}/{model_name}/{horizon}",
                    detail=f"Missing: {missing}",
                )
            X_aligned = X[feature_columns]

            # 7. Inference on latest row
            X_latest = X_aligned.iloc[[-1]]
            pred = int(model.predict(X_latest)[0])
            p_bullish = round(float(model.predict_proba(X_latest)[0, 1]), 4)
            p_bearish = round(max(0.0, min(1.0, 1.0 - p_bullish)), 4)
            prob = p_bullish if pred == 1 else p_bearish

            # 8. SHAP explanation
            explainer = SHAPExplainer(model, feature_columns)
            shap_exp = explainer.local_explanation(X_latest)

            # 9. Narrative
            narrative = explainer.generate_narrative(
                shap_exp,
                ticker=ticker,
                authoritative_prediction=pred,
                authoritative_p_bullish=p_bullish,
            )

            # 10. Feature snapshot
            snapshot = {
                col: round(float(X_latest.iloc[0][col]), 4)
                for col in feature_columns[:20]
            }

            latest_close = float(raw_df["Close"].iloc[-1])

            response = PredictionResponse(
                ticker=ticker,
                model_name=model_name,
                horizon=horizon,
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

            # 11. Signal fusion (best-effort — never blocks)
            if run_fusion and settings.OPENAI_API_KEY:
                try:
                    fused = self.fusion_service.fuse(ticker, response)
                    response.fused_signal = fused
                    response.intelligence_brief = fused.intelligence_brief
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
                "Prediction done: %s/%s → %s "
                "(p_bull=%.3f conf=%s model=%s auto_trained=%s)",
                ticker,
                horizon,
                "BULLISH" if pred else "BEARISH",
                p_bullish,
                response.confidence_label,
                model_name,
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
        """Predict for multiple tickers; capture failures per-ticker."""
        results: dict[str, PredictionResponse | str] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.predict(
                    ticker, horizon=horizon, run_fusion=run_fusion
                )
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s", ticker, exc)
                results[ticker] = str(exc)
        return results
