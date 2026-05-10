"""
FinSight AI — Prediction Service
Orchestrates data ingestion → feature engineering → auto model selection
→ model inference → SHAP explanation → signal fusion.

Changes in this revision
------------------------
* **Automatic model selection** — ``ModelSelector`` reads persisted training
  metadata and picks the highest-AUC model for the requested ticker.
  The caller no longer passes a model name; the service owns that decision.
  This removes a source of user confusion and makes the pipeline
  deterministic and auditable.

* **Signal fusion** — ``SignalFusionService`` combines the ML signal with
  live web-searched news via an LLM synthesis step, producing a
  ``FusedSignal`` that reconciles quantitative and qualitative evidence.
  The fused verdict is attached to ``PredictionResponse`` alongside the
  raw ML signal so downstream consumers (API, dashboard) can display both.

* **Graceful degradation** — signal fusion is best-effort.  If the web
  search or LLM call fails, ``PredictionResponse`` still carries the raw
  ML prediction; ``fused_signal.fusion_applied`` is ``False``.

Bug fixed in previous revision (retained)
-----------------------------------------
``SHAPExplainer.generate_narrative()`` now receives authoritative calibrated
values (``authoritative_prediction``, ``authoritative_p_bullish``) from
``model.predict()`` / ``model.predict_proba()``, preventing the narrative
from contradicting the signal card when Platt scaling shifts probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.core.exceptions import PredictionError
from app.core.logging_config import get_logger
from app.ml.data_ingestion import ingest_market_data
from app.ml.explainability import SHAPExplainer
from app.ml.feature_engineering import FeatureEngineer
from app.ml.training.trainer import ModelTrainer
from app.services.model_selector import FALLBACK_MODEL, ModelSelector
from app.services.signal_fusion import FusedSignal, SignalFusionService
from configs.settings import settings

logger = get_logger("prediction_service")


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResponse:
    """
    Structured prediction response.

    Attributes
    ----------
    ticker            : Stock ticker symbol.
    model_name        : Name of the model that produced the ML signal
                        (auto-selected by ``ModelSelector``).
    prediction        : Raw ML prediction: 0 = bearish, 1 = bullish.
    probability       : Directional probability of the *predicted* class.
    p_bullish         : Calibrated P(bullish) in [0, 1].
    p_bearish         : Calibrated P(bearish) = 1 − p_bullish.
    confidence_label  : "high" | "moderate" | "low" (ML-only).
    shap_explanation  : SHAP local explanation dict.
    narrative         : ML-only plain-English reasoning (SHAP-based).
    latest_close      : Most recent closing price.
    feature_snapshot  : Top-20 feature values at inference time.
    fused_signal      : Result of signal fusion (None if fusion was skipped
                        entirely, e.g. LLM not configured).
    """

    ticker:           str
    model_name:       str
    prediction:       int           # 0 = bearish, 1 = bullish
    probability:      float         # P(predicted direction)
    p_bullish:        float
    p_bearish:        float
    confidence_label: str
    shap_explanation: dict
    narrative:        str
    latest_close:     float
    feature_snapshot: dict
    fused_signal:     Optional[FusedSignal] = None


def _confidence_label(p_bullish: float) -> str:
    """
    Derive a confidence label from P(bullish).

    Symmetric around 0.5 — measures distance from the 50/50 baseline:

        |P(bullish) − 0.5| > 0.15  → 'high'
        |P(bullish) − 0.5| > 0.05  → 'moderate'
        otherwise                  → 'low'
    """
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
    End-to-end prediction pipeline with automatic model selection
    and news-fused signal synthesis.

    Attributes
    ----------
    trainer:        ``ModelTrainer`` for loading / saving artifacts.
    engineer:       ``FeatureEngineer`` instance.
    selector:       ``ModelSelector`` leaderboard.
    fusion_service: ``SignalFusionService`` for ML + news synthesis.
    """

    def __init__(self) -> None:
        self.trainer        = ModelTrainer()
        self.engineer       = FeatureEngineer()
        self.selector       = ModelSelector()
        self.fusion_service = SignalFusionService()

    def predict(
        self,
        ticker: str,
        use_cache: bool = True,
        run_fusion: bool = True,
    ) -> PredictionResponse:
        """
        Run the full prediction pipeline for a given ticker.

        The model is selected automatically by ``ModelSelector`` based on
        walk-forward AUC from persisted metadata.  The caller never passes
        a model name.

        Args:
            ticker:     Stock ticker symbol.
            use_cache:  Whether to use cached market data.
            run_fusion: Whether to run signal fusion (web search + LLM).
                        Set to ``False`` in test environments or when LLM
                        is not configured.

        Returns:
            ``PredictionResponse`` with ML signal, SHAP explanation,
            and (when enabled) the fused signal.

        Raises:
            PredictionError: On any unrecoverable pipeline failure.
        """
        try:
            logger.info("Prediction pipeline started: ticker=%s", ticker)

            # 1. Ingest
            raw_df = ingest_market_data(ticker, use_cache=use_cache)

            # 2. Feature engineering
            feature_df = self.engineer.build_features(raw_df)
            X, y       = self.engineer.split_X_y(feature_df)

            # 3. Auto-select best model
            model_name = self.selector.select(ticker)
            logger.info("[%s] Auto-selected model: %s", ticker, model_name)

            # 4. Load model — train on demand if no artifact exists yet
            try:
                model, feature_columns = self.trainer.load_model(ticker, model_name)
            except Exception:
                logger.info(
                    "[%s] No artifact for %s — training on demand…",
                    ticker, model_name,
                )
                _, train_result = self.trainer.train(
                    model_name=model_name,
                    X=X,
                    y=y,
                    ticker=ticker,
                )
                logger.info(
                    "[%s] On-demand training complete: AUC=%.3f",
                    ticker, train_result.mean_roc_auc,
                )
                model, feature_columns = self.trainer.load_model(ticker, model_name)

            # 5. Align features
            missing = set(feature_columns) - set(X.columns)
            if missing:
                raise PredictionError(
                    f"Feature mismatch for {ticker}/{model_name}",
                    detail=f"Missing columns: {missing}",
                )
            X_aligned = X[feature_columns]

            # 6. Inference on latest row
            X_latest  = X_aligned.iloc[[-1]]
            pred      = int(model.predict(X_latest)[0])

            p_bullish = round(float(model.predict_proba(X_latest)[0, 1]), 4)
            p_bearish = round(1.0 - p_bullish, 4)

            # Directional probability: confidence of the *predicted* direction.
            prob = p_bullish if pred == 1 else p_bearish

            # 7. SHAP explanation
            explainer = SHAPExplainer(model, feature_columns)
            shap_exp  = explainer.local_explanation(X_latest)

            # 8. Generate narrative with authoritative calibrated values
            narrative = explainer.generate_narrative(
                shap_exp,
                ticker=ticker,
                authoritative_prediction=pred,
                authoritative_p_bullish=p_bullish,
            )

            # 9. Feature snapshot (top 20 features for display)
            snapshot = {
                col: round(float(X_latest.iloc[0][col]), 4)
                for col in feature_columns[:20]
            }

            latest_close = float(raw_df["Close"].iloc[-1])

            response = PredictionResponse(
                ticker=ticker,
                model_name=model_name,
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
            )

            # 10. Signal fusion (best-effort — never blocks the ML result)
            if run_fusion and settings.OPENAI_API_KEY:
                try:
                    fused = self.fusion_service.fuse(ticker, response)
                    response.fused_signal = fused
                    logger.info(
                        "[%s] Fusion complete: %s → %s (fusion_applied=%s)",
                        ticker,
                        "BULLISH" if pred else "BEARISH",
                        fused.final_direction,
                        fused.fusion_applied,
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] Signal fusion skipped (non-critical): %s", ticker, exc
                    )
            else:
                logger.info(
                    "[%s] Signal fusion skipped (run_fusion=%s, api_key_set=%s)",
                    ticker, run_fusion, bool(settings.OPENAI_API_KEY),
                )

            logger.info(
                "Prediction complete: %s → %s "
                "(p_bull=%.3f, p_bear=%.3f, directional=%.3f, conf=%s, model=%s)",
                ticker,
                "BULLISH" if pred else "BEARISH",
                p_bullish,
                p_bearish,
                prob,
                response.confidence_label,
                model_name,
            )
            return response

        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(
                f"Prediction pipeline failed for {ticker}: {exc}"
            ) from exc

    def batch_predict(
        self,
        tickers: list[str],
        run_fusion: bool = True,
    ) -> dict[str, PredictionResponse | str]:
        """
        Run predictions for multiple tickers.

        Args:
            tickers:    List of ticker symbols.
            run_fusion: Whether to run signal fusion for each ticker.

        Returns:
            Dict mapping ticker → ``PredictionResponse`` or error string.
        """
        results: dict[str, PredictionResponse | str] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.predict(ticker, run_fusion=run_fusion)
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s", ticker, exc)
                results[ticker] = str(exc)
        return results