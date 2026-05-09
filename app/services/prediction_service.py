"""
FinSight AI — Prediction Service
Orchestrates data ingestion → feature engineering → model inference → explanation.
This is the core service layer consumed by the FastAPI backend and agent tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from app.core.exceptions import PredictionError
from app.core.logging_config import get_logger
from app.ml.data_ingestion import ingest_market_data
from app.ml.explainability import SHAPExplainer
from app.ml.feature_engineering import FeatureEngineer
from app.ml.training.trainer import ModelTrainer
from configs.settings import settings

logger = get_logger("prediction_service")


@dataclass
class PredictionResponse:
    """Structured prediction response."""
    ticker: str
    model_name: str
    prediction: int           # 0=bearish, 1=bullish
    probability: float        # Directional: P(bullish) if bullish, P(bearish) if bearish
    p_bullish: float          # Raw P(bullish) from model — always 0..1
    p_bearish: float          # Raw P(bearish) = 1 - p_bullish
    confidence_label: str     # 'high' | 'moderate' | 'low'
    shap_explanation: dict
    narrative: str
    latest_close: float
    feature_snapshot: dict    # last row feature values


def _confidence_label(p_bullish: float) -> str:
    """
    Derive a confidence label from P(bullish).

    Confidence measures how far the probability is from the 50/50 baseline —
    it is symmetric and independent of direction. After Platt calibration,
    probabilities are centred around 0.5 so thresholds are meaningful:

        |P(bullish) - 0.5| > 0.15  -> high      (e.g. >=0.65 or <=0.35)
        |P(bullish) - 0.5| > 0.05  -> moderate  (e.g. >=0.55 or <=0.45)
        otherwise                  -> low        (near coin-flip)
    """
    delta = abs(p_bullish - 0.5)
    if delta > 0.15:
        return "high"
    if delta > 0.05:
        return "moderate"
    return "low"


class PredictionService:
    """
    End-to-end prediction pipeline.

    Attributes:
        trainer: ModelTrainer instance for loading/saving artifacts.
        engineer: FeatureEngineer instance.
    """

    def __init__(self) -> None:
        self.trainer = ModelTrainer()
        self.engineer = FeatureEngineer()

    def predict(
        self,
        ticker: str,
        model_name: str = "xgboost",
        use_cache: bool = True,
    ) -> PredictionResponse:
        """
        Run full prediction pipeline for a given ticker.

        Args:
            ticker: Stock ticker symbol.
            model_name: Name of the trained model to use.
            use_cache: Whether to cache fetched market data.

        Returns:
            PredictionResponse with prediction, probability, and explanation.

        Raises:
            PredictionError: On any pipeline failure.
        """
        try:
            logger.info("Prediction pipeline started: ticker=%s model=%s", ticker, model_name)

            # 1. Ingest
            raw_df = ingest_market_data(ticker, use_cache=use_cache)

            # 2. Feature engineering
            # y is retained (not discarded) so it is available for
            # train-on-demand below if no saved artifact exists yet.
            feature_df = self.engineer.build_features(raw_df)
            X, y = self.engineer.split_X_y(feature_df)

            # 3. Load model — train on demand if no artifact exists yet
            try:
                model, feature_columns = self.trainer.load_model(ticker, model_name)
            except Exception:
                logger.info(
                    "No artifact found for %s/%s — training on demand...",
                    ticker, model_name,
                )
                _, train_result = self.trainer.train(
                    model_name=model_name,
                    X=X,
                    y=y,
                    ticker=ticker,
                )
                logger.info(
                    "On-demand training complete: AUC=%.3f",
                    train_result.mean_roc_auc,
                )
                model, feature_columns = self.trainer.load_model(ticker, model_name)

            # Align features
            missing = set(feature_columns) - set(X.columns)
            if missing:
                raise PredictionError(
                    f"Feature mismatch for {ticker}/{model_name}",
                    detail=f"Missing columns: {missing}",
                )
            X_aligned = X[feature_columns]

            # 4. Inference on latest row
            X_latest = X_aligned.iloc[[-1]]
            pred = int(model.predict(X_latest)[0])

            p_bullish = round(float(model.predict_proba(X_latest)[0, 1]), 4)
            p_bearish = round(1.0 - p_bullish, 4)

            # Directional probability: always show the confidence of the predicted
            # direction so the displayed figure reads intuitively.
            # e.g. BEARISH at P(bullish)=0.097 -> displays 90.3% (bearish confidence)
            #      BULLISH  at P(bullish)=0.720 -> displays 72.0% (bullish confidence)
            prob = p_bullish if pred == 1 else p_bearish

            # 5. SHAP explanation
            explainer = SHAPExplainer(model, feature_columns)
            shap_exp = explainer.local_explanation(X_latest)
            narrative = explainer.generate_narrative(shap_exp, ticker=ticker)

            # 6. Feature snapshot (top 20 features for display)
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
            )

            logger.info(
                "Prediction complete: %s -> %s (P_bull=%.3f, P_bear=%.3f, directional=%.3f, conf=%s)",
                ticker,
                "BULLISH" if pred else "BEARISH",
                p_bullish,
                p_bearish,
                prob,
                response.confidence_label,
            )
            return response

        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(f"Prediction pipeline failed for {ticker}: {exc}") from exc

    def batch_predict(
        self,
        tickers: list[str],
        model_name: str = "xgboost",
    ) -> dict[str, PredictionResponse | str]:
        """
        Run predictions for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            model_name: Model to use for all tickers.

        Returns:
            Dict mapping ticker -> PredictionResponse or error string.
        """
        results: dict[str, PredictionResponse | str] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.predict(ticker, model_name)
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s", ticker, exc)
                results[ticker] = str(exc)
        return results