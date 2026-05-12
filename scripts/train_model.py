"""
FinSight AI — Offline Model Training Script
CLI entry point for training and persisting a model artifact.

Usage:
    python scripts/train_model.py --ticker AAPL --model xgboost
    python scripts/train_model.py --ticker MSFT --model lightgbm --hpo --hpo-trials 50
    python scripts/train_model.py --tickers AAPL MSFT GOOGL --model random_forest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root without installing package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging_config import get_logger, setup_logging
from app.ml.data_ingestion import ingest_market_data
from app.ml.feature_engineering import FeatureEngineer
from app.ml.models.model_factory import list_models
from app.ml.training.trainer import ModelTrainer

setup_logging(level="INFO")
logger = get_logger("scripts.train")


def train_single(
    ticker: str,
    model_name: str,
    period_years: int,
    run_hpo: bool,
    hpo_trials: int,
) -> None:
    """Train a single ticker/model pair and report results."""
    logger.info("=" * 60)
    logger.info(
        "Training: ticker=%s  model=%s  years=%d  hpo=%s",
        ticker,
        model_name,
        period_years,
        run_hpo,
    )
    t0 = time.perf_counter()

    # 1. Ingest
    raw_df = ingest_market_data(ticker, period_years=period_years)
    logger.info("Ingested %d rows for %s", len(raw_df), ticker)

    # 2. Feature engineering
    engineer = FeatureEngineer()
    feature_df = engineer.build_features(raw_df)
    X, y = engineer.split_X_y(feature_df)
    logger.info("Feature matrix: %d rows × %d features", X.shape[0], X.shape[1])

    # 3. Train
    trainer = ModelTrainer()
    _, result = trainer.train(
        model_name=model_name,
        X=X,
        y=y,
        ticker=ticker,
        run_hpo=run_hpo,
        hpo_trials=hpo_trials,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done in %.1fs | Acc=%.3f | F1=%.3f | AUC=%.3f",
        elapsed,
        result.mean_accuracy,
        result.mean_f1,
        result.mean_roc_auc,
    )
    print(f"\n{'─' * 50}")
    print(f"  {ticker} / {model_name}")
    print(f"  Accuracy : {result.mean_accuracy:.4f}")
    print(f"  F1       : {result.mean_f1:.4f}")
    print(f"  ROC-AUC  : {result.mean_roc_auc:.4f}")
    print(f"  Folds    : {len(result.fold_results)}")
    if result.best_params:
        print(f"  Params   : {result.best_params}")
    print(f"{'─' * 50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinSight AI — Offline Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--ticker", type=str, help="Single stock ticker (e.g. AAPL)"
    )
    ticker_group.add_argument("--tickers", nargs="+", help="Multiple tickers")

    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=list_models(),
        help="ML model to train (default: xgboost)",
    )
    parser.add_argument(
        "--period-years", type=int, default=5, help="Years of history (default: 5)"
    )
    parser.add_argument(
        "--hpo", action="store_true", help="Run Optuna hyperparameter optimisation"
    )
    parser.add_argument(
        "--hpo-trials", type=int, default=30, help="Number of HPO trials (default: 30)"
    )

    args = parser.parse_args()
    tickers = (
        [args.ticker.upper()] if args.ticker else [t.upper() for t in args.tickers]
    )

    errors: list[str] = []
    for ticker in tickers:
        try:
            train_single(
                ticker=ticker,
                model_name=args.model,
                period_years=args.period_years,
                run_hpo=args.hpo,
                hpo_trials=args.hpo_trials,
            )
        except Exception as exc:
            logger.error("Failed for %s: %s", ticker, exc)
            errors.append(ticker)

    if errors:
        logger.warning("Training failed for: %s", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
