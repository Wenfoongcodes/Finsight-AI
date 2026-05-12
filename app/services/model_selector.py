"""
FinSight AI — Automatic Model Selection Service (v2)

Changes vs v1
-------------
* Multi-horizon support — ``select()`` and ``leaderboard()`` accept a
  ``horizon`` parameter so each prediction horizon has an independent
  model selection.  Artifact filenames are ``{TICKER}_{model}_{horizon}_meta.json``.

* ``has_any_model()`` accepts horizon.

Everything else (leaderboard pattern, MIN_AUC gate, preference order) is
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("model_selector")

MIN_AUC: float = 0.52

_PREFERENCE_ORDER: list[str] = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
]

FALLBACK_MODEL: str = "xgboost"


class ModelSelector:
    """
    Stateless leaderboard-based model selector with multi-horizon support.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = Path(model_dir or settings.MODELS_DIR)

    def select(self, ticker: str, horizon: str = "1d") -> str:
        leaderboard = self._build_leaderboard(ticker.upper(), horizon)
        if not leaderboard:
            logger.info(
                "[%s/%s] No trained models — falling back to %s",
                ticker,
                horizon,
                FALLBACK_MODEL,
            )
            return FALLBACK_MODEL
        best_name, best_auc = leaderboard[0]
        logger.info(
            "[%s/%s] Selected: %s (AUC=%.4f) from %d candidates",
            ticker,
            horizon,
            best_name,
            best_auc,
            len(leaderboard),
        )
        return best_name

    def leaderboard(self, ticker: str, horizon: str = "1d") -> list[dict]:
        raw = self._build_leaderboard(ticker.upper(), horizon)
        result = []
        for name, _ in raw:
            meta = self._load_meta(ticker.upper(), name, horizon)
            if meta:
                result.append(
                    {
                        "model": name,
                        "horizon": horizon,
                        "auc": round(meta.get("mean_roc_auc", 0.0), 4),
                        "accuracy": round(meta.get("mean_accuracy", 0.0), 4),
                        "f1": round(meta.get("mean_f1", 0.0), 4),
                        "trained_at": meta.get("trained_at", ""),
                    }
                )
        return result

    def has_any_model(self, ticker: str, horizon: str = "1d") -> bool:
        return bool(self._build_leaderboard(ticker.upper(), horizon))

    def _build_leaderboard(self, ticker: str, horizon: str) -> list[tuple[str, float]]:
        eligible: list[tuple[str, float]] = []
        all_entries: list[tuple[str, float]] = []

        for model_name in _PREFERENCE_ORDER:
            meta = self._load_meta(ticker, model_name, horizon)
            if meta is None:
                continue
            auc = meta.get("mean_roc_auc", 0.0)
            all_entries.append((model_name, auc))
            if auc >= MIN_AUC:
                eligible.append((model_name, auc))
            else:
                logger.debug(
                    "[%s/%s/%s] AUC %.4f below MIN_AUC %.2f",
                    ticker,
                    model_name,
                    horizon,
                    auc,
                    MIN_AUC,
                )

        selected = eligible if eligible else all_entries
        if not selected:
            return []

        if not eligible and all_entries:
            best_m, best_a = max(all_entries, key=lambda x: x[1])
            logger.warning(
                "[%s/%s] No models meet MIN_AUC %.2f — best available: %s (%.4f)",
                ticker,
                horizon,
                MIN_AUC,
                best_m,
                best_a,
            )

        selected.sort(key=lambda x: x[1], reverse=True)
        return selected

    def _load_meta(self, ticker: str, model_name: str, horizon: str) -> Optional[dict]:
        meta_path = self._model_dir / f"{ticker}_{model_name}_{horizon}_meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", meta_path.name, exc)
            return None

    def _score(self, meta: dict) -> float:
        return meta.get("mean_roc_auc", 0.0)
