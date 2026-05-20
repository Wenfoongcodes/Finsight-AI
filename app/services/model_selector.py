from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("model_selector")

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_AUC: float = 0.52

# All models trained when no artifact exists.  Order also acts as tiebreak.
ALL_TRAINING_MODELS: list[str] = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
]

# Kept for backward compat; callers that only need a single name can import this.
DEFAULT_TRAINING_MODEL: str = ALL_TRAINING_MODELS[0]
FALLBACK_MODEL: str = DEFAULT_TRAINING_MODEL

_PREFERENCE_ORDER: list[str] = ALL_TRAINING_MODELS  # same list, aliased for clarity

# Selection reason constants
REASON_LEADERBOARD = "leaderboard"
REASON_BELOW_THRESHOLD = "best_below_threshold"
REASON_NO_ARTIFACTS = "no_artifacts_default"


# ─────────────────────────────────────────────────────────────────────────────
# SelectionResult
# ─────────────────────────────────────────────────────────────────────────────


class SelectionResult(NamedTuple):
    """
    Structured output from ``ModelSelector.select()``.

    model_name       : Model registry key to use.
    reason           : One of REASON_* constants.
    auc              : Best known AUC, 0.0 when no artifacts exist.
    from_leaderboard : True only when at least one artifact passed MIN_AUC.
    """

    model_name: str
    reason: str
    auc: float
    from_leaderboard: bool


# ─────────────────────────────────────────────────────────────────────────────
# Selector
# ─────────────────────────────────────────────────────────────────────────────


class ModelSelector:
    """
    Stateless leaderboard-based model selector with multi-horizon support.

    When no artifacts exist, ``select()`` returns REASON_NO_ARTIFACTS with
    the first model in ALL_TRAINING_MODELS.  The caller (PredictionService)
    is responsible for iterating ALL_TRAINING_MODELS, training each one, and
    calling ``select()`` again to obtain the best result.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = Path(model_dir or settings.MODELS_DIR)

    # ── Primary API ───────────────────────────────────────────────────────────

    def select(self, ticker: str, horizon: str = "1d") -> SelectionResult:
        """
        Select the best available trained model for *ticker* / *horizon*.

        Three cases:
        1. Leaderboard hit  — artifact(s) pass MIN_AUC → return best by AUC.
        2. Below-threshold  — artifact(s) exist but all < MIN_AUC → return best.
        3. No artifacts     — return REASON_NO_ARTIFACTS so the caller trains all models.
        """
        eligible, all_entries = self._scan_leaderboard(ticker.upper(), horizon)

        if eligible:
            best_name, best_auc = eligible[0]
            logger.info(
                "[%s/%s] Model selected: %s (AUC=%.4f, candidates=%d)",
                ticker,
                horizon,
                best_name,
                best_auc,
                len(eligible),
            )
            return SelectionResult(
                model_name=best_name,
                reason=REASON_LEADERBOARD,
                auc=best_auc,
                from_leaderboard=True,
            )

        if all_entries:
            best_name, best_auc = all_entries[0]
            logger.warning(
                "[%s/%s] No model meets MIN_AUC %.2f. "
                "Best available: %s (AUC=%.4f). Proceeding with degraded confidence.",
                ticker,
                horizon,
                MIN_AUC,
                best_name,
                best_auc,
            )
            return SelectionResult(
                model_name=best_name,
                reason=REASON_BELOW_THRESHOLD,
                auc=best_auc,
                from_leaderboard=False,
            )

        # No artifacts at all — caller must train all models first
        logger.warning(
            "[%s/%s] No trained artifacts found. "
            "Caller should train all models in ALL_TRAINING_MODELS=%s, "
            "then call select() again.",
            ticker,
            horizon,
            ALL_TRAINING_MODELS,
        )
        return SelectionResult(
            model_name=ALL_TRAINING_MODELS[0],
            reason=REASON_NO_ARTIFACTS,
            auc=0.0,
            from_leaderboard=False,
        )

    def select_name(self, ticker: str, horizon: str = "1d") -> str:
        """Convenience shim — returns just the model name string."""
        return self.select(ticker, horizon).model_name

    def leaderboard(self, ticker: str, horizon: str = "1d") -> list[dict]:
        """Return all trained models for *ticker*/*horizon* sorted by AUC."""
        _, all_entries = self._scan_leaderboard(ticker.upper(), horizon)
        result = []
        for name, _ in all_entries:
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
        """Return True when at least one trained artifact exists."""
        _, all_entries = self._scan_leaderboard(ticker.upper(), horizon)
        return bool(all_entries)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _scan_leaderboard(
        self,
        ticker: str,
        horizon: str,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        eligible: list[tuple[str, float]] = []
        all_entries: list[tuple[str, float]] = []

        for model_name in _PREFERENCE_ORDER:
            meta = self._load_meta(ticker, model_name, horizon)
            if meta is None:
                continue
            auc = float(meta.get("mean_roc_auc", 0.0))
            all_entries.append((model_name, auc))
            if auc >= MIN_AUC:
                eligible.append((model_name, auc))
            else:
                logger.debug(
                    "[%s/%s/%s] AUC %.4f < MIN_AUC %.2f",
                    ticker,
                    model_name,
                    horizon,
                    auc,
                    MIN_AUC,
                )

        eligible.sort(key=lambda x: x[1], reverse=True)
        all_entries.sort(key=lambda x: x[1], reverse=True)
        return eligible, all_entries

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
