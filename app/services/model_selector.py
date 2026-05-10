"""
FinSight AI — Automatic Model Selection Service
Reads persisted training metadata and selects the best available model
for a given ticker based on walk-forward ROC-AUC.

Design rationale
----------------
Exposing model selection to end users is an anti-pattern for a financial
decision-support product:

1.  **Cognitive burden** — novice users cannot meaningfully distinguish
    XGBoost from LightGBM; the choice is noise to them.
2.  **Responsibility diffusion** — if the user "chose" the model, they
    shoulder blame for a bad prediction even when the model was genuinely
    inferior.  The system should own that decision.
3.  **Reproducibility** — an automated, metadata-driven selector is
    deterministic and auditable; a human click is not.

Selection algorithm (leaderboard pattern)
-----------------------------------------
For a given ticker:

1. Scan ``MODELS_DIR`` for ``{TICKER}_{model_name}_meta.json`` files.
2. Parse ``mean_roc_auc`` from each metadata file.
3. Apply a minimum-quality gate (``MIN_AUC``) to exclude models that
   are barely better than random.
4. Return the model name with the highest AUC.
5. If no trained model passes the gate, return the configured
   ``FALLBACK_MODEL`` constant so the caller knows to train on demand.

Thread safety
-------------
``ModelSelector`` is stateless — every call to ``select()`` reads fresh
metadata from disk.  This is intentional: metadata files are written by
``ModelTrainer`` after each training run, so a concurrent training job
automatically improves the leaderboard without requiring a cache
invalidation step.

Extension points
----------------
* Swap AUC for a composite score (e.g. Sharpe ratio of simulated returns)
  by overriding ``_score()``.
* Add an ensemble policy that returns a list of models instead of one.
* Integrate with MLflow or a database for centralized experiment tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("model_selector")

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum acceptable walk-forward ROC-AUC.
# A model below this threshold is treated as not trained / unusable.
MIN_AUC: float = 0.52

# Preference order used as a tiebreaker when multiple models share the
# same AUC (rounded to 4 dp).  More complex models are deprioritized
# to reduce inference latency when quality is equivalent.
_PREFERENCE_ORDER: list[str] = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
]

# Returned when no acceptable trained model exists for the ticker.
# The caller (PredictionService) uses this sentinel to trigger on-demand
# training with a sensible default.
FALLBACK_MODEL: str = "xgboost"


# ── Service ───────────────────────────────────────────────────────────────────

class ModelSelector:
    """
    Stateless leaderboard-based model selector.

    Scans persisted training metadata in ``MODELS_DIR`` and returns the
    name of the best-performing model for a given ticker.

    Usage::

        selector = ModelSelector()
        model_name = selector.select("AAPL")
        # → "xgboost"  (or whichever model has the highest AUC)
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = Path(model_dir or settings.MODELS_DIR)

    # ── Public API ────────────────────────────────────────────────────────────

    def select(self, ticker: str) -> str:
        """
        Return the name of the best available model for *ticker*.

        Args:
            ticker: Stock ticker symbol (case-insensitive).

        Returns:
            Model name string (e.g. ``"xgboost"``).  Falls back to
            ``FALLBACK_MODEL`` when no acceptable artifact is found.
        """
        leaderboard = self._build_leaderboard(ticker.upper())

        if not leaderboard:
            logger.info(
                "[%s] No trained models found — falling back to %s",
                ticker, FALLBACK_MODEL,
            )
            return FALLBACK_MODEL

        best_name, best_auc = leaderboard[0]
        logger.info(
            "[%s] Model selected: %s (AUC=%.4f) from %d candidate(s)",
            ticker, best_name, best_auc, len(leaderboard),
        )
        return best_name

    def leaderboard(self, ticker: str) -> list[dict]:
        """
        Return the full leaderboard for *ticker* as a list of dicts.

        Useful for logging, API introspection, or debugging.

        Returns:
            List of ``{"model": str, "auc": float, "accuracy": float,
            "f1": float, "trained_at": str}`` sorted by AUC descending.
        """
        raw = self._build_leaderboard(ticker.upper())
        result = []
        for name, _auc in raw:
            meta = self._load_meta(ticker.upper(), name)
            if meta:
                result.append({
                    "model":      name,
                    "auc":        round(meta.get("mean_roc_auc", 0.0), 4),
                    "accuracy":   round(meta.get("mean_accuracy", 0.0), 4),
                    "f1":         round(meta.get("mean_f1", 0.0), 4),
                    "trained_at": meta.get("trained_at", ""),
                })
        return result

    def has_any_model(self, ticker: str) -> bool:
        """Return True if at least one trained model artifact exists."""
        return bool(self._build_leaderboard(ticker.upper()))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_leaderboard(self, ticker: str) -> list[tuple[str, float]]:
        """
        Scan metadata files and return ``(model_name, auc)`` pairs sorted
        by AUC descending.

        Selection policy:
        -----------------
        1. Prefer models with AUC >= MIN_AUC.
        2. If none meet the threshold, fall back to the highest-AUC model
        overall instead of blindly using FALLBACK_MODEL.

        Tiebreaker:
            _PREFERENCE_ORDER index (lower = preferred).
        """
        eligible_entries: list[tuple[str, float]] = []
        all_entries: list[tuple[str, float]] = []

        for model_name in _PREFERENCE_ORDER:
            meta = self._load_meta(ticker, model_name)
            if meta is None:
                continue

            auc = meta.get("mean_roc_auc", 0.0)
            all_entries.append((model_name, auc))

            if auc < MIN_AUC:
                logger.debug(
                    "[%s/%s] AUC %.4f below MIN_AUC %.2f",
                    ticker, model_name, auc, MIN_AUC,
                )
                continue

            eligible_entries.append((model_name, auc))

        # Prefer threshold-qualified models
        selected_entries = eligible_entries if eligible_entries else all_entries

        # If nothing exists at all
        if not selected_entries:
            return []

        # Warn when forced to use sub-threshold models
        if not eligible_entries:
            best_model, best_auc = max(selected_entries, key=lambda x: x[1])
            logger.warning(
                "[%s] No models met MIN_AUC %.2f — selecting best available "
                "model: %s (AUC=%.4f)",
                ticker, MIN_AUC, best_model, best_auc,
            )

        # Sort descending by AUC
        selected_entries.sort(key=lambda x: x[1], reverse=True)

        return selected_entries

    def _load_meta(self, ticker: str, model_name: str) -> Optional[dict]:
        """Load and parse a metadata JSON file; return None on any failure."""
        meta_path = self._model_dir / f"{ticker}_{model_name}_meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to parse metadata %s: %s", meta_path.name, exc
            )
            return None

    def _score(self, meta: dict) -> float:
        """
        Composite score for ranking.  Currently uses raw AUC.

        Override this method to incorporate additional metrics such as
        calibration error, directional accuracy, or a Sharpe proxy.
        """
        return meta.get("mean_roc_auc", 0.0)