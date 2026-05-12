"""
FinSight AI — Automatic Model Selection Service (v3)

Bug fixed
---------
v2 returned ``FALLBACK_MODEL = "xgboost"`` silently when the leaderboard
was empty (no trained artifacts for the ticker/horizon).  This masked the
"no model exists" condition — the caller had no way to distinguish between
"xgboost was genuinely the best model" and "xgboost was picked because
nothing else existed."

Fix
---
``select()`` now returns a ``SelectionResult`` named-tuple that carries:

  model_name       : str   — the model to use
  reason           : str   — one of REASON_* constants
  auc              : float — 0.0 when no artifacts exist
  from_leaderboard : bool  — True only when MIN_AUC was satisfied

``PredictionService`` inspects ``result.reason`` and logs accordingly:
- REASON_NO_ARTIFACTS    → WARNING, proceeds to auto-train from scratch
- REASON_BELOW_THRESHOLD → WARNING, proceeds but flags low confidence
- REASON_LEADERBOARD     → INFO, standard path

``DEFAULT_TRAINING_MODEL`` replaces the old ``FALLBACK_MODEL`` constant and
is referenced by name so a single change cascades everywhere.

Backward compat
---------------
``select_name()`` shim returns just the model-name string for callers that
don't need the full SelectionResult (scripts, tests, CLI).
``FALLBACK_MODEL`` alias kept for any external import that references it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("model_selector")

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_AUC: float = 0.52

# What the system trains when no artifact exists — explicit policy, not a silent default.
DEFAULT_TRAINING_MODEL: str = "xgboost"
FALLBACK_MODEL: str = DEFAULT_TRAINING_MODEL  # backward-compat alias

_PREFERENCE_ORDER: list[str] = [
    "xgboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
]

# Selection reason constants consumed by PredictionService for branching logic
REASON_LEADERBOARD     = "leaderboard"          # normal best-AUC pick (passed MIN_AUC)
REASON_BELOW_THRESHOLD = "best_below_threshold" # best available, but under MIN_AUC
REASON_NO_ARTIFACTS    = "no_artifacts_default" # nothing trained yet for ticker/horizon


# ─────────────────────────────────────────────────────────────────────────────
# SelectionResult
# ─────────────────────────────────────────────────────────────────────────────

class SelectionResult(NamedTuple):
    """
    Structured output from ``ModelSelector.select()``.

    Always returned — never None, never raises.  Callers branch on ``reason``
    to handle the three distinct cases explicitly rather than guessing from
    the returned model name alone.
    """
    model_name:       str    # model registry key to pass to trainer
    reason:           str    # REASON_* constant above
    auc:              float  # best known AUC; 0.0 when no artifacts exist
    from_leaderboard: bool   # True only when MIN_AUC was satisfied


# ─────────────────────────────────────────────────────────────────────────────
# Selector
# ─────────────────────────────────────────────────────────────────────────────

class ModelSelector:
    """
    Stateless leaderboard-based model selector with multi-horizon support.

    Scans persisted ``*_meta.json`` training-result files from ``MODELS_DIR``
    and returns a ``SelectionResult`` so every caller knows *why* a model
    was chosen, not just which model.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = Path(model_dir or settings.MODELS_DIR)

    # ── Primary API ───────────────────────────────────────────────────────────

    def select(self, ticker: str, horizon: str = "1d") -> SelectionResult:
        """
        Select the best available model for *ticker* / *horizon*.

        Three distinct cases, all explicit:

        1. **Leaderboard hit** — at least one artifact passes MIN_AUC.
           Logs at INFO.  Returns best model by AUC.

        2. **Below-threshold fallback** — artifacts exist but none pass MIN_AUC.
           Logs at WARNING.  Returns best available so prediction can proceed,
           but caller should flag confidence as degraded.

        3. **No artifacts** — nothing trained yet for this ticker/horizon.
           Logs at WARNING with actionable message.  Returns
           ``DEFAULT_TRAINING_MODEL`` so ``load_or_train()`` knows what to train.
        """
        eligible, all_entries = self._scan_leaderboard(ticker.upper(), horizon)

        # ── Case 1: At least one model passes MIN_AUC ─────────────────────────
        if eligible:
            best_name, best_auc = eligible[0]
            logger.info(
                "[%s/%s] Model selected via leaderboard: %s "
                "(AUC=%.4f, %d candidate(s))",
                ticker, horizon, best_name, best_auc, len(eligible),
            )
            return SelectionResult(
                model_name=best_name,
                reason=REASON_LEADERBOARD,
                auc=best_auc,
                from_leaderboard=True,
            )

        # ── Case 2: Artifacts exist but all below MIN_AUC ────────────────────
        if all_entries:
            best_name, best_auc = all_entries[0]
            logger.warning(
                "[%s/%s] No model meets MIN_AUC %.2f. "
                "Best available: %s (AUC=%.4f). "
                "Consider retraining with more data or HPO. "
                "Proceeding with degraded confidence.",
                ticker, horizon, MIN_AUC, best_name, best_auc,
            )
            return SelectionResult(
                model_name=best_name,
                reason=REASON_BELOW_THRESHOLD,
                auc=best_auc,
                from_leaderboard=False,
            )

        # ── Case 3: No artifacts at all ───────────────────────────────────────
        logger.warning(
            "[%s/%s] No trained model artifacts found. "
            "System will auto-train '%s' from scratch. "
            "First prediction for this ticker/horizon will take longer.",
            ticker, horizon, DEFAULT_TRAINING_MODEL,
        )
        return SelectionResult(
            model_name=DEFAULT_TRAINING_MODEL,
            reason=REASON_NO_ARTIFACTS,
            auc=0.0,
            from_leaderboard=False,
        )

    def select_name(self, ticker: str, horizon: str = "1d") -> str:
        """
        Convenience shim that returns only the model name string.

        Prefer ``select()`` when the caller needs to branch on the selection
        reason.  Use this for CLI scripts and simple test assertions.
        """
        return self.select(ticker, horizon).model_name

    def leaderboard(self, ticker: str, horizon: str = "1d") -> list[dict]:
        """Return all trained models for *ticker*/*horizon* sorted by AUC."""
        _, all_entries = self._scan_leaderboard(ticker.upper(), horizon)
        result = []
        for name, _ in all_entries:
            meta = self._load_meta(ticker.upper(), name, horizon)
            if meta:
                result.append({
                    "model":      name,
                    "horizon":    horizon,
                    "auc":        round(meta.get("mean_roc_auc", 0.0), 4),
                    "accuracy":   round(meta.get("mean_accuracy", 0.0), 4),
                    "f1":         round(meta.get("mean_f1", 0.0), 4),
                    "trained_at": meta.get("trained_at", ""),
                })
        return result

    def has_any_model(self, ticker: str, horizon: str = "1d") -> bool:
        """Return True when at least one trained artifact exists."""
        _, all_entries = self._scan_leaderboard(ticker.upper(), horizon)
        return bool(all_entries)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _scan_leaderboard(
        self, ticker: str, horizon: str
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Scan meta files and return two sorted lists:
          eligible    — models whose AUC >= MIN_AUC, sorted desc
          all_entries — every model found, sorted desc

        Scans in _PREFERENCE_ORDER so tiebreaks at equal AUC are deterministic.
        """
        eligible:    list[tuple[str, float]] = []
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
                    "[%s/%s/%s] AUC %.4f < MIN_AUC %.2f — not eligible",
                    ticker, model_name, horizon, auc, MIN_AUC,
                )

        eligible.sort(key=lambda x: x[1], reverse=True)
        all_entries.sort(key=lambda x: x[1], reverse=True)
        return eligible, all_entries

    def _load_meta(
        self, ticker: str, model_name: str, horizon: str
    ) -> Optional[dict]:
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
        """Composite score hook — override to go beyond raw AUC."""
        return meta.get("mean_roc_auc", 0.0)