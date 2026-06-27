from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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
from app.ml.feature_engineering import HORIZONS, FeatureEngineer
from app.ml.training.trainer import ModelTrainer
from app.services.model_selector import (
    ALL_TRAINING_MODELS,
    REASON_BELOW_THRESHOLD,
    REASON_LEADERBOARD,
    REASON_NO_ARTIFACTS,
    ModelSelector,
)
from app.services.signal_fusion import FusedSignal, SignalFusionService
from configs.settings import settings

logger = get_logger("prediction_service")

# Type alias for the optional progress callback.
# Signature: (stage: str, message: str, pct: int) -> None
ProgressCallback = Callable[[str, str, int], None]


def _noop(stage: str, message: str, pct: int = 0) -> None:
    """Default no-op progress callback used when no callback is supplied."""


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
    feature_selection_meta: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────


class PredictionService:
    """
    End-to-end prediction pipeline:
      ModelSelector → (train all models if needed) → best model → SHAP → SignalFusion

    The ``progress_callback`` parameter allows callers to receive stage-by-stage
    progress notifications without coupling the service to any specific transport.
    The callback is invoked with ``(stage: str, message: str, pct: int)`` at each
    major pipeline checkpoint.  When no callback is supplied the default no-op
    is used so all existing call sites continue to work unchanged.

    Example (streaming endpoint)::

        def my_callback(stage, message, pct):
            queue.put_nowait(sse_event("progress", {...}))

        svc.predict("AAPL", progress_callback=my_callback)
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
        apply_feature_selection: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PredictionResponse:
        """
        Run the full prediction pipeline for *ticker* / *horizon*.

        When no artifacts exist the service trains all models in
        ALL_TRAINING_MODELS before selecting the best one by AUC.

        Parameters
        ----------
        ticker:             Stock ticker symbol.
        horizon:            Prediction horizon key.
        use_cache:          Whether to use cached market data.
        run_fusion:         Whether to run LLM signal fusion.
        apply_feature_selection: Whether to apply stability-based feature selection (default True — always on in production).
        progress_callback:  Optional ``(stage, message, pct) -> None`` callback
                            invoked at each major pipeline stage.  When
                            ``None`` (default), a no-op is used so all existing
                            call sites are unaffected.

        Raises:
            PredictionError: On any unrecoverable failure.
        """
        cb: ProgressCallback = progress_callback or _noop

        ticker = ticker.upper().strip()
        horizon = horizon.strip()

        if horizon not in HORIZONS:
            raise PredictionError(
                f"Invalid horizon '{horizon}'. Valid: {list(HORIZONS.keys())}"
            )

        try:
            logger.info("Prediction pipeline: ticker=%s horizon=%s", ticker, horizon)

            # ── 1. Ingest ─────────────────────────────────────────────────────
            cb("ingest", f"Fetching market data for {ticker}…", 5)
            raw_df = ingest_market_data(ticker, use_cache=use_cache)
            cb(
                "ingest",
                f"Loaded {len(raw_df):,} rows of market data for {ticker}",
                10,
            )

            # ── 2. Feature engineering ────────────────────────────────────────
            cb("features", f"Engineering features from {len(raw_df):,} rows…", 15)
            feature_df = self.engineer.build_features(raw_df)
            X, y = self.engineer.split_X_y(feature_df, horizon=horizon)
            cb(
                "features",
                f"Built {X.shape[1]} features × {X.shape[0]} rows",
                22,
            )

            # ── 3. Feature selection (Improvement 4) ──────────────────────────
            # Stability-based selection is now a standard pipeline stage baked
            # into ModelTrainer.train().  We pass apply_feature_selection through
            # to load_or_train so callers can disable it (e.g., in tests) but it
            # defaults to True in production.  The selection runs inside each
            # walk-forward training window so there is zero look-ahead bias.
            # No manual FeatureSelector call is needed here — the trainer handles
            # it and persists the stable feature set in the artifact, so on
            # subsequent calls we simply load the model and align X to its saved
            # feature_columns (done below in step 6).
            if apply_feature_selection:
                cb(
                    "features",
                    "Stability feature selection enabled (runs inside trainer)…",
                    25,
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
                cb(
                    "training",
                    f"No trained models found — training {len(ALL_TRAINING_MODELS)} "
                    f"models for {ticker}/{horizon}…",
                    30,
                )

                best_result = None
                n_models = len(ALL_TRAINING_MODELS)

                for model_idx, model_name in enumerate(ALL_TRAINING_MODELS):
                    pct_start = 30 + model_idx * (40 // n_models)
                    pct_end = 30 + (model_idx + 1) * (40 // n_models)

                    # Emit per-fold progress via a nested callback injected into
                    # the trainer.  We approximate fold progress within the
                    # model's pct range.
                    n_folds = settings.WALK_FORWARD_FOLDS

                    def _make_fold_cb(m_name, p_start, p_end, n_f):
                        _fold_counter = [0]

                        def _fold_cb(fold_result):
                            _fold_counter[0] += 1
                            fold_num = _fold_counter[0]
                            pct = p_start + int((fold_num / n_f) * (p_end - p_start))
                            cb(
                                "training",
                                f"Training {m_name} — fold {fold_num}/{n_f} complete "
                                f"(AUC {fold_result.roc_auc:.4f})",
                                pct,
                            )

                        return _fold_cb

                    fold_cb = _make_fold_cb(model_name, pct_start, pct_end, n_folds)

                    cb(
                        "training",
                        f"[{model_idx + 1}/{n_models}] Training {model_name}…",
                        pct_start,
                    )
                    try:
                        _, train_result = self.trainer.train(
                            model_name=model_name,
                            X=X,
                            y=y,
                            ticker=ticker,
                            horizon=horizon,
                            trigger_reason="no_artifacts_train_all",
                            fold_callback=fold_cb,
                            run_feature_selection=apply_feature_selection,
                        )
                        logger.info(
                            "[%s/%s] Trained %s: AUC=%.4f",
                            ticker,
                            horizon,
                            model_name,
                            train_result.mean_roc_auc,
                        )
                        cb(
                            "training",
                            f"{model_name} trained — AUC {train_result.mean_roc_auc:.4f}",
                            pct_end,
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
                        cb(
                            "training",
                            f"{model_name} training failed — skipping",
                            pct_end,
                        )

                if best_result is None:
                    raise PredictionError(
                        f"All model training attempts failed for {ticker}/{horizon}."
                    )

                auto_trained = True
                sel = self.selector.select(ticker, horizon=horizon)
                cb(
                    "training",
                    f"Best model: {sel.model_name} (AUC {sel.auc:.4f})",
                    72,
                )
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
                cb(
                    "model_select",
                    f"Using {sel.model_name} (AUC {sel.auc:.4f} — below threshold, "
                    f"confidence degraded)",
                    35,
                )
            else:
                logger.info(
                    "[%s/%s] Auto-selected: '%s' (AUC=%.4f).",
                    ticker,
                    horizon,
                    sel.model_name,
                    sel.auc,
                )
                cb(
                    "model_select",
                    f"Auto-selected {sel.model_name} (AUC {sel.auc:.4f})",
                    35,
                )

            confidence_degraded = sel.reason != REASON_LEADERBOARD

            # ── 5. Load or retrain on artifact issues ─────────────────────────
            cb("model_load", f"Loading {sel.model_name} model artifact…", 75)
            model, feature_columns, train_result_incr = self.trainer.load_or_train(
                ticker=ticker,
                model_name=sel.model_name,
                X=X,
                y=y,
                horizon=horizon,
                run_feature_selection=apply_feature_selection,
            )
            _fs_meta: Optional[dict] = None
            if train_result_incr is not None:
                auto_trained = True
                _fs_meta = train_result_incr.feature_selection_meta
                cb(
                    "training",
                    f"Incremental retrain complete — AUC {train_result_incr.mean_roc_auc:.4f}",
                    75,
                )
            else:
                try:
                    _active_id = self.trainer._store.get_active_version_id(
                        ticker, sel.model_name, horizon
                    )
                    if _active_id:
                        _entries = self.trainer._store.load_registry(
                            ticker, sel.model_name, horizon
                        )
                        _active_entry = next(
                            (e for e in _entries if e.get("version_id") == _active_id),
                            None,
                        )
                        if _active_entry:
                            _fs_meta = _active_entry.get("feature_selection")
                except Exception as _exc:
                    logger.debug("Could not read feature_selection_meta: %s", _exc)

            # ── 6. Align features ─────────────────────────────────────────────
            missing = set(feature_columns) - set(X.columns)
            if missing:
                raise PredictionError(
                    f"Feature mismatch for {ticker}/{sel.model_name}/{horizon}",
                    detail=f"Missing columns: {sorted(missing)}",
                )
            X_aligned = X[feature_columns]

            # ── 7. Inference ──────────────────────────────────────────────────
            cb("inference", f"Running inference with {sel.model_name}…", 80)
            X_latest = X_aligned.iloc[[-1]]
            pred = int(model.predict(X_latest)[0])

            p_bullish = round_prob(float(model.predict_proba(X_latest)[0, 1]))
            p_bearish = round_prob(1.0 - p_bullish)
            prob = round_prob(p_bullish if pred == 1 else p_bearish)

            # ── 8. SHAP ───────────────────────────────────────────────────────
            cb("shap", "Running SHAP analysis…", 84)
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
                confidence_label=confidence_label(p_bullish),
                shap_explanation=shap_exp,
                narrative=narrative,
                latest_close=latest_close,
                feature_snapshot=snapshot,
                fused_signal=None,
                auto_trained=auto_trained,
                feature_selection_meta=_fs_meta,
            )

            # ── 11. Signal fusion ─────────────────────────────────────────────
            if run_fusion and settings.OPENAI_API_KEY:
                cb("news", "Retrieving news intelligence…", 88)
                try:
                    fused = self.fusion_service.fuse(
                        ticker,
                        response,
                        horizon=horizon,
                    )
                    response.fused_signal = fused

                    if fused.fusion_applied:
                        cb(
                            "fusion",
                            f"LLM signal fusion complete — {fused.final_direction} "
                            f"({fused.final_confidence})",
                            95,
                        )
                    else:
                        cb(
                            "fusion",
                            f"Rule-based fusion — {fused.final_direction}",
                            95,
                        )

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
                    cb("fusion", "Signal fusion skipped — using ML-only signal", 95)
            else:
                logger.info(
                    "[%s/%s] Fusion skipped (run_fusion=%s, api_key=%s)",
                    ticker,
                    horizon,
                    run_fusion,
                    bool(settings.OPENAI_API_KEY),
                )
                cb("fusion", "Signal fusion skipped", 95)

            cb(
                "complete",
                f"Prediction complete — "
                f"{'BULLISH' if pred else 'BEARISH'} "
                f"(p={p_bullish:.1%}, conf={response.confidence_label})",
                100,
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
