"""
Unit tests for ``app.services.model_selector.ModelSelector``.

Covers the three selection branches (leaderboard hit, below-threshold,
no-artifacts), the versioned vs. legacy meta.json resolution order, and
the read-only convenience helpers (``select_name``, ``leaderboard``,
``has_any_model``).

All tests operate on a temporary directory standing in for
``settings.MODELS_DIR`` — no real training artifacts are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.model_selector import (
    ALL_TRAINING_MODELS,
    MIN_AUC,
    REASON_BELOW_THRESHOLD,
    REASON_LEADERBOARD,
    REASON_NO_ARTIFACTS,
    ModelSelector,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_legacy_meta(
    model_dir: Path, ticker: str, model_name: str, horizon: str, auc: float
) -> None:
    """Write a legacy flat ``{TICKER}_{model}_{horizon}_meta.json`` file."""
    meta_path = model_dir / f"{ticker}_{model_name}_{horizon}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "mean_roc_auc": auc,
                "mean_accuracy": 0.55,
                "mean_f1": 0.5,
                "trained_at": "2026-01-01T00:00:00Z",
                "version_id": "legacy",
            }
        ),
        encoding="utf-8",
    )


def _write_versioned_meta(
    model_dir: Path, ticker: str, model_name: str, horizon: str, auc: float
) -> None:
    """Write a versioned artifact tree with an ``active.json`` pointer."""
    version_id = "v1"
    version_dir = model_dir / ticker / model_name / horizon / "versions" / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "meta.json").write_text(
        json.dumps({"mean_roc_auc": auc, "mean_accuracy": 0.6, "mean_f1": 0.55}),
        encoding="utf-8",
    )
    active_path = model_dir / ticker / model_name / horizon / "active.json"
    active_path.write_text(json.dumps({"version_id": version_id}), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# No artifacts
# ─────────────────────────────────────────────────────────────────────────────


class TestNoArtifacts:
    def test_returns_no_artifacts_reason_when_directory_empty(self, tmp_path):
        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        assert result.reason == REASON_NO_ARTIFACTS
        assert result.model_name == ALL_TRAINING_MODELS[0]
        assert result.auc == 0.0
        assert result.from_leaderboard is False

    def test_has_any_model_false_when_empty(self, tmp_path):
        selector = ModelSelector(model_dir=tmp_path)
        assert selector.has_any_model("AAPL") is False

    def test_leaderboard_empty_list_when_no_artifacts(self, tmp_path):
        selector = ModelSelector(model_dir=tmp_path)
        assert selector.leaderboard("AAPL") == []


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard hit (>= MIN_AUC)
# ─────────────────────────────────────────────────────────────────────────────


class TestLeaderboardHit:
    def test_selects_highest_auc_model_above_threshold(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", MIN_AUC + 0.01)
        _write_legacy_meta(tmp_path, "AAPL", "lightgbm", "1d", MIN_AUC + 0.10)

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        assert result.reason == REASON_LEADERBOARD
        assert result.model_name == "lightgbm"
        assert result.auc == pytest.approx(MIN_AUC + 0.10)
        assert result.from_leaderboard is True

    def test_select_name_convenience_shim(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.75)
        selector = ModelSelector(model_dir=tmp_path)
        assert selector.select_name("AAPL", "1d") == "xgboost"

    def test_ticker_is_uppercased(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.80)
        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("aapl", horizon="1d")
        assert result.reason == REASON_LEADERBOARD
        assert result.model_name == "xgboost"

    def test_different_horizons_are_isolated(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.90)
        selector = ModelSelector(model_dir=tmp_path)

        result_1d = selector.select("AAPL", horizon="1d")
        result_7d = selector.select("AAPL", horizon="7d")

        assert result_1d.reason == REASON_LEADERBOARD
        assert result_7d.reason == REASON_NO_ARTIFACTS


# ─────────────────────────────────────────────────────────────────────────────
# Below-threshold branch
# ─────────────────────────────────────────────────────────────────────────────


class TestBelowThreshold:
    def test_returns_best_available_when_all_below_min_auc(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", MIN_AUC - 0.10)
        _write_legacy_meta(tmp_path, "AAPL", "lightgbm", "1d", MIN_AUC - 0.02)

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        assert result.reason == REASON_BELOW_THRESHOLD
        assert result.model_name == "lightgbm"  # closest to threshold
        assert result.from_leaderboard is False

    def test_boundary_auc_exactly_at_min_counts_as_eligible(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", MIN_AUC)
        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")
        assert result.reason == REASON_LEADERBOARD


# ─────────────────────────────────────────────────────────────────────────────
# Versioned layout takes priority over legacy
# ─────────────────────────────────────────────────────────────────────────────


class TestVersionedLayoutPriority:
    def test_versioned_meta_preferred_over_legacy(self, tmp_path):
        # Legacy says AUC below threshold; versioned says above — versioned
        # must win, proving the resolution order in ``_load_meta``.
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.10)
        _write_versioned_meta(tmp_path, "AAPL", "xgboost", "1d", 0.95)

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        assert result.reason == REASON_LEADERBOARD
        assert result.auc == pytest.approx(0.95)

    def test_corrupt_active_json_falls_back_to_legacy(self, tmp_path):
        active_dir = tmp_path / "AAPL" / "xgboost" / "1d"
        active_dir.mkdir(parents=True)
        (active_dir / "active.json").write_text("{not valid json", encoding="utf-8")
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.80)

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        assert result.reason == REASON_LEADERBOARD
        assert result.auc == pytest.approx(0.80)

    def test_active_json_missing_version_id_falls_back_to_legacy(self, tmp_path):
        active_dir = tmp_path / "AAPL" / "xgboost" / "1d"
        active_dir.mkdir(parents=True)
        (active_dir / "active.json").write_text(json.dumps({}), encoding="utf-8")
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.77)

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")
        assert result.auc == pytest.approx(0.77)


# ─────────────────────────────────────────────────────────────────────────────
# Leaderboard listing
# ─────────────────────────────────────────────────────────────────────────────


class TestLeaderboardListing:
    def test_leaderboard_sorted_descending_by_auc(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.55)
        _write_legacy_meta(tmp_path, "AAPL", "lightgbm", "1d", 0.70)
        _write_legacy_meta(tmp_path, "AAPL", "random_forest", "1d", 0.60)

        selector = ModelSelector(model_dir=tmp_path)
        board = selector.leaderboard("AAPL", "1d")

        aucs = [entry["auc"] for entry in board]
        assert aucs == sorted(aucs, reverse=True)
        assert board[0]["model"] == "lightgbm"

    def test_leaderboard_entries_include_expected_keys(self, tmp_path):
        _write_legacy_meta(tmp_path, "AAPL", "xgboost", "1d", 0.65)
        selector = ModelSelector(model_dir=tmp_path)
        board = selector.leaderboard("AAPL", "1d")

        assert len(board) == 1
        entry = board[0]
        for key in (
            "model",
            "horizon",
            "auc",
            "accuracy",
            "f1",
            "trained_at",
            "version_id",
        ):
            assert key in entry


# ─────────────────────────────────────────────────────────────────────────────
# Malformed / partial data handled gracefully
# ─────────────────────────────────────────────────────────────────────────────


class TestMalformedData:
    def test_corrupt_legacy_json_is_skipped_not_raised(self, tmp_path):
        bad_path = tmp_path / "AAPL_xgboost_1d_meta.json"
        bad_path.write_text("{this is not json", encoding="utf-8")

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        # Should degrade to "no artifacts" rather than raising.
        assert result.reason == REASON_NO_ARTIFACTS

    def test_missing_mean_roc_auc_key_defaults_to_zero(self, tmp_path):
        meta_path = tmp_path / "AAPL_xgboost_1d_meta.json"
        meta_path.write_text(json.dumps({"trained_at": "2026-01-01"}), encoding="utf-8")

        selector = ModelSelector(model_dir=tmp_path)
        result = selector.select("AAPL", horizon="1d")

        assert result.reason == REASON_BELOW_THRESHOLD
        assert result.auc == 0.0
