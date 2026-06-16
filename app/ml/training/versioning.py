"""
app/ml/training/versioning.py
==============================
Model versioning, promotion, rollback, and registry management.

Directory layout
----------------
data/models/
  AAPL/
    xgboost/
      1d/
        versions/
          20260514T094122_a3f7c2d1/
            model.pkl
            meta.json
          20260601T133045_b5e8f1a2/
            model.pkl
            meta.json
        active.json       ← {"version_id": "20260514T094122_a3f7c2d1"}
        versions.json     ← full history list, sorted by trained_at

Version identifier format
-------------------------
  {YYYYmmddTHHMMSS}_{8-char-hex}

  The timestamp prefix is sortable lexicographically.
  The hex suffix is the first 8 characters of SHA-256 over a stable
  string encoding of feature_columns + best_params, so two runs with
  identical configuration at different times produce different ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.logging_config import get_logger

logger = get_logger("versioning")

# ---------------------------------------------------------------------------
# Auto-promotion threshold: only promote a new version if its mean AUC
# exceeds the current active version's AUC by at least this much.
# Set to 0.0 to always promote (legacy behaviour).
# ---------------------------------------------------------------------------
DEFAULT_AUC_IMPROVEMENT_THRESHOLD: float = 0.005

# Maximum versions retained when prune() is called with keep_last=None
DEFAULT_KEEP_LAST: int = 10


# ---------------------------------------------------------------------------
# Version identifier helpers
# ---------------------------------------------------------------------------


def _feature_hash(feature_columns: list[str], best_params: dict) -> str:
    """8-char hex hash of (sorted feature columns, sorted params)."""
    stable = json.dumps(
        {
            "features": sorted(feature_columns),
            "params": {k: best_params[k] for k in sorted(best_params)},
        },
        sort_keys=True,
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:8]


def make_version_id(
    feature_columns: list[str],
    best_params: dict,
    trained_at: Optional[datetime] = None,
) -> str:
    """
    Generate a sortable, content-aware version identifier.

    Format: ``YYYYmmddTHHMMSS_{8-char-hash}``
    """
    ts = (trained_at or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    h = _feature_hash(feature_columns, best_params)
    return f"{ts}_{h}"


def parse_version_timestamp(version_id: str) -> datetime:
    """Parse the timestamp embedded in a version identifier."""
    ts_str = version_id.split("_")[0]
    return datetime.strptime(ts_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


class VersionedArtifactStore:
    """
    Manages the on-disk layout for versioned model artifacts.

    All public methods are atomic where possible (write-to-temp + os.replace).
    """

    def __init__(self, base_models_dir: Path) -> None:
        self._base = base_models_dir

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _slot_dir(self, ticker: str, model_name: str, horizon: str) -> Path:
        return self._base / ticker.upper() / model_name / horizon

    def _versions_dir(self, ticker: str, model_name: str, horizon: str) -> Path:
        return self._slot_dir(ticker, model_name, horizon) / "versions"

    def version_dir(
        self, ticker: str, model_name: str, horizon: str, version_id: str
    ) -> Path:
        return self._versions_dir(ticker, model_name, horizon) / version_id

    def model_path(
        self, ticker: str, model_name: str, horizon: str, version_id: str
    ) -> Path:
        return self.version_dir(ticker, model_name, horizon, version_id) / "model.pkl"

    def meta_path(
        self, ticker: str, model_name: str, horizon: str, version_id: str
    ) -> Path:
        return self.version_dir(ticker, model_name, horizon, version_id) / "meta.json"

    def _active_path(self, ticker: str, model_name: str, horizon: str) -> Path:
        return self._slot_dir(ticker, model_name, horizon) / "active.json"

    def _registry_path(self, ticker: str, model_name: str, horizon: str) -> Path:
        return self._slot_dir(ticker, model_name, horizon) / "versions.json"

    # ── Atomic JSON write ─────────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(path: Path, data: dict | list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── Active pointer ────────────────────────────────────────────────────────

    def get_active_version_id(
        self, ticker: str, model_name: str, horizon: str
    ) -> Optional[str]:
        """Return the active version identifier, or None if no pointer exists."""
        path = self._active_path(ticker, model_name, horizon)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("version_id")
        except Exception as exc:
            logger.warning("Failed to read active.json (%s): %s", path, exc)
            return None

    def set_active_version_id(
        self, ticker: str, model_name: str, horizon: str, version_id: str
    ) -> None:
        path = self._active_path(ticker, model_name, horizon)
        self._atomic_write(path, {"version_id": version_id})
        logger.info(
            "[%s/%s/%s] Active version set → %s",
            ticker,
            model_name,
            horizon,
            version_id,
        )

    # ── Version registry ──────────────────────────────────────────────────────

    def load_registry(self, ticker: str, model_name: str, horizon: str) -> list[dict]:
        """Return the version history list (sorted oldest → newest)."""
        path = self._registry_path(ticker, model_name, horizon)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Failed to read versions.json (%s): %s", path, exc)
            return []

    def save_registry(
        self, ticker: str, model_name: str, horizon: str, entries: list[dict]
    ) -> None:
        path = self._registry_path(ticker, model_name, horizon)
        self._atomic_write(path, entries)

    def add_registry_entry(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        entry: dict,
    ) -> None:
        """Append a new entry and persist the registry."""
        entries = self.load_registry(ticker, model_name, horizon)
        # Remove any stale entry with the same version_id (idempotent re-train)
        entries = [e for e in entries if e.get("version_id") != entry["version_id"]]
        entries.append(entry)
        # Keep sorted by version_id (lexicographic = chronological)
        entries.sort(key=lambda e: e.get("version_id", ""))
        self.save_registry(ticker, model_name, horizon, entries)

    def update_registry_active_flags(
        self, ticker: str, model_name: str, horizon: str, active_version_id: str
    ) -> None:
        """Set is_active=True for active_version_id, False for all others."""
        entries = self.load_registry(ticker, model_name, horizon)
        for e in entries:
            e["is_active"] = e.get("version_id") == active_version_id
        self.save_registry(ticker, model_name, horizon, entries)

    # ── Version existence / enumeration ───────────────────────────────────────

    def version_exists(
        self, ticker: str, model_name: str, horizon: str, version_id: str
    ) -> bool:
        return self.model_path(ticker, model_name, horizon, version_id).exists()

    def list_version_ids(self, ticker: str, model_name: str, horizon: str) -> list[str]:
        """Return all version IDs present on disk, sorted chronologically."""
        vdir = self._versions_dir(ticker, model_name, horizon)
        if not vdir.exists():
            return []
        ids = sorted(
            d.name for d in vdir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        return ids

    # ── Pruning ───────────────────────────────────────────────────────────────

    def prune(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        keep_last: int = DEFAULT_KEEP_LAST,
    ) -> list[str]:
        """
        Delete old version directories, preserving the ``keep_last`` most
        recent and always preserving the currently active version.

        Returns the list of version IDs that were deleted.
        """
        all_ids = self.list_version_ids(ticker, model_name, horizon)
        active_id = self.get_active_version_id(ticker, model_name, horizon)

        # Versions to preserve: the last N + the active one
        preserve = set(all_ids[-keep_last:])
        if active_id:
            preserve.add(active_id)

        deleted: list[str] = []
        for vid in all_ids:
            if vid in preserve:
                continue
            vdir = self.version_dir(ticker, model_name, horizon, vid)
            try:
                shutil.rmtree(vdir)
                deleted.append(vid)
                logger.info(
                    "[%s/%s/%s] Pruned version %s", ticker, model_name, horizon, vid
                )
            except Exception as exc:
                logger.warning(
                    "[%s/%s/%s] Failed to prune %s: %s",
                    ticker,
                    model_name,
                    horizon,
                    vid,
                    exc,
                )

        if deleted:
            # Rebuild registry from remaining versions on disk
            remaining = self.list_version_ids(ticker, model_name, horizon)
            entries = self.load_registry(ticker, model_name, horizon)
            entries = [e for e in entries if e.get("version_id") in set(remaining)]
            self.save_registry(ticker, model_name, horizon, entries)

        return deleted

    # ── Legacy flat-file detection ────────────────────────────────────────────

    def legacy_model_path(self, ticker: str, model_name: str, horizon: str) -> Path:
        """Path to the old-style flat ``{TICKER}_{MODEL}_{HORIZON}.pkl`` artifact."""
        return self._base / f"{ticker.upper()}_{model_name}_{horizon}.pkl"

    def legacy_meta_path(self, ticker: str, model_name: str, horizon: str) -> Path:
        return self._base / f"{ticker.upper()}_{model_name}_{horizon}_meta.json"

    def has_legacy_artifact(self, ticker: str, model_name: str, horizon: str) -> bool:
        return self.legacy_model_path(ticker, model_name, horizon).exists()

    def migrate_legacy_artifact(
        self,
        ticker: str,
        model_name: str,
        horizon: str,
        feature_columns: list[str],
        meta: dict,
    ) -> Optional[str]:
        """
        Migrate a legacy flat artifact into the versioned layout.

        The version ID is derived from the trained_at field in the meta dict
        (if present) so the migrated version retains its original timestamp.

        Returns the new version_id, or None if migration was not possible.
        """
        legacy_pkl = self.legacy_model_path(ticker, model_name, horizon)
        if not legacy_pkl.exists():
            return None

        trained_at_str = meta.get("trained_at", "")
        try:
            ts = datetime.strptime(trained_at_str[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        version_id = make_version_id(feature_columns, meta.get("best_params", {}), ts)
        vdir = self.version_dir(ticker, model_name, horizon, version_id)
        vdir.mkdir(parents=True, exist_ok=True)

        dest_pkl = vdir / "model.pkl"
        dest_meta = vdir / "meta.json"

        if not dest_pkl.exists():
            shutil.copy2(legacy_pkl, dest_pkl)
            logger.info(
                "[%s/%s/%s] Migrated legacy artifact → %s",
                ticker,
                model_name,
                horizon,
                version_id,
            )

        entry = _build_registry_entry(version_id, meta, feature_columns, is_active=True)
        self._atomic_write(dest_meta, meta)
        self.add_registry_entry(ticker, model_name, horizon, entry)
        self.set_active_version_id(ticker, model_name, horizon, version_id)

        # Remove legacy flat files so they don't confuse the old selector
        try:
            legacy_pkl.unlink(missing_ok=True)
            self.legacy_meta_path(ticker, model_name, horizon).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not remove legacy files: %s", exc)

        return version_id


# ---------------------------------------------------------------------------
# Registry entry builder
# ---------------------------------------------------------------------------


def _build_registry_entry(
    version_id: str,
    training_result_dict: dict,
    feature_columns: list[str],
    is_active: bool = False,
) -> dict:
    """Build a standardised registry entry from a TrainingResult.to_dict() output."""
    return {
        "version_id": version_id,
        "trained_at": training_result_dict.get("trained_at", ""),
        "trigger_reason": training_result_dict.get("trigger_reason", ""),
        "mean_roc_auc": training_result_dict.get("mean_roc_auc", 0.0),
        "mean_accuracy": training_result_dict.get("mean_accuracy", 0.0),
        "mean_f1": training_result_dict.get("mean_f1", 0.0),
        "n_features": training_result_dict.get("n_features", len(feature_columns)),
        "feature_hash": _feature_hash(
            feature_columns, training_result_dict.get("best_params", {})
        ),
        "best_params": training_result_dict.get("best_params", {}),
        "is_active": is_active,
        "training_duration_s": training_result_dict.get("training_duration_s", 0.0),
        "n_folds": training_result_dict.get("n_folds", 0),
    }
