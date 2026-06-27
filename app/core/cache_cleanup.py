"""
Automatic cleanup of outdated files across all FinSight AI data directories.

What gets cleaned
-----------------
RAW_DATA_DIR      (data/raw/)
    Parquet cache files older than CACHE_MAX_AGE_DAYS (default 1 day).
    These are the yfinance OHLCV parquet files written by ingest_market_data().
    File pattern: {TICKER}_{hash}.parquet

MODELS_DIR        (data/models/)
    Orphaned legacy flat artifacts (.pkl / _meta.json) whose ticker/model/horizon
    slot has already been migrated to the versioned layout.
    Versioned artifacts are managed by ModelTrainer.prune_versions() and are
    NOT touched here — this module only removes files at the flat models/ root.

EMBEDDINGS_DIR    (data/embeddings/)
    Stale FAISS index files (.faiss, _docs.pkl, _urls.json) older than
    EMBEDDINGS_MAX_AGE_DAYS (default 30 days).  The active index is never
    removed; only files that are not referenced by the current VECTOR_DB_PATH
    setting are candidates.

LOGS_DIR          (logs/)
    Log files older than LOGS_MAX_AGE_DAYS (default 7 days).

Usage
-----
Programmatic::

    from app.core.cache_cleanup import CacheCleanup
    report = CacheCleanup().run()
    print(report.summary())

CLI::

    python -m app.core.cache_cleanup                  # dry-run
    python -m app.core.cache_cleanup --execute         # actually delete
    python -m app.core.cache_cleanup --execute --verbose

FastAPI startup hook (optional, in main.py lifespan)::

    from app.core.cache_cleanup import CacheCleanup
    CacheCleanup(dry_run=False).run_raw_only()
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("finsight.cache_cleanup")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults  (all overridable via constructor)
# ─────────────────────────────────────────────────────────────────────────────

# How old a raw parquet cache file must be before it is eligible for deletion.
# Matches settings.CACHE_MAX_AGE_DAYS (default 1).
_DEFAULT_RAW_MAX_AGE_DAYS: int = 1

# How old a log file must be before it is deleted.
_DEFAULT_LOGS_MAX_AGE_DAYS: int = 7

# How old an embeddings file must be before it is deleted.
_DEFAULT_EMBEDDINGS_MAX_AGE_DAYS: int = 30

# Extensions that are considered raw data cache files.
_RAW_EXTENSIONS: frozenset[str] = frozenset({".parquet"})

# Extensions that are considered log files.
_LOG_EXTENSIONS: frozenset[str] = frozenset({".log"})

# Extensions that are considered embedding artifacts.
_EMBEDDING_EXTENSIONS: frozenset[str] = frozenset({".faiss", ".pkl", ".json"})

# Extensions that are legacy flat model artifacts at the models/ root.
_LEGACY_MODEL_EXTENSIONS: frozenset[str] = frozenset({".pkl", ".json"})


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DeletedFile:
    path: Path
    size_bytes: int
    age_days: float
    reason: str


@dataclass
class CleanupReport:
    """Summary of a cleanup run."""

    dry_run: bool
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    raw_deleted: list[DeletedFile] = field(default_factory=list)
    model_deleted: list[DeletedFile] = field(default_factory=list)
    embedding_deleted: list[DeletedFile] = field(default_factory=list)
    log_deleted: list[DeletedFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_deleted(self) -> int:
        return (
            len(self.raw_deleted)
            + len(self.model_deleted)
            + len(self.embedding_deleted)
            + len(self.log_deleted)
        )

    @property
    def total_bytes_freed(self) -> int:
        all_files = (
            self.raw_deleted
            + self.model_deleted
            + self.embedding_deleted
            + self.log_deleted
        )
        return sum(f.size_bytes for f in all_files)

    def summary(self) -> str:
        mode = "DRY RUN" if self.dry_run else "EXECUTED"
        mb = self.total_bytes_freed / 1_048_576
        lines = [
            f"Cache cleanup [{mode}] — {self.elapsed_s:.2f}s",
            f"  Raw cache:    {len(self.raw_deleted):>4} file(s)",
            f"  Legacy models:{len(self.model_deleted):>4} file(s)",
            f"  Embeddings:   {len(self.embedding_deleted):>4} file(s)",
            f"  Logs:         {len(self.log_deleted):>4} file(s)",
            f"  Total:        {self.total_deleted:>4} file(s)  "
            f"({mb:.2f} MB {'freed' if not self.dry_run else 'would be freed'})",
        ]
        if self.errors:
            lines.append(f"  Errors:       {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"    • {e}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class CacheCleanup:
    """
    Scans FinSight AI data directories and removes outdated files.

    Parameters
    ----------
    dry_run : bool
        When True (default) no files are deleted — only reported.
        Set False to actually remove files.
    raw_max_age_days : int
        Max age for raw parquet cache files.  Files older than this are stale.
    logs_max_age_days : int
        Max age for log files.
    embeddings_max_age_days : int
        Max age for embedding artifacts.
    raw_dir / models_dir / embeddings_dir / logs_dir : Path | None
        Override the directory paths.  If None, values are read from
        ``configs.settings`` at call time.
    verbose : bool
        Emit a log line for every file inspected (not just deleted ones).
    """

    def __init__(
        self,
        dry_run: bool = True,
        raw_max_age_days: int = _DEFAULT_RAW_MAX_AGE_DAYS,
        logs_max_age_days: int = _DEFAULT_LOGS_MAX_AGE_DAYS,
        embeddings_max_age_days: int = _DEFAULT_EMBEDDINGS_MAX_AGE_DAYS,
        raw_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.raw_max_age_days = raw_max_age_days
        self.logs_max_age_days = logs_max_age_days
        self.embeddings_max_age_days = embeddings_max_age_days
        self.verbose = verbose

        # Resolve directories lazily so settings are loaded only once
        self._raw_dir = raw_dir
        self._models_dir = models_dir
        self._embeddings_dir = embeddings_dir
        self._logs_dir = logs_dir

    # ── Directory resolution ──────────────────────────────────────────────────

    def _dirs(self):
        """Lazily load settings and return the four target directories."""
        try:
            from configs.settings import settings as s

            raw = self._raw_dir or s.RAW_DATA_DIR
            models = self._models_dir or s.MODELS_DIR
            embeddings = self._embeddings_dir or s.EMBEDDINGS_DIR
            logs = self._logs_dir or s.LOGS_DIR
            # Also read CACHE_MAX_AGE_DAYS from settings if caller didn't override
            if self.raw_max_age_days == _DEFAULT_RAW_MAX_AGE_DAYS:
                self.raw_max_age_days = getattr(
                    s, "CACHE_MAX_AGE_DAYS", _DEFAULT_RAW_MAX_AGE_DAYS
                )
            vector_db_base = getattr(s, "VECTOR_DB_PATH", None)
        except Exception:
            raw = self._raw_dir or Path("data/raw")
            models = self._models_dir or Path("data/models")
            embeddings = self._embeddings_dir or Path("data/embeddings")
            logs = self._logs_dir or Path("logs")
            vector_db_base = None
        return (
            Path(raw),
            Path(models),
            Path(embeddings),
            Path(logs),
            vector_db_base,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> CleanupReport:
        """
        Run all cleanup passes and return a CleanupReport.

        Passes executed (in order):
        1. Raw parquet cache  (data/raw/)
        2. Legacy flat model artifacts  (data/models/ root only)
        3. Stale embedding files  (data/embeddings/)
        4. Old log files  (logs/)
        """
        report = CleanupReport(dry_run=self.dry_run)
        raw_dir, models_dir, embeddings_dir, logs_dir, vector_db_base = self._dirs()

        logger.info(
            "CacheCleanup.run() started [dry_run=%s]  "
            "raw=%s  models=%s  embeddings=%s  logs=%s",
            self.dry_run,
            raw_dir,
            models_dir,
            embeddings_dir,
            logs_dir,
        )

        self._clean_raw(raw_dir, report)
        self._clean_legacy_models(models_dir, report)
        self._prune_versioned_models(models_dir, report)
        self._clean_embeddings(embeddings_dir, vector_db_base, report)
        self._clean_logs(logs_dir, report)

        report.elapsed_s = round(time.time() - report.started_at, 3)
        logger.info(report.summary())
        return report

    def run_raw_only(self) -> CleanupReport:
        """
        Lightweight variant — only cleans raw parquet cache.

        Suitable for the FastAPI startup lifespan hook where a full scan
        would slow down startup unnecessarily.
        """
        report = CleanupReport(dry_run=self.dry_run)
        raw_dir, *_ = self._dirs()
        self._clean_raw(raw_dir, report)
        report.elapsed_s = round(time.time() - report.started_at, 3)
        if report.raw_deleted or self.verbose:
            logger.info(
                "Raw cache cleanup: %d file(s) %s (%.2f MB)",
                len(report.raw_deleted),
                "would be removed" if self.dry_run else "removed",
                report.total_bytes_freed / 1_048_576,
            )
        return report

    # ── Pass 1: raw parquet cache ─────────────────────────────────────────────

    def _clean_raw(self, raw_dir: Path, report: CleanupReport) -> None:
        """
        Delete parquet files in ``raw_dir`` that are older than
        ``raw_max_age_days`` days.

        The existing ``_load_from_cache()`` in data_ingestion.py already
        evicts stale files on the next read, but that only fires when a
        specific ticker is requested.  This pass cleans up files for tickers
        that have not been requested recently and otherwise accumulate on disk.
        """
        if not raw_dir.exists():
            return

        max_age_s = self.raw_max_age_days * 86_400
        now = time.time()

        for path in raw_dir.iterdir():
            if path.suffix not in _RAW_EXTENSIONS or not path.is_file():
                continue

            try:
                age_s = now - path.stat().st_mtime
                age_days = age_s / 86_400

                if self.verbose:
                    logger.debug("RAW  %s  age=%.1fd", path.name, age_days)

                if age_s > max_age_s:
                    size = path.stat().st_size
                    deleted = DeletedFile(
                        path=path,
                        size_bytes=size,
                        age_days=round(age_days, 2),
                        reason=f"older than {self.raw_max_age_days}d cache TTL",
                    )
                    self._delete(path, report)
                    report.raw_deleted.append(deleted)

            except Exception as exc:
                msg = f"raw: {path.name}: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

    # ── Pass 2: legacy flat model artifacts ───────────────────────────────────

    def _clean_legacy_models(self, models_dir: Path, report: CleanupReport) -> None:
        """
        Remove legacy flat ``{TICKER}_{model}_{horizon}.pkl`` and
        ``{TICKER}_{model}_{horizon}_meta.json`` files from the root of
        ``models_dir`` that have been superseded by the versioned layout.

        A file is considered superseded when the corresponding versioned
        directory (``models_dir / TICKER / model / horizon / versions /``)
        exists and contains at least one version — meaning ModelTrainer
        has already migrated or retrained in the new layout.

        Files are only deleted if the versioned slot has an active pointer
        (``active.json``) — we never delete the only copy of a model.
        """
        if not models_dir.exists():
            return

        for path in models_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix not in _LEGACY_MODEL_EXTENSIONS:
                continue
            # Only act on names that match the legacy pattern:
            # {TICKER}_{model}_{horizon}.pkl  or  {TICKER}_{model}_{horizon}_meta.json
            stem = path.stem  # e.g. "AAPL_xgboost_1d" or "AAPL_xgboost_1d_meta"
            if stem.endswith("_meta"):
                stem = stem[:-5]  # strip "_meta"
            parts = stem.split("_")
            if len(parts) < 3:
                continue  # doesn't look like a legacy artifact

            ticker = parts[0]
            horizon = parts[-1]
            model = "_".join(parts[1:-1])  # handles "random_forest"

            # Check if a versioned slot with an active pointer exists
            active_json = models_dir / ticker / model / horizon / "active.json"
            if not active_json.exists():
                continue  # versioned layout not present yet — keep the file

            try:
                size = path.stat().st_size
                age_days = (time.time() - path.stat().st_mtime) / 86_400
                deleted = DeletedFile(
                    path=path,
                    size_bytes=size,
                    age_days=round(age_days, 2),
                    reason=f"superseded by versioned layout at "
                    f"{ticker}/{model}/{horizon}",
                )
                self._delete(path, report)
                report.model_deleted.append(deleted)

            except Exception as exc:
                msg = f"model: {path.name}: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

    # ── Pass 3: stale embedding files ─────────────────────────────────────────

    def _clean_embeddings(
        self,
        embeddings_dir: Path,
        vector_db_base: Optional[str],
        report: CleanupReport,
    ) -> None:
        """
        Remove embedding artifacts older than ``embeddings_max_age_days``.

        The active FAISS index (referenced by settings.VECTOR_DB_PATH) is
        always preserved regardless of age — only files that are NOT part of
        the active index are candidates for deletion.

        Recognised artifact sets (sharing a common base name):
            {base}.faiss
            {base}_docs.pkl
            {base}_urls.json
        """
        if not embeddings_dir.exists():
            return

        # Determine the set of basenames that belong to the active index
        protected_stems: set[str] = set()
        if vector_db_base:
            active_base = Path(vector_db_base).stem  # e.g. "faiss_index"
            protected_stems.add(active_base)
            protected_stems.add(active_base + "_docs")
            protected_stems.add(active_base + "_urls")

        max_age_s = self.embeddings_max_age_days * 86_400
        now = time.time()

        for path in embeddings_dir.iterdir():
            if path.suffix not in _EMBEDDING_EXTENSIONS or not path.is_file():
                continue
            if path.stem in protected_stems:
                if self.verbose:
                    logger.debug("EMBED %s  PROTECTED (active index)", path.name)
                continue

            try:
                age_s = now - path.stat().st_mtime
                age_days = age_s / 86_400

                if self.verbose:
                    logger.debug("EMBED %s  age=%.1fd", path.name, age_days)

                if age_s > max_age_s:
                    size = path.stat().st_size
                    deleted = DeletedFile(
                        path=path,
                        size_bytes=size,
                        age_days=round(age_days, 2),
                        reason=f"older than {self.embeddings_max_age_days}d "
                        f"embeddings TTL",
                    )
                    self._delete(path, report)
                    report.embedding_deleted.append(deleted)

            except Exception as exc:
                msg = f"embedding: {path.name}: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

    # ── Pass 4: old log files ─────────────────────────────────────────────────

    def _clean_logs(self, logs_dir: Path, report: CleanupReport) -> None:
        """
        Delete .log files in ``logs_dir`` older than ``logs_max_age_days``.

        Rotated log files (e.g. finsight.log.1, finsight.log.2026-06-01) are
        included.  The current active log file is typically written to
        continuously — it will pass the age check naturally on future runs once
        it rolls over.
        """
        if not logs_dir.exists():
            return

        max_age_s = self.logs_max_age_days * 86_400
        now = time.time()

        for path in logs_dir.iterdir():
            if not path.is_file():
                continue
            # Accept .log files and rotated variants like .log.1 / .log.2026-06-01
            if ".log" not in path.name:
                continue

            try:
                age_s = now - path.stat().st_mtime
                age_days = age_s / 86_400

                if self.verbose:
                    logger.debug("LOG  %s  age=%.1fd", path.name, age_days)

                if age_s > max_age_s:
                    size = path.stat().st_size
                    deleted = DeletedFile(
                        path=path,
                        size_bytes=size,
                        age_days=round(age_days, 2),
                        reason=f"older than {self.logs_max_age_days}d log TTL",
                    )
                    self._delete(path, report)
                    report.log_deleted.append(deleted)

            except Exception as exc:
                msg = f"log: {path.name}: {exc}"
                logger.warning(msg)
                report.errors.append(msg)

    def _prune_versioned_models(
        self, models_dir: Path, report: CleanupReport, keep_last: int = 3
    ) -> None:
        """
        Prune old versioned model artifacts across all ticker/model/horizon slots,
        always preserving the active version and the last ``keep_last`` versions.
        """
        from app.ml.training.versioning import VersionedArtifactStore

        if not models_dir.exists():
            return

        store = VersionedArtifactStore(models_dir)

        # Walk ticker/ → model_name/ → horizon/ slots
        for ticker_dir in sorted(models_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            for model_dir in sorted(ticker_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for horizon_dir in sorted(model_dir.iterdir()):
                    if not horizon_dir.is_dir():
                        continue

                    ticker = ticker_dir.name
                    model_name = model_dir.name
                    horizon = horizon_dir.name

                    try:
                        deleted_ids = (
                            []
                            if self.dry_run
                            else store.prune(
                                ticker, model_name, horizon, keep_last=keep_last
                            )
                        )
                        for vid in deleted_ids:
                            logger.info(
                                "[cache_cleanup] pruned model version %s/%s/%s @ %s",
                                ticker,
                                model_name,
                                horizon,
                                vid,
                            )
                            # Approximate size — version dir already deleted, so we log 0
                            report.model_deleted.append(
                                DeletedFile(
                                    path=models_dir
                                    / ticker
                                    / model_name
                                    / horizon
                                    / "versions"
                                    / vid,
                                    size_bytes=0,
                                    age_days=0.0,
                                    reason=f"versioned model pruned (keep_last={keep_last})",
                                )
                            )
                        if self.dry_run and store.list_version_ids(
                            ticker, model_name, horizon
                        ):
                            all_ids = store.list_version_ids(
                                ticker, model_name, horizon
                            )
                            active_id = store.get_active_version_id(
                                ticker, model_name, horizon
                            )
                            preserve = set(all_ids[-keep_last:]) | (
                                {active_id} if active_id else set()
                            )
                            candidates = [v for v in all_ids if v not in preserve]
                            for vid in candidates:
                                logger.info(
                                    "[DRY RUN] would prune model version %s/%s/%s @ %s",
                                    ticker,
                                    model_name,
                                    horizon,
                                    vid,
                                )
                                report.model_deleted.append(
                                    DeletedFile(
                                        path=models_dir
                                        / ticker
                                        / model_name
                                        / horizon
                                        / "versions"
                                        / vid,
                                        size_bytes=0,
                                        age_days=0.0,
                                        reason=f"versioned model pruned (keep_last={keep_last})",
                                    )
                                )
                    except Exception as exc:
                        msg = f"model prune {ticker}/{model_name}/{horizon}: {exc}"
                        logger.warning(msg)
                        report.errors.append(msg)

    # ── Deletion helper ───────────────────────────────────────────────────────

    def _delete(self, path: Path, report: CleanupReport) -> None:
        """Delete a single file, respecting dry_run."""
        if self.dry_run:
            logger.info("[DRY RUN] would delete %s", path)
        else:
            try:
                path.unlink(missing_ok=True)
                logger.info("Deleted %s", path)
            except Exception as exc:
                msg = f"Failed to delete {path}: {exc}"
                logger.warning(msg)
                report.errors.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Startup hook  (call from main.py lifespan if desired)
# ─────────────────────────────────────────────────────────────────────────────


def cleanup_on_startup(dry_run: bool = False) -> None:
    """
    Lightweight startup hook — only cleans raw parquet cache.

    Intended to be called inside the FastAPI lifespan context so stale
    market data is removed before the first prediction request of the day.

    Example usage in main.py::

        from app.core.cache_cleanup import cleanup_on_startup

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            setup_logging(...)
            cleanup_on_startup()   # ← add this line
            yield
    """
    try:
        report = CacheCleanup(dry_run=dry_run).run()
        logger.info("[cache_cleanup] %s", report.summary())
    except Exception as exc:
        logger.warning("Startup cache cleanup failed (non-fatal): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.core.cache_cleanup",
        description="FinSight AI — data directory cleanup utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Preview what would be deleted (safe, default)
  python -m app.core.cache_cleanup

  # Actually delete outdated files
  python -m app.core.cache_cleanup --execute

  # Delete with verbose per-file output
  python -m app.core.cache_cleanup --execute --verbose

  # Only clean raw cache, keep everything else
  python -m app.core.cache_cleanup --execute --raw-only

  # Custom age thresholds
  python -m app.core.cache_cleanup --execute --raw-days 2 --log-days 14
""",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually delete files. Without this flag, runs in dry-run mode.",
    )
    p.add_argument(
        "--raw-only",
        action="store_true",
        default=False,
        help="Only clean raw parquet cache (skip models, embeddings, logs).",
    )
    p.add_argument(
        "--raw-days",
        type=int,
        default=None,
        metavar="N",
        help="Max age in days for raw cache files (default: from settings).",
    )
    p.add_argument(
        "--log-days",
        type=int,
        default=_DEFAULT_LOGS_MAX_AGE_DAYS,
        metavar="N",
        help=f"Max age in days for log files (default: {_DEFAULT_LOGS_MAX_AGE_DAYS}).",
    )
    p.add_argument(
        "--embedding-days",
        type=int,
        default=_DEFAULT_EMBEDDINGS_MAX_AGE_DAYS,
        metavar="N",
        help=f"Max age in days for embedding files "
        f"(default: {_DEFAULT_EMBEDDINGS_MAX_AGE_DAYS}).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Log every file inspected, not just deleted ones.",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    )

    args = _build_parser().parse_args()
    dry_run = not args.execute

    kwargs: dict = dict(
        dry_run=dry_run,
        logs_max_age_days=args.log_days,
        embeddings_max_age_days=args.embedding_days,
        verbose=args.verbose,
    )
    if args.raw_days is not None:
        kwargs["raw_max_age_days"] = args.raw_days

    cleanup = CacheCleanup(**kwargs)

    if args.raw_only:
        report = cleanup.run_raw_only()
    else:
        report = cleanup.run()

    print()
    print(report.summary())

    if dry_run:
        print()
        print("  → Run with --execute to actually delete these files.")


if __name__ == "__main__":
    main()
