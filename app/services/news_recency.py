from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from app.core.logging_config import get_logger

logger = get_logger("news_recency")

# ─────────────────────────────────────────────────────────────────────────────
# Lookback window table  (days)
# ─────────────────────────────────────────────────────────────────────────────

HORIZON_MAX_AGE_DAYS: dict[str, int] = {
    "1d": 3,
    "7d": 7,
    "1m": 30,
    "6m": 90,
}

# Fallback when no horizon is supplied (agent / RAG chat contexts).
DEFAULT_MAX_AGE_DAYS: int = 7

# Weight multiplier applied to articles with unknown dates when policy is
# "accept_with_penalty".
UNKNOWN_DATE_WEIGHT_PENALTY: float = 0.5

UnknownDatePolicy = Literal["accept", "reject", "accept_with_penalty"]

# ─────────────────────────────────────────────────────────────────────────────
# Date extraction patterns
# ─────────────────────────────────────────────────────────────────────────────

# Ordered by specificity (most specific first).
_DATE_PATTERNS: list[tuple[str, str]] = [
    # ISO-8601: 2026-05-14  or  2026-05-14T09:41
    (r"\b(20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))", "%Y-%m-%d"),
    # US long form: May 14, 2026  /  May 14 2026
    (
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+20\d{2})",
        None,
    ),
    # US numeric: 05/14/2026
    (r"\b((?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/20\d{2})\b", "%m/%d/%Y"),
    # EU numeric: 14.05.2026
    (r"\b((?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.20\d{2})\b", "%d.%m.%Y"),
    # Relative: "2 days ago", "3 hours ago", "1 week ago"
    (r"\b(\d+)\s+(hour|day|week|month)s?\s+ago\b", "relative"),
]

_MONTH_ABBREVS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RecencyAuditEntry:
    """
    Attached to each ``NewsItem`` after recency filtering.

    Fields
    ------
    extracted_date   : Parsed publish date if found, else None.
    age_days         : Approximate age in days (None when date unknown).
    kept             : True if the article passed the lookback filter.
    rejection_reason : Human-readable reason when ``kept=False``.
    date_source      : Where the date came from (``"metadata"``, ``"snippet"``,
                       ``"relative_text"``, ``"unknown"``).
    penalty_applied  : True when ``unknown_date_policy="accept_with_penalty"``
                       caused a weight reduction.
    """

    extracted_date: Optional[datetime] = None
    age_days: Optional[float] = None
    kept: bool = True
    rejection_reason: str = ""
    date_source: str = "unknown"
    penalty_applied: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Date extractor
# ─────────────────────────────────────────────────────────────────────────────


class ArticleDateExtractor:
    """
    Best-effort article date extractor.

    Extraction order
    ----------------
    1. ``result_metadata`` dict keys: ``published``, ``date``,
       ``pubDate``, ``publishedAt``.
    2. Snippet text: regex patterns from most to least specific.
    3. Title text: same regex pass on the headline.

    Returns an *aware* UTC ``datetime`` or ``None``.
    """

    def extract(
        self,
        title: str,
        snippet: str,
        metadata: Optional[dict] = None,
    ) -> tuple[Optional[datetime], str]:
        """
        Try to extract a publish date.

        Returns:
            ``(datetime | None, source_label)``
            where ``source_label`` is ``"metadata"``, ``"snippet"``,
            ``"title"``, ``"relative_text"``, or ``"unknown"``.
        """
        # ── 1. Metadata fields ────────────────────────────────────────────────
        if metadata:
            for key in (
                "published",
                "date",
                "pubDate",
                "publishedAt",
                "article:published_time",
            ):
                raw = metadata.get(key)
                if raw and isinstance(raw, str):
                    dt = self._parse_iso(raw)
                    if dt:
                        return dt, "metadata"

        # ── 2. Snippet text ───────────────────────────────────────────────────
        dt, src = self._regex_extract(snippet)
        if dt:
            return dt, f"snippet/{src}"

        # ── 3. Title text ─────────────────────────────────────────────────────
        dt, src = self._regex_extract(title)
        if dt:
            return dt, f"title/{src}"

        return None, "unknown"

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_iso(raw: str) -> Optional[datetime]:
        """Try ISO-8601 parsing; normalise to UTC-aware."""
        raw = raw.strip().replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M%z",
            "%Y-%m-%d%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw[:25], fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        return None

    def _regex_extract(self, text: str) -> tuple[Optional[datetime], str]:
        """Apply ordered regex patterns to ``text``."""
        now = datetime.now(timezone.utc)

        for pattern, fmt in _DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            if fmt == "relative":
                # e.g. "3 days ago"
                count = int(match.group(1))
                unit = match.group(2).lower().rstrip("s")
                delta_map = {
                    "hour": timedelta(hours=count),
                    "day": timedelta(days=count),
                    "week": timedelta(weeks=count),
                    "month": timedelta(days=count * 30),
                }
                delta = delta_map.get(unit)
                if delta:
                    return (now - delta).replace(tzinfo=timezone.utc), "relative_text"
                continue

            if fmt is None:
                # Month-name long form: "May 14, 2026"
                dt = self._parse_month_name(match.group(1))
                if dt:
                    return dt, "snippet_long_date"
                continue

            try:
                date_str = match.group(1).split("T")[0]  # strip time component
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc), "snippet_numeric"
            except ValueError:
                continue

        return None, "unknown"

    @staticmethod
    def _parse_month_name(raw: str) -> Optional[datetime]:
        """Parse 'May 14, 2026' or 'May 14 2026' variants."""
        raw = raw.replace(",", "").strip()
        parts = raw.split()
        if len(parts) < 3:
            return None
        month_str = parts[0][:3].lower()
        month = _MONTH_ABBREVS.get(month_str)
        if not month:
            return None
        try:
            day = int(parts[1])
            year = int(parts[2])
            return datetime(year, month, day, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Recency filter
# ─────────────────────────────────────────────────────────────────────────────


class NewsRecencyFilter:
    """
    Filters a list of ``NewsItem`` objects to those within a lookback window.

    Usage
    -----
    ::

        filter_ = NewsRecencyFilter(horizon="1d")
        kept, dropped = filter_.apply(items)

    The ``audit`` attribute on each ``NewsItem`` (added in-place via
    ``setattr``) records the extraction outcome for downstream logging.

    Parameters
    ----------
    horizon:
        Prediction horizon key (``"1d"``, ``"7d"``, ``"1m"``, ``"6m"``).
        Determines ``max_age_days`` from ``HORIZON_MAX_AGE_DAYS``.
        Overrides ``max_age_days`` when both are supplied.
    max_age_days:
        Explicit override for the lookback window in calendar days.
    unknown_date_policy:
        What to do with articles whose publish date cannot be determined.
        See module docstring for policy descriptions.
    min_kept:
        Minimum number of articles to keep regardless of recency.
        Prevents an empty result when all articles lack dates and
        ``unknown_date_policy="reject"``.  The oldest available articles
        are kept to satisfy this floor.  Default: 2.
    """

    def __init__(
        self,
        horizon: str = "1d",
        max_age_days: Optional[int] = None,
        unknown_date_policy: UnknownDatePolicy = "accept_with_penalty",
        min_kept: int = 2,
    ) -> None:
        if max_age_days is not None:
            self.max_age_days = max_age_days
        else:
            self.max_age_days = HORIZON_MAX_AGE_DAYS.get(horizon, DEFAULT_MAX_AGE_DAYS)
        self.unknown_date_policy = unknown_date_policy
        self.min_kept = min_kept
        self._extractor = ArticleDateExtractor()

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply(self, items: list) -> tuple[list, list]:
        """
        Apply the recency filter to a list of ``NewsItem`` objects.

        Each item receives a ``recency_audit`` attribute (``RecencyAuditEntry``).
        Items are not mutated in any other way — ``final_weight`` is only
        reduced when ``unknown_date_policy="accept_with_penalty"``.

        Returns:
            ``(kept_items, dropped_items)``
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.max_age_days)

        kept: list = []
        dropped: list = []

        for item in items:
            audit = self._audit_item(item, now, cutoff)
            setattr(item, "recency_audit", audit)

            if audit.kept:
                if audit.penalty_applied:
                    item.final_weight *= UNKNOWN_DATE_WEIGHT_PENALTY
                kept.append(item)
            else:
                dropped.append(item)

        # Enforce the minimum-kept floor
        if len(kept) < self.min_kept and dropped:
            shortfall = self.min_kept - len(kept)
            # Prefer items with the most recent extracted dates
            rescued = self._rescue_newest(dropped, shortfall)
            for item in rescued:
                item.recency_audit.kept = True
                item.recency_audit.rejection_reason = (
                    f"Rescued to satisfy min_kept={self.min_kept}."
                )
                kept.append(item)
                dropped.remove(item)

        logger.info(
            "Recency filter (max_age=%dd, policy=%s): kept=%d dropped=%d",
            self.max_age_days,
            self.unknown_date_policy,
            len(kept),
            len(dropped),
        )
        if dropped:
            for item in dropped:
                logger.debug(
                    "Dropped '%s…' — %s",
                    item.title[:60],
                    item.recency_audit.rejection_reason,
                )

        return kept, dropped

    def apply_inplace(self, items: list) -> list:
        """Convenience wrapper — returns only kept items."""
        kept, _ = self.apply(items)
        return kept

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audit_item(self, item, now: datetime, cutoff: datetime) -> RecencyAuditEntry:
        """Build a ``RecencyAuditEntry`` for a single ``NewsItem``."""
        metadata = getattr(item, "metadata", None)
        dt, src = self._extractor.extract(item.title, item.snippet, metadata)

        if dt is None:
            # Date unknown — apply policy
            if self.unknown_date_policy == "reject":
                return RecencyAuditEntry(
                    extracted_date=None,
                    age_days=None,
                    kept=False,
                    rejection_reason="Date unknown and policy=reject.",
                    date_source="unknown",
                )
            elif self.unknown_date_policy == "accept_with_penalty":
                return RecencyAuditEntry(
                    extracted_date=None,
                    age_days=None,
                    kept=True,
                    date_source="unknown",
                    penalty_applied=True,
                )
            else:  # "accept"
                return RecencyAuditEntry(
                    extracted_date=None,
                    age_days=None,
                    kept=True,
                    date_source="unknown",
                )

        age = (now - dt).total_seconds() / 86_400
        if dt >= cutoff:
            return RecencyAuditEntry(
                extracted_date=dt,
                age_days=round(age, 1),
                kept=True,
                date_source=src,
            )
        else:
            return RecencyAuditEntry(
                extracted_date=dt,
                age_days=round(age, 1),
                kept=False,
                rejection_reason=(
                    f"Age {age:.1f}d exceeds max_age={self.max_age_days}d "
                    f"(published {dt.strftime('%Y-%m-%d')})."
                ),
                date_source=src,
            )

    @staticmethod
    def _rescue_newest(dropped: list, count: int) -> list:
        """
        Pick the ``count`` most recent (or date-unknown) articles from
        the dropped list to satisfy the ``min_kept`` floor.
        """

        def sort_key(item) -> float:
            audit: RecencyAuditEntry = getattr(
                item, "recency_audit", RecencyAuditEntry()
            )
            if audit.extracted_date:
                return audit.extracted_date.timestamp()
            return 0.0  # unknown-date items sort to the bottom

        return sorted(dropped, key=sort_key, reverse=True)[:count]
