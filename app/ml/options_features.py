"""
app/ml/options_features.py
============================
Options market data and implied-volatility feature engineering — Improvement 2.

Two data sources are blended, because they behave very differently:

1. VIX-family tickers (^VIX, ^VIX9D, ^VIX3M)
   Full historical daily series via yfinance — fetched exactly like any
   other price series (same pattern as ``_fetch_macro_series`` in
   ``fundamental_features.py``). These give genuine day-by-day granularity
   across the whole price index for free.

2. Per-ticker options-chain snapshot (ATM IV, constant-maturity IV,
   put/call ratios)
   yfinance only exposes a *live* options-chain snapshot — there is no
   historical-IV endpoint. To build a usable time series, snapshots must be
   captured once per trading day and persisted; ``OptionsHistoryStore``
   does that (one parquet file per ticker, appended to daily), and
   ``scripts/warm_options_cache.py`` is the operational job that should be
   scheduled once near market close to populate it. ``OptionsFeatureEngineer``
   then forward-fills that stored history onto the technical feature index
   and derives IV rank, IV change, and the IV/realized-vol spread from it.

Design principles (matching the rest of the codebase)
-------------------------------------------------------
* Fail-safe: every yfinance call is wrapped; failures return NaN/empty
  rather than raising. Callers should never have a pipeline crash because
  the options market was unavailable for a ticker.
* Liquidity-aware: contracts are quality-filtered (minimum volume/open
  interest, maximum bid-ask spread) before being used for ATM IV — see
  ``_quality_filter``. Tickers with no usable contracts after filtering are
  marked ``is_optionable=False`` rather than silently producing garbage IV.
* No side effects in the read path: ``OptionsFeatureEngineer.build()`` only
  *reads* the persisted history by default (``auto_snapshot=False``) because
  ``Ticker.option_chain()`` calls are slow (typically 1-3s per expiration)
  and unsuitable for the request/response path of a prediction API.
* Forward-fill ("last known value"), exactly like the fundamental and
  sector-correlation engineers: a snapshot taken today is the best estimate
  of options-market conditions until the next snapshot is captured.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("options_features")

# ── VIX-family tickers (full historical daily series via yfinance) ───────────
_VIX_TICKER = "^VIX"
_VIX9D_TICKER = "^VIX9D"
_VIX3M_TICKER = "^VIX3M"

# ── Quality filters for individual option contracts ───────────────────────────
_MIN_CONTRACT_VOLUME: int = 1
_MIN_OPEN_INTEREST: int = 10
_MAX_BID_ASK_SPREAD_PCT: float = 0.50  # reject quotes wider than 50% of mid

# Minimum price-index rows before attempting an options feature build
_MIN_ROWS_FOR_FEATURES: int = 20

# Target constant-maturity tenor (calendar days) for the interpolated IV
_CM_TARGET_DAYS: int = 30

# Rolling lookback window (calendar days of stored history) for IV rank
_IV_RANK_LOOKBACK_DAYS: int = 252

_OPTIONS_CACHE_DIRNAME: str = "options_cache"

# ── Feature name catalogue (for documentation / FeatureSelector awareness) ───
OPTIONS_FEATURE_NAMES: tuple[str, ...] = (
    "opt_is_optionable",
    "opt_atm_iv_near",
    "opt_atm_iv_cm30",
    "opt_iv_rank_252d",
    "opt_iv_change_5d",
    "opt_iv_rv_spread",
    "opt_put_call_vol_ratio",
    "opt_put_call_vol_ratio_ma5",
    "opt_put_call_oi_ratio",
    "vix_level",
    "vix_high_regime",
    "vix_elevated_regime",
    "vix9d_level",
    "vix3m_level",
    "vix_term_structure_slope",
    "vix_term_structure_inverted",
)


def _options_cache_dir() -> Path:
    d = Path(settings.DATA_DIR) / _OPTIONS_CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OptionsSnapshot:
    """
    Point-in-time options-chain snapshot for a single ticker.

    NaN/empty fields indicate the metric was unavailable (e.g. no listed
    options, or no contracts survived the liquidity quality filter).
    """

    ticker: str
    snapshot_date: str  # "YYYY-MM-DD" — calendar date the snapshot was taken
    is_optionable: bool = False
    near_expiry: str = ""
    next_expiry: str = ""
    near_days_to_expiry: float = np.nan
    next_days_to_expiry: float = np.nan
    atm_iv_near: float = np.nan
    atm_iv_next: float = np.nan
    atm_iv_cm30: float = np.nan
    put_call_volume_ratio: float = np.nan
    put_call_oi_ratio: float = np.nan
    total_call_volume: float = np.nan
    total_put_volume: float = np.nan
    n_contracts_used: int = 0
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "snapshot_date": self.snapshot_date,
            "is_optionable": self.is_optionable,
            "near_expiry": self.near_expiry,
            "next_expiry": self.next_expiry,
            "near_days_to_expiry": self.near_days_to_expiry,
            "next_days_to_expiry": self.next_days_to_expiry,
            "atm_iv_near": self.atm_iv_near,
            "atm_iv_next": self.atm_iv_next,
            "atm_iv_cm30": self.atm_iv_cm30,
            "put_call_volume_ratio": self.put_call_volume_ratio,
            "put_call_oi_ratio": self.put_call_oi_ratio,
            "total_call_volume": self.total_call_volume,
            "total_put_volume": self.total_put_volume,
            "n_contracts_used": self.n_contracts_used,
            "fetched_at": self.fetched_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quality filtering + ATM IV extraction
# ─────────────────────────────────────────────────────────────────────────────


def _quality_filter(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Drop illiquid or stale-quote contracts before they can pollute ATM IV.

    A contract is kept when it has *either* meaningful daily volume *or*
    meaningful open interest, AND its bid-ask spread (as a % of mid) is
    within tolerance, AND it reports a positive implied volatility.
    """
    if chain.empty:
        return chain

    df = chain.copy()
    df["volume"] = df.get("volume", pd.Series(0, index=df.index)).fillna(0)
    df["openInterest"] = df.get("openInterest", pd.Series(0, index=df.index)).fillna(0)
    df["bid"] = df.get("bid", pd.Series(np.nan, index=df.index))
    df["ask"] = df.get("ask", pd.Series(np.nan, index=df.index))

    mid = (df["bid"] + df["ask"]) / 2.0
    spread_pct = (df["ask"] - df["bid"]) / mid.replace(0, np.nan)

    liquid = (df["volume"] >= _MIN_CONTRACT_VOLUME) | (
        df["openInterest"] >= _MIN_OPEN_INTEREST
    )
    tight_quote = spread_pct.fillna(1.0) <= _MAX_BID_ASK_SPREAD_PCT
    has_iv = df.get("impliedVolatility", pd.Series(0.0, index=df.index)) > 0

    return df[liquid & tight_quote & has_iv]


def _atm_iv_from_chain(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float
) -> tuple[float, int]:
    """
    Distance-weighted average implied volatility from the two strikes
    nearest the spot price, pooled across calls and puts after quality
    filtering.

    Returns
    -------
    (atm_iv, n_contracts_used) — ``atm_iv`` is NaN when no contract survives
    filtering.
    """
    frames: list[pd.DataFrame] = []
    for chain in (calls, puts):
        filtered = _quality_filter(chain)
        if filtered.empty:
            continue
        filtered = filtered.assign(dist=(filtered["strike"] - spot).abs())
        frames.append(filtered.sort_values("dist").head(2))

    if not frames:
        return np.nan, 0

    combined = pd.concat(frames)
    if combined.empty:
        return np.nan, 0

    weights = 1.0 / (combined["dist"] + 1.0)
    atm_iv = float((combined["impliedVolatility"] * weights).sum() / weights.sum())
    return atm_iv, len(combined)


def _constant_maturity_iv(
    iv_near: float,
    days_near: float,
    iv_next: float,
    days_next: float,
    target_days: int = _CM_TARGET_DAYS,
) -> float:
    """
    Interpolate a "constant maturity" IV at ``target_days`` from the two
    nearest expirations, using the standard linear-interpolation-in-total-
    variance approach (interpolating IV directly would understate the
    target because variance — not vol — scales linearly with time).

        total_variance(T) = IV(T)^2 * T
        total_variance(target) is linearly interpolated between the two
        known maturities, then converted back to a vol.

    This avoids the discontinuity that occurs when the nearest expiration
    rolls over: each day's interpolated 30-day point moves smoothly even as
    the underlying contracts used to compute it change week to week.

    Falls back to the single available IV when only one expiration exists,
    or when interpolation isn't well-posed (e.g. tenors out of order).
    """
    if not np.isfinite(iv_near):
        return float(iv_next) if np.isfinite(iv_next) else np.nan
    if not np.isfinite(iv_next) or days_next <= days_near:
        return float(iv_near)

    t_target = target_days / 365.0
    t_near = days_near / 365.0
    t_next = days_next / 365.0

    if t_target <= t_near:
        return float(iv_near)
    if t_target >= t_next:
        return float(iv_next)

    var_near = iv_near**2 * t_near
    var_next = iv_next**2 * t_next
    var_target = var_near + (var_next - var_near) * (t_target - t_near) / (
        t_next - t_near
    )

    if var_target <= 0:
        return float(iv_near)
    return float(math.sqrt(var_target / t_target))


# ─────────────────────────────────────────────────────────────────────────────
# Live snapshot fetcher
# ─────────────────────────────────────────────────────────────────────────────


def fetch_options_snapshot(ticker: str) -> Optional[OptionsSnapshot]:
    """
    Fetch a live options-chain snapshot for *ticker* via yfinance.

    Never raises. Returns a snapshot with ``is_optionable=False`` (but
    populated date fields) when the ticker has no listed options, has no
    recent price history, or no contracts survive the liquidity filter.
    """
    today = datetime.now(timezone.utc).date()
    snap = OptionsSnapshot(ticker=ticker.upper(), snapshot_date=str(today))

    try:
        import yfinance as yf

        tkr = yf.Ticker(ticker)
        expiries = tkr.options
        if not expiries:
            logger.info("[%s] No listed options — not optionable.", ticker)
            return snap

        hist = tkr.history(period="1d")
        if hist.empty:
            logger.warning(
                "[%s] No recent price history for spot — skipping snapshot.", ticker
            )
            return snap
        spot = float(hist["Close"].iloc[-1])

        future_expiries = [
            e
            for e in expiries
            if datetime.strptime(e, "%Y-%m-%d").date() > today
        ]
        if not future_expiries:
            return snap

        near_str = future_expiries[0]
        near_days = (datetime.strptime(near_str, "%Y-%m-%d").date() - today).days

        chain_near = tkr.option_chain(near_str)
        atm_iv_near, n_near = _atm_iv_from_chain(chain_near.calls, chain_near.puts, spot)

        atm_iv_next: float = np.nan
        next_days: float = np.nan
        next_str: str = ""
        n_next = 0
        if len(future_expiries) > 1:
            next_str = future_expiries[1]
            next_days = float(
                (datetime.strptime(next_str, "%Y-%m-%d").date() - today).days
            )
            chain_next = tkr.option_chain(next_str)
            atm_iv_next, n_next = _atm_iv_from_chain(
                chain_next.calls, chain_next.puts, spot
            )

        cm30 = _constant_maturity_iv(atm_iv_near, near_days, atm_iv_next, next_days)

        call_vol = float(chain_near.calls.get("volume", pd.Series(dtype=float)).fillna(0).sum())
        put_vol = float(chain_near.puts.get("volume", pd.Series(dtype=float)).fillna(0).sum())
        call_oi = float(chain_near.calls.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
        put_oi = float(chain_near.puts.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())

        snap.is_optionable = (n_near + n_next) >= 2
        snap.near_expiry = near_str
        snap.next_expiry = next_str
        snap.near_days_to_expiry = float(near_days)
        snap.next_days_to_expiry = next_days
        snap.atm_iv_near = atm_iv_near
        snap.atm_iv_next = atm_iv_next
        snap.atm_iv_cm30 = cm30
        snap.put_call_volume_ratio = (put_vol / call_vol) if call_vol > 0 else np.nan
        snap.put_call_oi_ratio = (put_oi / call_oi) if call_oi > 0 else np.nan
        snap.total_call_volume = call_vol
        snap.total_put_volume = put_vol
        snap.n_contracts_used = n_near + n_next

        logger.info(
            "[%s] Options snapshot: atm_iv_near=%s cm30=%s pc_ratio=%s optionable=%s",
            ticker,
            f"{atm_iv_near:.4f}" if np.isfinite(atm_iv_near) else "nan",
            f"{cm30:.4f}" if np.isfinite(cm30) else "nan",
            f"{snap.put_call_volume_ratio:.3f}"
            if np.isfinite(snap.put_call_volume_ratio)
            else "nan",
            snap.is_optionable,
        )
        return snap

    except Exception as exc:
        logger.warning("[%s] Options snapshot fetch failed: %s", ticker, exc)
        return snap


# ─────────────────────────────────────────────────────────────────────────────
# VIX-family historical series
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_yf_series(ticker: str, start: str, end: str) -> pd.Series:
    """Daily Close series for a regular yfinance ticker (e.g. ^VIX). NaN-safe."""
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        series = df["Close"].squeeze()
        series.index = pd.to_datetime(series.index)
        return series.sort_index()
    except Exception as exc:
        logger.debug("VIX-family series fetch failed (%s): %s", ticker, exc)
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Persistent snapshot history store
# ─────────────────────────────────────────────────────────────────────────────


class OptionsHistoryStore:
    """
    Append-only persistent store of daily options snapshots.

    One parquet file per ticker: ``{DATA_DIR}/options_cache/{TICKER}.parquet``,
    indexed by snapshot date. Calling ``append()``/``update()`` more than once
    on the same calendar day overwrites that day's row (idempotent) rather
    than duplicating it.

    This store is what makes IV-rank and IV-change features possible at all:
    yfinance only ever exposes a *live* options-chain snapshot, never
    historical implied volatility, so the time series must be built up one
    trading day at a time. See ``scripts/warm_options_cache.py`` for the
    operational job that should populate this on a daily schedule.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._dir = cache_dir or _options_cache_dir()

    def _path(self, ticker: str) -> Path:
        return self._dir / f"{ticker.upper()}.parquet"

    def load(self, ticker: str) -> pd.DataFrame:
        """Return the full stored history for *ticker*, or an empty frame."""
        path = self._path(ticker)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception as exc:
            logger.warning("[%s] Failed to read options history (%s)", ticker, exc)
            return pd.DataFrame()

    def append(self, snapshot: OptionsSnapshot) -> None:
        """Idempotently upsert today's snapshot row, then persist atomically."""
        df = self.load(snapshot.ticker)

        row = pd.DataFrame([snapshot.to_dict()])
        row_index = pd.DatetimeIndex([pd.Timestamp(snapshot.snapshot_date)])
        row.index = row_index
        row = row.drop(columns=["ticker", "snapshot_date"])

        if not df.empty:
            df = df[df.index != row_index[0]]  # drop any existing same-day row
        df = pd.concat([df, row]).sort_index()

        path = self._path(snapshot.ticker)
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, prefix=".tmp_", suffix=".parquet")
        try:
            os.close(fd)
            df.to_parquet(tmp_path, index=True)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def update(self, ticker: str) -> Optional[OptionsSnapshot]:
        """Fetch today's live snapshot and append it to history."""
        snapshot = fetch_options_snapshot(ticker)
        if snapshot is not None:
            self.append(snapshot)
        return snapshot

    def has_snapshot_today(self, ticker: str) -> bool:
        today_ts = pd.Timestamp(datetime.now(timezone.utc).date())
        df = self.load(ticker)
        return (not df.empty) and (today_ts in df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Main feature engineer
# ─────────────────────────────────────────────────────────────────────────────


class OptionsFeatureEngineer:
    """
    Builds options-market and implied-volatility features aligned to a daily
    price index.

    Parameters
    ----------
    history_store:
        Pre-constructed ``OptionsHistoryStore``. Created automatically if None.
    include_vix:
        Whether to fetch and merge VIX-family features. Default True.
    auto_snapshot:
        If True, synchronously fetch+store a fresh snapshot when no snapshot
        exists for today. Default False — recommended for request-time use
        (``option_chain()`` calls are slow); prefer the pre-warmed cache via
        ``scripts/warm_options_cache.py`` for production traffic and only
        set True for ad-hoc analysis, notebooks, or backtests.
    """

    def __init__(
        self,
        history_store: Optional[OptionsHistoryStore] = None,
        include_vix: bool = True,
        auto_snapshot: bool = False,
    ) -> None:
        self._store = history_store or OptionsHistoryStore()
        self.include_vix = include_vix
        self.auto_snapshot = auto_snapshot

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        ticker: str,
        price_index: pd.DatetimeIndex,
        realized_vol_21d: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build all options-derived features for *ticker*, aligned to
        *price_index*.

        Parameters
        ----------
        ticker:
            Stock ticker symbol (e.g. "AAPL").
        price_index:
            DatetimeIndex of the technical feature matrix.
        realized_vol_21d:
            Optional 21-day annualised realized volatility series (already
            computed by ``FeatureEngineer``). When supplied, the IV/RV
            volatility-risk-premium spread feature is also produced.

        Returns
        -------
        pd.DataFrame, one row per date in *price_index*. Cells are NaN where
        data is unavailable — rows are never dropped, that's the caller's job.
        """
        ticker = ticker.upper().strip()
        if len(price_index) < _MIN_ROWS_FOR_FEATURES:
            logger.warning(
                "[%s] Price index too short (%d rows) for options features.",
                ticker,
                len(price_index),
            )
            return pd.DataFrame(index=price_index)

        frames: list[pd.DataFrame] = []

        snap_df = self._build_snapshot_features(ticker, price_index, realized_vol_21d)
        if not snap_df.empty:
            frames.append(snap_df)

        if self.include_vix:
            vix_df = self._build_vix_features(price_index)
            if not vix_df.empty:
                frames.append(vix_df)

        if not frames:
            return pd.DataFrame(index=price_index)

        result = pd.concat(frames, axis=1).reindex(price_index)
        logger.info(
            "[%s] Options feature matrix: %d features (%d with data) x %d rows",
            ticker,
            result.shape[1],
            result.notna().any(axis=0).sum(),
            len(result),
        )
        return result

    def get_feature_names(self) -> list[str]:
        return list(OPTIONS_FEATURE_NAMES)

    # ── Internal: snapshot history → features ─────────────────────────────────

    def _build_snapshot_features(
        self,
        ticker: str,
        price_index: pd.DatetimeIndex,
        realized_vol_21d: Optional[pd.Series],
    ) -> pd.DataFrame:
        if self.auto_snapshot and not self._store.has_snapshot_today(ticker):
            self._store.update(ticker)

        history = self._store.load(ticker)
        if history.empty:
            logger.info(
                "[%s] No options snapshot history yet — run "
                "scripts/warm_options_cache.py to populate it.",
                ticker,
            )
            return pd.DataFrame(index=price_index)

        history = history.sort_index()

        iv_rank = (
            history["atm_iv_cm30"]
            .rolling(_IV_RANK_LOOKBACK_DAYS, min_periods=20)
            .rank(pct=True)
        )
        iv_change_5d = history["atm_iv_cm30"].diff(5)
        pc_ratio_ma5 = history["put_call_volume_ratio"].rolling(5, min_periods=1).mean()

        feat = pd.DataFrame(
            {
                "opt_is_optionable": history["is_optionable"].astype(float),
                "opt_atm_iv_near": history["atm_iv_near"],
                "opt_atm_iv_cm30": history["atm_iv_cm30"],
                "opt_iv_rank_252d": iv_rank,
                "opt_iv_change_5d": iv_change_5d,
                "opt_put_call_vol_ratio": history["put_call_volume_ratio"],
                "opt_put_call_vol_ratio_ma5": pc_ratio_ma5,
                "opt_put_call_oi_ratio": history["put_call_oi_ratio"],
            },
            index=history.index,
        )

        # Snapshots are at most daily — forward-fill onto the full price index.
        feat = feat.reindex(price_index, method="ffill")

        if realized_vol_21d is not None:
            rv = realized_vol_21d.reindex(price_index, method="ffill")
            feat["opt_iv_rv_spread"] = feat["opt_atm_iv_cm30"] - rv

        return feat

    # ── Internal: VIX-family features ──────────────────────────────────────────

    def _build_vix_features(self, price_index: pd.DatetimeIndex) -> pd.DataFrame:
        start = str(price_index.min().date())
        end = str(price_index.max().date())

        vix = _fetch_yf_series(_VIX_TICKER, start, end)
        if vix.empty:
            logger.debug("VIX series unavailable — skipping VIX features.")
            return pd.DataFrame(index=price_index)

        df = pd.DataFrame(index=price_index)
        vix_a = vix.reindex(price_index, method="ffill")
        df["vix_level"] = vix_a
        df["vix_high_regime"] = (vix_a > 30).astype(int)
        df["vix_elevated_regime"] = (vix_a > 20).astype(int)

        vix9d = _fetch_yf_series(_VIX9D_TICKER, start, end)
        vix3m = _fetch_yf_series(_VIX3M_TICKER, start, end)

        if not vix9d.empty and not vix3m.empty:
            vix9d_a = vix9d.reindex(price_index, method="ffill")
            vix3m_a = vix3m.reindex(price_index, method="ffill")
            df["vix9d_level"] = vix9d_a
            df["vix3m_level"] = vix3m_a
            df["vix_term_structure_slope"] = vix9d_a - vix3m_a
            df["vix_term_structure_inverted"] = (
                df["vix_term_structure_slope"] > 0
            ).astype(int)
        else:
            logger.debug(
                "VIX9D/VIX3M unavailable — term-structure features skipped."
            )

        return df


# ─────────────────────────────────────────────────────────────────────────────
# LLM-facing narrative summary (for signal fusion)
# ─────────────────────────────────────────────────────────────────────────────


def build_options_context_narrative(
    ticker: str,
    history_store: Optional[OptionsHistoryStore] = None,
) -> str:
    """
    Build a short plain-English summary of current options-market conditions
    for *ticker*, suitable for injection into the LLM signal-fusion prompt
    (see ``app/services/signal_fusion.py``).

    Returns an empty string when no options snapshot history exists yet so
    callers can omit the section entirely rather than show stale or
    misleading placeholder text.
    """
    store = history_store or OptionsHistoryStore()
    history = store.load(ticker)
    if history.empty:
        return ""

    latest = history.iloc[-1]
    if not bool(latest.get("is_optionable", False)):
        return f"{ticker} options market: not liquid enough for a reliable IV signal."

    cm30 = latest.get("atm_iv_cm30", np.nan)
    pc_ratio = latest.get("put_call_volume_ratio", np.nan)

    iv_rank = np.nan
    if len(history) >= 20:
        iv_rank = (
            history["atm_iv_cm30"]
            .rolling(_IV_RANK_LOOKBACK_DAYS, min_periods=20)
            .rank(pct=True)
            .iloc[-1]
        )

    parts: list[str] = []
    if np.isfinite(cm30):
        parts.append(f"30-day ATM implied volatility is {cm30:.1%}")
    if np.isfinite(iv_rank):
        level = (
            "elevated" if iv_rank > 0.7 else "subdued" if iv_rank < 0.3 else "moderate"
        )
        parts.append(f"IV rank is {iv_rank:.0%} ({level} relative to the past year)")
    if np.isfinite(pc_ratio):
        skew = (
            "bearish" if pc_ratio > 1.1 else "bullish" if pc_ratio < 0.9 else "balanced"
        )
        parts.append(f"put/call volume ratio is {pc_ratio:.2f} ({skew} positioning)")

    if not parts:
        return ""

    return f"{ticker} options market: " + "; ".join(parts) + "."
