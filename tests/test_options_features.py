"""
tests/test_options_features.py
================================
Unit tests for OptionsFeatureEngineer, OptionsHistoryStore, and the IV
interpolation / quality-filtering primitives in app/ml/options_features.py.

All yfinance and network calls are mocked — no internet access required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml.feature_engineering import FeatureEngineer
from app.ml.options_features import (
    OPTIONS_FEATURE_NAMES,
    OptionsFeatureEngineer,
    OptionsHistoryStore,
    OptionsSnapshot,
    _atm_iv_from_chain,
    _constant_maturity_iv,
    _quality_filter,
    build_options_context_narrative,
    fetch_options_snapshot,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def price_index() -> pd.DatetimeIndex:
    return pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=300)


def _make_chain(strikes, ivs, volumes=None, ois=None, spread_pct=0.05) -> pd.DataFrame:
    n = len(strikes)
    volumes = volumes or [50] * n
    ois = ois or [200] * n
    mid = [s for s in strikes]  # arbitrary mid prices, not used directly
    bid = [m * (1 - spread_pct / 2) for m in mid]
    ask = [m * (1 + spread_pct / 2) for m in mid]
    return pd.DataFrame(
        {
            "strike": strikes,
            "impliedVolatility": ivs,
            "volume": volumes,
            "openInterest": ois,
            "bid": bid,
            "ask": ask,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# _quality_filter
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityFilter:
    def test_keeps_liquid_contracts(self):
        chain = _make_chain([100, 105], [0.25, 0.27], volumes=[10, 10])
        result = _quality_filter(chain)
        assert len(result) == 2

    def test_drops_illiquid_contracts(self):
        chain = _make_chain([100], [0.25], volumes=[0], ois=[0])
        result = _quality_filter(chain)
        assert result.empty

    def test_drops_wide_spread_contracts(self):
        chain = _make_chain([100], [0.25], spread_pct=2.0)  # 200% spread
        result = _quality_filter(chain)
        assert result.empty

    def test_drops_zero_iv_contracts(self):
        chain = _make_chain([100], [0.0])
        result = _quality_filter(chain)
        assert result.empty

    def test_empty_input_returns_empty(self):
        result = _quality_filter(pd.DataFrame())
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# _atm_iv_from_chain
# ─────────────────────────────────────────────────────────────────────────────


class TestAtmIvFromChain:
    def test_picks_strikes_nearest_spot(self):
        calls = _make_chain([90, 100, 110], [0.20, 0.25, 0.30])
        puts = _make_chain([90, 100, 110], [0.22, 0.26, 0.31])
        atm_iv, n = _atm_iv_from_chain(calls, puts, spot=100.0)
        assert n == 4  # 2 nearest calls + 2 nearest puts (all within range)
        assert 0.20 < atm_iv < 0.31

    def test_returns_nan_when_all_filtered_out(self):
        calls = _make_chain([100], [0.0])
        puts = _make_chain([100], [0.0])
        atm_iv, n = _atm_iv_from_chain(calls, puts, spot=100.0)
        assert np.isnan(atm_iv)
        assert n == 0

    def test_empty_chains_return_nan(self):
        atm_iv, n = _atm_iv_from_chain(pd.DataFrame(), pd.DataFrame(), spot=100.0)
        assert np.isnan(atm_iv)
        assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# _constant_maturity_iv
# ─────────────────────────────────────────────────────────────────────────────


class TestConstantMaturityIv:
    def test_interpolates_between_two_known_points(self):
        # near=10d @ 20%, next=60d @ 30% -> target 30d should land between them
        cm = _constant_maturity_iv(0.20, 10, 0.30, 60, target_days=30)
        assert 0.20 < cm < 0.30

    def test_falls_back_to_near_when_next_unavailable(self):
        cm = _constant_maturity_iv(0.20, 10, np.nan, np.nan, target_days=30)
        assert cm == pytest.approx(0.20)

    def test_falls_back_to_next_when_near_unavailable(self):
        cm = _constant_maturity_iv(np.nan, np.nan, 0.30, 60, target_days=30)
        assert cm == pytest.approx(0.30)

    def test_returns_nan_when_both_missing(self):
        cm = _constant_maturity_iv(np.nan, np.nan, np.nan, np.nan)
        assert np.isnan(cm)

    def test_target_before_near_returns_near(self):
        cm = _constant_maturity_iv(0.20, 30, 0.30, 60, target_days=10)
        assert cm == pytest.approx(0.20)

    def test_target_after_next_returns_next(self):
        cm = _constant_maturity_iv(0.20, 10, 0.30, 30, target_days=60)
        assert cm == pytest.approx(0.30)

    def test_out_of_order_tenors_falls_back_to_near(self):
        cm = _constant_maturity_iv(0.20, 60, 0.30, 10, target_days=30)
        assert cm == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────────────────────────
# fetch_options_snapshot
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchOptionsSnapshot:
    def test_returns_not_optionable_when_no_expiries(self):
        mock_tkr = MagicMock()
        mock_tkr.options = []
        with patch("yfinance.Ticker", return_value=mock_tkr):
            snap = fetch_options_snapshot("ZZZZ")
        assert snap is not None
        assert snap.is_optionable is False

    def test_returns_not_optionable_when_no_price_history(self):
        mock_tkr = MagicMock()
        tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=5)).strftime(
            "%Y-%m-%d"
        )
        mock_tkr.options = [tomorrow]
        mock_tkr.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_tkr):
            snap = fetch_options_snapshot("AAPL")
        assert snap.is_optionable is False

    def test_full_snapshot_with_two_expiries(self):
        today = datetime.now(timezone.utc).date()
        near = (today + timedelta(days=10)).strftime("%Y-%m-%d")
        nxt = (today + timedelta(days=45)).strftime("%Y-%m-%d")

        mock_tkr = MagicMock()
        mock_tkr.options = [near, nxt]
        mock_tkr.history.return_value = pd.DataFrame({"Close": [185.0]})

        near_chain = MagicMock(
            calls=_make_chain([180, 190], [0.22, 0.24]),
            puts=_make_chain([180, 190], [0.23, 0.25]),
        )
        next_chain = MagicMock(
            calls=_make_chain([180, 190], [0.26, 0.28]),
            puts=_make_chain([180, 190], [0.27, 0.29]),
        )

        def _option_chain(expiry):
            return near_chain if expiry == near else next_chain

        mock_tkr.option_chain.side_effect = _option_chain

        with patch("yfinance.Ticker", return_value=mock_tkr):
            snap = fetch_options_snapshot("AAPL")

        assert snap.is_optionable is True
        assert snap.near_expiry == near
        assert snap.next_expiry == nxt
        assert np.isfinite(snap.atm_iv_near)
        assert np.isfinite(snap.atm_iv_cm30)
        assert np.isfinite(snap.put_call_volume_ratio)

    def test_handles_yfinance_exception_gracefully(self):
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            snap = fetch_options_snapshot("AAPL")
        assert snap is not None
        assert snap.is_optionable is False


# ─────────────────────────────────────────────────────────────────────────────
# OptionsHistoryStore
# ─────────────────────────────────────────────────────────────────────────────


class TestOptionsHistoryStore:
    def test_load_returns_empty_when_no_file(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        df = store.load("AAPL")
        assert df.empty

    def test_append_then_load_roundtrip(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        snap = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date="2026-06-01",
            is_optionable=True,
            atm_iv_near=0.25,
            atm_iv_cm30=0.27,
            put_call_volume_ratio=0.95,
        )
        store.append(snap)
        df = store.load("AAPL")
        assert len(df) == 1
        assert df.iloc[0]["atm_iv_cm30"] == pytest.approx(0.27)

    def test_append_same_day_is_idempotent(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        snap1 = OptionsSnapshot(
            ticker="AAPL", snapshot_date="2026-06-01", atm_iv_cm30=0.20
        )
        snap2 = OptionsSnapshot(
            ticker="AAPL", snapshot_date="2026-06-01", atm_iv_cm30=0.30
        )
        store.append(snap1)
        store.append(snap2)
        df = store.load("AAPL")
        assert len(df) == 1
        assert df.iloc[0]["atm_iv_cm30"] == pytest.approx(0.30)  # latest wins

    def test_append_different_days_accumulates(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        store.append(OptionsSnapshot(ticker="AAPL", snapshot_date="2026-06-01"))
        store.append(OptionsSnapshot(ticker="AAPL", snapshot_date="2026-06-02"))
        df = store.load("AAPL")
        assert len(df) == 2

    def test_has_snapshot_today(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        today_str = str(datetime.now(timezone.utc).date())
        assert store.has_snapshot_today("AAPL") is False
        store.append(OptionsSnapshot(ticker="AAPL", snapshot_date=today_str))
        assert store.has_snapshot_today("AAPL") is True

    def test_update_calls_fetch_and_appends(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        fake_snap = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date=str(datetime.now(timezone.utc).date()),
            is_optionable=True,
            atm_iv_cm30=0.22,
        )
        with patch(
            "app.ml.options_features.fetch_options_snapshot", return_value=fake_snap
        ):
            result = store.update("AAPL")
        assert result is fake_snap
        df = store.load("AAPL")
        assert len(df) == 1

    def test_tickers_are_isolated(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        store.append(OptionsSnapshot(ticker="AAPL", snapshot_date="2026-06-01"))
        store.append(OptionsSnapshot(ticker="MSFT", snapshot_date="2026-06-01"))
        assert len(store.load("AAPL")) == 1
        assert len(store.load("MSFT")) == 1


# ─────────────────────────────────────────────────────────────────────────────
# OptionsFeatureEngineer.build()
# ─────────────────────────────────────────────────────────────────────────────


def _populate_history(store: OptionsHistoryStore, ticker: str, n_days: int = 60):
    """Write n_days of synthetic snapshots ending today."""
    np.random.seed(0)
    today = datetime.now(timezone.utc).date()
    for i in range(n_days, 0, -1):
        d = today - timedelta(days=i)
        iv = 0.20 + 0.05 * np.sin(i / 10) + np.random.normal(0, 0.01)
        store.append(
            OptionsSnapshot(
                ticker=ticker,
                snapshot_date=str(d),
                is_optionable=True,
                atm_iv_near=iv,
                atm_iv_cm30=iv,
                put_call_volume_ratio=1.0 + np.random.normal(0, 0.1),
                put_call_oi_ratio=1.1,
            )
        )


class TestOptionsFeatureEngineerBuild:
    def test_returns_dataframe_with_price_index(self, tmp_path, price_index):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        _populate_history(store, "AAPL")
        eng = OptionsFeatureEngineer(history_store=store, include_vix=False)
        result = eng.build("AAPL", price_index)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(price_index)
        assert result.index.equals(price_index)

    def test_contains_expected_snapshot_columns(self, tmp_path, price_index):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        _populate_history(store, "AAPL")
        eng = OptionsFeatureEngineer(history_store=store, include_vix=False)
        result = eng.build("AAPL", price_index)
        for col in [
            "opt_is_optionable",
            "opt_atm_iv_near",
            "opt_atm_iv_cm30",
            "opt_iv_rank_252d",
            "opt_iv_change_5d",
            "opt_put_call_vol_ratio",
            "opt_put_call_vol_ratio_ma5",
            "opt_put_call_oi_ratio",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_history_returns_empty_dataframe(self, tmp_path, price_index):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        eng = OptionsFeatureEngineer(history_store=store, include_vix=False)
        result = eng.build("AAPL", price_index)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(price_index)
        # No history -> all NaN, no snapshot columns populated
        assert "opt_atm_iv_cm30" not in result.columns

    def test_includes_iv_rv_spread_when_realized_vol_supplied(
        self, tmp_path, price_index
    ):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        _populate_history(store, "AAPL")
        rv = pd.Series(0.18, index=price_index)
        eng = OptionsFeatureEngineer(history_store=store, include_vix=False)
        result = eng.build("AAPL", price_index, realized_vol_21d=rv)
        assert "opt_iv_rv_spread" in result.columns

    def test_short_index_returns_empty(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        eng = OptionsFeatureEngineer(history_store=store, include_vix=False)
        tiny_index = pd.bdate_range("2026-01-01", periods=5)
        result = eng.build("AAPL", tiny_index)
        assert result.empty or len(result) == len(tiny_index)

    def test_vix_features_merged_when_enabled(self, tmp_path, price_index):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        _populate_history(store, "AAPL")

        vix_series = pd.Series(
            np.random.uniform(14, 28, len(price_index)), index=price_index
        )

        with patch(
            "app.ml.options_features._fetch_yf_series",
            side_effect=lambda ticker, start, end: (
                vix_series if ticker == "^VIX" else pd.Series(dtype=float)
            ),
        ):
            eng = OptionsFeatureEngineer(history_store=store, include_vix=True)
            result = eng.build("AAPL", price_index)

        assert "vix_level" in result.columns
        assert result["vix_level"].notna().any()

    def test_auto_snapshot_triggers_update_when_missing(self, tmp_path, price_index):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        fake_snap = OptionsSnapshot(
            ticker="AAPL",
            snapshot_date=str(datetime.now(timezone.utc).date()),
            is_optionable=True,
            atm_iv_cm30=0.21,
        )
        with patch(
            "app.ml.options_features.fetch_options_snapshot", return_value=fake_snap
        ):
            eng = OptionsFeatureEngineer(
                history_store=store, include_vix=False, auto_snapshot=True
            )
            result = eng.build("AAPL", price_index)

        assert store.has_snapshot_today("AAPL")
        assert "opt_atm_iv_cm30" in result.columns

    def test_get_feature_names(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        eng = OptionsFeatureEngineer(history_store=store)
        names = eng.get_feature_names()
        assert names == list(OPTIONS_FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# build_options_context_narrative
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildOptionsContextNarrative:
    def test_empty_history_returns_empty_string(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        assert build_options_context_narrative("AAPL", history_store=store) == ""

    def test_not_optionable_returns_message(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        store.append(
            OptionsSnapshot(
                ticker="AAPL",
                snapshot_date=str(datetime.now(timezone.utc).date()),
                is_optionable=False,
            )
        )
        text = build_options_context_narrative("AAPL", history_store=store)
        assert "not liquid enough" in text

    def test_optionable_history_produces_narrative(self, tmp_path):
        store = OptionsHistoryStore(cache_dir=tmp_path)
        _populate_history(store, "AAPL", n_days=40)
        text = build_options_context_narrative("AAPL", history_store=store)
        assert "implied volatility" in text
        assert text.startswith("AAPL options market:")


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer integration
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineerOptionsIntegration:
    def test_include_options_false_does_not_add_columns(self, sample_ohlcv):
        eng = FeatureEngineer(include_options=False)
        df = eng.build_features(sample_ohlcv)
        opt_cols = [
            c for c in df.columns if c.startswith("opt_") or c.startswith("vix")
        ]
        assert len(opt_cols) == 0

    def test_include_options_true_merges_columns(self, sample_ohlcv):
        mock_opt_eng = MagicMock()

        def _build(ticker, price_index, realized_vol_21d=None):
            return pd.DataFrame(
                {
                    "opt_atm_iv_cm30": 0.25,
                    "opt_iv_rank_252d": 0.6,
                    "vix_level": 18.0,
                },
                index=price_index,
            )

        mock_opt_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_options=True,
            ticker="AAPL",
            options_engineer=mock_opt_eng,
        )
        df = eng.build_features(sample_ohlcv)

        assert "opt_atm_iv_cm30" in df.columns
        assert "vix_level" in df.columns

    def test_options_failure_does_not_crash_pipeline(self, sample_ohlcv):
        mock_opt_eng = MagicMock()
        mock_opt_eng.build.side_effect = RuntimeError("yfinance options error")

        eng = FeatureEngineer(
            include_options=True,
            ticker="AAPL",
            options_engineer=mock_opt_eng,
        )
        df = eng.build_features(sample_ohlcv)
        # Technical features must still be present
        assert "rsi_14" in df.columns
        assert "macd" in df.columns

    def test_options_requires_ticker(self, sample_ohlcv):
        mock_opt_eng = MagicMock()
        mock_opt_eng.build.return_value = pd.DataFrame()

        eng = FeatureEngineer(
            include_options=True,
            ticker=None,
            options_engineer=mock_opt_eng,
        )
        df = eng.build_features(sample_ohlcv)
        assert not df.empty

    def test_no_inf_in_merged_output(self, sample_ohlcv):
        mock_opt_eng = MagicMock()

        def _build(ticker, price_index, realized_vol_21d=None):
            df = pd.DataFrame({"opt_atm_iv_cm30": 0.25}, index=price_index)
            df.loc[df.index[10:20], "opt_atm_iv_cm30"] = np.nan
            return df

        mock_opt_eng.build.side_effect = _build

        eng = FeatureEngineer(
            include_options=True,
            ticker="AAPL",
            options_engineer=mock_opt_eng,
        )
        df = eng.build_features(sample_ohlcv)
        feat_cols = eng.get_feature_columns(df)
        assert not np.isinf(df[feat_cols].values).any()


# ─────────────────────────────────────────────────────────────────────────────
# Feature name catalogue
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureNameCatalogue:
    def test_all_names_are_strings(self):
        for name in OPTIONS_FEATURE_NAMES:
            assert isinstance(name, str) and len(name) > 0

    def test_no_duplicate_names(self):
        assert len(OPTIONS_FEATURE_NAMES) == len(set(OPTIONS_FEATURE_NAMES))

    def test_prefixes_are_correct(self):
        for name in OPTIONS_FEATURE_NAMES:
            assert name.startswith("opt_") or name.startswith("vix"), (
                f"Feature {name!r} must start with 'opt_' or 'vix'"
            )
