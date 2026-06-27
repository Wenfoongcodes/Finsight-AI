"""
All yfinance and ingest_market_data calls are mocked — no internet access
required. Synthetic price series share a common market factor so the
correlation/covariance machinery has something non-trivial to estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import PortfolioAnalysisError
from app.services.portfolio_analysis import (
    LEDOIT_WOLF_THRESHOLD_STOCKS,
    CovarianceEstimator,
    MeanVarianceOptimizer,
    PortfolioAnalysisService,
    PortfolioPosition,
    RiskAttributor,
    VaREstimator,
    compute_efficient_frontier,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_correlated_ohlcv(n_tickers: int, n_days: int = 400, seed: int = 7) -> dict:
    """
    Synthetic OHLCV data for n_tickers, each driven by a shared market factor
    plus idiosyncratic noise -- gives the covariance estimator genuine
    cross-asset correlation to recover, rather than pure noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=n_days)
    market_factor = rng.normal(0, 0.01, n_days)

    data = {}
    for i in range(n_tickers):
        ticker = f"T{i:02d}"
        idio = rng.normal(0, 0.006, n_days)
        beta = 0.6 + 0.1 * (i % 3)
        rets = 0.0002 + beta * market_factor + idio
        prices = 100.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.005,
                "Low": prices * 0.995,
                "Close": prices,
                "Volume": np.full(n_days, 5_000_000.0),
            },
            index=dates,
        )
        data[ticker] = df
    return data


@pytest.fixture
def four_ticker_data():
    return _make_correlated_ohlcv(4, n_days=400)


@pytest.fixture
def twelve_ticker_data():
    return _make_correlated_ohlcv(12, n_days=400)


def _patch_ingest(data: dict):
    def _fake_ingest(ticker, period_years=2, min_rows=60, **kwargs):
        if ticker not in data:
            raise Exception(f"No data for {ticker}")
        return data[ticker]

    return patch(
        "app.services.portfolio_analysis.ingest_market_data", side_effect=_fake_ingest
    )


def _patch_sectors(mapping: dict[str, str]):
    return patch(
        "app.services.portfolio_analysis._resolve_sectors",
        return_value=mapping,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CovarianceEstimator
# ─────────────────────────────────────────────────────────────────────────────


class TestCovarianceEstimator:
    def _returns(self, data: dict) -> pd.DataFrame:
        prices = pd.DataFrame({t: df["Close"] for t, df in data.items()}).sort_index()
        return np.log(prices / prices.shift(1)).dropna(how="any")

    def test_correlation_diagonal_is_one(self, four_ticker_data):
        returns = self._returns(four_ticker_data)
        est = CovarianceEstimator().estimate(returns)
        diag = np.diag(est.correlation.values)
        assert np.allclose(diag, 1.0)

    def test_correlation_is_symmetric(self, four_ticker_data):
        returns = self._returns(four_ticker_data)
        est = CovarianceEstimator().estimate(returns)
        assert np.allclose(est.correlation.values, est.correlation.values.T)

    def test_correlation_bounded(self, four_ticker_data):
        returns = self._returns(four_ticker_data)
        est = CovarianceEstimator().estimate(returns)
        assert (est.correlation.values >= -1.0 - 1e-9).all()
        assert (est.correlation.values <= 1.0 + 1e-9).all()

    def test_uses_sample_method_below_threshold(self, four_ticker_data):
        returns = self._returns(four_ticker_data)
        est = CovarianceEstimator().estimate(returns)
        assert est.method == "sample"

    def test_uses_ledoit_wolf_above_threshold(self, twelve_ticker_data):
        assert len(twelve_ticker_data) > LEDOIT_WOLF_THRESHOLD_STOCKS
        returns = self._returns(twelve_ticker_data)
        est = CovarianceEstimator().estimate(returns)
        assert est.method == "ledoit_wolf"

    def test_uses_ewma_when_requested(self, four_ticker_data):
        returns = self._returns(four_ticker_data)
        est = CovarianceEstimator(use_ewma=True).estimate(returns)
        assert est.method == "ewma"

    def test_raises_on_insufficient_history(self, four_ticker_data):
        returns = self._returns(four_ticker_data).iloc[:10]
        with pytest.raises(PortfolioAnalysisError):
            CovarianceEstimator().estimate(returns)

    def test_correlated_assets_show_positive_correlation(self, four_ticker_data):
        """All four synthetic tickers share a positive-beta market factor."""
        returns = self._returns(four_ticker_data)
        est = CovarianceEstimator().estimate(returns)
        off_diag = est.correlation.values[~np.eye(4, dtype=bool)]
        assert (off_diag > 0).mean() > 0.5  # majority should be positively correlated


# ─────────────────────────────────────────────────────────────────────────────
# MeanVarianceOptimizer
# ─────────────────────────────────────────────────────────────────────────────


class TestMeanVarianceOptimizer:
    def _toy_problem(self):
        tickers = ["A", "B", "C", "D"]
        expected_returns = pd.Series([0.05, 0.10, -0.02, 0.03], index=tickers)
        cov = pd.DataFrame(
            np.array(
                [
                    [0.04, 0.01, 0.005, 0.0],
                    [0.01, 0.09, 0.0, 0.01],
                    [0.005, 0.0, 0.02, 0.0],
                    [0.0, 0.01, 0.0, 0.03],
                ]
            ),
            index=tickers,
            columns=tickers,
        )
        return expected_returns, cov

    def test_weights_sum_to_one(self):
        mu, cov = self._toy_problem()
        result = MeanVarianceOptimizer(max_position_weight=0.5).optimize(mu, cov)
        assert pytest.approx(result.weights.sum(), abs=1e-4) == 1.0

    def test_weights_respect_position_cap(self):
        mu, cov = self._toy_problem()
        cap = 0.4
        result = MeanVarianceOptimizer(max_position_weight=cap).optimize(mu, cov)
        assert (result.weights.values <= cap + 1e-6).all()

    def test_long_only_has_no_negative_weights(self):
        mu, cov = self._toy_problem()
        result = MeanVarianceOptimizer(
            max_position_weight=0.5, long_only=True
        ).optimize(mu, cov)
        assert (result.weights.values >= -1e-6).all()

    def test_converged_flag_is_bool(self):
        mu, cov = self._toy_problem()
        result = MeanVarianceOptimizer(max_position_weight=0.5).optimize(mu, cov)
        assert isinstance(result.converged, bool)

    def test_sector_constraint_respected(self):
        mu, cov = self._toy_problem()
        sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
        result = MeanVarianceOptimizer(
            max_position_weight=0.5, max_sector_weight=0.5
        ).optimize(mu, cov, sectors=sectors)
        tech_weight = result.weights["A"] + result.weights["B"]
        assert tech_weight <= 0.5 + 1e-3

    def test_constraints_applied_lists_position_cap(self):
        mu, cov = self._toy_problem()
        result = MeanVarianceOptimizer(max_position_weight=0.4).optimize(mu, cov)
        assert any("max_position_weight" in c for c in result.constraints_applied)


# ─────────────────────────────────────────────────────────────────────────────
# compute_efficient_frontier
# ─────────────────────────────────────────────────────────────────────────────


class TestEfficientFrontier:
    def _toy_problem(self):
        tickers = ["A", "B", "C"]
        mu = pd.Series([0.02, 0.08, 0.05], index=tickers)
        cov = pd.DataFrame(
            np.array([[0.03, 0.005, 0.0], [0.005, 0.06, 0.01], [0.0, 0.01, 0.04]]),
            index=tickers,
            columns=tickers,
        )
        return mu, cov

    def test_returns_points_sorted_by_volatility(self):
        mu, cov = self._toy_problem()
        frontier = compute_efficient_frontier(
            mu, cov, max_position_weight=1.0, n_points=10
        )
        vols = [p["volatility"] for p in frontier]
        assert vols == sorted(vols)

    def test_respects_n_points_upper_bound(self):
        mu, cov = self._toy_problem()
        frontier = compute_efficient_frontier(
            mu, cov, max_position_weight=1.0, n_points=8
        )
        assert len(frontier) <= 8

    def test_degenerate_equal_returns_yields_single_point(self):
        tickers = ["A", "B"]
        mu = pd.Series([0.03, 0.03], index=tickers)
        cov = pd.DataFrame(
            np.array([[0.02, 0.0], [0.0, 0.02]]), index=tickers, columns=tickers
        )
        frontier = compute_efficient_frontier(
            mu, cov, max_position_weight=1.0, n_points=10
        )
        assert len(frontier) == 1


# ─────────────────────────────────────────────────────────────────────────────
# RiskAttributor
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskAttributor:
    def _toy_problem(self):
        tickers = ["A", "B", "C"]
        weights = pd.Series([0.5, 0.3, 0.2], index=tickers)
        cov = pd.DataFrame(
            np.array([[0.05, 0.01, 0.0], [0.01, 0.08, 0.0], [0.0, 0.0, 0.02]]),
            index=tickers,
            columns=tickers,
        )
        return weights, cov

    def test_contributions_sum_to_total_risk(self):
        weights, cov = self._toy_problem()
        contributions = RiskAttributor.attribute(weights, cov)
        total_pct = sum(c.pct_of_total_risk for c in contributions)
        assert pytest.approx(total_pct, abs=1e-3) == 1.0

    def test_higher_weight_higher_vol_dominates(self):
        weights, cov = self._toy_problem()
        contributions = {c.ticker: c for c in RiskAttributor.attribute(weights, cov)}
        # B has the largest standalone variance (0.08) and meaningful weight (0.3)
        assert (
            contributions["B"].pct_of_total_risk > contributions["C"].pct_of_total_risk
        )

    def test_zero_weight_has_zero_contribution(self):
        tickers = ["A", "B"]
        weights = pd.Series([1.0, 0.0], index=tickers)
        cov = pd.DataFrame(
            np.array([[0.04, 0.0], [0.0, 0.09]]), index=tickers, columns=tickers
        )
        contributions = {c.ticker: c for c in RiskAttributor.attribute(weights, cov)}
        assert contributions["B"].pct_of_total_risk == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VaREstimator
# ─────────────────────────────────────────────────────────────────────────────


class TestVaREstimator:
    def _toy_problem(self, n_days=300):
        tickers = ["A", "B"]
        weights = pd.Series([0.6, 0.4], index=tickers)
        cov = pd.DataFrame(
            np.array([[0.04, 0.0], [0.0, 0.09]]), index=tickers, columns=tickers
        )
        rng = np.random.default_rng(3)
        returns = pd.DataFrame(
            {
                "A": rng.normal(0, 0.012, n_days),
                "B": rng.normal(0, 0.018, n_days),
            }
        )
        return weights, cov, returns

    def test_portfolio_volatility_positive(self):
        weights, cov, _ = self._toy_problem()
        vol = VaREstimator.portfolio_volatility(weights, cov)
        assert vol > 0

    def test_higher_confidence_gives_larger_var(self):
        weights, cov, returns = self._toy_problem()
        var_95 = VaREstimator.estimate(weights, cov, returns, confidence=0.95)
        var_99 = VaREstimator.estimate(weights, cov, returns, confidence=0.99)
        assert var_99.parametric_var_pct > var_95.parametric_var_pct

    def test_dollar_value_computed_when_portfolio_value_given(self):
        weights, cov, returns = self._toy_problem()
        var = VaREstimator.estimate(weights, cov, returns, portfolio_value=100_000.0)
        assert var.parametric_var_value is not None
        assert var.parametric_var_value == pytest.approx(
            var.parametric_var_pct * 100_000.0, rel=1e-3
        )

    def test_no_dollar_value_when_portfolio_value_omitted(self):
        weights, cov, returns = self._toy_problem()
        var = VaREstimator.estimate(weights, cov, returns)
        assert var.parametric_var_value is None

    def test_longer_horizon_gives_larger_var(self):
        weights, cov, returns = self._toy_problem()
        var_1d = VaREstimator.estimate(weights, cov, returns, horizon_days=1)
        var_10d = VaREstimator.estimate(weights, cov, returns, horizon_days=10)
        assert var_10d.parametric_var_pct > var_1d.parametric_var_pct


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioAnalysisService — integration
# ─────────────────────────────────────────────────────────────────────────────


class TestPortfolioAnalysisService:
    def test_analyze_returns_expected_structure(self, four_ticker_data):
        positions = [PortfolioPosition(ticker=t) for t in four_ticker_data]
        probs = {t: 0.6 for t in four_ticker_data}

        with (
            _patch_ingest(four_ticker_data),
            _patch_sectors({t: "Technology" for t in four_ticker_data}),
        ):
            svc = PortfolioAnalysisService()
            result = svc.analyze(positions, prediction_probs=probs)

        assert set(result.tickers) == set(four_ticker_data.keys())
        assert pytest.approx(sum(result.current_weights.values()), abs=1e-3) == 1.0
        assert pytest.approx(sum(result.optimal_weights.values()), abs=1e-3) == 1.0
        assert pytest.approx(sum(result.sector_exposure.values()), abs=1e-3) == 1.0
        assert result.current_portfolio_volatility >= 0
        assert result.optimal_portfolio_volatility >= 0
        assert len(result.efficient_frontier) > 0
        assert result.var["confidence"] == 0.95

    def test_drops_failed_ticker_and_warns(self, four_ticker_data):
        tickers = list(four_ticker_data.keys()) + ["BADTICKER"]
        positions = [PortfolioPosition(ticker=t) for t in tickers]

        with (
            _patch_ingest(four_ticker_data),
            _patch_sectors({t: "Technology" for t in four_ticker_data}),
        ):
            svc = PortfolioAnalysisService()
            result = svc.analyze(positions)

        assert "BADTICKER" in result.dropped_tickers
        assert any("BADTICKER" in w for w in result.warnings)

    def test_missing_predictions_produce_neutral_expected_returns(
        self, four_ticker_data
    ):
        positions = [PortfolioPosition(ticker=t) for t in four_ticker_data]

        with (
            _patch_ingest(four_ticker_data),
            _patch_sectors({t: "Technology" for t in four_ticker_data}),
        ):
            svc = PortfolioAnalysisService()
            result = svc.analyze(positions, prediction_probs=None)

        assert all(abs(v) < 1e-9 for v in result.expected_returns.values())
        assert any("neutral" in w.lower() for w in result.warnings)

    def test_infeasible_max_position_weight_raises(self, four_ticker_data):
        positions = [PortfolioPosition(ticker=t) for t in four_ticker_data]

        with (
            _patch_ingest(four_ticker_data),
            _patch_sectors({t: "Technology" for t in four_ticker_data}),
        ):
            svc = PortfolioAnalysisService()
            with pytest.raises(PortfolioAnalysisError):
                # 4 tickers * 10% cap = 40% max achievable -- infeasible.
                svc.analyze(positions, max_position_weight=0.10)

    def test_raises_with_fewer_than_two_valid_tickers(self, four_ticker_data):
        positions = [PortfolioPosition(ticker="ONLY_ONE")]
        single = {"ONLY_ONE": next(iter(four_ticker_data.values()))}

        with _patch_ingest(single):
            svc = PortfolioAnalysisService()
            with pytest.raises(PortfolioAnalysisError):
                svc.analyze(positions)

    def test_shares_based_sizing_uses_latest_price(self, four_ticker_data):
        tickers = list(four_ticker_data.keys())
        # Give the first ticker a large share count -> should dominate current_weights
        positions = [
            PortfolioPosition(ticker=t, shares=1000.0 if i == 0 else 1.0)
            for i, t in enumerate(tickers)
        ]

        with (
            _patch_ingest(four_ticker_data),
            _patch_sectors({t: "Technology" for t in tickers}),
        ):
            svc = PortfolioAnalysisService()
            result = svc.analyze(positions)

        assert result.current_weights[tickers[0]] > result.current_weights[tickers[1]]

    def test_sector_exposure_aggregates_across_tickers(self, four_ticker_data):
        tickers = list(four_ticker_data.keys())
        positions = [PortfolioPosition(ticker=t) for t in tickers]
        sectors = {
            tickers[0]: "Tech",
            tickers[1]: "Tech",
            tickers[2]: "Health",
            tickers[3]: "Health",
        }

        with _patch_ingest(four_ticker_data), _patch_sectors(sectors):
            svc = PortfolioAnalysisService()
            result = svc.analyze(positions)

        assert set(result.sector_exposure.keys()) == {"Tech", "Health"}
        assert pytest.approx(sum(result.sector_exposure.values()), abs=1e-3) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# API route smoke tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPortfolioRoute:
    def test_rejects_fewer_than_two_positions(self, api_client):
        resp = api_client.post(
            "/api/v1/portfolio/analyze",
            json={"positions": [{"ticker": "AAPL"}]},
        )
        assert resp.status_code == 422

    def test_rejects_duplicate_tickers(self, api_client):
        resp = api_client.post(
            "/api/v1/portfolio/analyze",
            json={"positions": [{"ticker": "AAPL"}, {"ticker": "AAPL"}]},
        )
        assert resp.status_code == 422

    def test_successful_analysis_mocked(self, api_client, four_ticker_data):
        tickers = list(four_ticker_data.keys())

        mock_pred = MagicMock()
        mock_pred.p_bullish = 0.62
        mock_pred.prediction = 1
        mock_pred.confidence_label = "moderate"
        mock_pred.model_name = "xgboost"

        with (
            patch(
                "app.api.v1.endpoints.portfolio._get_prediction_service"
            ) as mock_get_svc,
            _patch_ingest(four_ticker_data),
            _patch_sectors({t: "Technology" for t in tickers}),
        ):
            mock_get_svc.return_value.predict.return_value = mock_pred

            resp = api_client.post(
                "/api/v1/portfolio/analyze",
                json={
                    "positions": [{"ticker": t} for t in tickers],
                    "max_position_weight": 0.5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert set(data["tickers"]) == set(tickers)
        assert pytest.approx(sum(data["current_weights"].values()), abs=1e-3) == 1.0
        assert len(data["predictions"]) == len(tickers)
