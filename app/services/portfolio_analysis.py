"""
Turns a list of independent single-stock predictions into a coherent
portfolio view:

1. ReturnSeriesBuilder      — fetches + aligns daily return series for all
                               portfolio constituents (reuses the existing
                               parquet-cached ingest_market_data() pipeline).
2. CovarianceEstimator      — rolling / EWMA correlation & covariance matrix.
                               Automatically switches to Ledoit-Wolf shrinkage
                               (sklearn.covariance.LedoitWolf) once the number
                               of assets exceeds LEDOIT_WOLF_THRESHOLD_STOCKS,
                               where the sample covariance matrix becomes
                               poorly conditioned relative to the available
                               history.
3. MeanVarianceOptimizer    — Markowitz mean-variance optimization (SLSQP via
                               scipy.optimize — no extra dependency, scipy
                               ships transitively with scikit-learn) with
                               position-size, long-only, sector-concentration,
                               and turnover constraints.
4. compute_efficient_frontier — sweeps target returns and solves the
                               minimum-variance portfolio at each, tracing
                               the efficient frontier for visualisation.
5. RiskAttributor           — decomposes total portfolio variance into
                               per-asset contributions (accounts for
                               cross-asset correlation, not just standalone
                               volatility).
6. VaREstimator             — parametric (analytical) and historical
                               (empirical) Value-at-Risk.

Important caveat — expected returns are a proxy
-------------------------------------------------
The mean-variance optimizer needs an "expected return" estimate per asset.
FinSight AI derives this from the ML model's calibrated P(bullish): a stock
at p_bullish=0.70 is treated as having a higher expected-return proxy than
one at p_bullish=0.52. This is **not** a real forecast of magnitude — it is
a monotonic transform of a directional probability, scaled by
``return_scale_factor``. It is documented here, in the API response
(``expected_returns``), and should be treated by callers as a relative
ranking signal, not as guaranteed future performance.

This module never raises to a caller that hasn't supplied valid input — all
the way down to RiskAttributor and VaREstimator. The boundary that *can*
raise is data ingestion (insufficient tickers / insufficient history) and
infeasible constraint configurations, both surfaced as
``PortfolioAnalysisError`` so the API layer can return a clean 422 instead
of a 500.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from app.core.exceptions import PortfolioAnalysisError
from app.core.formatting import round_metric, utc_now_iso
from app.core.logging_config import get_logger
from app.ml.data_ingestion import ingest_market_data

logger = get_logger("portfolio_analysis")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_LOOKBACK_DAYS: int = 252
MIN_LOOKBACK_DAYS: int = 60
LEDOIT_WOLF_THRESHOLD_STOCKS: int = 10  # > this many assets -> shrinkage estimator
DEFAULT_MAX_POSITION_WEIGHT: float = (
    0.50  # feasible by default down to 2 positions (the
)
# documented minimum portfolio size); a fixed 10% cap would mathematically force
# the optimizer to fail for any portfolio smaller than 10 positions (10% * 9 < 100%),
# which is a poor out-of-the-box default. Callers building larger, more diversified
# portfolios should pass a tighter explicit max_position_weight (e.g. 0.10-0.15).
DEFAULT_MAX_SECTOR_WEIGHT: float = 1.0  # opt-in (disabled) by default. Unlike
# max_position_weight, there's no portfolio-size-independent default that's
# guaranteed feasible here: a perfectly legitimate request (e.g. "analyze my
# 5 favorite tech stocks") can span just a single sector, and any binding
# default cap below 1.0 would reject that out of the box. Callers building
# genuinely multi-sector portfolios should pass an explicit tighter value
# (e.g. 0.30-0.40) to enforce sector diversification.
DEFAULT_RISK_FREE_RATE: float = 0.0
DEFAULT_VAR_CONFIDENCE: float = 0.95
DEFAULT_RETURN_SCALE_FACTOR: float = 0.50
DEFAULT_EFFICIENT_FRONTIER_POINTS: int = 15
TANGENCY_SWEEP_POINTS: int = 21  # internal resolution for the optimizer's own
# min-variance-for-target sweep used to locate the max-Sharpe (tangency) point;
# independent of DEFAULT_EFFICIENT_FRONTIER_POINTS, which controls the coarser
# frontier returned to the caller for visualization.


# ─────────────────────────────────────────────────────────────────────────────
# Sector resolution — lightweight, self-contained, TTL-free in-process cache
# ─────────────────────────────────────────────────────────────────────────────
#
# Deliberately separate from app.ml.sector_correlation_features._resolve_sector_etf
# (which maps sector -> ETF for feature engineering). This module only needs
# the raw sector string for grouping / concentration constraints, so it keeps
# its own tiny cache rather than importing a private helper from another
# subsystem.

_SECTOR_CACHE: dict[str, str] = {}


def _resolve_sector(ticker: str) -> str:
    """Best-effort sector lookup via yfinance. Never raises — returns 'Unknown'."""
    if ticker in _SECTOR_CACHE:
        return _SECTOR_CACHE[ticker]
    sector = "Unknown"
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector") or "Unknown"
    except Exception as exc:
        logger.debug("[portfolio] Sector lookup failed for %s: %s", ticker, exc)
    _SECTOR_CACHE[ticker] = sector
    return sector


def _resolve_sectors(tickers: list[str]) -> dict[str, str]:
    return {t: _resolve_sector(t) for t in tickers}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Return series construction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReturnSeriesBundle:
    """Aligned daily log-return matrix for a portfolio's constituents."""

    tickers: list[str]
    returns: pd.DataFrame  # columns=tickers, index=dates, daily log returns
    latest_prices: pd.Series  # ticker -> most recent Close
    dropped_tickers: list[str] = field(default_factory=list)


class ReturnSeriesBuilder:
    """
    Fetches and aligns daily price/return series for a list of tickers.

    Reuses ``ingest_market_data()`` so the parquet cache (and the
    CACHE_MAX_AGE_DAYS TTL) is shared with the rest of the prediction
    pipeline — zero additional HTTP cost for tickers already fetched today.

    Tickers that fail to ingest (delisted, bad symbol, network error,
    insufficient history) are dropped with a warning rather than failing the
    whole portfolio request. Analysis proceeds with at least 2 valid tickers;
    fewer than that raises ``PortfolioAnalysisError``.
    """

    def __init__(self, period_years: int = 2) -> None:
        self.period_years = period_years

    def build(self, tickers: list[str]) -> ReturnSeriesBundle:
        closes: dict[str, pd.Series] = {}
        dropped: list[str] = []

        for ticker in tickers:
            try:
                df = ingest_market_data(
                    ticker,
                    period_years=self.period_years,
                    min_rows=MIN_LOOKBACK_DAYS,
                )
                closes[ticker] = df["Close"]
            except Exception as exc:
                logger.warning(
                    "[portfolio] Dropping %s — market data unavailable: %s",
                    ticker,
                    exc,
                )
                dropped.append(ticker)

        if len(closes) < 2:
            raise PortfolioAnalysisError(
                "At least 2 tickers with valid market data are required for "
                "portfolio analysis.",
                detail=f"Dropped: {dropped}",
            )

        price_df = pd.DataFrame(closes).sort_index()
        price_df = price_df.ffill().dropna(how="any")

        if len(price_df) < MIN_LOOKBACK_DAYS:
            raise PortfolioAnalysisError(
                f"Only {len(price_df)} overlapping trading days across the "
                f"requested tickers; minimum required is {MIN_LOOKBACK_DAYS}.",
                detail="Tickers may have mismatched listing histories.",
            )

        # A single zero or negative price tick (a known data-source quirk,
        # e.g. around stock splits or brief data glitches) produces
        # log(0) = -inf or log(negative) = NaN. .dropna() alone only
        # catches the latter -- an infinite return would otherwise sail
        # through untouched and corrupt every downstream covariance/risk
        # calculation with a non-finite value, eventually surfacing as a
        # confusing "Out of range float values are not JSON compliant"
        # error at the API boundary rather than a clear data-quality one.
        # Replacing inf -> NaN first means a single bad tick is dropped
        # like any other missing value instead of propagating silently.
        returns = np.log(price_df / price_df.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        kept = list(price_df.columns)

        return ReturnSeriesBundle(
            tickers=kept,
            returns=returns,
            latest_prices=price_df.iloc[-1],
            dropped_tickers=dropped,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Covariance / correlation estimation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CovarianceEstimate:
    tickers: list[str]
    correlation: pd.DataFrame
    covariance: pd.DataFrame  # annualized
    method: str  # "sample" | "ledoit_wolf" | "ewma"
    lookback_days: int
    annualized_volatility: pd.Series


class CovarianceEstimator:
    """
    Estimates the correlation and (annualized) covariance matrix for a set
    of aligned daily return series.

    Method selection
    ----------------
    - ``len(tickers) > LEDOIT_WOLF_THRESHOLD_STOCKS``
        Ledoit-Wolf shrinkage (``sklearn.covariance.LedoitWolf``). As the
        number of assets approaches the number of usable observations the
        sample covariance matrix becomes poorly conditioned (or singular);
        shrinking it toward a structured target produces a more reliable,
        better-conditioned estimate.
    - ``len(tickers) <= LEDOIT_WOLF_THRESHOLD_STOCKS``
        Sample covariance, optionally exponentially weighted
        (``use_ewma=True``) so recent observations dominate while older
        history is not discarded outright — more responsive to regime
        changes than a flat rolling window without the noise of a very
        short window.
    """

    def __init__(
        self,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        use_ewma: bool = False,
        ewma_halflife_days: int = 63,
    ) -> None:
        self.lookback_days = lookback_days
        self.use_ewma = use_ewma
        self.ewma_halflife_days = ewma_halflife_days

    def estimate(self, returns: pd.DataFrame) -> CovarianceEstimate:
        window = (
            returns.iloc[-self.lookback_days :]
            if len(returns) > self.lookback_days
            else returns
        )
        if len(window) < MIN_LOOKBACK_DAYS:
            raise PortfolioAnalysisError(
                f"Insufficient overlapping history ({len(window)} days) for "
                f"covariance estimation; minimum required is "
                f"{MIN_LOOKBACK_DAYS}."
            )

        tickers = list(window.columns)
        n_assets = len(tickers)

        if n_assets > LEDOIT_WOLF_THRESHOLD_STOCKS:
            method = "ledoit_wolf"
            lw = LedoitWolf().fit(window.values)
            cov_daily_df = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
            logger.info(
                "[portfolio] %d assets > threshold (%d) — using Ledoit-Wolf "
                "shrinkage (shrinkage=%.4f).",
                n_assets,
                LEDOIT_WOLF_THRESHOLD_STOCKS,
                lw.shrinkage_,
            )
        elif self.use_ewma:
            method = "ewma"
            cov_daily_df = _ewma_covariance(window, self.ewma_halflife_days)
        else:
            method = "sample"
            cov_daily_df = window.cov()

        cov_annual = cov_daily_df * TRADING_DAYS_PER_YEAR
        std = np.sqrt(np.clip(np.diag(cov_annual.values), 0.0, None))
        std_safe = np.where(std == 0, np.nan, std)
        corr_values = cov_annual.values / np.outer(std_safe, std_safe)
        # Set the diagonal on the raw numpy array *before* wrapping it in a
        # DataFrame. Under pandas' Copy-on-Write semantics (mandatory as of
        # pandas 3.0, and opt-in on 2.x), `DataFrame.values` after a
        # `.fillna()` call returns a read-only array, so mutating it
        # in-place via `np.fill_diagonal` raises `ValueError: underlying
        # array is read-only`. Mutating the freshly-allocated numpy array
        # directly sidesteps that entirely and is correct regardless of
        # pandas version/CoW setting.
        np.fill_diagonal(corr_values, 1.0)
        corr_df = pd.DataFrame(corr_values, index=tickers, columns=tickers).fillna(0.0)

        return CovarianceEstimate(
            tickers=tickers,
            correlation=corr_df,
            covariance=cov_annual,
            method=method,
            lookback_days=len(window),
            annualized_volatility=pd.Series(std, index=tickers),
        )


def _ewma_covariance(returns: pd.DataFrame, halflife_days: int) -> pd.DataFrame:
    """
    Exponentially weighted covariance — recent observations weighted more
    heavily than older ones, without discarding the older history outright
    (unlike a hard rolling-window cutoff).

    ``returns`` is assumed sorted ascending by date (oldest first); the
    last row receives the highest weight.
    """
    decay = math.log(2) / halflife_days
    lam = math.exp(-decay)
    n = len(returns)
    # weights[0] (oldest) -> lam^(n-1) (smallest); weights[-1] (newest) -> lam^0 = 1
    raw_weights = np.array([lam**i for i in range(n)][::-1])
    weights = raw_weights / raw_weights.sum()

    values = returns.values
    weighted_mean = np.average(values, axis=0, weights=weights)
    centered = values - weighted_mean
    weighted_cov = (centered * weights[:, None]).T @ centered
    return pd.DataFrame(weighted_cov, index=returns.columns, columns=returns.columns)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mean-variance optimization
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    converged: bool
    constraints_applied: list[str]


class MeanVarianceOptimizer:
    """
    Solves a constrained mean-variance (Markowitz) optimization via SLSQP.

    Objective
    ---------
    Maximize the Sharpe ratio ``(E[r] - risk_free) / vol`` subject to:
        sum(weights) == 1
        position bounds      -> long-only and/or max_position_weight cap
        sector concentration -> sum of weights per sector <= max_sector_weight
        turnover (optional)  -> sum(|w - w_current|) <= turnover_limit

    No extra dependency is required — ``scipy.optimize`` ships transitively
    with scikit-learn, which is already a hard requirement of FinSight AI.
    """

    def __init__(
        self,
        max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
        max_sector_weight: float = DEFAULT_MAX_SECTOR_WEIGHT,
        long_only: bool = True,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        turnover_limit: Optional[float] = None,
    ) -> None:
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.long_only = long_only
        self.risk_free_rate = risk_free_rate
        self.turnover_limit = turnover_limit

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        sectors: Optional[dict[str, str]] = None,
        current_weights: Optional[pd.Series] = None,
    ) -> OptimizationResult:
        """
        Find the weights that maximize the Sharpe ratio under the configured
        constraints.

        Implementation note — why this isn't a single direct SLSQP call on
        the Sharpe ratio
        -----------------------------------------------------------------
        The Sharpe ratio ``(w.mu - rf) / sqrt(w'Σw)`` is scale-invariant:
        scaling ``w`` by any positive constant leaves it unchanged. SLSQP's
        line search assumes the objective changes along a descent
        direction; near the optimum (and especially whenever expected
        returns are close to equal across assets, which is common given the
        proxy expected-return construction here), the objective gradient
        becomes near-parallel to the active equality-constraint gradient.
        That made a direct single-shot SLSQP solve on the raw Sharpe ratio
        fail ("Positive directional derivative for linesearch") on the
        large majority of realistic portfolios in testing — not just
        contrived edge cases.

        Instead this uses the standard, numerically well-behaved two-anchor
        tangency search: solve the global minimum-variance portfolio and
        the maximum-expected-return portfolio (both convex/linear, very
        reliable for SLSQP), then sweep minimum-variance-for-a-target-return
        solves between those two anchors (each itself convex with a smooth
        quadratic objective) and keep whichever swept point has the highest
        Sharpe ratio. This reconstructs the constrained efficient frontier
        internally and picks the tangency point off of it, rather than
        searching directly in Sharpe-ratio space.
        """
        tickers = list(expected_returns.index)
        n = len(tickers)
        cov = covariance.loc[tickers, tickers].values
        mu = expected_returns.values

        lower = 0.0 if self.long_only else -1.0
        bounds = [(lower, self.max_position_weight) for _ in range(n)]

        constraints_applied = [
            "sum_to_one",
            "long_only" if self.long_only else "long_short",
            f"max_position_weight={self.max_position_weight}",
        ]

        extra_cons: list[dict] = []
        if sectors and self.max_sector_weight < 1.0:
            sector_groups: dict[str, list[int]] = {}
            for idx, t in enumerate(tickers):
                sector_groups.setdefault(sectors.get(t, "Unknown"), []).append(idx)
            for idxs in sector_groups.values():
                extra_cons.append(
                    {
                        "type": "ineq",
                        "fun": (
                            lambda w, idxs=idxs: (
                                self.max_sector_weight - float(np.sum(w[idxs]))
                            )
                        ),
                    }
                )
            constraints_applied.append(f"max_sector_weight={self.max_sector_weight}")

        if self.turnover_limit is not None and current_weights is not None:
            current = current_weights.reindex(tickers).fillna(0.0).values
            extra_cons.append(
                {
                    "type": "ineq",
                    "fun": (
                        lambda w, current=current: (
                            self.turnover_limit - float(np.sum(np.abs(w - current)))
                        )
                    ),
                }
            )
            constraints_applied.append(f"turnover_limit={self.turnover_limit}")

        sum_to_one = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        w_equal = np.full(n, 1.0 / n)
        current_arr = (
            current_weights.reindex(tickers).fillna(0.0).values
            if current_weights is not None
            else None
        )

        def _feasible(w: np.ndarray, cons: list[dict], tol: float = 1e-4) -> bool:
            """
            Checks actual constraint satisfaction directly, independent of
            SLSQP's own ``success`` flag. The L1 turnover constraint
            (``sum(|w - current|) <= limit``) has a non-differentiable kink
            exactly at its own feasible starting point, which in testing
            made SLSQP report "Iteration limit reached" / non-convergence
            even when it had already found (or nearly found) a perfectly
            usable, constraint-satisfying point. Trusting the solver's flag
            alone in that case silently discarded good solutions; checking
            feasibility explicitly recovers them.
            """
            for c in cons:
                val = c["fun"](w)
                if c["type"] == "eq" and abs(val) > tol:
                    return False
                if c["type"] == "ineq" and val < -tol:
                    return False
            lo, hi = bounds[0]
            return bool(np.all(w >= lo - tol) and np.all(w <= hi + tol))

        def _min_variance(target: Optional[float], w0: np.ndarray):
            cons = [sum_to_one] + extra_cons
            if target is not None:
                cons = cons + [
                    {
                        "type": "eq",
                        "fun": (lambda w, target=target: float(np.dot(w, mu) - target)),
                    }
                ]
            res = minimize(
                lambda w: float(w @ cov @ w),
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            res.is_usable = bool(res.success or _feasible(res.x, cons))
            return res

        def _sharpe(w: np.ndarray) -> float:
            port_ret = float(np.dot(w, mu))
            port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
            return (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        # Anchor 1 — global minimum-variance portfolio (no return target).
        min_var_res = _min_variance(target=None, w0=w_equal)

        # Anchor 2 — maximum achievable expected return under the same
        # constraints. Linear objective -> normally the most numerically
        # reliable of the three problem shapes used here (the turnover
        # constraint's non-smooth kink is the main exception, handled by
        # the feasibility check above rather than the solver's own flag).
        base_cons = [sum_to_one] + extra_cons
        max_ret_res = minimize(
            lambda w: -float(np.dot(w, mu)),
            w_equal,
            method="SLSQP",
            bounds=bounds,
            constraints=base_cons,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        max_ret_res.is_usable = bool(
            max_ret_res.success or _feasible(max_ret_res.x, base_cons)
        )

        candidates: list[tuple[np.ndarray, float]] = []

        # Safety-net candidate: holding the current allocation is always a
        # zero-turnover, trivially-feasible-under-turnover option. Only
        # offered as a candidate if it also respects the position/sector
        # constraints — those are constraints on the *recommended* weights,
        # not a license to keep an existing out-of-bounds position.
        if current_arr is not None and _feasible(current_arr, base_cons):
            candidates.append((current_arr, _sharpe(current_arr)))

        if min_var_res.is_usable:
            candidates.append((min_var_res.x, _sharpe(min_var_res.x)))

        if min_var_res.is_usable and max_ret_res.is_usable:
            r_min = float(np.dot(min_var_res.x, mu))
            r_max = float(np.dot(max_ret_res.x, mu))
            if not math.isclose(r_min, r_max, abs_tol=1e-9):
                warm_start = min_var_res.x
                for target in np.linspace(r_min, r_max, TANGENCY_SWEEP_POINTS)[1:-1]:
                    res = _min_variance(target=float(target), w0=warm_start)
                    if res.is_usable:
                        candidates.append((res.x, _sharpe(res.x)))
                        warm_start = res.x
                candidates.append((max_ret_res.x, _sharpe(max_ret_res.x)))

        if candidates:
            weights, _ = max(candidates, key=lambda c: c[1])
            weights = np.clip(weights, lower, self.max_position_weight)
            total = weights.sum()
            if total > 0:
                weights = weights / total
            converged = True
        else:
            logger.warning(
                "[portfolio] Tangency-portfolio search found no feasible "
                "candidate under the given constraints — falling back to "
                "equal-weight allocation."
            )
            weights = w_equal
            converged = False

        port_ret = float(np.dot(weights, mu))
        port_vol = float(np.sqrt(max(weights @ cov @ weights, 1e-12)))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=pd.Series(weights, index=tickers),
            expected_return=round_metric(port_ret),
            expected_volatility=round_metric(port_vol),
            sharpe_ratio=round_metric(sharpe),
            method="mean_variance_slsqp",
            converged=converged,
            constraints_applied=constraints_applied,
        )


def compute_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
    long_only: bool = True,
    n_points: int = DEFAULT_EFFICIENT_FRONTIER_POINTS,
) -> list[dict[str, float]]:
    """
    Trace the efficient frontier by sweeping target returns between the
    minimum and maximum achievable expected return and solving the
    minimum-variance portfolio at each target.

    Returns a list of ``{"expected_return": ..., "volatility": ...}`` points
    sorted by volatility (ascending), suitable for plotting directly.
    """
    tickers = list(expected_returns.index)
    n = len(tickers)
    cov = covariance.loc[tickers, tickers].values
    mu = expected_returns.values

    lo, hi = float(mu.min()), float(mu.max())
    degenerate_returns = math.isclose(lo, hi, abs_tol=1e-12)
    targets = [lo] if degenerate_returns else list(np.linspace(lo, hi, n_points))

    lower = 0.0 if long_only else -1.0
    bounds = [(lower, max_position_weight) for _ in range(n)]

    frontier: list[dict[str, float]] = []
    for target in targets:
        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        if not degenerate_returns:
            # When every asset shares (numerically) the same expected return,
            # dot(w, mu) == mu_value * sum(w) == mu_value for ANY weight
            # vector that already satisfies sum(w) == 1 -- the constraint is
            # then linearly dependent on the sum-to-one constraint, which
            # makes the combined constraint Jacobian singular and causes
            # SLSQP to report "Positive directional derivative for
            # linesearch" (success=False) even when it lands exactly on the
            # true minimum-variance solution. Omitting the redundant
            # constraint in that case avoids the spurious failure entirely.
            cons.append(
                {
                    "type": "eq",
                    "fun": (lambda w, target=target: float(np.dot(w, mu) - target)),
                }
            )

        def _variance(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        w0 = np.full(n, 1.0 / n)
        res = minimize(
            _variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if res.success:
            vol = math.sqrt(max(res.fun, 0.0))
            frontier.append(
                {
                    "expected_return": round_metric(target),
                    "volatility": round_metric(vol),
                }
            )

    frontier.sort(key=lambda p: p["volatility"])
    return frontier


# ─────────────────────────────────────────────────────────────────────────────
# 4. Risk attribution
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RiskContribution:
    ticker: str
    weight: float
    marginal_contribution: float
    contribution_to_variance: float
    pct_of_total_risk: float


class RiskAttributor:
    """
    Decomposes total portfolio variance into per-asset contributions.

    For weight vector ``w`` and covariance matrix ``Σ``:
        portfolio_variance      = w^T Σ w
        marginal_contribution_i = (Σ w)_i
        contribution_to_var_i   = w_i * (Σ w)_i
        sum(contribution_to_var) == portfolio_variance

    A position's contribution depends on its correlation with every other
    holding, not just its own standalone volatility — a stock highly
    correlated with several other positions amplifies their simultaneous
    moves and so contributes more to total risk than its weight alone would
    suggest.
    """

    @staticmethod
    def attribute(
        weights: pd.Series, covariance: pd.DataFrame
    ) -> list[RiskContribution]:
        tickers = list(weights.index)
        w = weights.values
        cov = covariance.loc[tickers, tickers].values

        port_variance = float(w @ cov @ w)
        if port_variance <= 0:
            return [
                RiskContribution(
                    ticker=t,
                    weight=round_metric(float(wi)),
                    marginal_contribution=0.0,
                    contribution_to_variance=0.0,
                    pct_of_total_risk=0.0,
                )
                for t, wi in zip(tickers, w)
            ]

        marginal = cov @ w
        contributions = w * marginal

        results = [
            RiskContribution(
                ticker=t,
                weight=round_metric(float(wi)),
                marginal_contribution=round_metric(float(m)),
                contribution_to_variance=round_metric(float(c)),
                pct_of_total_risk=round_metric(float(c / port_variance)),
            )
            for t, wi, m, c in zip(tickers, w, marginal, contributions)
        ]
        results.sort(key=lambda r: r.pct_of_total_risk, reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Volatility & Value-at-Risk
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VaRResult:
    confidence: float
    horizon_days: int
    parametric_var_pct: float
    parametric_var_value: Optional[float]
    historical_var_pct: float
    historical_var_value: Optional[float]
    method_note: str


class VaREstimator:
    """Portfolio volatility and Value-at-Risk (parametric + historical)."""

    @staticmethod
    def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
        """Annualized portfolio volatility from the standard quadratic form."""
        tickers = list(weights.index)
        w = weights.values
        cov = covariance.loc[tickers, tickers].values
        return float(np.sqrt(max(w @ cov @ w, 0.0)))

    @staticmethod
    def estimate(
        weights: pd.Series,
        covariance: pd.DataFrame,
        returns: pd.DataFrame,
        portfolio_value: Optional[float] = None,
        confidence: float = DEFAULT_VAR_CONFIDENCE,
        horizon_days: int = 1,
    ) -> VaRResult:
        """
        Parametric VaR assumes normally distributed portfolio returns,
        scaled from the annualized volatility down to the requested horizon
        via the square-root-of-time rule.

        Historical (empirical) VaR resamples the actual portfolio return
        history at the requested horizon and reads off the relevant tail
        percentile — no distributional assumption, but requires enough
        history to populate the tail reliably.
        """
        annual_vol = VaREstimator.portfolio_volatility(weights, covariance)
        daily_vol = annual_vol / math.sqrt(TRADING_DAYS_PER_YEAR)
        horizon_vol = daily_vol * math.sqrt(horizon_days)

        z = norm.ppf(1 - confidence)
        parametric_pct = float(-z * horizon_vol)  # positive = magnitude of loss

        tickers = list(weights.index)
        aligned = returns[tickers]
        port_daily_returns = aligned.values @ weights.values

        if horizon_days > 1 and len(port_daily_returns) > horizon_days:
            port_returns_h = (
                pd.Series(port_daily_returns)
                .rolling(horizon_days)
                .sum()
                .dropna()
                .values
            )
        else:
            port_returns_h = port_daily_returns

        if len(port_returns_h) > 0:
            historical_pct = float(
                -np.percentile(port_returns_h, (1 - confidence) * 100)
            )
        else:
            historical_pct = parametric_pct

        return VaRResult(
            confidence=confidence,
            horizon_days=horizon_days,
            parametric_var_pct=round_metric(parametric_pct),
            parametric_var_value=(
                round_metric(parametric_pct * portfolio_value)
                if portfolio_value
                else None
            ),
            historical_var_pct=round_metric(historical_pct),
            historical_var_value=(
                round_metric(historical_pct * portfolio_value)
                if portfolio_value
                else None
            ),
            method_note=(
                f"Parametric VaR assumes normally distributed returns "
                f"(annualized volatility={annual_vol:.4f}). Historical VaR uses "
                f"the empirical distribution of {len(port_returns_h)} "
                f"{horizon_days}-day return window(s)."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Portfolio positions & facade service
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PortfolioPosition:
    """
    A single portfolio holding. Sizing can be specified any of three ways
    (checked in this priority order): explicit ``current_weight``,
    ``market_value``, or ``shares`` (converted to market value using the
    latest fetched Close price). If none are supplied for any position in
    the portfolio, all positions fall back to an equal-weight allocation.
    """

    ticker: str
    shares: float = 0.0
    market_value: Optional[float] = None
    current_weight: Optional[float] = None


def _compute_current_weights(
    pos_by_ticker: dict[str, PortfolioPosition],
    kept_tickers: list[str],
    latest_prices: pd.Series,
) -> pd.Series:
    values: dict[str, float] = {}
    for t in kept_tickers:
        pos = pos_by_ticker.get(t)
        if pos is None:
            values[t] = 0.0
        elif pos.current_weight is not None:
            values[t] = pos.current_weight
        elif pos.market_value is not None:
            values[t] = pos.market_value
        elif pos.shares:
            values[t] = pos.shares * float(latest_prices.get(t, 0.0))
        else:
            values[t] = 0.0

    total = sum(values.values())
    if total <= 0:
        n = len(kept_tickers)
        return pd.Series({t: 1.0 / n for t in kept_tickers})
    return pd.Series({t: v / total for t, v in values.items()})


def _aggregate_sector_exposure(
    weights: pd.Series, sectors: dict[str, str]
) -> dict[str, float]:
    agg: dict[str, float] = {}
    for t, w in weights.items():
        sector = sectors.get(t, "Unknown")
        agg[sector] = agg.get(sector, 0.0) + float(w)
    return agg


@dataclass
class PortfolioAnalysisResult:
    tickers: list[str]
    dropped_tickers: list[str]
    expected_returns: dict[str, float]
    correlation_matrix: dict[str, dict[str, float]]
    covariance_method: str
    lookback_days: int
    current_weights: dict[str, float]
    optimal_weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    constraints_applied: list[str]
    optimizer_converged: bool
    current_risk_attribution: list[dict]
    optimal_risk_attribution: list[dict]
    current_portfolio_volatility: float
    current_expected_return: float
    optimal_portfolio_volatility: float
    efficient_frontier: list[dict[str, float]]
    var: dict
    sector_exposure: dict[str, float]
    generated_at: str
    warnings: list[str]


class PortfolioAnalysisService:
    """
    Facade that wires together return-series construction, covariance
    estimation, optimization, risk attribution, and VaR into a single
    portfolio-level analysis call.

    ML prediction probabilities are supplied by the caller (typically the
    API route, which already has access to ``PredictionService``) rather
    than fetched internally — this keeps the portfolio math decoupled from
    the ML pipeline and easy to test in isolation.
    """

    def __init__(
        self,
        return_builder: Optional[ReturnSeriesBuilder] = None,
        covariance_estimator: Optional[CovarianceEstimator] = None,
    ) -> None:
        self._return_builder = return_builder or ReturnSeriesBuilder()
        self._cov_estimator = covariance_estimator or CovarianceEstimator()

    def analyze(
        self,
        positions: list[PortfolioPosition],
        prediction_probs: Optional[dict[str, float]] = None,
        portfolio_value: Optional[float] = None,
        max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
        max_sector_weight: float = DEFAULT_MAX_SECTOR_WEIGHT,
        long_only: bool = True,
        turnover_limit: Optional[float] = None,
        var_confidence: float = DEFAULT_VAR_CONFIDENCE,
        var_horizon_days: int = 1,
        return_scale_factor: float = DEFAULT_RETURN_SCALE_FACTOR,
        frontier_points: int = DEFAULT_EFFICIENT_FRONTIER_POINTS,
    ) -> PortfolioAnalysisResult:
        warnings: list[str] = []
        tickers_requested = [p.ticker.upper().strip() for p in positions]

        bundle = self._return_builder.build(tickers_requested)
        if bundle.dropped_tickers:
            warnings.append(
                f"Dropped (no usable market data): {bundle.dropped_tickers}"
            )

        cov_est = self._cov_estimator.estimate(bundle.returns)
        kept_tickers = cov_est.tickers
        n = len(kept_tickers)

        if max_position_weight * n < 1.0 - 1e-6:
            raise PortfolioAnalysisError(
                f"max_position_weight={max_position_weight} is infeasible for "
                f"{n} position(s) — the maximum achievable total weight is "
                f"{max_position_weight * n:.2f}, which is below 1.0.",
                detail="Increase max_position_weight or include more positions.",
            )

        pos_by_ticker = {p.ticker.upper().strip(): p for p in positions}
        current_weights = _compute_current_weights(
            pos_by_ticker, kept_tickers, bundle.latest_prices
        )

        probs = prediction_probs or {}
        missing_probs = [t for t in kept_tickers if t not in probs]
        if missing_probs:
            warnings.append(
                f"No ML prediction supplied for {missing_probs} — treated as "
                "neutral (p_bullish=0.5 -> expected_return proxy=0)."
            )
        expected_returns = pd.Series(
            {
                t: (probs.get(t, 0.5) - 0.5) * 2 * return_scale_factor
                for t in kept_tickers
            }
        )

        sectors = _resolve_sectors(kept_tickers)
        n_sectors = len(set(sectors.values())) if sectors else 0
        if (
            n_sectors > 0
            and max_sector_weight < 1.0
            and (max_sector_weight * n_sectors < 1.0 - 1e-6)
        ):
            raise PortfolioAnalysisError(
                f"max_sector_weight={max_sector_weight} is infeasible — "
                f"these {n} position(s) span only {n_sectors} distinct "
                f"sector(s), and the maximum achievable total weight is "
                f"{max_sector_weight * n_sectors:.2f}, which is below 1.0.",
                detail=(
                    "Increase max_sector_weight, include positions from more "
                    "sectors, or set max_sector_weight=1.0 to disable the "
                    "sector constraint."
                ),
            )

        optimizer = MeanVarianceOptimizer(
            max_position_weight=max_position_weight,
            max_sector_weight=max_sector_weight,
            long_only=long_only,
            turnover_limit=turnover_limit,
        )
        opt_result = optimizer.optimize(
            expected_returns=expected_returns,
            covariance=cov_est.covariance,
            sectors=sectors,
            current_weights=current_weights,
        )
        if not opt_result.converged:
            warnings.append(
                "Optimizer did not converge under the given constraints — "
                "optimal_weights fall back to an equal-weight allocation."
            )

        frontier = compute_efficient_frontier(
            expected_returns=expected_returns,
            covariance=cov_est.covariance,
            max_position_weight=max_position_weight,
            long_only=long_only,
            n_points=frontier_points,
        )

        current_risk = RiskAttributor.attribute(current_weights, cov_est.covariance)
        optimal_risk = RiskAttributor.attribute(opt_result.weights, cov_est.covariance)

        current_vol = VaREstimator.portfolio_volatility(
            current_weights, cov_est.covariance
        )
        optimal_vol = VaREstimator.portfolio_volatility(
            opt_result.weights, cov_est.covariance
        )
        current_expected_return = float(
            np.dot(
                current_weights.reindex(kept_tickers).values, expected_returns.values
            )
        )

        var_result = VaREstimator.estimate(
            weights=current_weights,
            covariance=cov_est.covariance,
            returns=bundle.returns,
            portfolio_value=portfolio_value,
            confidence=var_confidence,
            horizon_days=var_horizon_days,
        )

        sector_exposure = _aggregate_sector_exposure(current_weights, sectors)

        return PortfolioAnalysisResult(
            tickers=kept_tickers,
            dropped_tickers=bundle.dropped_tickers,
            expected_returns={t: round_metric(v) for t, v in expected_returns.items()},
            correlation_matrix=cov_est.correlation.round(4).to_dict(),
            covariance_method=cov_est.method,
            lookback_days=cov_est.lookback_days,
            current_weights={
                t: round_metric(float(w)) for t, w in current_weights.items()
            },
            optimal_weights={
                t: round_metric(float(w)) for t, w in opt_result.weights.items()
            },
            expected_return=opt_result.expected_return,
            expected_volatility=opt_result.expected_volatility,
            sharpe_ratio=opt_result.sharpe_ratio,
            constraints_applied=opt_result.constraints_applied,
            optimizer_converged=opt_result.converged,
            current_risk_attribution=[asdict(r) for r in current_risk],
            optimal_risk_attribution=[asdict(r) for r in optimal_risk],
            current_portfolio_volatility=round_metric(current_vol),
            current_expected_return=round_metric(current_expected_return),
            optimal_portfolio_volatility=round_metric(optimal_vol),
            efficient_frontier=frontier,
            var=asdict(var_result),
            sector_exposure={k: round_metric(v) for k, v in sector_exposure.items()},
            generated_at=utc_now_iso(),
            warnings=warnings,
        )
