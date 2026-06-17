"""
scripts/warm_options_cache.py
================================
Daily options-snapshot cache warmup — operational companion to
``app/ml/options_features.py``.

Why this script exists
-----------------------
yfinance only exposes a *live* options-chain snapshot (current bids/asks,
implied volatility, volume, open interest) — there is no historical-IV
endpoint. To build the time series needed for IV rank, IV change, and the
IV/realized-vol spread, a snapshot must be captured once per trading day and
persisted. This script is that daily job: it fetches a fresh
``OptionsSnapshot`` for each ticker and appends it to
``{DATA_DIR}/options_cache/{TICKER}.parquet`` via ``OptionsHistoryStore``.

Run it once near market close (e.g. 15 minutes before the close, when the
day's volume/open-interest figures are most representative) via cron,
Windows Task Scheduler, or a HuggingFace Spaces scheduled job.

Usage
-----
    # Default watchlist
    python scripts/warm_options_cache.py --defaults

    # Single ticker
    python scripts/warm_options_cache.py --ticker AAPL

    # Multiple tickers, slower politeness delay between requests
    python scripts/warm_options_cache.py --tickers AAPL MSFT NVDA --sleep-s 2.0

Cron example (weekdays at 15:45 ET)
------------------------------------
    45 15 * * 1-5  cd /path/to/finsight-ai && \
        .venv/bin/python scripts/warm_options_cache.py --defaults >> logs/options_warm.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging_config import get_logger, setup_logging
from app.ml.options_features import OptionsHistoryStore

setup_logging(level="INFO")
logger = get_logger("scripts.warm_options")

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "JPM",
    "GS",
    "SPY",
]


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'Ticker':<8} {'Optionable':<11} {'ATM IV (30d)':>13} "
        f"{'P/C Ratio':>10} {'Contracts':>10}"
    )
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in rows:
        iv_str = f"{r['atm_iv_cm30'] * 100:.2f}%" if r["atm_iv_cm30"] == r["atm_iv_cm30"] else "n/a"
        pc_str = f"{r['put_call_volume_ratio']:.2f}" if r["put_call_volume_ratio"] == r["put_call_volume_ratio"] else "n/a"
        print(
            f"{r['ticker']:<8} {str(r['is_optionable']):<11} {iv_str:>13} "
            f"{pc_str:>10} {r['n_contracts_used']:>10}"
        )
    print("─" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinSight AI — Daily Options Snapshot Cache Warmup"
    )
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker", type=str, help="Single ticker")
    ticker_group.add_argument("--tickers", nargs="+", help="Multiple tickers")
    ticker_group.add_argument(
        "--defaults",
        action="store_true",
        help=f"Use default watchlist: {DEFAULT_TICKERS}",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=1.0,
        help="Seconds to sleep between tickers — politeness delay (default: 1.0)",
    )

    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.defaults:
        tickers = DEFAULT_TICKERS
    else:
        parser.print_help()
        sys.exit(1)

    logger.info("Warming options cache for %d ticker(s): %s", len(tickers), tickers)

    store = OptionsHistoryStore()
    rows: list[dict] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] Snapshotting %s…", i + 1, len(tickers), ticker)
        try:
            snap = store.update(ticker)
            if snap is None:
                failed.append(ticker)
                continue
            rows.append(snap.to_dict())
            if not snap.is_optionable:
                logger.warning("  %s: not optionable (no liquid contracts).", ticker)
        except Exception as exc:
            logger.error("  %s: snapshot failed (%s)", ticker, exc)
            failed.append(ticker)

        if i < len(tickers) - 1:
            time.sleep(args.sleep_s)

    print_summary(rows)

    succeeded = [r["ticker"] for r in rows if r["is_optionable"]]
    logger.info(
        "Warmup complete. %d/%d tickers optionable, %d failed outright.",
        len(succeeded),
        len(tickers),
        len(failed),
    )

    if failed:
        print(f"Hard failures: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
