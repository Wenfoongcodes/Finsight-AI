from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging_config import get_logger, setup_logging
from app.ml.data_ingestion import get_data_summary, ingest_multiple_tickers

setup_logging(level="INFO")
logger = get_logger("scripts.ingest")

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


def print_summary(summaries: list[dict]) -> None:
    header = f"{'Ticker':<8} {'Rows':>6} {'Start':<12} {'End':<12} {'Min Close':>10} {'Max Close':>10}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for s in summaries:
        print(
            f"{s['ticker']:<8} {s['rows']:>6} {s['start_date']:<12} {s['end_date']:<12} "
            f"{s['close_min']:>10.2f} {s['close_max']:>10.2f}"
        )
    print("─" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinSight AI — Offline Data Ingestion & Cache Warmup"
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
        "--years", type=int, default=5, help="Years of history (default: 5)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Skip cache, force re-download"
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

    use_cache = not args.no_cache
    logger.info(
        "Ingesting %d ticker(s) | years=%d | cache=%s",
        len(tickers),
        args.years,
        use_cache,
    )

    results = ingest_multiple_tickers(
        tickers, period_years=args.years, use_cache=use_cache
    )

    summaries = [get_data_summary(df, ticker) for ticker, df in results.items()]
    print_summary(summaries)

    failed = set(tickers) - set(results.keys())
    if failed:
        logger.warning("Failed tickers: %s", sorted(failed))
        sys.exit(1)

    logger.info(
        "Ingestion complete. %d/%d tickers succeeded.", len(results), len(tickers)
    )


if __name__ == "__main__":
    main()
