from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from app.core.exceptions import (
    DataIngestionError,
    DataValidationError,
    InsufficientDataError,
)
from app.ml.data_ingestion import (
    get_data_summary,
    ingest_market_data,
    validate_ohlcv,
)


class TestValidateOHLCV:
    def test_passes_valid_dataframe(self, sample_ohlcv):
        validate_ohlcv(sample_ohlcv, "TEST")  # should not raise

    def test_raises_on_empty_df(self):
        with pytest.raises(DataValidationError, match="Empty"):
            validate_ohlcv(pd.DataFrame(), "TEST")

    def test_raises_on_missing_columns(self, sample_ohlcv):
        df = sample_ohlcv.drop(columns=["Volume"])
        with pytest.raises(DataValidationError, match="Missing columns"):
            validate_ohlcv(df, "TEST")

    def test_raises_on_insufficient_rows(self, sample_ohlcv):
        df = sample_ohlcv.iloc[:10]
        with pytest.raises(InsufficientDataError):
            validate_ohlcv(df, "TEST")

    def test_raises_on_nonpositive_close(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.iloc[0, df.columns.get_loc("Close")] = -1.0
        with pytest.raises(DataValidationError, match="Non-positive"):
            validate_ohlcv(df, "TEST")


class TestGetDataSummary:
    def test_returns_expected_keys(self, sample_ohlcv):
        summary = get_data_summary(sample_ohlcv, "AAPL")
        expected = {
            "ticker",
            "start_date",
            "end_date",
            "rows",
            "columns",
            "close_min",
            "close_max",
            "close_mean",
            "null_count",
        }
        assert expected.issubset(summary.keys())

    def test_correct_ticker(self, sample_ohlcv):
        summary = get_data_summary(sample_ohlcv, "MSFT")
        assert summary["ticker"] == "MSFT"

    def test_row_count_matches(self, sample_ohlcv):
        summary = get_data_summary(sample_ohlcv, "X")
        assert summary["rows"] == len(sample_ohlcv)


class TestIngestMarketData:
    def test_ingest_uses_cache(self, sample_ohlcv, tmp_path, monkeypatch):
        """Should return cached parquet on second call without hitting yfinance."""
        import app.ml.data_ingestion as ingestion_mod

        monkeypatch.setattr(ingestion_mod.settings, "RAW_DATA_DIR", tmp_path)

        with patch(
            "app.ml.data_ingestion.fetch_yfinance", return_value=sample_ohlcv
        ) as mock_fetch:
            df1 = ingest_market_data("AAPL", period_years=2, use_cache=True)
            df2 = ingest_market_data("AAPL", period_years=2, use_cache=True)
            assert mock_fetch.call_count == 1
            assert len(df1) == len(df2)

    def test_ingest_raises_on_fetch_failure(self, tmp_path, monkeypatch):
        import app.ml.data_ingestion as ingestion_mod

        monkeypatch.setattr(ingestion_mod.settings, "RAW_DATA_DIR", tmp_path)

        with patch(
            "app.ml.data_ingestion.fetch_yfinance",
            side_effect=DataIngestionError("network error"),
        ):
            with pytest.raises(DataIngestionError):
                ingest_market_data("FAKE", period_years=1, use_cache=False)
