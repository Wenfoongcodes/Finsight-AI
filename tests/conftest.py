"""
FinSight AI — Pytest Configuration & Shared Fixtures
Provides reusable fixtures for unit and integration tests.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with 500 rows.
    Simulates a realistic price series using geometric Brownian motion.
    """
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range(end=date.today(), periods=n)

    price = 150.0
    prices = [price]
    for _ in range(n - 1):
        price *= np.exp(np.random.normal(0.0002, 0.015))
        prices.append(price)

    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens = closes * (1 + np.random.normal(0, 0.003, n))
    volumes = np.random.randint(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df.sort_index()


@pytest.fixture
def small_ohlcv(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Subset of 300 rows for faster tests."""
    return sample_ohlcv.iloc[-300:].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def feature_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix built from synthetic OHLCV data."""
    from app.ml.feature_engineering import FeatureEngineer
    return FeatureEngineer().build_features(sample_ohlcv)


@pytest.fixture
def X_y(feature_df: pd.DataFrame):
    """Split feature matrix into X and y."""
    from app.ml.feature_engineering import FeatureEngineer
    return FeatureEngineer().split_X_y(feature_df)


# ─────────────────────────────────────────────────────────────────────────────
# Trained Model Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_rf(X_y):
    """A fitted RandomForest model for use in explainability / eval tests."""
    from app.ml.models.model_factory import get_model
    X, y = X_y
    model = get_model("random_forest", n_estimators=20)
    model.fit(X, y)
    return model, list(X.columns)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    """FastAPI test client with mocked service layer."""
    from main import app
    with TestClient(app) as client:
        yield client
