from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Environment Patch — session-scoped, autouse
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _patch_openai_env():
    """
    Ensure OPENAI_API_KEY is set for the entire test session.

    This prevents ``OpenAIClient.__init__`` from raising ``LLMError`` on
    the missing-key guard in environments where the real key is absent
    (e.g. a developer's local machine without a ``.env`` file).

    The original value of the environment variable (if any) is restored
    after the session so this fixture has zero side-effects on the shell
    that launched pytest.

    The ``get_settings`` lru_cache is cleared so the patched value is
    visible to any ``Settings()`` instantiation during the session.
    """
    _DUMMY_KEY = "sk-test-dummy"
    _ENV_KEY = "OPENAI_API_KEY"
    _original = os.environ.get(_ENV_KEY)

    # Only patch when no real key is present so developer environments
    # with a valid key are not affected.
    if not _original:
        os.environ[_ENV_KEY] = _DUMMY_KEY

        # Clear the lru_cache so the patched env var is visible to settings.
        try:
            from configs.settings import get_settings

            get_settings.cache_clear()
        except Exception:
            pass  # Non-fatal — settings may already reflect the env.

    yield

    # Restore original state.
    if not _original:
        os.environ.pop(_ENV_KEY, None)
        try:
            from configs.settings import get_settings

            get_settings.cache_clear()
        except Exception:
            pass


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
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
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
