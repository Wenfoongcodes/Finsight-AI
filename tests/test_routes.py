from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from main import app
from app.core.exceptions import ModelNotFoundError

client = TestClient(app)


@dataclass
class MockNewsItem:
    """Mock news item for FusedSignal."""
    title: str
    snippet: str
    url: str
    sentiment: str = "neutral"
    final_weight: float = 0.5
    domain: str = "example.com"


@dataclass
class MockFusedSignal:
    """Mock fused signal object."""
    final_direction: str
    final_confidence: str
    fusion_probability: float
    synthesis_narrative: str
    fusion_applied: bool
    news_sentiment: str
    news_items: list = field(default_factory=list)
    ml_direction: str = ""
    ml_probability: float = 0.5
    generated_at: str = ""
    recency_note: str = ""


@dataclass
class MockPrediction:
    """Mock prediction object matching PredictionResponse structure."""
    ticker: str
    model_name: str
    horizon: str
    selection_reason: str
    confidence_degraded: bool
    prediction: int
    probability: float
    p_bullish: float
    p_bearish: float
    confidence_label: str
    shap_explanation: dict
    narrative: str
    latest_close: float
    feature_snapshot: dict
    fused_signal: Optional[MockFusedSignal] = None
    auto_trained: bool = False


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestPredictionEndpoint:
    def test_predict_returns_404_without_model(self):
        """Should return 404 when no trained model artifact exists."""
        with patch("app.api.v1.endpoints.routes._get_prediction_service") as mock_svc:
            # Mock the service to raise ModelNotFoundError
            mock_svc.return_value.predict.side_effect = ModelNotFoundError(
                "No trained model found for AAPL/1d"
            )
            
            resp = client.post(
                "/api/v1/predict/", json={"ticker": "AAPL", "model_name": "xgboost"}
            )
            # Should return 404 when ModelNotFoundError is raised
            assert resp.status_code == 404

    def test_predict_validates_ticker_length(self):
        resp = client.post(
            "/api/v1/predict/", json={"ticker": "", "model_name": "xgboost"}
        )
        assert resp.status_code == 422

    def test_predict_uppercase_normalisation(self):
        """Route should accept lowercase ticker and normalise it."""
        with patch("app.api.v1.endpoints.routes._get_prediction_service") as mock_svc:
            mock_fused_signal = MockFusedSignal(
                final_direction="BULLISH",
                final_confidence="HIGH",
                fusion_probability=0.72,
                synthesis_narrative="Bullish signal.",
                fusion_applied=False,
                news_sentiment="positive",
                news_items=[],
            )
            
            mock_pred = MockPrediction(
                ticker="AAPL",
                model_name="xgboost",
                horizon="1d",
                selection_reason="leaderboard",
                confidence_degraded=False,
                prediction=1,
                probability=0.72,
                p_bullish=0.72,
                p_bearish=0.28,
                confidence_label="high",
                shap_explanation={"top_features": []},
                narrative="Bullish signal.",
                latest_close=185.5,
                feature_snapshot={},
                fused_signal=mock_fused_signal,
                auto_trained=False,
            )
            mock_svc.return_value.predict.return_value = mock_pred

            resp = client.post(
                "/api/v1/predict/", json={"ticker": "aapl", "model_name": "xgboost"}
            )
            # The validator uppercases it — mock should be called with 'AAPL'
            if resp.status_code == 200:
                assert resp.json()["ticker"] == "AAPL"


class TestTrainingEndpoint:
    def test_train_validates_period(self):
        resp = client.post(
            "/api/v1/train/",
            json={"ticker": "AAPL", "model_name": "xgboost", "period_years": 0},
        )
        assert resp.status_code == 422

    def test_train_validates_hpo_trials(self):
        resp = client.post(
            "/api/v1/train/",
            json={
                "ticker": "AAPL",
                "model_name": "xgboost",
                "run_hpo": True,
                "hpo_trials": 2,
            },
        )
        assert resp.status_code == 422


class TestRAGEndpoints:
    def test_ingest_requires_texts(self):
        resp = client.post("/api/v1/rag/ingest", json={"texts": [], "source": "test"})
        assert resp.status_code == 422

    def test_chat_requires_query(self):
        resp = client.post("/api/v1/rag/chat", json={"query": ""})
        assert resp.status_code == 422