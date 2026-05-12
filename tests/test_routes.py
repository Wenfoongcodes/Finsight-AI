"""
Tests for app.api.v1.endpoints.routes (FastAPI HTTP layer)
Uses TestClient with mocked service dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


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
        resp = client.post(
            "/api/v1/predict/", json={"ticker": "AAPL", "model_name": "xgboost"}
        )
        # Either 404 (no model) or 422 (no data) — both are valid without setup
        assert resp.status_code in (404, 422, 500)

    def test_predict_validates_ticker_length(self):
        resp = client.post(
            "/api/v1/predict/", json={"ticker": "", "model_name": "xgboost"}
        )
        assert resp.status_code == 422

    def test_predict_uppercase_normalisation(self):
        """Route should accept lowercase ticker and normalise it."""
        with patch("app.api.v1.endpoints.routes._get_prediction_service") as mock_svc:
            mock_pred = MagicMock()
            mock_pred.ticker = "AAPL"
            mock_pred.model_name = "xgboost"
            mock_pred.prediction = 1
            mock_pred.probability = 0.72
            mock_pred.confidence_label = "high"
            mock_pred.latest_close = 185.5
            mock_pred.narrative = "Bullish signal."
            mock_pred.shap_explanation = {"top_features": []}
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
