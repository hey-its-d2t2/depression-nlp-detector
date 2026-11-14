"""Integration tests for the FastAPI prediction endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from depression_detector.api.main import create_app
from depression_detector.api.routes import predict as predict_module


@pytest.fixture(scope="module")
def client(fitted_model):
    """Create a test client with a pre-loaded model."""
    predict_module.set_model(fitted_model)
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_info_returns_version(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "name" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "I feel so hopeless and empty every single day."},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "I feel great today. Everything is wonderful!"},
        )
        data = resp.json()
        assert "label" in data
        assert "label_text" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["label"] in [0, 1]

    def test_predict_depressed_text(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "I feel worthless and hopeless. Nothing matters anymore."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["probabilities"]["depressed"] > 0.0

    def test_predict_rejects_empty_text(self, client):
        resp = client.post("/api/v1/predict", json={"text": "   "})
        assert resp.status_code == 422

    def test_predict_rejects_too_short_text(self, client):
        resp = client.post("/api/v1/predict", json={"text": "Hi"})
        assert resp.status_code == 422

    def test_predict_with_custom_threshold(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "I am not feeling well lately.", "threshold": 0.3},
        )
        assert resp.status_code == 200
        assert resp.json()["threshold_used"] == 0.3


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        resp = client.post(
            "/api/v1/predict/batch",
            json={
                "texts": [
                    "I feel hopeless.",
                    "Had a great day today!",
                    "I can't stop crying.",
                ]
            },
        )
        assert resp.status_code == 200

    def test_batch_response_count_matches_input(self, client):
        texts = ["Text one.", "Text two.", "Text three."]
        resp = client.post("/api/v1/predict/batch", json={"texts": texts})
        assert resp.json()["total"] == len(texts)

    def test_batch_rejects_empty_list(self, client):
        resp = client.post("/api/v1/predict/batch", json={"texts": []})
        assert resp.status_code == 422
