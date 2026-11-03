"""Tests for production hardening: security headers, request tracing, feedback, metrics."""

from __future__ import annotations

import io
from app.main import app
from tests.conftest import make_jpeg


def test_security_headers_and_request_id(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert "x-process-time" in resp.headers


def test_metrics_endpoint(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "average_confidence" in data
    assert "latency_ms" in data
    assert data["status"] in ("healthy", "degraded")


def test_feedback_loop_submission(client):
    payload = {
        "request_id": "test-req-123",
        "food_name": "butter chicken",
        "is_correct": False,
        "selected_food": "paneer butter masala",
        "comments": "Vegetarian curry, not chicken.",
    }
    resp = client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "Feedback recorded" in data["message"]


def test_prediction_logging_on_analyze(client):
    data = make_jpeg()
    resp = client.post(
        "/api/v1/analyze-food",
        files={"image": ("meal.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "request_id" in body
    assert body["success"] is True

    # Check metrics recorded prediction
    metrics_resp = client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    m_data = metrics_resp.json()
    assert m_data["total_predictions"] >= 1

