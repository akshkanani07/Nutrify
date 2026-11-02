"""Unit and integration tests for MLModelProvider and production inference pipeline."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image

from app.providers.ml_provider import MLModelProvider
from app.exceptions import ModelNotReadyError


def test_ml_provider_initialization(monkeypatch):
    """MLModelProvider should be loaded and ready when ENABLE_ML_INTEGRATION=1."""
    monkeypatch.setenv("ENABLE_ML_INTEGRATION", "1")
    provider = MLModelProvider()
    assert provider.name == "ml_model"
    assert provider.ready is True
    assert provider.model_version is not None


def test_ml_provider_predict_food_json_format():
    """predict_food_json should return exactly the required JSON keys."""
    provider = MLModelProvider()

    # Create dummy food-like image
    img = Image.new("RGB", (224, 224), color=(240, 200, 50))
    result = provider.predict_food_json(img)

    required_keys = {"food_name", "confidence", "calories", "protein", "carbs", "fat", "fiber"}
    assert set(result.keys()) == required_keys

    assert isinstance(result["food_name"], str)
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["calories"], (int, float))
    assert isinstance(result["protein"], (int, float))
    assert isinstance(result["carbs"], (int, float))
    assert isinstance(result["fat"], (int, float))
    assert isinstance(result["fiber"], (int, float))


def test_ml_provider_analyze_schema(monkeypatch):
    """analyze(img) should return dictionary conforming to backend response structure when active."""
    monkeypatch.setenv("ENABLE_ML_INTEGRATION", "1")
    provider = MLModelProvider()

    img = Image.new("RGB", (224, 224), color=(50, 150, 50))
    result = provider.analyze(img)

    assert "status" in result
    assert "items" in result
    assert "totals" in result
    assert "score" in result
    assert "advice" in result

    item = result["items"][0]
    assert "food_key" in item
    assert "name" in item
    assert "confidence" in item
    assert "nutrition" in item


def test_ml_provider_not_ready_fallback():
    """MLModelProvider should raise ModelNotReadyError when checkpoint does not exist."""
    fake_provider = MLModelProvider(checkpoint_path=Path("non_existent_model.pt"))
    assert fake_provider.ready is False

    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    with pytest.raises(ModelNotReadyError):
        fake_provider.predict_food_json(img)

    with pytest.raises(ModelNotReadyError):
        fake_provider.analyze(img)

