"""ML model provider — Production Nutrition AI Provider.

Connects the trained EfficientNet-B0 + MLP classification pipeline to FastAPI,
providing:
1. predict_food_json(image_input) -> Exact JSON format:
   {
       "food_name": str,
       "confidence": float,
       "calories": float,
       "protein": float,
       "carbs": float,
       "fat": float,
       "fiber": float
   }
2. analyze(img) -> Full analysis dictionary matching backend AnalysisResponse schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, BinaryIO, Union

from PIL import Image

from app.exceptions import ModelNotReadyError

logger = logging.getLogger("nutrify.provider.ml")

MODEL_CHECKPOINT_PATH = Path("ml/models/production/nutrition_model_best.pt")
CANDIDATES_CSV_PATH = Path("ml/data/raw/candidates.csv")


class MLModelProvider:
    """Production ML model provider backed by EfficientNet-B0 + MLP classifier."""

    name = "ml_model"

    def __init__(
        self,
        checkpoint_path: Path | str = MODEL_CHECKPOINT_PATH,
        candidates_csv: Path | str = CANDIDATES_CSV_PATH,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.candidates_csv = Path(candidates_csv)
        self._model_version = "v1.0-efficientnet-b0-mlp"
        self._service: Any = None
        self._ready = False

        self._initialize_service()

    def _initialize_service(self) -> None:
        """Lazily load the production inference pipeline."""
        if not self.checkpoint_path.exists():
            logger.warning(f"Production weights missing at {self.checkpoint_path}.")
            self._ready = False
            return

        try:
            from ml.src.production_pipeline import ProductionInferenceService

            self._service = ProductionInferenceService(
                checkpoint_path=self.checkpoint_path,
                candidates_csv=self.candidates_csv,
                device="cpu",
                confidence_threshold=0.40,
            )
            # Ready is active when enabled or in production ML mode
            import os
            self._ready = os.getenv("ENABLE_ML_INTEGRATION", "false").lower() in ("true", "1", "yes")
            logger.info("MLModelProvider successfully loaded production inference pipeline.")
        except Exception as exc:
            logger.error(f"Failed to load ML production inference pipeline: {exc}", exc_info=True)
            self._ready = False

    @property
    def ready(self) -> bool:
        """Returns True if the ML model is loaded and ready to serve inference requests."""
        return self._ready and self._service is not None

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def predict_food_json(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    ) -> dict[str, Any]:
        """Standard production backend integration interface.

        Input:
            Image file (file path, raw bytes, file-like stream, or PIL Image)

        Output JSON:
            {
                "food_name": str,
                "confidence": float,
                "calories": float,
                "protein": float,
                "carbs": float,
                "fat": float,
                "fiber": float
            }
        """
        if self._service is None:
            raise ModelNotReadyError("The ML model is not loaded. Weights may be missing.")

        res = self._service.predict(image_input)

        return {
            "food_name": res["food_name"],
            "confidence": res["confidence"],
            "calories": res["calories"],
            "protein": res["protein"],
            "carbs": res["carbs"],
            "fat": res["fat"],
            "fiber": res["fiber"],
        }

    def analyze(self, img: Image.Image) -> dict[str, Any]:
        """Full FastAPI backend endpoint analyzer matching AnalysisResponse schema."""
        if not self.ready:
            raise ModelNotReadyError("The ML model is not loaded. Weights may be missing.")

        pred = self._service.predict(img)

        food_name = pred["food_name"]
        confidence = pred["confidence"]
        status = "ok" if pred["is_confident"] else "unsure"

        # Build candidate items
        candidate_items = [
            {
                "food_key": c["food_name"].replace(" ", "_"),
                "name": c["food_name"],
                "score": c["confidence"],
            }
            for c in pred["candidates"][1:5]
        ]

        nutrition = {
            "calories": pred["calories"],
            "protein_g": pred["protein"],
            "carbs_g": pred["carbs"],
            "fat_g": pred["fat"],
            "fiber_g": pred["fiber"],
            "sugar_g": pred["candidates"][0].get("sugar", 0.0) if pred["candidates"] else 0.0,
            "sodium_mg": pred["candidates"][0].get("sodium", 0.0) if pred["candidates"] else 0.0,
        }

        food_item = {
            "food_key": food_name.replace(" ", "_"),
            "name": food_name,
            "confidence": confidence,
            "grams": pred.get("serving_weight_g", 100.0),
            "estimate": True,
            "match": "curated",
            "nutrition": nutrition,
            "candidates": candidate_items,
        }

        # Calculate a nutritional meal score (0-100)
        calories = pred["calories"]
        protein = pred["protein"]
        fiber = pred["fiber"]
        fat = pred["fat"]

        protein_pts = min(30.0, (protein / 20.0) * 30.0)
        fiber_pts = min(30.0, (fiber / 8.0) * 30.0)
        cal_pts = max(0.0, 40.0 - abs(calories - 300.0) * 0.08)
        score_val = int(round(protein_pts + fiber_pts + cal_pts))
        score_val = max(10, min(95, score_val))

        band = "excellent" if score_val >= 80 else ("good" if score_val >= 65 else ("fair" if score_val >= 45 else "poor"))

        advice = []
        if fiber < 3.0:
            advice.append("Consider pairing with a high-fiber vegetable or whole grain side.")
        if protein > 15.0:
            advice.append("Great source of dietary protein.")
        if calories > 400.0:
            advice.append("Hearty energy-dense portion.")

        return {
            "status": status,
            "items": [food_item],
            "totals": nutrition,
            "score": {
                "value": score_val,
                "band": band,
                "breakdown": [
                    {"label": "Protein Quality", "points": round(protein_pts, 1), "max": 30.0},
                    {"label": "Fiber Density", "points": round(fiber_pts, 1), "max": 30.0},
                    {"label": "Caloric Balance", "points": round(cal_pts, 1), "max": 40.0},
                ],
            },
            "advice": advice,
        }
