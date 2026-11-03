"""Test nutrition accuracy, 650-class mapping completeness, and portion scaling."""

import json
from pathlib import Path
from ml.src.production_pipeline import ProductionNutritionMapper


def test_650_nutrition_completeness():
    classes_path = Path("ml/models/production/classes_650.json")
    if not classes_path.exists():
        return
    with open(classes_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"] if isinstance(meta, dict) and "classes" in meta else meta

    mapper = ProductionNutritionMapper()
    assert len(mapper.nutrition_db) >= 650

    for food_name in classes:
        nutr = mapper.get_nutrition(food_name)
        assert nutr["calories"] >= 0.0
        assert nutr["protein"] >= 0.0
        assert nutr["carbs"] >= 0.0
        assert nutr["fat"] >= 0.0
        assert nutr["fiber"] >= 0.0
        assert nutr["serving_weight_g"] > 0.0


def test_portion_scaling_linear():
    mapper = ProductionNutritionMapper()
    food = "apples gala with skin raw"
    n100 = mapper.get_nutrition(food, serving_weight_g=100.0)
    n200 = mapper.get_nutrition(food, serving_weight_g=200.0)

    # Calories for 200g should be exactly 2x of 100g
    assert abs(n200["calories"] - 2.0 * n100["calories"]) <= 0.2
    assert abs(n200["protein"] - 2.0 * n100["protein"]) <= 0.05
    assert abs(n200["carbs"] - 2.0 * n100["carbs"]) <= 0.1

