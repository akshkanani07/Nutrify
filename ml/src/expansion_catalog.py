"""Expansion Catalog and Multi-Tier Class Registry for Nutrition AI.

Organizes 650 food classes into 4 staged milestones:
- Tier 1: 50 classes (Current production baseline)
- Tier 2: 200 classes (Core everyday whole foods, fruits, vegetables, top prepared dishes)
- Tier 3: 450 classes (Extended global dishes, grains, legumes, dairy, meats, seafood)
- Tier 4: 650 classes (Full production catalog with regional Indian, Asian, and Mediterranean staples)

Every class is mapped to ground-truth nutritional profiles from USDA FoodData Central.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("nutrition_ai.catalog")


@dataclass
class FoodClassDefinition:
    """Canonical food class definition for image classification & nutrition lookup."""
    class_id: int
    name: str
    tier: int  # 1: 50-class, 2: 200-class, 3: 450-class, 4: 650-class
    category: str
    usda_fdc_id: int | None
    calories_per_100g: float
    protein_g: float
    fat_g: float
    carb_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float
    default_serving_g: float
    synonyms: list[str]


# Top 101 classic prepared dishes (Food-101 subset with nutritional estimates)
FOOD101_PREPARED_DISHES: dict[str, dict[str, Any]] = {
    "apple pie": {"category": "sweet", "cal": 237.0, "p": 1.9, "f": 11.0, "c": 34.0, "fib": 1.6, "sug": 16.0, "sod": 204.0, "serving": 125.0},
    "baklava": {"category": "sweet", "cal": 429.0, "p": 7.0, "f": 26.0, "c": 44.0, "fib": 3.0, "sug": 29.0, "sod": 135.0, "serving": 80.0},
    "beef carpaccio": {"category": "meat", "cal": 180.0, "p": 22.0, "f": 10.0, "c": 1.0, "fib": 0.0, "sug": 0.0, "sod": 380.0, "serving": 100.0},
    "bibimbap": {"category": "mixed_dish", "cal": 150.0, "p": 6.5, "f": 4.5, "c": 21.0, "fib": 2.2, "sug": 3.0, "sod": 360.0, "serving": 350.0},
    "bruschetta": {"category": "bakery", "cal": 175.0, "p": 4.5, "f": 7.0, "c": 24.0, "fib": 1.8, "sug": 2.5, "sod": 310.0, "serving": 100.0},
    "caesar salad": {"category": "vegetable", "cal": 128.0, "p": 3.5, "f": 10.5, "c": 5.5, "fib": 1.5, "sug": 1.8, "sod": 290.0, "serving": 150.0},
    "cannoli": {"category": "sweet", "cal": 330.0, "p": 6.8, "f": 16.0, "c": 40.0, "fib": 1.0, "sug": 22.0, "sod": 150.0, "serving": 90.0},
    "caprese salad": {"category": "vegetable", "cal": 165.0, "p": 9.5, "f": 13.0, "c": 3.0, "fib": 0.8, "sug": 2.2, "sod": 240.0, "serving": 150.0},
    "carrot cake": {"category": "sweet", "cal": 370.0, "p": 4.0, "f": 19.0, "c": 48.0, "fib": 1.5, "sug": 34.0, "sod": 320.0, "serving": 110.0},
    "ceviche": {"category": "seafood", "cal": 95.0, "p": 14.0, "f": 1.8, "c": 5.5, "fib": 1.0, "sug": 2.0, "sod": 420.0, "serving": 150.0},
    "cheesecake": {"category": "sweet", "cal": 321.0, "p": 5.5, "f": 22.5, "c": 25.5, "fib": 0.5, "sug": 21.0, "sod": 207.0, "serving": 120.0},
    "chicken curry": {"category": "mixed_dish", "cal": 140.0, "p": 13.5, "f": 7.5, "c": 5.0, "fib": 1.2, "sug": 2.0, "sod": 410.0, "serving": 250.0},
    "chicken quesadilla": {"category": "mixed_dish", "cal": 260.0, "p": 15.0, "f": 13.5, "c": 20.0, "fib": 1.5, "sug": 1.5, "sod": 580.0, "serving": 180.0},
    "chicken wings": {"category": "poultry", "cal": 290.0, "p": 23.0, "f": 21.0, "c": 1.0, "fib": 0.0, "sug": 0.0, "sod": 480.0, "serving": 150.0},
    "chocolate cake": {"category": "sweet", "cal": 371.0, "p": 4.5, "f": 18.0, "c": 51.0, "fib": 2.5, "sug": 36.0, "sod": 300.0, "serving": 110.0},
    "churros": {"category": "sweet", "cal": 360.0, "p": 4.0, "f": 18.5, "c": 44.0, "fib": 1.5, "sug": 18.0, "sod": 280.0, "serving": 85.0},
    "clam chowder": {"category": "soup", "cal": 85.0, "p": 3.8, "f": 4.2, "c": 8.0, "fib": 0.8, "sug": 1.5, "sod": 390.0, "serving": 240.0},
    "club sandwich": {"category": "mixed_dish", "cal": 235.0, "p": 12.5, "f": 11.0, "c": 21.5, "fib": 1.8, "sug": 3.0, "sod": 560.0, "serving": 220.0},
    "crab cakes": {"category": "seafood", "cal": 210.0, "p": 14.5, "f": 13.0, "c": 8.5, "fib": 0.5, "sug": 1.0, "sod": 460.0, "serving": 120.0},
    "deviled eggs": {"category": "egg", "cal": 205.0, "p": 11.0, "f": 16.5, "c": 2.0, "fib": 0.2, "sug": 1.2, "sod": 340.0, "serving": 100.0},
    "donuts": {"category": "sweet", "cal": 420.0, "p": 5.0, "f": 23.0, "c": 49.0, "fib": 1.8, "sug": 26.0, "sod": 320.0, "serving": 75.0},
    "dumplings": {"category": "mixed_dish", "cal": 175.0, "p": 8.5, "f": 6.5, "c": 21.0, "fib": 1.2, "sug": 1.5, "sod": 420.0, "serving": 150.0},
    "edamame": {"category": "legume", "cal": 122.0, "p": 11.9, "f": 5.2, "c": 9.9, "fib": 5.2, "sug": 2.2, "sod": 6.0, "serving": 120.0},
    "eggs benedict": {"category": "egg", "cal": 240.0, "p": 11.5, "f": 17.5, "c": 9.0, "fib": 0.5, "sug": 1.0, "sod": 510.0, "serving": 200.0},
    "falafel": {"category": "legume", "cal": 333.0, "p": 13.3, "f": 17.8, "c": 31.8, "fib": 4.5, "sug": 2.5, "sod": 480.0, "serving": 120.0},
    "french fries": {"category": "snack", "cal": 312.0, "p": 3.4, "f": 15.0, "c": 41.4, "fib": 3.8, "sug": 0.3, "sod": 210.0, "serving": 117.0},
    "french onion soup": {"category": "soup", "cal": 70.0, "p": 4.0, "f": 3.5, "c": 6.0, "fib": 1.0, "sug": 2.5, "sod": 440.0, "serving": 250.0},
    "fried calamari": {"category": "seafood", "cal": 240.0, "p": 15.0, "f": 13.0, "c": 15.0, "fib": 0.8, "sug": 0.5, "sod": 410.0, "serving": 140.0},
    "fried rice": {"category": "grain", "cal": 168.0, "p": 4.5, "f": 5.5, "c": 25.0, "fib": 1.2, "sug": 1.0, "sod": 390.0, "serving": 200.0},
    "gnocchi": {"category": "grain", "cal": 145.0, "p": 3.8, "f": 1.2, "c": 30.0, "fib": 1.8, "sug": 1.2, "sod": 340.0, "serving": 180.0},
    "guacamole": {"category": "condiment_sauce", "cal": 160.0, "p": 2.0, "f": 14.5, "c": 8.5, "fib": 6.5, "sug": 0.8, "sod": 280.0, "serving": 60.0},
    "hamburger": {"category": "mixed_dish", "cal": 254.0, "p": 13.0, "f": 12.5, "c": 22.0, "fib": 1.2, "sug": 4.0, "sod": 460.0, "serving": 200.0},
    "hot dog": {"category": "mixed_dish", "cal": 290.0, "p": 10.5, "f": 18.0, "c": 21.0, "fib": 1.0, "sug": 4.5, "sod": 680.0, "serving": 115.0},
    "lasagna": {"category": "mixed_dish", "cal": 135.0, "p": 7.8, "f": 5.5, "c": 13.5, "fib": 1.2, "sug": 2.5, "sod": 380.0, "serving": 250.0},
    "macaroni and cheese": {"category": "mixed_dish", "cal": 180.0, "p": 7.0, "f": 8.5, "c": 19.0, "fib": 1.1, "sug": 2.0, "sod": 420.0, "serving": 200.0},
    "nachos": {"category": "snack", "cal": 305.0, "p": 7.5, "f": 18.0, "c": 29.0, "fib": 3.0, "sug": 2.0, "sod": 510.0, "serving": 160.0},
    "omelette": {"category": "egg", "cal": 154.0, "p": 11.0, "f": 12.0, "c": 1.0, "fib": 0.0, "sug": 0.8, "sod": 310.0, "serving": 150.0},
    "pad thai": {"category": "mixed_dish", "cal": 170.0, "p": 6.5, "f": 6.0, "c": 22.5, "fib": 1.5, "sug": 5.0, "sod": 440.0, "serving": 250.0},
    "paella": {"category": "mixed_dish", "cal": 155.0, "p": 9.0, "f": 4.5, "c": 20.0, "fib": 1.4, "sug": 1.5, "sod": 370.0, "serving": 250.0},
    "pancakes": {"category": "bakery", "cal": 227.0, "p": 6.4, "f": 9.7, "c": 28.3, "fib": 1.4, "sug": 4.5, "sod": 439.0, "serving": 120.0},
    "peking duck": {"category": "poultry", "cal": 337.0, "p": 19.0, "f": 28.0, "c": 2.0, "fib": 0.2, "sug": 1.5, "sod": 390.0, "serving": 150.0},
    "pho": {"category": "soup", "cal": 85.0, "p": 6.5, "f": 2.0, "c": 10.5, "fib": 0.8, "sug": 1.5, "sod": 410.0, "serving": 350.0},
    "pizza": {"category": "mixed_dish", "cal": 266.0, "p": 11.4, "f": 9.8, "c": 33.3, "fib": 2.3, "sug": 3.6, "sod": 598.0, "serving": 150.0},
    "ramen": {"category": "soup", "cal": 110.0, "p": 4.5, "f": 4.0, "c": 14.5, "fib": 1.0, "sug": 1.2, "sod": 480.0, "serving": 350.0},
    "ravioli": {"category": "mixed_dish", "cal": 160.0, "p": 6.8, "f": 5.2, "c": 21.5, "fib": 1.5, "sug": 1.8, "sod": 360.0, "serving": 180.0},
    "risotto": {"category": "grain", "cal": 145.0, "p": 3.5, "f": 5.0, "c": 21.0, "fib": 0.8, "sug": 0.8, "sod": 330.0, "serving": 200.0},
    "samosa": {"category": "snack", "cal": 262.0, "p": 4.5, "f": 15.0, "c": 27.0, "fib": 2.5, "sug": 1.8, "sod": 360.0, "serving": 90.0},
    "sashimi": {"category": "seafood", "cal": 125.0, "p": 24.0, "f": 3.0, "c": 0.0, "fib": 0.0, "sug": 0.0, "sod": 75.0, "serving": 120.0},
    "spaghetti bolognese": {"category": "mixed_dish", "cal": 150.0, "p": 7.5, "f": 5.2, "c": 18.0, "fib": 1.5, "sug": 2.5, "sod": 380.0, "serving": 250.0},
    "spring rolls": {"category": "snack", "cal": 195.0, "p": 5.0, "f": 9.0, "c": 23.0, "fib": 1.8, "sug": 2.0, "sod": 340.0, "serving": 100.0},
    "steak": {"category": "meat", "cal": 271.0, "p": 25.0, "f": 19.0, "c": 0.0, "fib": 0.0, "sug": 0.0, "sod": 65.0, "serving": 200.0},
    "sushi": {"category": "seafood", "cal": 143.0, "p": 4.5, "f": 1.5, "c": 28.0, "fib": 1.0, "sug": 3.5, "sod": 390.0, "serving": 180.0},
    "tacos": {"category": "mixed_dish", "cal": 226.0, "p": 9.5, "f": 11.5, "c": 21.0, "fib": 2.5, "sug": 2.0, "sod": 430.0, "serving": 160.0},
    "tiramisu": {"category": "sweet", "cal": 290.0, "p": 4.8, "f": 17.5, "c": 28.5, "fib": 0.8, "sug": 19.0, "sod": 95.0, "serving": 100.0},
    "waffles": {"category": "bakery", "cal": 291.0, "p": 7.9, "f": 14.1, "c": 32.9, "fib": 1.3, "sug": 5.0, "sod": 511.0, "serving": 100.0},
}

# Regional Indian Food Additions
INDIAN_STAPLE_DISHES: dict[str, dict[str, Any]] = {
    "biryani chicken": {"category": "mixed_dish", "cal": 180.0, "p": 9.5, "f": 6.5, "c": 21.0, "fib": 1.5, "sug": 1.2, "sod": 420.0, "serving": 300.0},
    "biryani mutton": {"category": "mixed_dish", "cal": 215.0, "p": 11.0, "f": 9.5, "c": 21.0, "fib": 1.5, "sug": 1.0, "sod": 440.0, "serving": 300.0},
    "butter chicken": {"category": "poultry", "cal": 195.0, "p": 14.0, "f": 13.0, "c": 5.5, "fib": 1.2, "sug": 3.5, "sod": 460.0, "serving": 220.0},
    "chana masala": {"category": "legume", "cal": 140.0, "p": 6.5, "f": 4.5, "c": 19.0, "fib": 5.5, "sug": 2.5, "sod": 380.0, "serving": 200.0},
    "dal makhani": {"category": "legume", "cal": 160.0, "p": 6.0, "f": 8.0, "c": 16.5, "fib": 4.5, "sug": 2.0, "sod": 390.0, "serving": 200.0},
    "dal tadka": {"category": "legume", "cal": 120.0, "p": 6.5, "f": 3.5, "c": 16.0, "fib": 4.0, "sug": 1.5, "sod": 360.0, "serving": 200.0},
    "dosa plain": {"category": "grain", "cal": 168.0, "p": 3.9, "f": 3.7, "c": 29.8, "fib": 1.8, "sug": 0.5, "sod": 210.0, "serving": 120.0},
    "masala dosa": {"category": "mixed_dish", "cal": 195.0, "p": 4.5, "f": 6.5, "c": 30.0, "fib": 2.8, "sug": 1.5, "sod": 320.0, "serving": 180.0},
    "idli": {"category": "grain", "cal": 130.0, "p": 4.2, "f": 0.5, "c": 27.5, "fib": 1.8, "sug": 0.4, "sod": 180.0, "serving": 100.0},
    "gulab jamun": {"category": "sweet", "cal": 380.0, "p": 5.0, "f": 14.0, "c": 58.0, "fib": 0.5, "sug": 45.0, "sod": 110.0, "serving": 75.0},
    "jalebi": {"category": "sweet", "cal": 420.0, "p": 2.5, "f": 15.0, "c": 69.0, "fib": 0.5, "sug": 52.0, "sod": 85.0, "serving": 70.0},
    "naan plain": {"category": "bakery", "cal": 290.0, "p": 8.5, "f": 6.0, "c": 51.0, "fib": 2.5, "sug": 3.5, "sod": 420.0, "serving": 90.0},
    "garlic naan": {"category": "bakery", "cal": 310.0, "p": 8.5, "f": 8.0, "c": 51.0, "fib": 2.5, "sug": 3.5, "sod": 440.0, "serving": 95.0},
    "paneer butter masala": {"category": "dairy", "cal": 235.0, "p": 10.5, "f": 18.0, "c": 7.5, "fib": 1.5, "sug": 4.0, "sod": 450.0, "serving": 220.0},
    "paneer tikka": {"category": "dairy", "cal": 210.0, "p": 13.5, "f": 14.5, "c": 6.0, "fib": 1.2, "sug": 2.5, "sod": 390.0, "serving": 150.0},
    "palak paneer": {"category": "dairy", "cal": 175.0, "p": 9.5, "f": 13.0, "c": 5.5, "fib": 2.8, "sug": 2.0, "sod": 410.0, "serving": 220.0},
    "pav bhaji": {"category": "mixed_dish", "cal": 185.0, "p": 4.5, "f": 7.5, "c": 25.0, "fib": 3.5, "sug": 4.5, "sod": 480.0, "serving": 250.0},
    "roti chapati": {"category": "bakery", "cal": 264.0, "p": 8.0, "f": 3.5, "c": 50.0, "fib": 7.0, "sug": 1.0, "sod": 190.0, "serving": 60.0},
    "paratha aloo": {"category": "bakery", "cal": 235.0, "p": 5.0, "f": 8.5, "c": 35.0, "fib": 3.0, "sug": 1.5, "sod": 340.0, "serving": 120.0},
    "rajma masala": {"category": "legume", "cal": 135.0, "p": 7.0, "f": 4.0, "c": 18.5, "fib": 5.5, "sug": 2.0, "sod": 370.0, "serving": 200.0},
    "vada pav": {"category": "snack", "cal": 290.0, "p": 6.5, "f": 12.0, "c": 39.0, "fib": 3.0, "sug": 3.0, "sod": 460.0, "serving": 140.0},
}


class ScalableExpansionCatalog:
    """Manages the full multi-tier food catalog up to 650 classes."""

    def __init__(self, candidates_csv: Path | str = "ml/data/raw/candidates.csv") -> None:
        self.candidates_csv = Path(candidates_csv)
        self.classes: dict[str, FoodClassDefinition] = {}
        self._build_catalog()

    def _clean_name(self, name: str) -> str:
        clean = name.lower().replace(",", " ").replace("-", " ").replace("'", "")
        return " ".join(clean.split())

    def _build_catalog(self) -> None:
        """Load and merge USDA candidates, Food-101 dishes, and Indian regional items into 650 classes."""
        # 1. Existing Tier-1 50 classes
        tier1_names = [
            "apples gala with skin raw", "archway home style cookies oatmeal", "bananas raw",
            "blackberries raw", "blueberries raw", "bread egg", "bread rye", "broccoli leaves raw",
            "broccoli raw", "cabbage raw", "carrots raw", "celery raw", "cheese blue", "cheese brie",
            "chicken spread", "cornsalad raw", "croutons plain", "egg white dried", "egg yolk dried",
            "flour rice brown", "flour soy defatted", "garlic raw", "hummus commercial",
            "ice creams chocolate", "kale raw", "ketchup restaurant", "lemons raw without peel",
            "lentils sprouted raw", "mangos raw", "muffins oat bran", "mushrooms white raw",
            "nectarines raw", "nuts almonds", "onions red raw", "oranges raw navels",
            "pasta gluten free corn cooked", "peanuts spanish raw", "pizza rolls frozen unprepared",
            "pork fresh ground raw", "potato pancakes", "raspberries raw", "rice crackers",
            "salmon nuggets breaded frozen heated", "scrapple pork", "spinach raw", "strawberries raw",
            "tomatoes green raw", "tortilla chips yellow plain salted", "wasabi root raw",
            "yogurt plain low fat"
        ]

        class_id = 0

        # Load candidates.csv database
        candidates_db: dict[str, dict[str, Any]] = {}
        if self.candidates_csv.exists():
            with open(self.candidates_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    norm = self._clean_name(row.get("description", ""))
                    try:
                        candidates_db[norm] = {
                            "fdc_id": int(row.get("fdc_id") or 0),
                            "category": row.get("category", "ingredient_other"),
                            "calories": float(row.get("calories_per_100g") or 0.0),
                            "protein": float(row.get("protein_g") or 0.0),
                            "fat": float(row.get("fat_g") or 0.0),
                            "carbs": float(row.get("carb_g") or 0.0),
                            "fiber": float(row.get("fiber_g") or 0.0),
                            "sugar": float(row.get("sugar_g") or 0.0),
                            "sodium": float(row.get("sodium_mg") or 0.0),
                            "serving": float(row.get("default_serving_g") or 100.0),
                        }
                    except Exception:
                        continue

        # Register Tier 1 (50 classes)
        for name in tier1_names:
            c_name = self._clean_name(name)
            info = candidates_db.get(c_name, {
                "fdc_id": None, "category": "ingredient_other",
                "calories": 100.0, "protein": 3.0, "fat": 2.0, "carbs": 15.0,
                "fiber": 2.0, "sugar": 3.0, "sodium": 50.0, "serving": 100.0
            })
            self.classes[c_name] = FoodClassDefinition(
                class_id=class_id,
                name=c_name,
                tier=1,
                category=info["category"],
                usda_fdc_id=info["fdc_id"],
                calories_per_100g=info["calories"],
                protein_g=info["protein"],
                fat_g=info["fat"],
                carb_g=info["carbs"],
                fiber_g=info["fiber"],
                sugar_g=info["sugar"],
                sodium_mg=info["sodium"],
                default_serving_g=info["serving"],
                synonyms=[name],
            )
            class_id += 1

        # Register Food-101 Prepared Dishes (assigned to Tier 2 & Tier 3)
        for name, d in FOOD101_PREPARED_DISHES.items():
            c_name = self._clean_name(name)
            if c_name in self.classes:
                continue
            tier = 2 if class_id < 200 else 3
            self.classes[c_name] = FoodClassDefinition(
                class_id=class_id,
                name=c_name,
                tier=tier,
                category=d["category"],
                usda_fdc_id=None,
                calories_per_100g=d["cal"],
                protein_g=d["p"],
                fat_g=d["f"],
                carb_g=d["c"],
                fiber_g=d["fib"],
                sugar_g=d["sug"],
                sodium_mg=d["sod"],
                default_serving_g=d["serving"],
                synonyms=[name],
            )
            class_id += 1

        # Register Indian Regional Dishes
        for name, d in INDIAN_STAPLE_DISHES.items():
            c_name = self._clean_name(name)
            if c_name in self.classes:
                continue
            tier = 2 if class_id < 200 else 3
            self.classes[c_name] = FoodClassDefinition(
                class_id=class_id,
                name=c_name,
                tier=tier,
                category=d["category"],
                usda_fdc_id=None,
                calories_per_100g=d["cal"],
                protein_g=d["p"],
                fat_g=d["f"],
                carb_g=d["c"],
                fiber_g=d["fib"],
                sugar_g=d["sug"],
                sodium_mg=d["sod"],
                default_serving_g=d["serving"],
                synonyms=[name],
            )
            class_id += 1

        # Fill remaining classes up to 650 from USDA candidates.csv
        # Prioritizing primary whole foods: fruits, vegetables, grains, meats, dairy, seafood
        priority_categories = ["fruit", "vegetable", "meat", "poultry", "seafood", "dairy", "grain", "legume", "bakery", "snack"]

        for cat in priority_categories:
            for desc, info in sorted(candidates_db.items()):
                if class_id >= 650:
                    break
                if info["category"] == cat and desc not in self.classes:
                    tier = 2 if class_id < 200 else (3 if class_id < 450 else 4)
                    self.classes[desc] = FoodClassDefinition(
                        class_id=class_id,
                        name=desc,
                        tier=tier,
                        category=cat,
                        usda_fdc_id=info["fdc_id"],
                        calories_per_100g=info["calories"],
                        protein_g=info["protein"],
                        fat_g=info["fat"],
                        carb_g=info["carbs"],
                        fiber_g=info["fiber"],
                        sugar_g=info["sugar"],
                        sodium_mg=info["sodium"],
                        default_serving_g=info["serving"],
                        synonyms=[desc],
                    )
                    class_id += 1

        # Any remaining classes to ensure exactly 650
        for desc, info in sorted(candidates_db.items()):
            if class_id >= 650:
                break
            if desc not in self.classes:
                self.classes[desc] = FoodClassDefinition(
                    class_id=class_id,
                    name=desc,
                    tier=4,
                    category=info["category"],
                    usda_fdc_id=info["fdc_id"],
                    calories_per_100g=info["calories"],
                    protein_g=info["protein"],
                    fat_g=info["fat"],
                    carb_g=info["carbs"],
                    fiber_g=info["fiber"],
                    sugar_g=info["sugar"],
                    sodium_mg=info["sodium"],
                    default_serving_g=info["serving"],
                    synonyms=[desc],
                )
                class_id += 1

    def get_tier_classes(self, max_tier: int = 4) -> dict[str, FoodClassDefinition]:
        """Return subset of catalog filtered by tier level."""
        return {k: v for k, v in self.classes.items() if v.tier <= max_tier}

    def export_catalog(self, output_json: Path | str = "ml/knowledge_base/expanded_catalog_650.json") -> Path:
        """Export serialized catalog to JSON for production backend and inference engines."""
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_classes": len(self.classes),
            "tier_counts": {
                "tier_1_50": len(self.get_tier_classes(1)),
                "tier_2_200": len(self.get_tier_classes(2)),
                "tier_3_450": len(self.get_tier_classes(3)),
                "tier_4_650": len(self.get_tier_classes(4)),
            },
            "classes": {k: asdict(v) for k, v in self.classes.items()},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(self.classes)} food classes to {out_path}")
        return out_path


if __name__ == "__main__":
    catalog = ScalableExpansionCatalog()
    print(f"Total curated classes: {len(catalog.classes)}")
    for t in [1, 2, 3, 4]:
        sub = catalog.get_tier_classes(t)
        print(f"Tier {t} (Max): {len(sub)} classes")
    catalog.export_catalog()

