"""Nutrition Database Mapper.

Decouples nutritional metadata from image classification.
Given a predicted food class name or FDC ID, looks up nutritional values
from candidates.csv and computes serving-adjusted macronutrients.
"""

from __future__ import annotations

import csv
import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FoodNutrition:
    """Nutritional profile per 100 grams and default serving."""

    fdc_id: str
    description: str
    normalised_name: str
    category: str
    usda_category: str
    calories_per_100g: float
    protein_g: float
    fat_g: float
    carb_g: float
    fiber_g: float | None = None
    sugar_g: float | None = None
    sodium_mg: float | None = None
    default_serving_g: float | None = None
    source: str = "USDA"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def calculate_serving(self, serving_weight_g: float | None = None) -> dict[str, Any]:
        """Calculate nutritional breakdown for a specific serving weight in grams.

        If serving_weight_g is not specified, uses default_serving_g if available,
        otherwise defaults to 100g.
        """
        weight = serving_weight_g if serving_weight_g is not None else (self.default_serving_g or 100.0)
        factor = weight / 100.0

        return {
            "fdc_id": self.fdc_id,
            "description": self.description,
            "category": self.category,
            "serving_weight_g": round(weight, 1),
            "calories": round(self.calories_per_100g * factor, 1),
            "protein_g": round(self.protein_g * factor, 2),
            "fat_g": round(self.fat_g * factor, 2),
            "carbs_g": round(self.carb_g * factor, 2),
            "fiber_g": round(self.fiber_g * factor, 2) if self.fiber_g is not None else None,
            "sugar_g": round(self.sugar_g * factor, 2) if self.sugar_g is not None else None,
            "sodium_mg": round(self.sodium_mg * factor, 1) if self.sodium_mg is not None else None,
            "per_100g": {
                "calories": self.calories_per_100g,
                "protein_g": self.protein_g,
                "fat_g": self.fat_g,
                "carbs_g": self.carb_g,
                "fiber_g": self.fiber_g,
                "sugar_g": self.sugar_g,
                "sodium_mg": self.sodium_mg,
            },
        }


class NutritionMapper:
    """Lookup table and query engine for food nutrition profiles."""

    def __init__(self, csv_path: str | Path | None = None) -> None:
        if csv_path is None:
            # Default to canonical candidates.csv location
            default_path = Path("ml/data/raw/candidates.csv")
            if not default_path.exists():
                # Try finding from root workspace
                candidate_paths = [
                    Path("C:/Nutrify/ml/data/raw/candidates.csv"),
                    Path("../data/raw/candidates.csv"),
                    Path("data/raw/candidates.csv"),
                ]
                for p in candidate_paths:
                    if p.exists():
                        default_path = p
                        break
            csv_path = default_path

        self.csv_path = Path(csv_path)
        self.by_norm_name: dict[str, FoodNutrition] = {}
        self.by_fdc_id: dict[str, FoodNutrition] = {}
        self.by_description: dict[str, FoodNutrition] = {}
        self.all_foods: list[FoodNutrition] = []

        if self.csv_path.exists():
            self._load_csv()

    def _safe_float(self, value: Any) -> float | None:
        """Parse float safely, returning None for empty or invalid strings."""
        if value is None:
            return None
        val_str = str(value).strip()
        if not val_str:
            return None
        try:
            return float(val_str)
        except ValueError:
            return None

    def _load_csv(self) -> None:
        """Parse CSV rows into FoodNutrition objects."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                norm_name = r.get("normalised_name", "").strip().lower()
                fdc_id = r.get("fdc_id", "").strip()
                desc = r.get("description", "").strip()

                calories = self._safe_float(r.get("calories_per_100g")) or 0.0
                protein = self._safe_float(r.get("protein_g")) or 0.0
                fat = self._safe_float(r.get("fat_g")) or 0.0
                carb = self._safe_float(r.get("carb_g")) or 0.0
                fiber = self._safe_float(r.get("fiber_g"))
                sugar = self._safe_float(r.get("sugar_g"))
                sodium = self._safe_float(r.get("sodium_mg"))
                serving = self._safe_float(r.get("default_serving_g"))

                food = FoodNutrition(
                    fdc_id=fdc_id,
                    description=desc,
                    normalised_name=norm_name,
                    category=r.get("category", "ingredient_other"),
                    usda_category=r.get("usda_category", ""),
                    calories_per_100g=calories,
                    protein_g=protein,
                    fat_g=fat,
                    carb_g=carb,
                    fiber_g=fiber,
                    sugar_g=sugar,
                    sodium_mg=sodium,
                    default_serving_g=serving,
                    source=r.get("source", "USDA"),
                )

                self.all_foods.append(food)
                if norm_name:
                    self.by_norm_name[norm_name] = food
                if fdc_id:
                    self.by_fdc_id[fdc_id] = food
                if desc:
                    self.by_description[desc.lower()] = food

    def normalise(self, text: str) -> str:
        """Convert food query text to canonical normalised name."""
        s = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", s).strip()

    def lookup(self, query: str) -> FoodNutrition | None:
        """Look up nutrition by exact normalized name, FDC ID, description, or fuzzy match."""
        if not self.all_foods:
            return None

        # 1. Direct FDC ID lookup
        if query in self.by_fdc_id:
            return self.by_fdc_id[query]

        # 2. Normalized name lookup
        norm = self.normalise(query)
        if norm in self.by_norm_name:
            return self.by_norm_name[norm]

        # 3. Description lookup
        q_lower = query.strip().lower()
        if q_lower in self.by_description:
            return self.by_description[q_lower]

        # 4. Prefix or substring match in normalized names
        for k, food in self.by_norm_name.items():
            if norm and (k.startswith(norm) or norm in k):
                return food

        # 5. Fuzzy match against normalized names
        matches = difflib.get_close_matches(norm, list(self.by_norm_name.keys()), n=1, cutoff=0.6)
        if matches:
            return self.by_norm_name[matches[0]]

        return None

    def __len__(self) -> int:
        return len(self.all_foods)

