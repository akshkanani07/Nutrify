"""Nutrition Mapping Engine for Nutrition AI.

Decouples nutritional attributes from image classification labels.
Maintains standardized schema:
    food_name,USDA_ID,calories,protein,carbs,fat,fiber
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("nutrition_ai.nutrition_mapping")


@dataclass
class NutritionRecord:
    """Standardized nutrition entry per 100g."""

    food_name: str
    USDA_ID: str
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_nutrition_mapping(
    candidates_csv: str | Path = "ml/data/raw/candidates.csv",
    output_csv: str | Path = "ml/data/nutrition_mapping.csv",
) -> int:
    """Generate standardized nutrition_mapping.csv from verified candidates.csv."""
    candidates_path = Path(candidates_csv)
    output_path = Path(output_csv)

    if not candidates_path.exists():
        raise FileNotFoundError(f"Source candidates file not found: {candidates_path}")

    records: list[NutritionRecord] = []
    seen_names: set[str] = set()

    def parse_float(val: Any) -> float | None:
        if val is None:
            return None
        val_str = str(val).strip()
        if not val_str:
            return None
        try:
            return float(val_str)
        except ValueError:
            return None

    with open(candidates_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_name = row.get("normalised_name", "").strip().lower()
            if not food_name or food_name in seen_names:
                continue

            seen_names.add(food_name)
            usda_id = row.get("fdc_id", "").strip()
            calories = parse_float(row.get("calories_per_100g")) or 0.0
            protein = parse_float(row.get("protein_g")) or 0.0
            fat = parse_float(row.get("fat_g")) or 0.0
            carbs = parse_float(row.get("carb_g")) or 0.0
            fiber = parse_float(row.get("fiber_g"))

            records.append(
                NutritionRecord(
                    food_name=food_name,
                    USDA_ID=usda_id,
                    calories=round(calories, 1),
                    protein=round(protein, 2),
                    carbs=round(carbs, 2),
                    fat=round(fat, 2),
                    fiber=round(fiber, 2) if fiber is not None else None,
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["food_name", "USDA_ID", "calories", "protein", "carbs", "fat", "fiber"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "food_name": r.food_name,
                "USDA_ID": r.USDA_ID,
                "calories": r.calories,
                "protein": r.protein,
                "carbs": r.carbs,
                "fat": r.fat,
                "fiber": r.fiber if r.fiber is not None else "",
            })

    logger.info(f"Generated {len(records)} nutrition mappings at {output_path}")
    return len(records)


class NutritionCatalog:
    """Lookup index for standardized nutrition mappings."""

    def __init__(self, mapping_csv: str | Path = "ml/data/nutrition_mapping.csv") -> None:
        self.mapping_path = Path(mapping_csv)
        self.by_name: dict[str, NutritionRecord] = {}
        self.by_usda_id: dict[str, NutritionRecord] = {}

        if not self.mapping_path.exists():
            # Auto-generate if candidates.csv exists
            candidates = Path("ml/data/raw/candidates.csv")
            if candidates.exists():
                generate_nutrition_mapping(candidates, self.mapping_path)

        if self.mapping_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("food_name", "").strip().lower()
                usda_id = row.get("USDA_ID", "").strip()

                def _to_f(v: str) -> float | None:
                    try:
                        return float(v) if v.strip() else None
                    except ValueError:
                        return None

                rec = NutritionRecord(
                    food_name=name,
                    USDA_ID=usda_id,
                    calories=_to_f(row.get("calories", "")) or 0.0,
                    protein=_to_f(row.get("protein", "")) or 0.0,
                    carbs=_to_f(row.get("carbs", "")) or 0.0,
                    fat=_to_f(row.get("fat", "")) or 0.0,
                    fiber=_to_f(row.get("fiber", "")),
                )
                if name:
                    self.by_name[name] = rec
                if usda_id:
                    self.by_usda_id[usda_id] = rec

    def get(self, food_name_or_id: str) -> NutritionRecord | None:
        """Lookup nutrition record by name or USDA ID."""
        key = food_name_or_id.strip().lower()
        if key in self.by_name:
            return self.by_name[key]
        if food_name_or_id.strip() in self.by_usda_id:
            return self.by_usda_id[food_name_or_id.strip()]
        return None

    def contains(self, food_name: str) -> bool:
        return food_name.strip().lower() in self.by_name

    def __len__(self) -> int:
        return len(self.by_name)


if __name__ == "__main__":
    count = generate_nutrition_mapping()
    print(f"Created ml/data/nutrition_mapping.csv with {count} verified entries.")

