"""USDA FoodData Central Reference Image Mapper & Collector.

Aligns USDA Foundation and SR Legacy food entries with visual datasets.
Allows mapping candidate classes directly into the image hierarchy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("nutrition_ai.usda_collector")


def list_usda_target_classes(
    candidates_csv: str | Path = "ml/data/raw/candidates.csv",
    category_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Extract list of verified USDA food classes available for visual dataset expansion."""
    p = Path(candidates_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing candidates database: {p}")

    foods = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cat = r.get("category", "")
            if category_filter and cat.lower() != category_filter.lower():
                continue
            foods.append({
                "fdc_id": r.get("fdc_id"),
                "food_name": r.get("normalised_name"),
                "description": r.get("description"),
                "category": cat,
                "calories": r.get("calories_per_100g"),
            })
    return foods


def setup_usda_image_folders(
    dest_images_dir: str | Path = "ml/data/images",
    candidates_csv: str | Path = "ml/data/raw/candidates.csv",
    limit_classes: int | None = None,
) -> int:
    """Create directory structure in ml/data/images/ for all USDA candidate food classes."""
    dest = Path(dest_images_dir)
    dest.mkdir(parents=True, exist_ok=True)

    foods = list_usda_target_classes(candidates_csv)
    if limit_classes:
        foods = foods[:limit_classes]

    created = 0
    for f in foods:
        name = f["food_name"]
        if not name:
            continue
        folder_name = name.replace(" ", "_").replace("/", "_")
        folder = dest / folder_name
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            created += 1

    logger.info(f"Initialized {created} class directories under {dest}")
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USDA Image Directory Setup")
    parser.add_argument("--dest-dir", type=str, default="ml/data/images", help="Target images folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of classes to create")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    c = setup_usda_image_folders(args.dest_dir, limit_classes=args.limit)
    print(f"Created {c} class directories for USDA dataset expansion.")

