"""Food-101 Dataset Importer and Label Mapper.

Imports the classic 101-class food dataset into the Nutrition AI image directory,
mapping Food-101 class keys to normalized names in the Nutrition AI catalog.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger("nutrition_ai.food101")

# Standard Food-101 download URL
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

# Canonical mapping from Food-101 class keys to Nutrition AI normalized names
FOOD101_LABEL_MAP: dict[str, str] = {
    "apple_pie": "apple pie",
    "baby_back_ribs": "pork ribs",
    "baklava": "baklava",
    "beef_carpaccio": "beef carpaccio",
    "beef_tartare": "beef tartare",
    "beet_salad": "beet salad",
    "beignets": "beignets",
    "bibimbap": "bibimbap",
    "bread_pudding": "bread pudding",
    "breakfast_burrito": "breakfast burrito",
    "bruschetta": "bruschetta",
    "caesar_salad": "caesar salad",
    "cannoli": "cannoli",
    "caprese_salad": "caprese salad",
    "carrot_cake": "carrot cake",
    "ceviche": "ceviche",
    "cheesecake": "cheesecake",
    "cheese_plate": "cheese plate",
    "chicken_curry": "chicken curry",
    "chicken_quesadilla": "chicken quesadilla",
    "chicken_wings": "chicken wings",
    "chocolate_cake": "chocolate cake",
    "chocolate_mousse": "chocolate mousse",
    "churros": "churros",
    "clam_chowder": "clam chowder",
    "club_sandwich": "club sandwich",
    "crab_cakes": "crab cakes",
    "creme_brulee": "creme brulee",
    "croque_madame": "croque madame",
    "cup_cakes": "cupcakes",
    "deviled_eggs": "deviled eggs",
    "donuts": "donuts",
    "dumplings": "dumplings",
    "edamame": "edamame",
    "eggs_benedict": "eggs benedict",
    "escargots": "escargot",
    "falafel": "falafel",
    "filet_mignon": "filet mignon",
    "fish_and_chips": "fish and chips",
    "foie_gras": "foie gras",
    "french_fries": "french fries",
    "french_onion_soup": "french onion soup",
    "french_toast": "french toast",
    "fried_calamari": "fried calamari",
    "fried_rice": "fried rice",
    "frozen_yogurt": "frozen yogurt",
    "garlic_bread": "garlic bread",
    "gnocchi": "gnocchi",
    "greek_salad": "greek salad",
    "grilled_cheese_sandwich": "grilled cheese sandwich",
    "grilled_salmon": "grilled salmon",
    "guacamole": "guacamole",
    "gyoza": "gyoza",
    "hamburger": "hamburger",
    "hot_and_sour_soup": "hot and sour soup",
    "hot_dog": "hot dog",
    "huevos_rancheros": "huevos rancheros",
    "hummus": "hummus commercial",
    "ice_cream": "ice cream",
    "lasagna": "lasagna",
    "lobster_bisque": "lobster bisque",
    "lobster_roll_sandwich": "lobster roll",
    "macaroni_and_cheese": "macaroni and cheese",
    "macarons": "macarons",
    "miso_soup": "miso soup",
    "mussels": "mussels",
    "nachos": "nachos",
    "omelette": "omelette",
    "onion_rings": "onion rings",
    "oysters": "oysters",
    "pad_thai": "pad thai",
    "paella": "paella",
    "pancakes": "pancakes",
    "panna_cotta": "panna cotta",
    "peking_duck": "peking duck",
    "pho": "pho",
    "pizza": "pizza meat topping thick crust frozen cooked",
    "pork_chop": "pork chop",
    "poutine": "poutine",
    "prime_rib": "prime rib",
    "pulled_pork_sandwich": "pulled pork sandwich",
    "ramen": "ramen",
    "ravioli": "ravioli",
    "red_velvet_cake": "red velvet cake",
    "risotto": "risotto",
    "samosa": "samosa",
    "sashimi": "sashimi",
    "scallops": "scallops",
    "seaweed_salad": "seaweed salad",
    "shrimp_and_grits": "shrimp and grits",
    "spaghetti_bolognese": "spaghetti bolognese",
    "spaghetti_carbonara": "spaghetti carbonara",
    "spring_rolls": "spring rolls",
    "steak": "steak",
    "strawberry_shortcake": "strawberry shortcake",
    "sushi": "sushi",
    "tacos": "tacos",
    "takoyaki": "takoyaki",
    "tiramisu": "tiramisu",
    "tuna_tartare": "tuna tartare",
    "waffles": "waffles",
}


def import_food101(
    extracted_food101_dir: str | Path,
    dest_images_dir: str | Path = "ml/data/images",
    max_images_per_class: int | None = None,
    classes_subset: list[str] | None = None,
) -> dict[str, int]:
    """Import an extracted Food-101 dataset into Nutrition AI image structure.

    Args:
        extracted_food101_dir: Path to directory containing 'images' folder from Food-101.
        dest_images_dir: Target ml/data/images directory.
        max_images_per_class: Maximum images to import per class (None for all).
        classes_subset: Specific class keys to import (None for all 101).

    Returns:
        Dictionary mapping imported class name to imported image count.
    """
    src_dir = Path(extracted_food101_dir)
    images_src = src_dir / "images" if (src_dir / "images").exists() else src_dir
    dest_dir = Path(dest_images_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    imported_stats: dict[str, int] = {}
    valid_exts = {".jpg", ".jpeg", ".png"}

    target_classes = classes_subset or list(FOOD101_LABEL_MAP.keys())

    for class_key in target_classes:
        class_folder = images_src / class_key
        if not class_folder.exists() or not class_folder.is_dir():
            continue

        normalized_label = FOOD101_LABEL_MAP.get(class_key, class_key.replace("_", " "))
        target_folder = dest_dir / normalized_label.replace(" ", "_")
        target_folder.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in valid_exts]
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]

        count = 0
        for img_file in image_files:
            target_file = target_folder / img_file.name
            if not target_file.exists():
                shutil.copy2(img_file, target_file)
            count += 1

        imported_stats[normalized_label] = count
        logger.info(f"Imported {count} images for '{normalized_label}'")

    return imported_stats


def download_and_extract_food101(
    download_dir: str | Path = "ml/cache/food101",
    dest_images_dir: str | Path = "ml/data/images",
    max_images_per_class: int = 100,
) -> dict[str, int]:
    """Download Food-101 tarball and import into Nutrition AI directory."""
    dl_path = Path(download_dir)
    dl_path.mkdir(parents=True, exist_ok=True)
    archive_path = dl_path / "food-101.tar.gz"

    if not archive_path.exists():
        logger.info(f"Downloading Food-101 dataset from {FOOD101_URL} (4.6 GB)...")
        urllib.request.urlretrieve(FOOD101_URL, archive_path)

    extract_path = dl_path / "extracted"
    if not (extract_path / "food-101").exists():
        logger.info("Extracting Food-101 tarball...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_path)

    return import_food101(
        extracted_food101_dir=extract_path / "food-101",
        dest_images_dir=dest_images_dir,
        max_images_per_class=max_images_per_class,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Food-101 Dataset")
    parser.add_argument("--source-dir", type=str, default=None, help="Path to pre-downloaded Food-101 directory")
    parser.add_argument("--dest-dir", type=str, default="ml/data/images", help="Target images directory")
    parser.add_argument("--max-per-class", type=int, default=150, help="Max images per class to import")
    parser.add_argument("--download", action="store_true", help="Download official Food-101 tarball")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.download:
        download_and_extract_food101(dest_images_dir=args.dest_dir, max_images_per_class=args.max_per_class)
    elif args.source_dir:
        import_food101(args.source_dir, dest_images_dir=args.dest_dir, max_images_per_class=args.max_per_class)
    else:
        print("Specify --source-dir <path> or --download.")

