"""Orchestrator to generate and package the initial 50-class food image dataset."""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset_synthesizer import generate_food_image
from ml.src.nutrition_mapping import NutritionCatalog
from ml.src.split_generator import generate_splits
from ml.src.utils import setup_logger

logger = setup_logger("nutrition_ai.build_dataset")

# 50 Curated Food Classes with visual profile definitions matching candidates.csv / nutrition_mapping.csv
CLASSES_50_CONFIG: dict[str, dict] = {
    # Fruits
    "apples gala with skin raw": {"shape": "circle", "color1": (210, 45, 40), "color2": (150, 30, 25), "accent": (80, 140, 40)},
    "bananas raw": {"shape": "curved", "color1": (245, 215, 60), "color2": (190, 160, 40), "accent": (100, 140, 40)},
    "oranges raw navels": {"shape": "circle", "color1": (245, 130, 25), "color2": (200, 95, 15), "accent": (60, 140, 50)},
    "strawberries raw": {"shape": "circle", "color1": (220, 30, 45), "color2": (170, 20, 30), "accent": (50, 160, 50)},
    "nectarines raw": {"shape": "circle", "color1": (225, 90, 45), "color2": (180, 50, 30), "accent": (230, 160, 50)},
    "mangos raw": {"shape": "circle", "color1": (240, 175, 40), "color2": (210, 80, 35), "accent": (100, 150, 45)},
    "lemons raw without peel": {"shape": "circle", "color1": (250, 230, 65), "color2": (215, 190, 45), "accent": (230, 240, 180)},
    "blueberries raw": {"shape": "circle", "color1": (45, 55, 115), "color2": (30, 35, 80), "accent": (85, 95, 150)},
    "raspberries raw": {"shape": "circle", "color1": (195, 25, 65), "color2": (150, 15, 45), "accent": (220, 60, 95)},
    "blackberries raw": {"shape": "circle", "color1": (28, 24, 38), "color2": (18, 15, 25), "accent": (60, 45, 75)},

    # Vegetables
    "broccoli raw": {"shape": "florets", "color1": (45, 115, 45), "color2": (30, 80, 30), "accent": (110, 165, 85)},
    "carrots raw": {"shape": "tapered", "color1": (240, 110, 20), "color2": (195, 80, 15), "accent": (65, 150, 55)},
    "spinach raw": {"shape": "leaves", "color1": (35, 105, 40), "color2": (25, 75, 30), "accent": (90, 155, 75)},
    "kale raw": {"shape": "leaves", "color1": (30, 85, 50), "color2": (20, 60, 35), "accent": (70, 130, 80)},
    "garlic raw": {"shape": "circle", "color1": (240, 235, 220), "color2": (200, 190, 170), "accent": (160, 145, 120)},
    "onions red raw": {"shape": "circle", "color1": (150, 40, 85), "color2": (105, 25, 60), "accent": (225, 200, 215)},
    "tomatoes green raw": {"shape": "circle", "color1": (115, 160, 65), "color2": (85, 125, 45), "accent": (160, 200, 105)},
    "celery raw": {"shape": "tapered", "color1": (135, 195, 95), "color2": (100, 155, 70), "accent": (175, 225, 140)},
    "cabbage raw": {"shape": "circle", "color1": (165, 210, 140), "color2": (125, 170, 105), "accent": (210, 240, 195)},
    "wasabi root raw": {"shape": "tapered", "color1": (120, 175, 85), "color2": (90, 135, 60), "accent": (150, 205, 110)},
    "cornsalad raw": {"shape": "leaves", "color1": (50, 125, 55), "color2": (35, 90, 40), "accent": (100, 170, 90)},
    "mushrooms white raw": {"shape": "circle", "color1": (235, 228, 215), "color2": (195, 185, 170), "accent": (150, 140, 125)},
    "lentils sprouted raw": {"shape": "bowl_dip", "color1": (145, 165, 110), "color2": (110, 130, 80), "accent": (200, 220, 160)},
    "broccoli leaves raw": {"shape": "leaves", "color1": (40, 100, 45), "color2": (28, 70, 32), "accent": (85, 145, 75)},

    # Bakery & Grains
    "bread rye": {"shape": "rectangle", "color1": (145, 105, 75), "color2": (105, 75, 50), "accent": (60, 45, 35)},
    "bread egg": {"shape": "rectangle", "color1": (215, 165, 80), "color2": (175, 125, 55), "accent": (120, 80, 35)},
    "croutons plain": {"shape": "rectangle", "color1": (190, 145, 85), "color2": (150, 105, 55), "accent": (220, 180, 115)},
    "potato pancakes": {"shape": "rectangle", "color1": (185, 135, 65), "color2": (140, 95, 45), "accent": (220, 170, 90)},
    "muffins oat bran": {"shape": "circle", "color1": (155, 110, 65), "color2": (115, 80, 45), "accent": (85, 55, 30)},
    "rice crackers": {"shape": "circle", "color1": (230, 210, 175), "color2": (190, 165, 130), "accent": (140, 115, 85)},
    "pasta gluten free corn cooked": {"shape": "bowl_dip", "color1": (235, 205, 95), "color2": (195, 165, 70), "accent": (250, 225, 135)},
    "archway home style cookies oatmeal": {"shape": "circle", "color1": (175, 130, 80), "color2": (135, 95, 55), "accent": (95, 65, 40)},
    "flour rice brown": {"shape": "bowl_dip", "color1": (225, 215, 195), "color2": (190, 175, 155), "accent": (155, 140, 120)},
    "flour soy defatted": {"shape": "bowl_dip", "color1": (235, 225, 185), "color2": (195, 185, 145), "accent": (160, 150, 115)},

    # Legumes & Nuts
    "hummus commercial": {"shape": "bowl_dip", "color1": (220, 195, 145), "color2": (185, 155, 110), "accent": (165, 55, 40)},
    "peanuts spanish raw": {"shape": "rectangle", "color1": (195, 145, 95), "color2": (155, 110, 70), "accent": (140, 70, 50)},
    "nuts almonds": {"shape": "rectangle", "color1": (165, 110, 65), "color2": (125, 80, 45), "accent": (210, 165, 115)},

    # Dairy & Eggs
    "egg yolk dried": {"shape": "bowl_dip", "color1": (245, 175, 30), "color2": (210, 135, 20), "accent": (255, 210, 70)},
    "egg white dried": {"shape": "bowl_dip", "color1": (245, 240, 230), "color2": (215, 210, 195), "accent": (190, 185, 170)},
    "cheese blue": {"shape": "rectangle", "color1": (230, 225, 205), "color2": (190, 185, 165), "accent": (75, 115, 110)},
    "cheese brie": {"shape": "rectangle", "color1": (240, 235, 210), "color2": (210, 200, 175), "accent": (250, 248, 235)},
    "yogurt plain low fat": {"shape": "bowl_dip", "color1": (248, 248, 245), "color2": (220, 220, 215), "accent": (180, 180, 175)},

    # Proteins & Prepared Meals
    "salmon nuggets breaded frozen heated": {"shape": "rectangle", "color1": (210, 140, 65), "color2": (170, 105, 45), "accent": (235, 105, 85)},
    "chicken spread": {"shape": "bowl_dip", "color1": (225, 195, 165), "color2": (190, 155, 125), "accent": (135, 95, 75)},
    "pork fresh ground raw": {"shape": "rectangle", "color1": (215, 120, 130), "color2": (180, 85, 95), "accent": (245, 200, 205)},
    "pizza rolls frozen unprepared": {"shape": "rectangle", "color1": (215, 160, 80), "color2": (175, 120, 55), "accent": (180, 45, 35)},
    "tortilla chips yellow plain salted": {"shape": "pie_slice", "color1": (230, 195, 95), "color2": (190, 155, 65), "accent": (245, 225, 140)},
    "scrapple pork": {"shape": "rectangle", "color1": (135, 100, 80), "color2": (95, 70, 50), "accent": (165, 135, 110)},
    "ice creams chocolate": {"shape": "bowl_dip", "color1": (90, 55, 35), "color2": (60, 35, 20), "accent": (130, 85, 55)},
    "ketchup restaurant": {"shape": "bowl_dip", "color1": (195, 25, 20), "color2": (150, 15, 12), "accent": (225, 55, 45)},
}


def build_initial_dataset(
    images_per_class: int = 320,
    dest_images_dir: str | Path = "ml/data/images",
    output_annotations_dir: str | Path = "ml/data/annotations",
) -> dict:
    """Generate 50 food classes with specified images per class, and create stratified splits."""
    images_dir = Path(dest_images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    catalog = NutritionCatalog("ml/data/nutrition_mapping.csv")
    logger.info(f"Loaded nutrition catalog with {len(catalog)} verified foods.")

    total_classes = len(CLASSES_50_CONFIG)
    total_target_images = total_classes * images_per_class

    logger.info("=" * 60)
    logger.info(f"Generating Initial Dataset: {total_classes} Classes x {images_per_class} Images = {total_target_images} Images")
    logger.info("=" * 60)

    start_time = time.time()
    generated_counts: dict[str, int] = {}

    for idx, (food_name, class_cfg) in enumerate(CLASSES_50_CONFIG.items(), start=1):
        class_folder_name = food_name.replace(" ", "_")
        class_dir = images_dir / class_folder_name
        class_dir.mkdir(parents=True, exist_ok=True)

        t_class_start = time.time()
        for i in range(1, images_per_class + 1):
            filename = f"img_{i:04d}.jpg"
            img_path = class_dir / filename

            # Generate distinct image
            img = generate_food_image(food_name, class_cfg, image_idx=i, width=224, height=224)
            img.save(img_path, format="JPEG", quality=90)

        elapsed_class = time.time() - t_class_start
        generated_counts[food_name] = images_per_class

        if idx % 5 == 0 or idx == total_classes:
            logger.info(f"[{idx:2d}/{total_classes}] Generated 320 images for '{food_name}' ({elapsed_class:.2f}s)")

    total_time = time.time() - start_time
    logger.info(f"Image generation completed in {total_time:.2f} seconds ({total_target_images / total_time:.1f} img/s).")

    # Step 2: Generate Stratified Splits
    logger.info("Generating stratified annotations (70% Train, 15% Val, 15% Test)...")
    split_res = generate_splits(
        images_dir=images_dir,
        output_dir=output_annotations_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    return {
        "total_classes": total_classes,
        "total_images": total_target_images,
        "images_per_class": images_per_class,
        "splits": split_res,
        "generation_time_s": round(total_time, 2),
    }


if __name__ == "__main__":
    res = build_initial_dataset(images_per_class=320)
    print("\nDataset generation finished:")
    print(res)

