"""Automated CSV Split Generator for Nutrition AI Datasets.

Scans the ml/data/images/ directory, extracts class labels,
and creates stratified train.csv, val.csv, and test.csv annotations.
"""

from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml.src.nutrition_mapping import NutritionCatalog

logger = logging.getLogger("nutrition_ai.split_generator")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_image_directory(
    images_dir: str | Path = "ml/data/images",
    label_map: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Scan directory for class subfolders and extract (relative_image_path, label) pairs.

    Expects structure:
        images_dir/
            <class_name_or_folder>/
                image_001.jpg
                image_002.jpg
    """
    img_dir = Path(images_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    samples: list[tuple[str, str]] = []

    # Iterate through class subdirectories
    subdirs = [d for d in img_dir.iterdir() if d.is_dir()]

    for subdir in sorted(subdirs, key=lambda d: d.name):
        folder_name = subdir.name
        # Compute canonical label: underscore to space or lookup in custom label_map
        if label_map and folder_name in label_map:
            label = label_map[folder_name]
        else:
            label = folder_name.replace("_", " ").lower().strip()

        for img_file in subdir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTENSIONS:
                # Relative path: class_folder/filename.jpg
                rel_path = f"{folder_name}/{img_file.name}"
                samples.append((rel_path, label))

    logger.info(f"Discovered {len(samples)} images across {len(subdirs)} class folders in {img_dir}")
    return samples


def generate_splits(
    images_dir: str | Path = "ml/data/images",
    output_dir: str | Path = "ml/data/annotations",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate stratified train.csv, val.csv, and test.csv from images_dir.

    Args:
        images_dir: Directory containing class subfolders of images.
        output_dir: Destination folder for train.csv, val.csv, and test.csv.
        train_ratio: Fraction for training split (e.g. 0.70).
        val_ratio: Fraction for validation split (e.g. 0.15).
        test_ratio: Fraction for test split (e.g. 0.15).
        random_seed: Deterministic shuffle seed.
        label_map: Optional mapping from folder name to canonical label.

    Returns:
        Summary dict containing counts and class breakdown per split.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0, got: {total_ratio}")

    samples = scan_image_directory(images_dir, label_map=label_map)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by class label for stratified splitting
    by_class: dict[str, list[str]] = defaultdict(list)
    for rel_path, label in samples:
        by_class[label].append(rel_path)

    rng = random.Random(random_seed)

    train_rows: list[tuple[str, str]] = []
    val_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []

    for label, paths in sorted(by_class.items()):
        rng.shuffle(paths)
        n = len(paths)

        if n == 1:
            # Single sample goes to train
            train_rows.append((paths[0], label))
        elif n == 2:
            train_rows.append((paths[0], label))
            val_rows.append((paths[1], label))
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            n_train = n - n_val - n_test

            # Adjust if rounding caused negative or zero train
            if n_train <= 0:
                n_train = 1
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1

            train_slice = paths[:n_train]
            val_slice = paths[n_train : n_train + n_val]
            test_slice = paths[n_train + n_val :]

            for p in train_slice:
                train_rows.append((p, label))
            for p in val_slice:
                val_rows.append((p, label))
            for p in test_slice:
                test_rows.append((p, label))

    # Shuffle rows within each split
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    # Write CSV files
    def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(rows)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    write_csv(train_path, train_rows)
    write_csv(val_path, val_rows)
    write_csv(test_path, test_rows)

    logger.info(
        f"Splits written: Train={len(train_rows)}, Val={len(val_rows)}, Test={len(test_rows)}"
    )

    return {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "total_images": len(samples),
        "total_classes": len(by_class),
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

