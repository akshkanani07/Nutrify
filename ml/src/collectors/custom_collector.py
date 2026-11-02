"""Custom Food Image Importer and Sanitizer.

Imports, sanitizes, and indexes custom collected food photos (phone cameras,
user uploads, or field data) into the Nutrition AI image architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger("nutrition_ai.custom_collector")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of file content for deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_food_name(name: str) -> str:
    """Normalize input folder/file name into canonical food label."""
    s = re.sub(r"[_\-]+", " ", name)
    s = re.sub(r"[^\w\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def import_custom_images(
    source_dir: str | Path,
    dest_images_dir: str | Path = "ml/data/images",
    default_label: str | None = None,
    max_dimension: int = 1024,
) -> dict[str, Any]:
    """Import and sanitize custom images.

    If source_dir contains subfolders, each subfolder name is treated as a food class label.
    If source_dir contains raw images without subfolders, default_label must be provided.

    Args:
        source_dir: Directory containing images or class folders.
        dest_images_dir: Target ml/data/images directory.
        default_label: Label to apply if images are not in class subfolders.
        max_dimension: Resize max edge if larger (saves disk & speeds training).

    Returns:
        Summary dict containing counts of imported, skipped, and duplicate images.
    """
    src = Path(source_dir)
    dest = Path(dest_images_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src}")

    # Build known hashes in dest_images_dir to avoid duplicates
    existing_hashes: set[str] = set()
    for existing_img in dest.rglob("*"):
        if existing_img.suffix.lower() in VALID_EXTENSIONS and existing_img.is_file():
            try:
                existing_hashes.add(compute_sha256(existing_img))
            except Exception:
                pass

    total_imported = 0
    total_duplicates = 0
    total_corrupt = 0
    per_class_counts: dict[str, int] = {}

    # Check if directory has subfolders
    subdirs = [d for d in src.iterdir() if d.is_dir()]

    if subdirs:
        # Structured by class folder
        tasks = []
        for subdir in subdirs:
            label = sanitize_food_name(subdir.name)
            for f in subdir.iterdir():
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                    tasks.append((f, label))
    else:
        # Flat directory of images
        if not default_label:
            raise ValueError(
                "Flat image folder detected. Please specify --default-label for these images."
            )
        clean_label = sanitize_food_name(default_label)
        tasks = [
            (f, clean_label)
            for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

    for img_path, label in tasks:
        # 1. Deduplication check
        file_hash = compute_sha256(img_path)
        if file_hash in existing_hashes:
            total_duplicates += 1
            continue

        # 2. Image integrity check
        try:
            with Image.open(img_path) as img:
                img.verify()
            # Reopen for saving/converting
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                # Downsample if overly large
                if max(rgb_img.size) > max_dimension:
                    rgb_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

                target_folder = dest / label.replace(" ", "_")
                target_folder.mkdir(parents=True, exist_ok=True)

                out_filename = f"{label.replace(' ', '_')}_{file_hash[:10]}.jpg"
                out_path = target_folder / out_filename

                rgb_img.save(out_path, format="JPEG", quality=92)

            existing_hashes.add(file_hash)
            total_imported += 1
            per_class_counts[label] = per_class_counts.get(label, 0) + 1

        except Exception as e:
            logger.warning(f"Corrupt or invalid image skipped {img_path}: {e}")
            total_corrupt += 1

    summary = {
        "imported": total_imported,
        "duplicates_skipped": total_duplicates,
        "corrupted_skipped": total_corrupt,
        "classes_updated": len(per_class_counts),
        "per_class": per_class_counts,
    }
    logger.info(
        f"Custom Import Complete: {total_imported} imported, {total_duplicates} duplicates, {total_corrupt} corrupt."
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Custom Food Images")
    parser.add_argument("--source-dir", type=str, required=True, help="Folder containing raw photos or class subfolders")
    parser.add_argument("--dest-dir", type=str, default="ml/data/images", help="Target ml/data/images directory")
    parser.add_argument("--default-label", type=str, default=None, help="Class label if source folder is flat")
    parser.add_argument("--max-size", type=int, default=1024, help="Max image width/height")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = import_custom_images(args.source_dir, args.dest_dir, args.default_label, args.max_size)
    print(f"Import Summary:\n{stats}")

