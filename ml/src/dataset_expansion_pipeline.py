"""Production-Ready Dataset Expansion, Cleaning, Annotation & Split Pipeline.

Supports incremental scaling from 50 to 650+ food classes across multi-source datasets:
1. Food-101 (Classic prepared meals)
2. Open Food Facts (Packaged and branded foods)
3. USDA FoodData Central (Whole foods, fruits, vegetables, cuts)
4. Custom Community / User submissions (Regional Indian, Asian, Mediterranean)

Automates:
- SHA-256 deduplication
- Resolution, aspect ratio, and blur filtering
- Color mode normalization (RGBA, CMYK, Palette -> RGB)
- Stratified 70/15/15 train/val/test split generation
- Full validation report generation
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.expansion_catalog import ScalableExpansionCatalog

logger = logging.getLogger("nutrition_ai.expansion_pipeline")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class CleaningStats:
    """Summary of dataset sanitation."""
    total_scanned: int = 0
    passed_validation: int = 0
    duplicates_removed: int = 0
    corrupted_removed: int = 0
    sub_resolution_removed: int = 0
    extreme_aspect_removed: int = 0
    blurry_removed: int = 0
    modes_converted: int = 0


class ScalableDatasetPipeline:
    """End-to-end dataset preparation and split generator for scaling up to 650+ classes."""

    def __init__(
        self,
        images_dir: Path | str = "ml/data/images",
        annotations_dir: Path | str = "ml/data/annotations",
        candidates_csv: Path | str = "ml/data/raw/candidates.csv",
        min_resolution: int = 64,
        max_aspect_ratio: float = 4.0,
        blur_variance_threshold: float = 35.0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.catalog = ScalableExpansionCatalog(candidates_csv)
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.blur_variance_threshold = blur_variance_threshold

    def compute_sha256(self, file_path: Path) -> str:
        """Calculate content hash for duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def sanitize_image(self, file_path: Path) -> tuple[bool, str | None, Image.Image | None]:
        """Validate and sanitize an image file.

        Returns (is_valid, rejection_reason, sanitized_pil_image).
        """
        # 1. Extension check
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            return False, "INVALID_EXTENSION", None

        # 2. Decode verification
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            if len(img_bytes) == 0:
                return False, "EMPTY_FILE", None
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.load()
        except Exception:
            return False, "CORRUPTED_BYTES", None

        # 3. EXIF orientation
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass

        w, h = pil_img.size
        # 4. Dimension & aspect check
        if w < self.min_resolution or h < self.min_resolution:
            return False, f"SUB_RESOLUTION_{w}x{h}", None

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > self.max_aspect_ratio:
            return False, f"EXTREME_ASPECT_{aspect:.1f}", None

        # 5. Color normalization
        if pil_img.mode in ("RGBA", "LA", "P"):
            rgba = pil_img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            blended = Image.alpha_composite(bg, rgba)
            pil_img = blended.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # 6. Blur detection via Laplacian
        gray = pil_img.convert("L")
        lap = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0))
        lap_var = float(np.var(np.array(lap, dtype=np.float32)))
        if lap_var < self.blur_variance_threshold:
            return False, f"SEVERE_BLUR_VAR_{lap_var:.1f}", None

        return True, None, pil_img

    def clean_and_index_dataset(
        self,
        target_classes: list[str] | None = None,
        max_images_per_class: int = 400,
    ) -> tuple[dict[str, list[Path]], CleaningStats]:
        """Scan, deduplicate, sanitize, and index images for target classes."""
        stats = CleaningStats()
        seen_hashes: set[str] = set()
        indexed: dict[str, list[Path]] = defaultdict(list)

        active_classes = set(target_classes or self.catalog.classes.keys())
        logger.info(f"Indexing images directory across {len(active_classes)} target classes...")

        if not self.images_dir.exists():
            self.images_dir.mkdir(parents=True, exist_ok=True)
            return indexed, stats

        class_folders = [d for d in self.images_dir.iterdir() if d.is_dir()]

        for folder in class_folders:
            folder_class = folder.name.replace("_", " ").lower().strip()
            if folder_class not in active_classes:
                continue

            for img_file in folder.iterdir():
                if not img_file.is_file():
                    continue

                stats.total_scanned += 1

                # 1. Check deduplication hash
                try:
                    file_hash = self.compute_sha256(img_file)
                except Exception:
                    stats.corrupted_removed += 1
                    continue

                if file_hash in seen_hashes:
                    stats.duplicates_removed += 1
                    continue

                # 2. Validation & sanitation
                is_valid, reason, sanitized_img = self.sanitize_image(img_file)
                if not is_valid:
                    if "CORRUPTED" in (reason or "") or "EMPTY" in (reason or ""):
                        stats.corrupted_removed += 1
                    elif "SUB_RESOLUTION" in (reason or ""):
                        stats.sub_resolution_removed += 1
                    elif "EXTREME_ASPECT" in (reason or ""):
                        stats.extreme_aspect_removed += 1
                    elif "SEVERE_BLUR" in (reason or ""):
                        stats.blurry_removed += 1
                    continue

                seen_hashes.add(file_hash)
                stats.passed_validation += 1
                indexed[folder_class].append(img_file)

                if len(indexed[folder_class]) >= max_images_per_class:
                    break

        return indexed, stats

    def generate_stratified_splits(
        self,
        indexed_samples: dict[str, list[Path]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        output_prefix: str = "",
        seed: int = 42,
    ) -> dict[str, int]:
        """Generate stratified train.csv, val.csv, and test.csv annotations."""
        rng = random.Random(seed)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        train_records: list[tuple[str, str]] = []
        val_records: list[tuple[str, str]] = []
        test_records: list[tuple[str, str]] = []

        for class_name, paths in sorted(indexed_samples.items()):
            shuffled = list(paths)
            rng.shuffle(shuffled)
            n = len(shuffled)
            if n == 0:
                continue

            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
            n_test = max(1, n - n_train - n_val) if n >= 2 else 0

            # Partition
            train_paths = shuffled[:n_train]
            val_paths = shuffled[n_train:n_train + n_val]
            test_paths = shuffled[n_train + n_val:]

            for p in train_paths:
                rel = f"{p.parent.name}/{p.name}"
                train_records.append((rel, class_name))
            for p in val_paths:
                rel = f"{p.parent.name}/{p.name}"
                val_records.append((rel, class_name))
            for p in test_paths:
                rel = f"{p.parent.name}/{p.name}"
                test_records.append((rel, class_name))

        # Shuffle each split
        rng.shuffle(train_records)
        rng.shuffle(val_records)
        rng.shuffle(test_records)

        def write_csv(filename: str, records: list[tuple[str, str]]):
            path = self.annotations_dir / f"{output_prefix}{filename}"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "label"])
                writer.writerows(records)
            return path

        write_csv("train.csv", train_records)
        write_csv("val.csv", val_records)
        write_csv("test.csv", test_records)

        counts = {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
            "total_classes": len(indexed_samples),
        }

        logger.info(
            f"Generated stratified splits: Train={counts['train']}, Val={counts['val']}, Test={counts['test']} across {counts['total_classes']} classes."
        )
        return counts

    def run_pipeline(
        self,
        tier: int = 4,
        max_images_per_class: int = 320,
    ) -> dict[str, Any]:
        """Run complete dataset expansion and validation."""
        target_classes = list(self.catalog.get_tier_classes(tier).keys())
        logger.info(f"Starting pipeline for Tier {tier} ({len(target_classes)} target classes)...")

        t0 = time.perf_counter()
        indexed, stats = self.clean_and_index_dataset(target_classes, max_images_per_class)
        split_counts = self.generate_stratified_splits(indexed)
        duration = round(time.perf_counter() - t0, 2)

        report = {
            "tier": tier,
            "target_classes_count": len(target_classes),
            "indexed_classes_count": len(indexed),
            "cleaning_stats": asdict(stats),
            "split_counts": split_counts,
            "duration_seconds": duration,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        report_path = Path("ml/data/reports/dataset_expansion_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Expansion report saved to: {report_path.resolve()}")
        return report


if __name__ == "__main__":
    pipeline = ScalableDatasetPipeline()
    res = pipeline.run_pipeline(tier=1)
    print("Pipeline Execution Summary:")
    print(json.dumps(res, indent=2))
