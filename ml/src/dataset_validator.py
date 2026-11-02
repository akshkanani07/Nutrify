"""Comprehensive Dataset Validator for Food & Drink Classification.

Validates:
- Missing image files
- Corrupted images (PIL verify)
- Duplicate images (SHA-256 hash collision)
- Empty or malformed labels
- Class distribution and imbalance
- Nutrition mapping coverage
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ml.src.nutrition_mapping import NutritionCatalog

logger = logging.getLogger("nutrition_ai.validator")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 checksum for image content deduplication."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DatasetValidationReport:
    """Structured results from validation run."""

    total_images: int
    total_classes: int
    images_per_class: dict[str, Any]
    missing_files: list[str]
    corrupted_images: list[str]
    duplicate_images: list[dict[str, str]]
    empty_labels: list[int]
    unmapped_nutrition_labels: list[str]
    split_counts: dict[str, int]
    training_readiness_status: str  # 'READY' or 'NOT READY'
    blockers: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetValidator:
    """Validates physical image assets, CSV annotations, and labels."""

    def __init__(
        self,
        images_dir: str | Path = "ml/data/images",
        annotations_dir: str | Path = "ml/data/annotations",
        nutrition_csv: str | Path = "ml/data/nutrition_mapping.csv",
        min_samples_per_class: int = 5,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.nutrition_catalog = NutritionCatalog(nutrition_csv)
        self.min_samples_per_class = min_samples_per_class

    def validate_csv(self, csv_path: Path) -> list[tuple[int, str, str]]:
        """Read and validate CSV rows. Returns list of (line_num, rel_path, label)."""
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 2:
                raise ValueError(f"Malformed header in {csv_path}: {header}")

            for idx, r in enumerate(reader, start=2):
                if not r or len(r) < 2:
                    continue
                rows.append((idx, r[0].strip(), r[1].strip()))
        return rows

    def run_validation(
        self,
        check_images_on_disk: bool = True,
        check_corrupt: bool = True,
        check_duplicates: bool = True,
    ) -> DatasetValidationReport:
        """Run complete validation suite across train.csv, val.csv, and test.csv."""
        logger.info(f"Running validation on annotations in {self.annotations_dir}")

        missing_files: list[str] = []
        corrupted_images: list[str] = []
        empty_labels: list[int] = []
        duplicate_pairs: list[dict[str, str]] = []
        blockers: list[str] = []
        warnings: list[str] = []

        split_counts: dict[str, int] = {}
        all_samples: list[tuple[str, str, str]] = []  # (split_name, rel_path, label)

        # 1. Check annotation files
        expected_splits = ["train.csv", "val.csv", "test.csv"]
        found_splits = [s for s in expected_splits if (self.annotations_dir / s).exists()]

        if not found_splits:
            blockers.append(
                f"No annotation CSV files (train.csv, val.csv) found in {self.annotations_dir}"
            )
            return DatasetValidationReport(
                total_images=0,
                total_classes=0,
                images_per_class={"min": 0, "max": 0, "avg": 0, "counts": {}},
                missing_files=[],
                corrupted_images=[],
                duplicate_images=[],
                empty_labels=[],
                unmapped_nutrition_labels=[],
                split_counts={},
                training_readiness_status="NOT READY",
                blockers=blockers,
                warnings=warnings,
            )

        if "train.csv" not in found_splits:
            blockers.append("train.csv is missing.")
        if "val.csv" not in found_splits:
            blockers.append("val.csv is missing.")

        for split_name in found_splits:
            csv_file = self.annotations_dir / split_name
            split_key = split_name.replace(".csv", "")
            try:
                rows = self.validate_csv(csv_file)
                split_counts[split_key] = len(rows)
                for line_no, path_str, label_str in rows:
                    if not label_str:
                        empty_labels.append(line_no)
                    all_samples.append((split_key, path_str, label_str))
            except Exception as e:
                blockers.append(f"Error reading {split_name}: {e}")

        if empty_labels:
            blockers.append(f"Found {len(empty_labels)} empty labels in annotations.")

        # 2. Check image existence and integrity
        hash_to_path: dict[str, str] = {}
        class_counter: Counter[str] = Counter()

        for split_key, rel_path, label in all_samples:
            if label:
                class_counter[label] += 1

            if not check_images_on_disk:
                continue

            full_path = self.images_dir / rel_path
            if not full_path.exists():
                missing_files.append(f"{split_key}:{rel_path}")
                continue

            if check_corrupt:
                try:
                    with Image.open(full_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted_images.append(f"{rel_path} ({e})")
                    continue

            if check_duplicates:
                try:
                    h = compute_file_hash(full_path)
                    if h in hash_to_path:
                        duplicate_pairs.append({
                            "original": hash_to_path[h],
                            "duplicate": str(rel_path),
                        })
                    else:
                        hash_to_path[h] = str(rel_path)
                except Exception:
                    pass

        # 3. Nutrition mapping verification
        unmapped_labels: list[str] = []
        for class_label in class_counter:
            if not self.nutrition_catalog.contains(class_label):
                unmapped_labels.append(class_label)

        if unmapped_labels:
            warnings.append(
                f"{len(unmapped_labels)} class labels do not currently have entries in nutrition_mapping.csv"
            )

        # 4. Class imbalance and minimum counts
        counts = list(class_counter.values())
        min_c = min(counts) if counts else 0
        max_c = max(counts) if counts else 0
        avg_c = round(sum(counts) / len(counts), 1) if counts else 0.0

        low_sample_classes = [c for c, n in class_counter.items() if n < self.min_samples_per_class]
        if low_sample_classes:
            warnings.append(
                f"{len(low_sample_classes)} classes have fewer than {self.min_samples_per_class} images"
            )

        # 5. Evaluate Readiness
        if missing_files:
            blockers.append(f"{len(missing_files)} image files referenced in CSV are missing from disk.")
        if corrupted_images:
            blockers.append(f"{len(corrupted_images)} image files are corrupted.")
        if len(class_counter) < 2:
            blockers.append(
                f"Classification requires at least 2 distinct classes. Found {len(class_counter)}."
            )
        if split_counts.get("train", 0) == 0:
            blockers.append("Training set has 0 valid samples.")
        if split_counts.get("val", 0) == 0:
            blockers.append("Validation set has 0 valid samples.")

        readiness = "READY" if not blockers else "NOT READY"

        report = DatasetValidationReport(
            total_images=len(all_samples),
            total_classes=len(class_counter),
            images_per_class={
                "min": min_c,
                "max": max_c,
                "avg": avg_c,
                "counts": dict(class_counter.most_common(20)),
            },
            missing_files=missing_files[:50],  # sample up to 50
            corrupted_images=corrupted_images[:50],
            duplicate_images=duplicate_pairs[:50],
            empty_labels=empty_labels,
            unmapped_nutrition_labels=unmapped_labels,
            split_counts=split_counts,
            training_readiness_status=readiness,
            blockers=blockers,
            warnings=warnings,
        )

        return report

    def save_report(
        self,
        report: DatasetValidationReport,
        reports_dir: str | Path = "ml/data/reports",
    ) -> tuple[Path, Path]:
        """Save report in both JSON and Markdown format."""
        rep_dir = Path(reports_dir)
        rep_dir.mkdir(parents=True, exist_ok=True)

        json_path = rep_dir / "dataset_validation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        md_path = rep_dir / "dataset_validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Nutrition AI Dataset Validation Report\n\n")
            f.write(f"**Training Readiness Status:** `{report.training_readiness_status}`\n\n")
            f.write("## Overview\n")
            f.write(f"- **Total Images:** {report.total_images}\n")
            f.write(f"- **Total Classes:** {report.total_classes}\n")
            f.write(f"- **Split Counts:** {report.split_counts}\n")
            f.write(
                f"- **Images Per Class:** Min={report.images_per_class['min']}, "
                f"Max={report.images_per_class['max']}, Avg={report.images_per_class['avg']}\n\n"
            )
            f.write(f"- **Missing Files:** {len(report.missing_files)}\n")
            f.write(f"- **Corrupted Images:** {len(report.corrupted_images)}\n")
            f.write(f"- **Duplicate Pairs:** {len(report.duplicate_images)}\n\n")

            if report.blockers:
                f.write("## Blockers (Must Fix)\n")
                for b in report.blockers:
                    f.write(f"- ❌ {b}\n")
                f.write("\n")

            if report.warnings:
                f.write("## Warnings\n")
                for w in report.warnings:
                    f.write(f"- ⚠️ {w}\n")
                f.write("\n")

        return json_path, md_path

