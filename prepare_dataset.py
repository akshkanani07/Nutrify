"""Training Readiness and Dataset Preparation Script for Nutrition AI.

Verifies:
1. Dataset directory structure
2. CSV annotations correctness (train.csv, val.csv, test.csv)
3. Physical image availability and integrity
4. Label consistency with nutrition catalog

Outputs:
READY FOR TRAINING: YES / NO
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset_validator import DatasetValidator
from ml.src.nutrition_mapping import NutritionCatalog, generate_nutrition_mapping
from ml.src.split_generator import generate_splits


def check_directory_structure(base_dir: Path) -> tuple[bool, list[str]]:
    """Verify standard ML data architecture exists."""
    required_dirs = [
        base_dir / "ml" / "data" / "images",
        base_dir / "ml" / "data" / "annotations",
        base_dir / "ml" / "data" / "raw",
        base_dir / "ml" / "data" / "reports",
    ]
    missing = []
    for d in required_dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            missing.append(f"Created missing directory: {d}")

    # Check candidates.csv
    candidates_csv = base_dir / "ml" / "data" / "raw" / "candidates.csv"
    if not candidates_csv.exists():
        return False, [f"Missing core nutrition database: {candidates_csv}"]

    # Ensure nutrition_mapping.csv exists
    nutrition_csv = base_dir / "ml" / "data" / "nutrition_mapping.csv"
    if not nutrition_csv.exists():
        generate_nutrition_mapping(candidates_csv, nutrition_csv)

    return True, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Nutrition AI Training Readiness Verification")
    parser.add_argument("--data-dir", type=str, default="ml/data", help="Root data directory")
    parser.add_argument("--auto-split", action="store_true", help="Auto-generate splits if images exist")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    nutrition_csv = data_dir / "nutrition_mapping.csv"
    reports_dir = data_dir / "reports"

    print("\n" + "=" * 60)
    print("NUTRITION AI -- DATASET PREPARATION & READINESS CHECK")
    print("=" * 60)

    # 1. Structure check
    struct_ok, struct_notes = check_directory_structure(PROJECT_ROOT)
    for note in struct_notes:
        print(f"  [Structure] {note}")

    if not struct_ok:
        print("\n" + "=" * 60)
        print("READY FOR TRAINING: NO")
        print("Reason: Core nutrition dataset (candidates.csv) is missing.")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("  [Structure] Dataset directory hierarchy: OK")

    # 2. Check if annotations exist or need auto-generation
    train_csv = annotations_dir / "train.csv"
    val_csv = annotations_dir / "val.csv"
    test_csv = annotations_dir / "test.csv"

    has_annotations = train_csv.exists() and val_csv.exists()

    # Check for images on disk
    subdirs = [d for d in images_dir.iterdir() if d.is_dir()] if images_dir.exists() else []
    total_raw_images = sum(len(list(d.glob("*"))) for d in subdirs)

    if not has_annotations or args.auto_split:
        if total_raw_images > 0:
            print(f"  [Splits] Discovered {total_raw_images} images across {len(subdirs)} classes. Generating splits...")
            generate_splits(
                images_dir=images_dir,
                output_dir=annotations_dir,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )
            has_annotations = True
        else:
            print("  [Splits] No existing image assets found in ml/data/images/.")

    # 3. Validation check
    validator = DatasetValidator(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        nutrition_csv=nutrition_csv,
    )

    report = validator.run_validation(check_images_on_disk=True)
    validator.save_report(report, reports_dir=reports_dir)

    print(f"  [Annotations] Total split samples: {report.total_images} across {report.total_classes} classes")
    print(f"  [Nutrition] Verified nutrition mapping loaded ({len(validator.nutrition_catalog)} foods)")

    # 4. Final Verdict
    print("\n" + "=" * 60)
    print("DATASET DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total Discovered Images : {report.total_images}")
    print(f"Total Active Classes    : {report.total_classes}")
    print(f"Missing Image Files     : {len(report.missing_files)}")
    print(f"Corrupted Images        : {len(report.corrupted_images)}")
    print(f"Duplicate Image Pairs   : {len(report.duplicate_images)}")
    print(f"Splits Available        : {report.split_counts}")

    if report.warnings:
        print("\nOperational Notices / Warnings:")
        for w in report.warnings:
            print(f"  * {w}")

    if report.blockers:
        print("\nAction Required Before Training:")
        for b in report.blockers:
            print(f"  - {b}")

    is_ready = report.training_readiness_status == "READY"

    print("\n" + "=" * 60)
    if is_ready:
        print("READY FOR TRAINING: YES")
        print("All structural, CSV, and image integrity constraints passed.")
        print("You can start model training with: python ml/train.py")
    else:
        print("READY FOR TRAINING: NO")
        print("Dataset preparation is configured, but image dataset must be added.")
        print("To add food images, see dataset collection instructions in ml/README.md.")
    print("=" * 60 + "\n")

    sys.exit(0 if is_ready else 1)


if __name__ == "__main__":
    main()
