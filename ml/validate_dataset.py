"""CLI Dataset Validation Script for Nutrition AI.

Checks dataset integrity, missing images, corrupt files, duplicates, class distribution,
and prints the standardized Dataset Report.

Usage:
    python ml/validate_dataset.py
    python ml/validate_dataset.py --images-dir ml/data/images --annotations-dir ml/data/annotations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset_validator import DatasetValidator
from ml.src.utils import setup_logger


def print_report_summary(rep) -> None:
    """Print the exact standardized Dataset Report required."""
    print("\n" + "=" * 60)
    print("Dataset Report:")
    print(f"- Total images                : {rep.total_images}")
    print(f"- Total classes               : {rep.total_classes}")
    print(
        f"- Images per class            : min={rep.images_per_class['min']}, "
        f"max={rep.images_per_class['max']}, avg={rep.images_per_class['avg']}"
    )
    print(f"- Missing files               : {len(rep.missing_files)}")
    print(f"- Corrupted images            : {len(rep.corrupted_images)}")
    print(f"- Duplicate image pairs       : {len(rep.duplicate_images)}")
    print(f"- Splits                      : {rep.split_counts}")
    print(f"- Training readiness status   : {rep.training_readiness_status}")
    print("=" * 60)

    if rep.blockers:
        print("\nBlockers to Resolve:")
        for b in rep.blockers:
            print(f"  [X] {b}")

    if rep.warnings:
        print("\nWarnings:")
        for w in rep.warnings:
            print(f"  [!] {w}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Nutrition AI Dataset")
    parser.add_argument("--images-dir", type=str, default="ml/data/images", help="Path to images directory")
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="ml/data/annotations",
        help="Path to folder containing train.csv, val.csv",
    )
    parser.add_argument(
        "--nutrition-csv",
        type=str,
        default="ml/data/nutrition_mapping.csv",
        help="Path to nutrition mapping CSV",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="ml/data/reports",
        help="Path to save validation reports",
    )
    parser.add_argument("--skip-corrupt-check", action="store_true", help="Skip detailed PIL image verification")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip SHA256 duplicate checking")
    args = parser.parse_args()

    logger = setup_logger("nutrition_ai.validate")
    logger.info("Starting Dataset Integrity Validation...")

    validator = DatasetValidator(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        nutrition_csv=args.nutrition_csv,
    )

    report = validator.run_validation(
        check_images_on_disk=True,
        check_corrupt=not args.skip_corrupt_check,
        check_duplicates=not args.skip_duplicates,
    )

    json_p, md_p = validator.save_report(report, reports_dir=args.reports_dir)
    logger.info(f"Saved validation reports to {json_p} and {md_p}")

    print_report_summary(report)

    if report.training_readiness_status != "READY":
        sys.exit(1)


if __name__ == "__main__":
    main()
