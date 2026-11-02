"""CLI Script to generate train.csv, val.csv, and test.csv from ml/data/images/.

Usage:
    python ml/generate_splits.py --images-dir ml/data/images --output-dir ml/data/annotations
    python ml/generate_splits.py --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.split_generator import generate_splits
from ml.src.utils import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Train, Val, and Test CSV Annotations")
    parser.add_argument("--images-dir", type=str, default="ml/data/images", help="Folder containing class image folders")
    parser.add_argument("--output-dir", type=str, default="ml/data/annotations", help="Destination folder for split CSVs")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split fraction")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split fraction")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random shuffle seed")
    args = parser.parse_args()

    logger = setup_logger("nutrition_ai.generate_splits")
    logger.info("Starting CSV Split Generation...")

    res = generate_splits(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("CSV SPLIT GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total Discovered Images  : {res['total_images']}")
    print(f"Total Classes            : {res['total_classes']}")
    print(f"Train Split ({args.train_ratio*100:.0f}%)       : {res['train_count']} images -> {res['files']['train']}")
    print(f"Validation Split ({args.val_ratio*100:.0f}%)  : {res['val_count']} images -> {res['files']['val']}")
    print(f"Test Split ({args.test_ratio*100:.0f}%)        : {res['test_count']} images -> {res['files']['test']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

