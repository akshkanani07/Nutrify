"""Model Evaluation Script for Nutrition AI PyTorch Pipeline.

Evaluates a saved checkpoint on a test or validation dataset and generates
Top-1, Top-5, per-class metrics, and performance summaries.

Usage:
    python ml/evaluate.py --checkpoint ml/models/checkpoints/best_model.pt --test-csv ml/data/annotations/test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset import FoodImageDataset, get_val_transforms
from ml.src.metrics import AverageMeter, calculate_topk_accuracy
from ml.src.model import build_model
from ml.src.utils import get_device, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Nutrition AI Model Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/models/checkpoints/best_model.pt",
        help="Path to trained model checkpoint file (.pt)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to evaluation CSV file containing annotations",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="ml/data/images",
        help="Root directory for image files",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Evaluation image size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics report JSON",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx_to_class: dict[int, str],
) -> dict[str, Any]:
    """Run full evaluation pass collecting accuracy, loss, and per-class stats."""
    model.eval()
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_meter = AverageMeter("Top-1", ":.2f")
    top5_meter = AverageMeter("Top-5", ":.2f")

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
        batch_size = images.size(0)

        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(top1, batch_size)
        top5_meter.update(top5, batch_size)

        _, preds = torch.max(outputs, 1)
        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            class_total[int(t)] += 1
            if t == p:
                class_correct[int(t)] += 1

    # Per-class accuracy breakdown
    per_class_metrics = {}
    for idx, class_name in idx_to_class.items():
        total = class_total[idx]
        correct = class_correct[idx]
        acc = (correct / total * 100.0) if total > 0 else 0.0
        per_class_metrics[class_name] = {
            "total_samples": total,
            "correct_samples": correct,
            "accuracy_pct": round(acc, 2),
        }

    return {
        "loss": round(loss_meter.avg, 4),
        "top1_accuracy_pct": round(top1_meter.avg, 2),
        "top5_accuracy_pct": round(top5_meter.avg, 2),
        "total_evaluated_samples": len(data_loader.dataset),
        "per_class": per_class_metrics,
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger("nutrition_ai.eval")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    device = get_device(args.device)
    logger.info(f"Loading checkpoint from {checkpoint_path} onto {device}")

    # Load checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {int(k): v for k, v in checkpoint.get("idx_to_class", {}).items()}

    # If class_to_idx not in checkpoint, look for classes.json in checkpoint dir
    if not class_to_idx:
        classes_json = checkpoint_path.parent / "classes.json"
        if classes_json.exists():
            with open(classes_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                class_to_idx = data["class_to_idx"]
                idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}

    num_classes = len(class_to_idx)
    logger.info(f"Loaded {num_classes} classes from checkpoint.")

    # Build model and load weights
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Prepare evaluation DataLoader
    transform = get_val_transforms(args.image_size)
    dataset = FoodImageDataset(
        annotations_file=args.test_csv,
        images_dir=args.images_dir,
        transform=transform,
        class_to_idx=class_to_idx,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()

    logger.info(f"Evaluating {len(dataset)} samples...")
    results = evaluate_model(model, data_loader, criterion, device, idx_to_class)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Evaluated Samples : {results['total_evaluated_samples']}")
    logger.info(f"Average Loss            : {results['loss']:.4f}")
    logger.info(f"Top-1 Accuracy          : {results['top1_accuracy_pct']:.2f}%")
    logger.info(f"Top-5 Accuracy          : {results['top5_accuracy_pct']:.2f}%")
    logger.info("=" * 60)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()

