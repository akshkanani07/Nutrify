"""Complete Two-Stage Training Pipeline for Nutrition AI EfficientNet-B0.

Stage 1:
  - Freeze EfficientNet-B0 backbone
  - Train classifier head for 10 epochs
  - AdamW optimizer + Cosine Annealing scheduler
  - Save best checkpoint to ml/models/checkpoints/stage1_best.pt
  - Evaluate and report: Training loss, Validation loss, Top-1 accuracy, Top-5 accuracy

Stage 2:
  - Unfreeze backbone
  - Fine-tune model with lower learning rate
  - Save final production model to ml/models/production/nutrition_model_b0.pt and ml/models/checkpoints/best_model.pt
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset import FoodImageDataset, get_train_transforms, get_val_transforms
from ml.src.metrics import AverageMeter, calculate_topk_accuracy
from ml.src.model import build_model
from ml.src.utils import get_device, set_seed, setup_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nutrition_ai.training_run")


def load_annotation_subset(csv_path: str, max_per_class: int | None = None) -> list[tuple[str, str]]:
    """Load annotations from CSV, optionally capping samples per class for balanced training."""
    by_class = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            by_class[r["label"]].append(r["image_path"])

    samples = []
    for label, paths in sorted(by_class.items()):
        selected = paths[:max_per_class] if max_per_class else paths
        for p in selected:
            samples.append((p, label))
    return samples


def extract_dataset_features(
    model: nn.Module,
    samples: list[tuple[str, str]],
    class_to_idx: dict[str, int],
    images_dir: Path,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 1280-d features through frozen EfficientNet-B0 backbone."""
    transform = get_val_transforms(224)
    features_list = []
    labels_list = []

    total = len(samples)
    model.eval()

    logger.info(f"Extracting frozen backbone features for {total} images (batch size {batch_size})...")
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch_samples = samples[i : i + batch_size]
        batch_imgs = []
        batch_lbls = []

        for rel_path, label in batch_samples:
            img_path = images_dir / rel_path
            try:
                with Image.open(img_path) as img:
                    tensor = transform(img.convert("RGB"))
                    batch_imgs.append(tensor)
                    batch_lbls.append(class_to_idx[label])
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")

        if not batch_imgs:
            continue

        imgs_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feat = model.extract_features(imgs_tensor)
            feat = feat.cpu()

        features_list.append(feat)
        labels_list.extend(batch_lbls)

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= total:
            pct = min(100.0, (i + len(batch_samples)) / total * 100)
            elapsed = time.time() - start_time
            logger.info(f"  Feature extraction: {pct:5.1f}% ({min(i + batch_size, total)}/{total}) in {elapsed:.1f}s")

    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)
    logger.info(f"Features extracted: shape {all_features.shape} in {time.time() - start_time:.1f}s")
    return all_features, all_labels


def main() -> None:
    set_seed(42)
    torch.set_num_threads(4)

    device = get_device("auto")
    logger.info("=" * 65)
    logger.info(f"Starting Nutrition AI EfficientNet-B0 Model Training on {device}")
    logger.info("=" * 65)

    images_dir = Path("ml/data/images")
    train_csv = "ml/data/annotations/train.csv"
    val_csv = "ml/data/annotations/val.csv"
    test_csv = "ml/data/annotations/test.csv"

    # Build class mapping from train.csv
    train_samples_all = load_annotation_subset(train_csv)
    classes = sorted(list(set(label for _, label in train_samples_all)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)

    logger.info(f"Discovered {num_classes} classes from {len(train_samples_all)} total training samples.")

    # Select balanced subsets for efficient CPU transfer learning:
    # 50 classes x 50 images = 2,500 train images (rich variation, fast on CPU)
    # 50 classes x 15 images = 750 val images
    train_samples = load_annotation_subset(train_csv, max_per_class=50)
    val_samples = load_annotation_subset(val_csv, max_per_class=15)
    test_samples = load_annotation_subset(test_csv, max_per_class=15)

    logger.info(f"Training subset   : {len(train_samples)} images ({len(train_samples)//num_classes} per class)")
    logger.info(f"Validation subset : {len(val_samples)} images ({len(val_samples)//num_classes} per class)")
    logger.info(f"Test subset       : {len(test_samples)} images ({len(test_samples)//num_classes} per class)")

    # Output directories
    checkpoints_dir = Path("ml/models/checkpoints")
    production_dir = Path("ml/models/production")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    production_dir.mkdir(parents=True, exist_ok=True)

    # Save class mappings
    with open(checkpoints_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)

    # Initialize model
    model = build_model(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    model.to(device)

    # =========================================================================
    # STAGE 1: Freeze EfficientNet backbone, Train classifier head (10 epochs)
    # =========================================================================
    logger.info("\n" + "=" * 65)
    logger.info("STAGE 1: Freeze EfficientNet Backbone -- Train Classifier Head")
    logger.info("Epochs: 10 | Optimizer: AdamW | Scheduler: CosineAnnealing")
    logger.info("=" * 65)

    # Extract backbone features
    train_feats, train_labels = extract_dataset_features(
        model, train_samples, class_to_idx, images_dir, device, batch_size=64
    )
    val_feats, val_labels = extract_dataset_features(
        model, val_samples, class_to_idx, images_dir, device, batch_size=64
    )

    feat_train_ds = TensorDataset(train_feats, train_labels)
    feat_val_ds = TensorDataset(val_feats, val_labels)

    feat_train_loader = DataLoader(feat_train_ds, batch_size=32, shuffle=True)
    feat_val_loader = DataLoader(feat_val_ds, batch_size=32, shuffle=False)

    classifier = model.backbone.classifier.to(device)
    optimizer = AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_top1 = 0.0
    stage1_history = []

    logger.info("\nStarting Stage 1 Training Loop:")
    logger.info("-" * 65)

    for epoch in range(1, 11):
        classifier.train()
        train_loss_meter = AverageMeter("Loss", ":.4f")
        train_top1_meter = AverageMeter("Top1", ":.2f")
        train_top5_meter = AverageMeter("Top5", ":.2f")

        for feats, targets in feat_train_loader:
            feats, targets = feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = classifier(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            bsz = feats.size(0)
            train_loss_meter.update(loss.item(), bsz)
            train_top1_meter.update(top1, bsz)
            train_top5_meter.update(top5, bsz)

        scheduler.step()

        # Validation
        classifier.eval()
        val_loss_meter = AverageMeter("ValLoss", ":.4f")
        val_top1_meter = AverageMeter("ValTop1", ":.2f")
        val_top5_meter = AverageMeter("ValTop5", ":.2f")

        with torch.no_grad():
            for feats, targets in feat_val_loader:
                feats, targets = feats.to(device), targets.to(device)
                outputs = classifier(feats)
                loss = criterion(outputs, targets)

                top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
                bsz = feats.size(0)
                val_loss_meter.update(loss.item(), bsz)
                val_top1_meter.update(top1, bsz)
                val_top5_meter.update(top5, bsz)

        pct = (epoch / 10) * 100
        is_best = val_loss_meter.avg < best_val_loss

        checkpoint_status = "No change"
        if is_best:
            best_val_loss = val_loss_meter.avg
            best_val_top1 = val_top1_meter.avg
            checkpoint_status = f"BEST UPDATED (val_loss={best_val_loss:.4f})"

            # Save stage 1 best checkpoint
            stage1_ckpt_path = checkpoints_dir / "stage1_best.pt"
            torch.save(
                {
                    "stage": 1,
                    "epoch": epoch,
                    "model_name": "efficientnet_b0",
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss_meter.avg,
                    "val_top1": val_top1_meter.avg,
                    "val_top5": val_top5_meter.avg,
                },
                stage1_ckpt_path,
            )

        logger.info(
            f"Epoch [{epoch:2d}/10] ({pct:3.0f}%) | "
            f"Train Loss: {train_loss_meter.avg:.4f} | Train Acc: {train_top1_meter.avg:5.2f}% | "
            f"Val Loss: {val_loss_meter.avg:.4f} | Val Acc: {val_top1_meter.avg:5.2f}% (Top-5: {val_top5_meter.avg:5.2f}%) | "
            f"Checkpoint: {checkpoint_status}"
        )

        stage1_history.append({
            "epoch": epoch,
            "train_loss": train_loss_meter.avg,
            "train_top1": train_top1_meter.avg,
            "val_loss": val_loss_meter.avg,
            "val_top1": val_top1_meter.avg,
            "val_top5": val_top5_meter.avg,
        })

    # =========================================================================
    # STAGE 1 EVALUATION: Report required metrics
    # =========================================================================
    logger.info("\n" + "=" * 65)
    logger.info("STAGE 1 EVALUATION REPORT")
    logger.info("=" * 65)
    last_ep = stage1_history[-1]
    logger.info(f"Training Loss     : {last_ep['train_loss']:.4f}")
    logger.info(f"Validation Loss   : {best_val_loss:.4f}")
    logger.info(f"Top-1 Accuracy    : {best_val_top1:.2f}%")
    logger.info(f"Top-5 Accuracy    : {last_ep['val_top5']:.2f}%")
    logger.info(f"Best Checkpoint   : {checkpoints_dir / 'stage1_best.pt'}")
    logger.info("=" * 65)

    # =========================================================================
    # STAGE 2: Unfreeze backbone & Fine-tune with lower learning rate
    # =========================================================================
    logger.info("\n" + "=" * 65)
    logger.info("STAGE 2: Unfreeze Backbone & Fine-Tune Model")
    logger.info("Unfreezing top backbone layers for fine-tuning...")
    logger.info("=" * 65)

    # Unfreeze backbone
    model.unfreeze_backbone()

    # Differential learning rate: lower for backbone features, moderate for classifier head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_ft = AdamW(
        [
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=3, eta_min=1e-6)

    # Load 10 images per class for fast, effective CPU end-to-end fine-tuning
    ft_samples = load_annotation_subset(train_csv, max_per_class=10)
    transform_train = get_train_transforms(224)
    transform_val = get_val_transforms(224)

    class FastDataset(Dataset):
        def __init__(self, samples_list, transform_fn):
            self.samples = samples_list
            self.transform = transform_fn

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            rel_p, lbl = self.samples[idx]
            with Image.open(images_dir / rel_p) as img:
                return self.transform(img.convert("RGB")), class_to_idx[lbl]

    ft_train_loader = DataLoader(
        FastDataset(ft_samples, transform_train), batch_size=32, shuffle=True
    )
    ft_val_loader = DataLoader(
        FastDataset(val_samples[:250], transform_val), batch_size=32, shuffle=False
    )

    stage2_epochs = 3
    logger.info(f"Fine-tuning for {stage2_epochs} epochs over {len(ft_samples)} diverse images...")
    logger.info("-" * 65)

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        train_loss_meter = AverageMeter("Loss", ":.4f")
        train_top1_meter = AverageMeter("Top1", ":.2f")
        train_top5_meter = AverageMeter("Top5", ":.2f")

        for batch_idx, (imgs, targets) in enumerate(ft_train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer_ft.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_ft.step()

            top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            bsz = imgs.size(0)
            train_loss_meter.update(loss.item(), bsz)
            train_top1_meter.update(top1, bsz)
            train_top5_meter.update(top5, bsz)

        scheduler_ft.step()

        # Validation
        model.eval()
        val_loss_meter = AverageMeter("ValLoss", ":.4f")
        val_top1_meter = AverageMeter("ValTop1", ":.2f")
        val_top5_meter = AverageMeter("ValTop5", ":.2f")

        with torch.no_grad():
            for imgs, targets in ft_val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
                bsz = imgs.size(0)
                val_loss_meter.update(loss.item(), bsz)
                val_top1_meter.update(top1, bsz)
                val_top5_meter.update(top5, bsz)

        pct = (epoch / stage2_epochs) * 100
        logger.info(
            f"Stage 2 Epoch [{epoch:2d}/{stage2_epochs}] ({pct:3.0f}%) | "
            f"Train Loss: {train_loss_meter.avg:.4f} | Train Acc: {train_top1_meter.avg:5.2f}% | "
            f"Val Loss: {val_loss_meter.avg:.4f} | Val Acc: {val_top1_meter.avg:5.2f}% (Top-5: {val_top5_meter.avg:5.2f}%) | "
            f"Checkpoint: Updated"
        )

    # =========================================================================
    # SAVE PRODUCTION MODEL
    # =========================================================================
    logger.info("\n" + "=" * 65)
    logger.info("SAVING FINAL PRODUCTION MODEL")
    logger.info("=" * 65)

    prod_checkpoint_path = production_dir / "nutrition_model_b0.pt"
    best_checkpoint_path = checkpoints_dir / "best_model.pt"

    final_payload = {
        "model_name": "efficientnet_b0",
        "num_classes": num_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "model_state_dict": model.state_dict(),
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "final_val_loss": val_loss_meter.avg,
        "final_val_top1": val_top1_meter.avg,
        "final_val_top5": val_top5_meter.avg,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    torch.save(final_payload, prod_checkpoint_path)
    torch.save(final_payload, best_checkpoint_path)

    logger.info(f"Saved production model to: {prod_checkpoint_path.resolve()}")
    logger.info(f"Saved best checkpoint to  : {best_checkpoint_path.resolve()}")

    # Model info metadata JSON
    info_json_path = production_dir / "model_info.json"
    with open(info_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": "EfficientNet-B0 Transfer Learning",
                "num_classes": num_classes,
                "classes": classes,
                "val_loss": round(val_loss_meter.avg, 4),
                "top1_accuracy_pct": round(val_top1_meter.avg, 2),
                "top5_accuracy_pct": round(val_top5_meter.avg, 2),
                "checkpoint": str(prod_checkpoint_path),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved model metadata JSON to: {info_json_path.resolve()}")

    # Quick test inference verification
    logger.info("\nVerifying production inference on a test image...")
    sample_test_path = images_dir / test_samples[0][0]
    with Image.open(sample_test_path) as img:
        test_t = transform_val(img.convert("RGB")).unsqueeze(0).to(device)
    topk_idx, topk_probs = model.predict_topk(test_t, k=3)
    preds = [f"{idx_to_class[idx.item()]} ({prob.item()*100:.1f}%)" for idx, prob in zip(topk_idx[0], topk_probs[0])]
    logger.info(f"Test sample: {test_samples[0][0]} (True label: {test_samples[0][1]})")
    logger.info(f"Predictions: {', '.join(preds)}")

    logger.info("\n" + "=" * 65)
    logger.info("ALL TRAINING PHASES COMPLETED SUCCESSFULLY")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

