"""Phase 7: Real Transfer Learning Training Pipeline.

Trains 650-class food classifier strictly on the cleaned real-world photographic dataset.
Techniques:
- Pre-trained EfficientNet-B0 backbone feature extraction
- 2-layer MLP head with Dropout (0.3) & BatchNorm1d
- Optimizer: AdamW (lr=2e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=20, eta_min=1e-5)
- Loss: CrossEntropyLoss with Label Smoothing (0.05)
- Gradient Clipping (max_norm=1.0)
- Early Stopping (patience=5 on validation loss)
- Saves:
  - ml/models/checkpoints/real_train_best.pt
  - ml/models/checkpoints/real_train_last.pt
  - ml/data/reports/real_training_history.json
  - training_report.md
"""

from __future__ import annotations

import json
import logging
import os
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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_DATASET_DIR = PROJECT_ROOT / "ml" / "data" / "real_world_dataset"
CATALOG_PATH = PROJECT_ROOT / "ml" / "knowledge_base" / "expanded_catalog_650.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "checkpoints"
HISTORY_OUTPUT_PATH = PROJECT_ROOT / "ml" / "data" / "reports" / "real_training_history.json"
TRAIN_REPORT_PATH = PROJECT_ROOT / "training_report.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("real_trainer")


def run_real_training(num_epochs: int = 20, batch_size: int = 64) -> dict[str, Any]:
    torch.set_num_threads(4)
    device = torch.device("cpu")
    t0 = time.time()
    logger.info("Executing Phase 7: Real Transfer Learning Training Pipeline...")

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    all_classes = sorted(list(catalog["classes"].keys()))
    num_classes = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for i, c in enumerate(all_classes)}

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Backbone Extractor
    weights = models.EfficientNet_B0_Weights.DEFAULT
    m = models.efficientnet_b0(weights=weights)
    extractor = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
    extractor.eval()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    flip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Partition Real Dataset: 70% Train, 15% Val, 15% Test
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for cname in all_classes:
        c_idx = class_to_idx[cname]
        folder = REAL_DATASET_DIR / cname.replace(" ", "_")
        if not folder.exists():
            continue
        imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in valid_exts])
        n = len(imgs)
        if n >= 3:
            train_samples.extend([(p, c_idx) for p in imgs[: int(n * 0.70)]])
            val_samples.extend([(p, c_idx) for p in imgs[int(n * 0.70) : int(n * 0.85)]])
            test_samples.extend([(p, c_idx) for p in imgs[int(n * 0.85) :]])
        elif n == 2:
            train_samples.append((imgs[0], c_idx))
            val_samples.append((imgs[1], c_idx))
            test_samples.append((imgs[1], c_idx))
        elif n == 1:
            train_samples.append((imgs[0], c_idx))
            val_samples.append((imgs[0], c_idx))
            test_samples.append((imgs[0], c_idx))

    logger.info(f"Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    def extract_dataset_features(samples: list[tuple[Path, int]], desc: str, augment: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        feats_list = []
        labels_list = []
        total = len(samples)
        logger.info(f"Extracting features for {desc} ({total} samples, augment={augment})...")
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch = samples[i : i + batch_size]
                tensors = []
                targets = []
                for p, target in batch:
                    try:
                        with Image.open(p) as img:
                            rgb = img.convert("RGB")
                            tensors.append(val_transform(rgb))
                            targets.append(target)
                            if augment:
                                tensors.append(flip_transform(rgb))
                                targets.append(target)
                    except Exception:
                        pass
                if not tensors:
                    continue
                bx = torch.stack(tensors)
                f = extractor(bx)
                feats_list.append(f)
                labels_list.extend(targets)
        return torch.cat(feats_list, dim=0), torch.tensor(labels_list, dtype=torch.long)

    train_feats, train_targets = extract_dataset_features(train_samples, "Train Set", augment=True)
    val_feats, val_targets = extract_dataset_features(val_samples, "Val Set", augment=False)
    test_feats, test_targets = extract_dataset_features(test_samples, "Test Set", augment=False)

    logger.info(f"Extracted feature matrices: Train={train_feats.shape}, Val={val_feats.shape}, Test={test_feats.shape}")

    # 3. Enhanced Classifier Head
    class ProductionHead(nn.Module):
        def __init__(self, in_features: int = 1280, hidden_dim: int = 512, num_classes: int = 650):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            return x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            return x

    head = ProductionHead(in_features=1280, hidden_dim=512, num_classes=num_classes)

    optimizer = AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_loader = DataLoader(TensorDataset(train_feats, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_targets), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 5
    patience_counter = 0

    history: list[dict[str, Any]] = []

    logger.info("Starting training loop with early stopping...")
    for ep in range(1, num_epochs + 1):
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += bx.size(0)

        scheduler.step()
        train_acc = (train_correct / train_total) * 100.0 if train_total else 0.0
        avg_train_loss = train_loss / train_total if train_total else 0.0

        # Validation
        head.eval()
        val_loss = 0.0
        val_correct_1 = 0
        val_correct_5 = 0
        val_total = 0

        with torch.no_grad():
            for bx, by in val_loader:
                logits = head(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)

                top5 = torch.topk(logits, k=5, dim=-1).indices
                preds = top5[:, 0]
                val_correct_1 += (preds == by).sum().item()
                val_correct_5 += sum(by[i].item() in top5[i].tolist() for i in range(by.size(0)))
                val_total += bx.size(0)

        avg_val_loss = val_loss / val_total if val_total else 0.0
        val_top1 = (val_correct_1 / val_total) * 100.0 if val_total else 0.0
        val_top5 = (val_correct_5 / val_total) * 100.0 if val_total else 0.0

        current_lr = scheduler.get_last_lr()[0]
        is_best = avg_val_loss < best_val_loss
        status_msg = "No change"
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = ep
            patience_counter = 0
            status_msg = f"BEST UPDATED (val_loss={best_val_loss:.4f})"
            # Save best checkpoint
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": {
                        **{f"extractor.{k}": v for k, v in extractor.state_dict().items()},
                        **{f"head.{k}": v for k, v in head.state_dict().items()},
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                },
                CHECKPOINTS_DIR / "real_train_best.pt",
            )
        else:
            patience_counter += 1

        logger.info(
            f"Epoch [{ep:2d}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:5.2f}%) | "
            f"Val Loss: {avg_val_loss:.4f} (Top-1: {val_top1:5.2f}%, Top-5: {val_top5:5.2f}%) | "
            f"LR: {current_lr:.6f} | {status_msg}"
        )

        history.append({
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(avg_val_loss, 4),
            "val_top1": round(val_top1, 2),
            "val_top5": round(val_top5, 2),
            "learning_rate": round(current_lr, 7),
            "is_best": is_best,
        })

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {ep} (patience={patience})")
            break

    # Save last checkpoint
    torch.save(
        {
            "epoch": ep,
            "model_state_dict": {
                **{f"extractor.{k}": v for k, v in extractor.state_dict().items()},
                **{f"head.{k}": v for k, v in head.state_dict().items()},
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        },
        CHECKPOINTS_DIR / "real_train_last.pt",
    )

    # Save history
    training_summary = {
        "title": "Phase 7 Real Transfer Learning Execution Summary",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "total_duration_seconds": round(time.time() - t0, 2),
        "history": history,
    }

    with open(HISTORY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    # Generate training_report.md
    with open(TRAIN_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Nutrition AI — Phase 7 Real Model Training Report\n\n")
        f.write(f"- **Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"- **Dataset:** Cleaned Real-World Photographic Food Dataset (5,286 images)\n")
        f.write(f"- **Architecture:** EfficientNet-B0 + 2-Layer Enhanced MLP Head (512-d hidden, Dropout 0.3)\n")
        f.write(f"- **Classes:** {num_classes}\n")
        f.write(f"- **Optimizer:** AdamW (lr=2e-3, weight_decay=1e-4) with CosineAnnealingLR\n")
        f.write(f"- **Loss Function:** CrossEntropy with Label Smoothing (0.05)\n")
        f.write(f"- **Total Epochs:** {len(history)} (Best Epoch: {best_epoch})\n")
        f.write(f"- **Best Validation Loss:** {best_val_loss:.4f}\n")
        f.write(f"- **Checkpoints Saved:**\n")
        f.write(f"  - `ml/models/checkpoints/real_train_best.pt`\n")
        f.write(f"  - `ml/models/checkpoints/real_train_last.pt`\n\n")
        f.write("## Empirical Training History\n\n")
        f.write("| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Top-1 (%) | Val Top-5 (%) | Learning Rate |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for h in history:
            star = " ★" if h["is_best"] else ""
            f.write(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.2f}% | {h['val_loss']:.4f}{star} | {h['val_top1']:.2f}% | {h['val_top5']:.2f}% | {h['learning_rate']:.6f} |\n")

    # Also deploy best checkpoint to production locations
    prod_dir = PROJECT_ROOT / "ml" / "models" / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(CHECKPOINTS_DIR / "real_train_best.pt", prod_dir / "nutrition_model_650_realworld.pt")
    shutil.copy(CHECKPOINTS_DIR / "real_train_best.pt", prod_dir / "nutrition_model_650_best.pt")
    logger.info("Successfully deployed best checkpoint to production models.")

    logger.info(f"Saved training history to {HISTORY_OUTPUT_PATH.resolve()}")
    logger.info(f"Saved training report to {TRAIN_REPORT_PATH.resolve()}")
    return training_summary


if __name__ == "__main__":
    run_real_training()
