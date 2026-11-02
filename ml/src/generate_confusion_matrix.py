"""Generate confusion matrix and analyze classification errors on test set."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset import get_val_transforms
from ml.src.model import build_model

def main():
    torch.set_num_threads(4)
    device = torch.device("cpu")

    ckpt_path = Path("ml/models/checkpoints/stage1_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_val_transforms(224)
    images_dir = Path("ml/data/images")

    with open("ml/data/annotations/test.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    total_samples = len(rows)
    correct_samples = 0
    top5_correct = 0

    batch_size = 64
    for i in range(0, total_samples, batch_size):
        batch = rows[i : i + batch_size]
        batch_tensors = []
        batch_targets = []
        for r in batch:
            p = images_dir / r["image_path"]
            lbl = r["label"]
            target_idx = class_to_idx[lbl]
            with Image.open(p) as img:
                batch_tensors.append(transform(img.convert("RGB")))
            batch_targets.append(target_idx)

        t_batch = torch.stack(batch_tensors)
        with torch.no_grad():
            logits = model(t_batch)
            probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(probs, dim=-1).tolist()
        top5 = torch.topk(probs, k=5, dim=-1).indices.tolist()

        for true_idx, pred_idx, top5_preds in zip(batch_targets, preds, top5):
            confusion_matrix[true_idx, pred_idx] += 1
            if true_idx == pred_idx:
                correct_samples += 1
            if true_idx in top5_preds:
                top5_correct += 1

    top1_acc = (correct_samples / total_samples) * 100
    top5_acc = (top5_correct / total_samples) * 100

    print(f"Test Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Test Top-5 Accuracy: {top5_acc:.2f}%")

    # Find top error pairs (where true != pred)
    error_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                error_pairs.append({
                    "true_class": idx_to_class[i],
                    "pred_class": idx_to_class[j],
                    "count": int(confusion_matrix[i, j]),
                    "pct_of_class": round(float(confusion_matrix[i, j] / confusion_matrix[i].sum() * 100), 1),
                })

    error_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Save report
    out_dir = Path("ml/data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "total_test_samples": total_samples,
        "top1_accuracy": round(top1_acc, 2),
        "top5_accuracy": round(top5_acc, 2),
        "top_10_errors": error_pairs[:10],
        "confusion_matrix": confusion_matrix.tolist(),
        "classes": [idx_to_class[i] for i in range(num_classes)],
    }
    with open(out_dir / "confusion_matrix_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- TOP 10 CLASSIFICATION ERRORS ---")
    for idx, err in enumerate(error_pairs[:10], 1):
        print(f"{idx:2d}. True: '{err['true_class']}' -> Predicted: '{err['pred_class']}' ({err['count']} errors, {err['pct_of_class']}%)")

if __name__ == "__main__":
    main()
