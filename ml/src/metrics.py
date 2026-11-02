"""Evaluation metrics and training progress tracking."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "Metric", fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} (avg: {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def calculate_topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> list[float]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    if batch_size == 0:
        return [0.0] * len(topk)

    # If num_classes < maxk, cap maxk
    num_classes = output.size(1)
    effective_maxk = min(maxk, num_classes)

    _, pred = output.topk(effective_maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        effective_k = min(k, num_classes)
        correct_k = correct[:effective_k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return res


class MetricTracker:
    """Logs and persists epoch-by-epoch loss and accuracy metrics."""

    def __init__(self, save_dir: str | Path | None = None) -> None:
        self.save_dir = Path(save_dir) if save_dir else None
        self.history: list[dict[str, Any]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_top1: float,
        train_top5: float,
        val_loss: float,
        val_top1: float,
        val_top5: float,
        learning_rate: float,
        epoch_time_s: float = 0.0,
    ) -> None:
        """Record metrics for an epoch."""
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 2),
            "train_top5": round(train_top5, 2),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 2),
            "val_top5": round(val_top5, 2),
            "learning_rate": learning_rate,
            "time_seconds": round(epoch_time_s, 2),
        }
        self.history.append(record)

        if self.save_dir:
            self.save_csv(self.save_dir / "training_log.csv")
            self.save_json(self.save_dir / "training_history.json")

    def save_csv(self, path: str | Path) -> None:
        """Export history to CSV table."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            return

        fieldnames = list(self.history[0].keys())
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def save_json(self, path: str | Path) -> None:
        """Export history to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get_best(self, metric: str = "val_loss", mode: str = "min") -> dict[str, Any] | None:
        """Retrieve best epoch record according to chosen metric."""
        if not self.history:
            return None
        if mode == "min":
            return min(self.history, key=lambda x: x.get(metric, float("inf")))
        return max(self.history, key=lambda x: x.get(metric, float("-inf")))

