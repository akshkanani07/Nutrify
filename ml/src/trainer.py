"""Training engine, validation loops, checkpointing, and early stopping."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader

from ml.configs.train_config import PipelineConfig
from ml.src.metrics import AverageMeter, MetricTracker, calculate_topk_accuracy
from ml.src.model import FoodClassifier

logger = logging.getLogger("nutrition_ai.trainer")


class Trainer:
    """Production PyTorch Trainer with validation, checkpointing, and metrics tracking."""

    def __init__(
        self,
        model: FoodClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: PipelineConfig,
        class_to_idx: dict[str, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        self.device = device

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = MetricTracker(save_dir=self.save_dir)

        self.best_metric = float("inf") if config.training.save_best_metric == "val_loss" else 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _build_optimizer(self) -> Optimizer:
        """Initialize optimizer based on configuration."""
        name = self.config.training.optimizer_name.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        # Separate parameters with weight decay from bias and norm layers
        decay_params = []
        no_decay_params = []
        for name_param, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name_param or "bn" in name_param or "norm" in name_param:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if name == "adamw":
            return AdamW(optim_groups, lr=lr)
        elif name == "adam":
            return Adam(optim_groups, lr=lr)
        elif name == "sgd":
            return SGD(optim_groups, lr=lr, momentum=self.config.training.momentum, nesterov=True)
        else:
            logger.warning(f"Unknown optimizer {name}, defaulting to AdamW.")
            return AdamW(optim_groups, lr=lr)

    def _build_scheduler(self) -> _LRScheduler | ReduceLROnPlateau | None:
        """Initialize learning rate scheduler."""
        name = self.config.training.scheduler_name.lower()
        epochs = self.config.training.epochs

        if name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=self.config.training.lr_min)
        elif name == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        elif name == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma,
            )
        return None

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Execute one complete training epoch."""
        self.model.train()
        loss_meter = AverageMeter("Loss", ":.4f")
        top1_meter = AverageMeter("Top1", ":.2f")
        top5_meter = AverageMeter("Top5", ":.2f")

        total_batches = len(self.train_loader)
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()

            if self.config.training.gradient_clip_val:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_val
                )

            self.optimizer.step()

            # Measure accuracy
            top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(top1, batch_size)
            top5_meter.update(top5, batch_size)

            if (batch_idx + 1) % self.config.training.log_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch [{epoch}/{self.config.training.epochs}] "
                    f"Step [{batch_idx + 1}/{total_batches}] | "
                    f"Loss: {loss_meter.val:.4f} (avg: {loss_meter.avg:.4f}) | "
                    f"Top-1: {top1_meter.avg:.2f}% | "
                    f"Top-5: {top5_meter.avg:.2f}% | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        return loss_meter.avg, top1_meter.avg, top5_meter.avg

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        """Run validation evaluation over val_loader."""
        self.model.eval()
        loss_meter = AverageMeter("ValLoss", ":.4f")
        top1_meter = AverageMeter("ValTop1", ":.2f")
        top5_meter = AverageMeter("ValTop5", ":.2f")

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(top1, batch_size)
            top5_meter.update(top5, batch_size)

        return loss_meter.avg, top1_meter.avg, top5_meter.avg

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_top1: float,
        is_best: bool = False,
        filename: str = "last_checkpoint.pt",
    ) -> Path:
        """Persist full model checkpoint with state dicts and metadata."""
        checkpoint_path = self.save_dir / filename
        state = {
            "epoch": epoch,
            "model_name": self.config.model.model_name,
            "num_classes": len(self.class_to_idx),
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "config": self.config.to_dict(),
        }

        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved new best model checkpoint to {best_path}")

        return checkpoint_path

    def fit(self) -> dict[str, Any]:
        """Main training loop orchestrating epochs, validation, and early stopping."""
        logger.info(
            f"Starting training on device: {self.device} for {self.config.training.epochs} epochs"
        )
        logger.info(f"Target classes: {len(self.class_to_idx)} | Batch size: {self.config.data.batch_size}")

        total_start = time.time()

        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start = time.time()

            # Handle dynamic backbone unfreezing if configured
            if (
                self.config.model.unfreeze_at_epoch is not None
                and epoch == self.config.model.unfreeze_at_epoch
            ):
                logger.info(f"Epoch {epoch}: Unfreezing backbone for fine-tuning.")
                self.model.unfreeze_backbone()
                # Rebuild optimizer with lower learning rate for backbone
                self.optimizer = self._build_optimizer()

            # 1. Train
            train_loss, train_top1, train_top5 = self.train_epoch(epoch)

            # 2. Validate
            val_loss, val_top1, val_top5 = self.validate()

            # 3. Step LR Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # 4. Log & Track
            self.tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                train_top5=train_top5,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                learning_rate=current_lr,
                epoch_time_s=epoch_time,
            )

            logger.info(
                f"--> Epoch [{epoch}/{self.config.training.epochs}] Summary: "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_top1:.2f}% | Val Acc: {val_top1:.2f}% (Top-5: {val_top5:.2f}%) | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            # 5. Checkpoint & Best Metric evaluation
            is_best = False
            if self.config.training.save_best_metric == "val_loss":
                if val_loss < (self.best_metric - self.config.training.min_delta):
                    self.best_metric = val_loss
                    self.best_epoch = epoch
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:  # val_acc
                if val_top1 > (self.best_metric + self.config.training.min_delta):
                    self.best_metric = val_top1
                    self.best_epoch = epoch
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, val_top1, is_best=is_best)

            # 6. Early Stopping Check
            if self.patience_counter >= self.config.training.patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs (no improvement for {self.patience_counter} epochs)."
                )
                break

        total_time = time.time() - total_start
        logger.info(
            f"Training finished in {total_time / 60:.1f} minutes. "
            f"Best {self.config.training.save_best_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}."
        )

        return {
            "total_epochs": epoch,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "history": self.tracker.history,
        }

