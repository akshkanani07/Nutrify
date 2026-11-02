"""Configuration management for PyTorch Nutrition AI training pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Dataset and data loader configurations."""

    data_dir: str = "ml/data"
    annotations_csv: str = "ml/data/annotations/annotations.csv"
    train_csv: str | None = None
    val_csv: str | None = None
    test_csv: str | None = None
    images_dir: str = "ml/data/images"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    val_split_ratio: float = 0.15
    test_split_ratio: float = 0.10
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture and transfer learning configurations."""

    model_name: str = "efficientnet_b0"
    num_classes: int = 1051
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False
    unfreeze_at_epoch: int | None = None


@dataclass
class TrainingConfig:
    """Optimization and training hyperparameter configurations."""

    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"  # 'adamw', 'adam', 'sgd'
    momentum: float = 0.9  # for sgd
    scheduler_name: str = "cosine"  # 'cosine', 'plateau', 'step', 'none'
    lr_min: float = 1e-6
    step_size: int = 7
    gamma: float = 0.1
    patience: int = 5
    min_delta: float = 1e-4
    gradient_clip_val: float | None = 1.0
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'
    save_dir: str = "ml/models/checkpoints"
    save_best_metric: str = "val_loss"  # 'val_loss' or 'val_acc'
    log_interval: int = 10


@dataclass
class PipelineConfig:
    """Composite configuration for complete training pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create PipelineConfig from nested dictionary."""
        data_cfg = DataConfig(**data.get("data", {}))
        model_cfg = ModelConfig(**data.get("model", {}))
        train_cfg = TrainingConfig(**data.get("training", {}))
        return cls(data=data_cfg, model=model_cfg, training=train_cfg)

    def save_json(self, path: str | Path) -> None:
        """Save configuration as formatted JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration as YAML if pyyaml is installed, else JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except ImportError:
            self.save_json(path.with_suffix(".json"))

    @classmethod
    def load_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except ImportError:
            return cls.load_json(path)

