"""CLI Training Script for Nutrition AI PyTorch Pipeline.

Usage:
    python ml/train.py --annotations-csv ml/data/annotations/annotations.csv --images-dir ml/data/images
    python ml/train.py --train-csv ml/data/annotations/train.csv --val-csv ml/data/annotations/val.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.configs.train_config import PipelineConfig
from ml.src.dataset import create_dataloaders
from ml.src.model import build_model
from ml.src.trainer import Trainer
from ml.src.utils import count_parameters, get_device, save_json, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Food & Drink Dataset")

    # Configuration file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML or JSON config file")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="ml/data", help="Root data folder")
    parser.add_argument(
        "--annotations-csv",
        type=str,
        default="ml/data/annotations/annotations.csv",
        help="Path to single CSV with image annotations",
    )
    parser.add_argument("--train-csv", type=str, default=None, help="Path to separate train CSV")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to separate val CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to separate test CSV")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="ml/data/images",
        help="Root directory containing image files",
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "step", "none"])
    parser.add_argument("--image-size", type=int, default=224, help="Input image dimension (224 for B0)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader subprocess workers")

    # Model & Transfer learning
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pre-trained weights")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout probability")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone feature extractor")

    # System & Output
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--save-dir", type=str, default="ml/models/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random reproducibility seed")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("nutrition_ai.train")

    logger.info("=" * 60)
    logger.info("Starting Nutrition AI Model Training Pipeline")
    logger.info("=" * 60)

    # 1. Load base configuration or default
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = PipelineConfig.load_yaml(args.config)
    else:
        config = PipelineConfig()

    # 2. Apply CLI overrides
    if args.train_csv:
        config.data.train_csv = args.train_csv
    if args.val_csv:
        config.data.val_csv = args.val_csv
    if args.test_csv:
        config.data.test_csv = args.test_csv
    if args.annotations_csv:
        config.data.annotations_csv = args.annotations_csv
    if args.images_dir:
        config.data.images_dir = args.images_dir

    config.data.batch_size = args.batch_size
    config.data.image_size = args.image_size
    config.data.num_workers = args.num_workers
    config.training.epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.optimizer_name = args.optimizer
    config.training.scheduler_name = args.scheduler
    config.training.save_dir = args.save_dir
    config.training.device = args.device
    config.model.pretrained = not args.no_pretrained
    config.model.dropout = args.dropout
    config.model.freeze_backbone = args.freeze_backbone

    # 3. Environment & Hardware setup
    set_seed(args.seed)
    device = get_device(config.training.device)
    logger.info(f"Target compute hardware: {device}")

    # 4. DataLoaders creation
    train_source = config.data.train_csv or config.data.annotations_csv
    val_source = config.data.val_csv

    logger.info(f"Initializing datasets from {train_source}")
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        train_annotations=train_source,
        val_annotations=val_source,
        test_annotations=config.data.test_csv,
        images_dir=config.data.images_dir,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        random_seed=args.seed,
    )

    num_classes = len(class_to_idx)
    config.model.num_classes = num_classes
    logger.info(f"Total training classes detected: {num_classes}")

    # Save class mapping alongside checkpoints
    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    save_json({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, save_dir / "classes.json")
    config.save_json(save_dir / "train_config.json")

    # 5. Model building
    model = build_model(
        num_classes=num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
    )

    params = count_parameters(model)
    logger.info(
        f"Model parameters: Total={params['total']:,} | "
        f"Trainable={params['trainable']:,} | Frozen={params['frozen']:,}"
    )

    # 6. Trainer initialization and execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_to_idx=class_to_idx,
        device=device,
    )

    results = trainer.fit()

    logger.info("=" * 60)
    logger.info("Training Run Completed Successfully")
    logger.info(f"Best Validation Metric: {results['best_metric']:.4f} at epoch {results['best_epoch']}")
    logger.info(f"Checkpoints and artifacts saved in: {save_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

