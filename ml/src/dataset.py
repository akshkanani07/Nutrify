"""PyTorch Dataset and DataLoader pipeline for Food & Drink Recognition."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

logger = logging.getLogger("nutrition_ai.dataset")


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Standard transfer-learning augmentation pipeline for training."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Validation and evaluation transform pipeline with deterministic cropping."""
    crop_size = image_size
    resize_size = int(image_size * 1.14)  # 256 for 224 crop
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(image_size: int = 224) -> T.Compose:
    """Transform pipeline for single-image or batch production inference."""
    return get_val_transforms(image_size)


class FoodImageDataset(Dataset):
    """PyTorch Dataset for food images with CSV annotation metadata."""

    def __init__(
        self,
        annotations_file: str | Path,
        images_dir: str | Path | None = None,
        transform: Callable | None = None,
        class_to_idx: dict[str, int] | None = None,
        split_filter: str | None = None,
    ) -> None:
        """Initialize dataset from CSV.

        Args:
            annotations_file: Path to CSV containing image paths and labels.
            images_dir: Root directory for relative image paths.
            transform: Optional torchvision transforms.
            class_to_idx: Pre-existing class mapping (e.g. from training set).
            split_filter: Optional filter for 'split' column (e.g. 'train', 'val', 'test').
        """
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir) if images_dir else self.annotations_file.parent
        self.transform = transform
        self.split_filter = split_filter.lower() if split_filter else None

        self.samples: list[dict[str, Any]] = []
        self._load_annotations(class_to_idx)

    def _detect_columns(self, fieldnames: list[str]) -> tuple[str, str, str | None]:
        """Detect image path, label, and split column names flexibly."""
        lower_map = {name.lower().strip(): name for name in fieldnames}

        # Candidate names for image path
        path_col = None
        for cand in ["image_path", "filepath", "filename", "path", "image", "image_id", "file"]:
            if cand in lower_map:
                path_col = lower_map[cand]
                break

        # Candidate names for label
        label_col = None
        for cand in ["label", "food_name", "normalised_name", "class", "category", "class_name"]:
            if cand in lower_map:
                label_col = lower_map[cand]
                break

        # Candidate names for split
        split_col = None
        for cand in ["split", "subset", "dataset_split", "phase"]:
            if cand in lower_map:
                split_col = lower_map[cand]
                break

        if not path_col or not label_col:
            raise ValueError(
                f"CSV must contain image path and label columns. Found fields: {fieldnames}"
            )

        return path_col, label_col, split_col

    def _load_annotations(self, class_to_idx: dict[str, int] | None) -> None:
        """Read CSV and build sample index and class maps."""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Empty or headerless CSV: {self.annotations_file}")

            path_col, label_col, split_col = self._detect_columns(reader.fieldnames)

            raw_samples = []
            unique_labels = set()

            for row in reader:
                # Apply split filter if requested
                if self.split_filter and split_col:
                    row_split = row.get(split_col, "").strip().lower()
                    if row_split and row_split != self.split_filter:
                        continue

                img_path_str = row[path_col].strip()
                label_str = row[label_col].strip()

                if not img_path_str or not label_str:
                    continue

                raw_samples.append((img_path_str, label_str, row))
                unique_labels.add(label_str)

        # Build or use class_to_idx
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            sorted_labels = sorted(unique_labels)
            self.class_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

        self.idx_to_class = {idx: lbl for lbl, idx in self.class_to_idx.items()}

        # Verify images and build verified samples
        missing_count = 0
        for img_rel_path, label_str, raw_row in raw_samples:
            if label_str not in self.class_to_idx:
                continue

            full_path = self.images_dir / img_rel_path
            # If not relative to images_dir, check absolute path
            if not full_path.exists() and Path(img_rel_path).exists():
                full_path = Path(img_rel_path)

            if not full_path.exists():
                missing_count += 1
                continue

            self.samples.append({
                "path": str(full_path),
                "label": label_str,
                "label_idx": self.class_to_idx[label_str],
                "metadata": raw_row,
            })

        if missing_count > 0:
            logger.warning(
                f"Skipped {missing_count} missing image files in {self.annotations_file}"
            )

        logger.info(
            f"Loaded {len(self.samples)} samples across {len(self.class_to_idx)} classes from {self.annotations_file}"
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def classes(self) -> list[str]:
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

    def save_class_mapping(self, out_path: str | Path) -> None:
        """Export class mapping to JSON."""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"class_to_idx": self.class_to_idx, "idx_to_class": self.idx_to_class}, f, indent=2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img_path = sample["path"]
        label_idx = sample["label_idx"]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Fallback black image in case of I/O error
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


def create_dataloaders(
    train_annotations: str | Path,
    val_annotations: str | Path | None = None,
    test_annotations: str | Path | None = None,
    images_dir: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_split_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, int]]:
    """Factory creating training, validation, and optional test DataLoaders."""
    train_path = Path(train_annotations)

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Scenario 1: Explicit val_annotations provided
    if val_annotations is not None:
        train_ds = FoodImageDataset(
            annotations_file=train_path,
            images_dir=images_dir,
            transform=train_transform,
        )
        val_ds = FoodImageDataset(
            annotations_file=Path(val_annotations),
            images_dir=images_dir,
            transform=val_transform,
            class_to_idx=train_ds.class_to_idx,
        )
    else:
        # Scenario 2: Single CSV with split column or automatic random split
        # Check if single CSV has 'split' column
        has_split_column = False
        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.lower().strip() for h in next(reader, [])]
            has_split_column = any(s in header for s in ["split", "subset", "dataset_split", "phase"])

        if has_split_column:
            train_ds = FoodImageDataset(
                annotations_file=train_path,
                images_dir=images_dir,
                transform=train_transform,
                split_filter="train",
            )
            val_ds = FoodImageDataset(
                annotations_file=train_path,
                images_dir=images_dir,
                transform=val_transform,
                class_to_idx=train_ds.class_to_idx,
                split_filter="val",
            )
        else:
            # Build full dataset and split indices
            full_ds = FoodImageDataset(
                annotations_file=train_path,
                images_dir=images_dir,
                transform=None,
            )
            total_samples = len(full_ds)
            val_size = int(total_samples * val_split_ratio)
            train_size = total_samples - val_size

            generator = torch.Generator().manual_seed(random_seed)
            train_sub, val_sub = torch.utils.data.random_split(
                full_ds, [train_size, val_size], generator=generator
            )

            # Subsets wrapping transformed dataset
            class TransformedSubset(Dataset):
                def __init__(self, subset, transform):
                    self.subset = subset
                    self.transform = transform

                def __len__(self):
                    return len(self.subset)

                def __getitem__(self, idx):
                    sample = self.subset.dataset.samples[self.subset.indices[idx]]
                    with Image.open(sample["path"]) as img:
                        image = img.convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    return image, sample["label_idx"]

            train_ds = TransformedSubset(train_sub, train_transform)
            val_ds = TransformedSubset(val_sub, val_transform)
            class_to_idx = full_ds.class_to_idx

    class_to_idx = getattr(train_ds, "class_to_idx", class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_ds) > batch_size else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = None
    if test_annotations is not None:
        test_ds = FoodImageDataset(
            annotations_file=Path(test_annotations),
            images_dir=images_dir,
            transform=val_transform,
            class_to_idx=class_to_idx,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader, class_to_idx

