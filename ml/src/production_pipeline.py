"""Production Preprocessing and Inference Pipeline for Nutrition AI.

Features:
- Robust image loading: accepts file paths, raw bytes, byte streams, and PIL Images.
- Color mode conversion: handles RGBA (alpha flattening onto white background), CMYK, Grayscale, and Palette modes.
- EXIF orientation correction.
- Image quality assessment: resolution check, blur detection (Laplacian variance), contrast/blank check.
- Production transform: aspect-ratio preserving resize and ImageNet normalization.
- Nutrition mapping: decoupled nutritional lookup from candidates.csv.
- Standard backend JSON output format.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger("nutrition_ai.production")


# =====================================================================
# CUSTOM EXCEPTIONS
# =====================================================================

class ProductionInferenceError(Exception):
    """Base exception for production inference pipeline."""

    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


class InvalidImageFormatError(ProductionInferenceError):
    """Raised when uploaded file is not a supported image format."""

    def __init__(self, message: str = "Unsupported or invalid image format.", hint: str | None = None) -> None:
        super().__init__("INVALID_IMAGE_FORMAT", message, hint or "Please upload a JPEG, PNG, or WEBP image.")


class CorruptedImageError(ProductionInferenceError):
    """Raised when image bytes are corrupted, truncated, or unreadable."""

    def __init__(self, message: str = "Image file is corrupted.", hint: str | None = None) -> None:
        super().__init__("CORRUPTED_IMAGE", message, hint or "The uploaded file could not be decoded.")


class ImageResolutionError(ProductionInferenceError):
    """Raised when image resolution is too low for reliable classification."""

    def __init__(self, message: str = "Image resolution is too low.", hint: str | None = None) -> None:
        super().__init__("IMAGE_TOO_SMALL", message, hint or "Please provide an image of at least 32x32 pixels.")


# =====================================================================
# IMAGE QUALITY DATA STRUCTURE
# =====================================================================

@dataclass
class QualityReport:
    """Diagnostic report on input image quality."""
    width: int
    height: int
    aspect_ratio: float
    is_valid: bool
    blur_score: float
    contrast_score: float
    is_blurry: bool
    is_low_contrast: bool
    format: str
    warnings: list[str]


# =====================================================================
# PRODUCTION PREPROCESSOR & QUALITY VALIDATOR
# =====================================================================

class ProductionImagePreprocessor:
    """Production image validator, sanitizer, and preprocessor."""

    def __init__(
        self,
        target_size: int = 224,
        min_dimension: int = 32,
        blur_threshold: float = 60.0,
        contrast_threshold: float = 12.0,
    ) -> None:
        self.target_size = target_size
        self.min_dimension = min_dimension
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold

        # PyTorch normalization transform
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load_and_validate(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    ) -> tuple[Image.Image, QualityReport]:
        """Decode, sanitize, and validate an arbitrary food image input."""
        warnings: list[str] = []
        raw_format = "UNKNOWN"

        # 1. Decode into PIL Image
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise CorruptedImageError(f"Image file not found: {path}")
            try:
                with open(path, "rb") as f:
                    data = f.read()
                raw_format = path.suffix.lstrip(".").upper() or "IMAGE"
                pil_img = Image.open(io.BytesIO(data))
                pil_img.load()
            except Exception as e:
                raise CorruptedImageError(f"Failed to read image from {path}: {e}")
        elif isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise InvalidImageFormatError("Received empty image byte buffer.")
            try:
                pil_img = Image.open(io.BytesIO(image_input))
                pil_img.load()
                raw_format = pil_img.format or "BYTES"
            except Exception as e:
                raise CorruptedImageError(f"Failed to decode image bytes: {e}")
        elif hasattr(image_input, "read"):
            try:
                data = image_input.read()
                pil_img = Image.open(io.BytesIO(data))
                pil_img.load()
                raw_format = pil_img.format or "STREAM"
            except Exception as e:
                raise CorruptedImageError(f"Failed to decode image stream: {e}")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.copy()
            raw_format = pil_img.format or "PIL"
        else:
            raise InvalidImageFormatError(f"Unsupported input type: {type(image_input)}")

        # 2. Check resolution
        w, h = pil_img.size
        if w < self.min_dimension or h < self.min_dimension:
            raise ImageResolutionError(
                f"Image dimensions ({w}x{h}) are below minimum requirement ({self.min_dimension}x{self.min_dimension} px)."
            )

        aspect_ratio = round(max(w, h) / max(1, min(w, h)), 2)
        if aspect_ratio > 10.0:
            warnings.append(f"Extreme aspect ratio ({aspect_ratio}:1) detected.")

        # 3. EXIF orientation correction
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
            w, h = pil_img.size
        except Exception:
            pass

        # 4. Color normalization and alpha handling
        if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
            # Alpha flatten onto clean white background
            rgba = pil_img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            blended = Image.alpha_composite(background, rgba)
            pil_img = blended.convert("RGB")
        elif pil_img.mode == "P":
            # Palette image
            rgba = pil_img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            blended = Image.alpha_composite(background, rgba)
            pil_img = blended.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # 5. Quality checks: blur & contrast
        grayscale = pil_img.convert("L")
        np_gray = np.array(grayscale, dtype=np.float32)

        # Contrast via standard deviation of pixel intensities
        contrast_score = float(np.std(np_gray))
        is_low_contrast = contrast_score < self.contrast_threshold
        if is_low_contrast:
            warnings.append("Low contrast image: scene may be underexposed or blank.")

        # Blur estimation via Laplacian edge response variance
        # Apply standard 3x3 discrete Laplacian filter
        lap_filtered = grayscale.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0))
        np_lap = np.array(lap_filtered, dtype=np.float32)
        blur_score = float(np.var(np_lap))
        is_blurry = blur_score < self.blur_threshold
        if is_blurry:
            warnings.append(f"Image appears blurry (Laplacian variance {blur_score:.1f} < {self.blur_threshold}).")

        report = QualityReport(
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            is_valid=True,
            blur_score=round(blur_score, 2),
            contrast_score=round(contrast_score, 2),
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            format=raw_format,
            warnings=warnings,
        )

        return pil_img, report

    def preprocess_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Resize with aspect preservation (letterbox) and normalize into PyTorch tensor (1, C, H, W)."""
        target = self.target_size
        w, h = pil_img.size

        # Aspect-ratio preserving resize (fit inside target x target)
        scale = target / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Pad onto square canvas with ImageNet neutral gray background (124, 116, 104)
        canvas = Image.new("RGB", (target, target), (124, 116, 104))
        paste_x = (target - new_w) // 2
        paste_y = (target - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))

        tensor = self.tensor_transform(canvas)
        return tensor.unsqueeze(0)  # [1, 3, 224, 224]


# =====================================================================
# FULL PRODUCTION MODEL ARCHITECTURE
# =====================================================================

class EnhancedClassifierHead(nn.Module):
    """Deep 2-layer classifier head matching trained checkpoint."""

    def __init__(self, in_features: int, num_classes: int = 50, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        return self.fc2(out)


class ProductionFoodModel(nn.Module):
    """Full end-to-end model encapsulating EfficientNet-B0 backbone + EnhancedClassifierHead."""

    def __init__(self, extractor: nn.Module, head: nn.Module):
        super().__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(x)
        return self.head(feat)

    @torch.inference_mode()
    def predict_topk(self, x: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        topk_res = torch.topk(probs, min(k, logits.size(-1)), dim=-1)
        return topk_res.indices, topk_res.values


# =====================================================================
# NUTRITION DATABASE MAPPER
# =====================================================================

class ProductionNutritionMapper:
    """Nutritional profile resolver mapping food classes to macronutrients."""

    def __init__(
        self,
        csv_path: Union[str, Path] = "ml/data/raw/candidates.csv",
        json_path: Union[str, Path] = "ml/models/production/nutrition_mapping_650.json",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.nutrition_db: dict[str, dict[str, Any]] = {}
        self._load()

    def _normalize_name(self, name: str) -> str:
        clean = name.lower().replace(",", " ").replace("-", " ").replace("'", "")
        return " ".join(clean.split())

    def _load(self) -> None:
        # 1. Prefer comprehensive 650-class USDA JSON mapping if present
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for food_key, val in data.items():
                    norm = self._normalize_name(food_key)
                    self.nutrition_db[norm] = {
                        "food_name": val.get("food_name", food_key),
                        "category": val.get("category", "ingredient_other"),
                        "category_group": val.get("category_group", "General"),
                        "usda_fdc_id": val.get("usda_fdc_id"),
                        "calories": float(val.get("calories", 0.0)),
                        "protein": float(val.get("protein", 0.0)),
                        "carbs": float(val.get("carbs", 0.0)),
                        "fat": float(val.get("fat", 0.0)),
                        "fiber": float(val.get("fiber", 0.0)),
                        "sugar": float(val.get("sugar", 0.0)),
                        "sodium": float(val.get("sodium", 0.0)),
                        "default_serving_g": float(val.get("serving_weight_g", 100.0)),
                    }
                logger.info(f"Loaded {len(self.nutrition_db)} nutrition profiles from {self.json_path}")
                return
            except Exception as e:
                logger.warning(f"Failed loading {self.json_path}: {e}. Falling back to CSV.")

        if not self.csv_path.exists():
            logger.warning(f"Nutrition CSV not found at {self.csv_path}. Using fallback values.")
            return

        import csv

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                desc = row.get("description", "").strip()
                norm = self._normalize_name(desc)
                try:
                    profile = {
                        "category": row.get("category", "ingredient_other"),
                        "calories": float(row.get("calories_per_100g") or 0.0),
                        "protein": float(row.get("protein_g") or 0.0),
                        "carbs": float(row.get("carb_g") or 0.0),
                        "fat": float(row.get("fat_g") or 0.0),
                        "fiber": float(row.get("fiber_g") or 0.0),
                        "sugar": float(row.get("sugar_g") or 0.0),
                        "sodium": float(row.get("sodium_mg") or 0.0),
                        "default_serving_g": float(row.get("default_serving_g") or 100.0),
                    }
                    self.nutrition_db[norm] = profile
                except Exception:
                    continue

    def get_nutrition_per_100g(self, food_name: str) -> dict[str, Any]:
        """Returns baseline unscaled per-100g nutritional profile."""
        return self.get_nutrition(food_name, serving_weight_g=100.0)

    def get_nutrition(self, food_name: str, serving_weight_g: float | None = None) -> dict[str, Any]:
        """Lookup nutritional values scaled by serving weight (defaults to reference serving size)."""
        target = self._normalize_name(food_name)

        # 1. Exact lookup
        match = self.nutrition_db.get(target)

        # 2. Token overlap lookup
        if not match:
            target_tokens = set(target.split())
            best_score = 0
            best_cand = None
            for cand, prof in self.nutrition_db.items():
                cand_tokens = set(cand.split())
                score = len(target_tokens & cand_tokens)
                if score > best_score:
                    best_score = score
                    best_cand = prof
            match = best_cand

        # Fallback profile
        if not match:
            match = {
                "category": "ingredient_other",
                "calories": 100.0,
                "protein": 3.0,
                "carbs": 15.0,
                "fat": 2.0,
                "fiber": 2.0,
                "sugar": 3.0,
                "sodium": 50.0,
                "default_serving_g": 100.0,
            }

        ref_serving = float(match.get("default_serving_g", 100.0))
        target_weight = serving_weight_g if serving_weight_g is not None else ref_serving
        scale = target_weight / 100.0

        return {
            "category": match.get("category", "general"),
            "category_group": match.get("category_group", "General"),
            "serving_weight_g": round(target_weight, 1),
            "reference_serving_g": round(ref_serving, 1),
            "calories": round(match["calories"] * scale, 1),
            "protein": round(match["protein"] * scale, 2),
            "carbs": round(match["carbs"] * scale, 2),
            "fat": round(match["fat"] * scale, 2),
            "fiber": round(match["fiber"] * scale, 2),
            "sugar": round(match.get("sugar", 0.0) * scale, 2),
            "sodium": round(match.get("sodium", 0.0) * scale, 1),
        }


# =====================================================================
# PRODUCTION INFERENCE SERVICE (BACKEND INTERFACE)
# =====================================================================

class ProductionInferenceService:
    """Production Inference Service providing the backend integration interface."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "ml/models/production/nutrition_model_best.pt",
        candidates_csv: Union[str, Path] = "ml/data/raw/candidates.csv",
        device: str = "cpu",
        confidence_threshold: float = 0.50,
        num_threads: int | None = 4,
    ) -> None:
        if device == "cpu" and num_threads:
            try:
                import os
                n_threads = min(num_threads, os.cpu_count() or 4)
                torch.set_num_threads(n_threads)
            except Exception:
                pass

        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold

        self.preprocessor = ProductionImagePreprocessor(target_size=224)
        self.nutrition_mapper = ProductionNutritionMapper(csv_path=candidates_csv)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.class_to_idx = checkpoint.get("class_to_idx", {})
        idx_to_class_raw = checkpoint.get("idx_to_class", {})
        self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
        self.num_classes = len(self.class_to_idx)

        # Build architecture
        m = models.efficientnet_b0(weights=None)
        extractor = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        head = EnhancedClassifierHead(in_features=1280, num_classes=self.num_classes, hidden_dim=512, dropout=0.3)
        self.model = ProductionFoodModel(extractor=extractor, head=head)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"ProductionInferenceService ready with {self.num_classes} classes on {self.device}.")

    def _determine_confidence_tier(self, conf: float) -> tuple[str, str, bool]:
        """Calibrate confidence score into High (>=80%), Medium (50-80%), Low (<50%)."""
        if conf >= 0.80:
            return "high", "High confidence match.", True
        elif conf >= 0.50:
            return "medium", "Likely match. Review nutrition or select alternate if needed.", True
        else:
            return "low", "Prediction uncertain. Try another photo.", False

    def predict(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
        top_k: int = 5,
        serving_weight_g: float | None = None,
    ) -> dict[str, Any]:
        """Classify food image and return standardized backend JSON structure."""
        # 1. Validation & Preprocessing
        pil_img, quality = self.preprocessor.load_and_validate(image_input)
        input_tensor = self.preprocessor.preprocess_tensor(pil_img).to(self.device)

        # 2. Forward inference (torch.inference_mode inside predict_topk)
        topk_idx, topk_probs = self.model.predict_topk(input_tensor, k=top_k)

        top1_idx = int(topk_idx[0][0].item())
        top1_prob = float(topk_probs[0][0].item())
        top1_name = self.idx_to_class[top1_idx]

        # 3. Nutrition calculation
        top1_nutrition = self.nutrition_mapper.get_nutrition(top1_name, serving_weight_g=serving_weight_g)
        top1_per_100g = self.nutrition_mapper.get_nutrition_per_100g(top1_name)

        # 4. Top-5 candidates list
        candidates = []
        for rank, (idx, prob) in enumerate(zip(topk_idx[0], topk_probs[0]), 1):
            c_idx = int(idx.item())
            c_prob = float(prob.item())
            c_name = self.idx_to_class[c_idx]
            c_nut = self.nutrition_mapper.get_nutrition(c_name, serving_weight_g=serving_weight_g)
            candidates.append({
                "rank": rank,
                "food_name": c_name,
                "confidence": round(c_prob, 4),
                "calories": c_nut["calories"],
                "protein": c_nut["protein"],
                "carbs": c_nut["carbs"],
                "fat": c_nut["fat"],
                "fiber": c_nut["fiber"],
                "category": c_nut["category"],
            })

        tier, user_msg, is_confident = self._determine_confidence_tier(top1_prob)

        # 5. Build standardized response
        response = {
            "food_name": top1_name,
            "confidence": round(top1_prob, 4),
            "calories": top1_nutrition["calories"],
            "protein": top1_nutrition["protein"],
            "carbs": top1_nutrition["carbs"],
            "fat": top1_nutrition["fat"],
            "fiber": top1_nutrition["fiber"],
            # Production diagnostics & metadata
            "category": top1_nutrition["category"],
            "category_group": top1_nutrition.get("category_group", "General"),
            "serving_weight_g": top1_nutrition["serving_weight_g"],
            "reference_serving_g": top1_nutrition.get("reference_serving_g", 100.0),
            "confidence_tier": tier,
            "user_message": user_msg,
            "is_confident": is_confident,
            "candidates": candidates,
            "quality": asdict(quality),
            "nutrition_per_100g": {
                "calories": top1_per_100g["calories"],
                "protein": top1_per_100g["protein"],
                "carbs": top1_per_100g["carbs"],
                "fat": top1_per_100g["fat"],
                "fiber": top1_per_100g["fiber"],
            },
        }

        return response

    def predict_batch(
        self,
        image_inputs: list[Union[str, Path, bytes, BinaryIO, Image.Image]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Batch inference for processing multiple images efficiently."""
        if not image_inputs:
            return []

        tensors = []
        qualities = []
        for inp in image_inputs:
            pil_img, qual = self.preprocessor.load_and_validate(inp)
            tensors.append(self.preprocessor.preprocess_tensor(pil_img))
            qualities.append(qual)

        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        topk_idx, topk_probs = self.model.predict_topk(batch_tensor, k=top_k)

        results = []
        for i in range(len(image_inputs)):
            top1_idx = int(topk_idx[i][0].item())
            top1_prob = float(topk_probs[i][0].item())
            top1_name = self.idx_to_class[top1_idx]
            top1_nutrition = self.nutrition_mapper.get_nutrition(top1_name)
            tier, user_msg, is_confident = self._determine_confidence_tier(top1_prob)

            candidates = []
            for rank, (idx, prob) in enumerate(zip(topk_idx[i], topk_probs[i]), 1):
                c_name = self.idx_to_class[int(idx.item())]
                c_nut = self.nutrition_mapper.get_nutrition(c_name)
                candidates.append({
                    "rank": rank,
                    "food_name": c_name,
                    "confidence": round(float(prob.item()), 4),
                    "calories": c_nut["calories"],
                    "protein": c_nut["protein"],
                    "carbs": c_nut["carbs"],
                    "fat": c_nut["fat"],
                    "fiber": c_nut["fiber"],
                })

            results.append({
                "food_name": top1_name,
                "confidence": round(top1_prob, 4),
                "calories": top1_nutrition["calories"],
                "protein": top1_nutrition["protein"],
                "carbs": top1_nutrition["carbs"],
                "fat": top1_nutrition["fat"],
                "fiber": top1_nutrition["fiber"],
                "confidence_tier": tier,
                "user_message": user_msg,
                "is_confident": is_confident,
                "candidates": candidates,
                "quality": asdict(qualities[i]),
            })

        return results

