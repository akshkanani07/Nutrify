"""Production Inference Engine for Nutrition AI.

Classifies food images using trained PyTorch EfficientNet-B0 weights,
and independently attaches nutritional breakdown from candidates.csv.

Usage:
    python ml/inference.py --image path/to/salad.jpg --checkpoint ml/models/checkpoints/best_model.pt
    python ml/inference.py --image-dir path/to/test_images/ --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.dataset import get_inference_transforms
from ml.src.model import build_model
from ml.src.nutrition import NutritionMapper
from ml.src.utils import get_device, setup_logger

logger = setup_logger("nutrition_ai.inference")


class FoodPredictor:
    """End-to-end inference engine decoupling image classification from nutrition mapping."""

    def __init__(
        self,
        checkpoint_path: str | Path = "ml/models/checkpoints/best_model.pt",
        nutrition_csv: str | Path | None = None,
        device: str = "auto",
        image_size: int = 224,
    ) -> None:
        self.device = get_device(device)
        self.image_size = image_size
        self.transform = get_inference_transforms(image_size)

        # 1. Initialize Nutrition Mapper (Decoupled lookup)
        self.nutrition_mapper = NutritionMapper(csv_path=nutrition_csv)
        logger.info(
            f"Loaded nutrition database with {len(self.nutrition_mapper)} reference foods from {self.nutrition_mapper.csv_path}"
        )

        # 2. Load PyTorch model checkpoint
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.checkpoint_path}. Train model first using ml/train.py"
            )

        logger.info(f"Loading weights from {self.checkpoint_path} onto {self.device}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load class index mappings
        self.class_to_idx = checkpoint.get("class_to_idx", {})
        idx_to_class_raw = checkpoint.get("idx_to_class", {})
        self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}

        if not self.class_to_idx:
            # Fallback to classes.json
            classes_json = self.checkpoint_path.parent / "classes.json"
            if classes_json.exists():
                with open(classes_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.class_to_idx = data["class_to_idx"]
                    self.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}

        self.num_classes = len(self.class_to_idx)
        logger.info(f"Initialized predictor with {self.num_classes} output classes.")

        # Instantiate model architecture and load state dict
        self.model = build_model(num_classes=self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_input: str | Path | Image.Image,
        top_k: int = 5,
        serving_weight_g: float | None = None,
    ) -> dict[str, Any]:
        """Classify a food image and map top predictions to nutrition profiles.

        Args:
            image_input: File path (str/Path) or PIL Image instance.
            top_k: Number of highest-probability candidate food classes to return.
            serving_weight_g: Optional serving size in grams for nutrition scaling.

        Returns:
            Structured dictionary with top prediction, full top-K list, and nutrition details.
        """
        # Load image if file path provided
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            image = Image.open(img_path).convert("RGB")
            image_source = str(img_path)
        else:
            image = image_input.convert("RGB")
            image_source = "memory_buffer"

        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Model forward pass
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1)[0]

        # Extract top-K predictions
        k = min(top_k, self.num_classes)
        topk_probs, topk_indices = torch.topk(probs, k=k)

        candidates = []
        for prob, idx in zip(topk_probs.cpu().tolist(), topk_indices.cpu().tolist()):
            class_name = self.idx_to_class.get(idx, f"class_{idx}")
            confidence = round(prob, 4)

            # Look up decoupled nutrition profile
            food_nutr = self.nutrition_mapper.lookup(class_name)
            if food_nutr:
                nutrition_data = food_nutr.calculate_serving(serving_weight_g)
            else:
                nutrition_data = {
                    "fdc_id": None,
                    "description": class_name,
                    "category": "unknown",
                    "serving_weight_g": serving_weight_g or 100.0,
                    "calories": None,
                    "protein_g": None,
                    "fat_g": None,
                    "carbs_g": None,
                    "per_100g": None,
                }

            candidates.append({
                "rank": len(candidates) + 1,
                "class_id": idx,
                "class_name": class_name,
                "confidence": confidence,
                "confidence_pct": round(confidence * 100.0, 2),
                "nutrition": nutrition_data,
            })

        top_cand = candidates[0] if candidates else None

        return {
            "image_source": image_source,
            "top_prediction": {
                "class_name": top_cand["class_name"] if top_cand else None,
                "confidence": top_cand["confidence"] if top_cand else 0.0,
                "confidence_pct": top_cand["confidence_pct"] if top_cand else 0.0,
                "nutrition": top_cand["nutrition"] if top_cand else None,
            },
            "candidates": candidates,
        }


def print_prediction_table(result: dict[str, Any]) -> None:
    """Print clean terminal summary of prediction and nutrition."""
    top = result["top_prediction"]
    nutr = top["nutrition"]

    print("\n" + "=" * 65)
    print(f"PREDICTION: {top['class_name'].upper()} ({top['confidence_pct']}%)")
    print("=" * 65)
    print(f"Source Image : {result['image_source']}")
    print(f"Food Category: {nutr.get('category', 'N/A')}")
    print(f"Serving Size : {nutr.get('serving_weight_g', 100)}g")
    print("-" * 65)
    print(f"Calories     : {nutr.get('calories', 'N/A')} kcal")
    print(f"Protein      : {nutr.get('protein_g', 'N/A')} g")
    print(f"Carbohydrates: {nutr.get('carbs_g', 'N/A')} g")
    print(f"Fat          : {nutr.get('fat_g', 'N/A')} g")
    if nutr.get("fiber_g") is not None:
        print(f"Dietary Fiber: {nutr.get('fiber_g')} g")
    if nutr.get("sugar_g") is not None:
        print(f"Sugar        : {nutr.get('sugar_g')} g")
    if nutr.get("sodium_mg") is not None:
        print(f"Sodium       : {nutr.get('sodium_mg')} mg")
    print("=" * 65)

    if len(result["candidates"]) > 1:
        print("\nOther Top Candidates:")
        for cand in result["candidates"][1:]:
            c_nutr = cand["nutrition"]
            cals = f"{c_nutr.get('calories')} kcal" if c_nutr.get("calories") is not None else "N/A"
            print(f"  #{cand['rank']} {cand['class_name']:<30} {cand['confidence_pct']:>5.1f}%  ({cals})")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nutrition AI Food Recognition Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file to analyze")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to directory containing test images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/models/checkpoints/best_model.pt",
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--nutrition-csv",
        type=str,
        default="ml/data/raw/candidates.csv",
        help="Path to reference nutrition CSV",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate classes to return")
    parser.add_argument(
        "--serving-size",
        type=float,
        default=None,
        help="Custom food weight in grams to calculate nutrition",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image and not args.image_dir:
        print("Please provide --image <path> or --image-dir <path>.")
        sys.exit(1)

    predictor = FoodPredictor(
        checkpoint_path=args.checkpoint,
        nutrition_csv=args.nutrition_csv,
        device=args.device,
    )

    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        dir_p = Path(args.image_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths.extend([p for p in dir_p.iterdir() if p.suffix.lower() in valid_exts])

    all_results = []
    for p in image_paths:
        res = predictor.predict(p, top_k=args.top_k, serving_weight_g=args.serving_size)
        all_results.append(res)
        print_prediction_table(res)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(all_results if len(all_results) > 1 else all_results[0], f, indent=2)
        print(f"Results saved to {out_p}")


if __name__ == "__main__":
    main()

