# Nutrition AI -- Dataset Preparation, Training & Inference Pipeline

Production-ready PyTorch pipeline for food and beverage image classification with decoupled nutritional profiling using transfer learning on **EfficientNet-B0**.

---

## 1. Directory Architecture

```
ml/
├── configs/
│   ├── __init__.py
│   └── train_config.py            # Dataclass configurations (Data, Model, Training)
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset, transforms, split logic & DataLoader factory
│   ├── dataset_validator.py       # Dataset validation engine (integrity, hashes, class distribution)
│   ├── model.py                   # Transfer learning architecture with EfficientNet-B0
│   ├── nutrition.py               # NutritionMapper (lookup from candidates.csv)
│   ├── nutrition_mapping.py       # NutritionCatalog & standardized schema generator
│   ├── split_generator.py         # Automated stratified CSV split generator
│   ├── metrics.py                 # MetricTracker, AverageMeter, Top-1 / Top-5 accuracy
│   ├── trainer.py                 # Training loop, validation, checkpointing & early stopping
│   └── utils.py                   # Reproducibility seed, device resolution & logging
│   └── collectors/                # Dataset collection & ingestion tools
│       ├── __init__.py
│       ├── food101_collector.py   # Food-101 importer & label mapper
│       ├── openfoodfacts_collector.py # Open Food Facts category image fetcher
│       ├── usda_collector.py      # USDA reference directory initializer
│       └── custom_collector.py    # Custom user photo importer, sanitizer & deduplicator
├── generate_splits.py             # CLI: Generates train.csv, val.csv, test.csv
├── validate_dataset.py            # CLI: Validates dataset integrity and writes report
├── train.py                       # CLI: Model training entry point
├── evaluate.py                    # CLI: Model evaluation & test set reporting
├── inference.py                   # CLI: Standalone inference engine (predicts food + nutrition)
├── prepare_dataset.py             # CLI wrapper: System readiness verification
├── requirements.txt               # PyTorch, Torchvision, Pillow & data dependencies
├── README.md                      # Pipeline documentation & operator manual
├── data/
│   ├── annotations/               # CSV annotations (train.csv, val.csv, test.csv)
│   ├── images/                    # Image storage organized by class subfolders
│   ├── nutrition_mapping.csv      # Standardized nutrition mapping (food_name, USDA_ID, macros)
│   ├── raw/
│   │   └── candidates.csv         # Verified USDA nutrition database (1,051 foods)
│   └── reports/                   # Validation reports (JSON and Markdown)
└── models/
    └── checkpoints/               # Model weights, classes.json & training metrics
```

---

## 2. Dataset Architecture & Nutrition Mapping

### Dataset Hierarchy (`ml/data/`):
- `images/`: Food photos stored in class subfolders (e.g. `images/apple_raw/001.jpg`).
- `annotations/`: Generated `train.csv`, `val.csv`, and `test.csv`.
- `raw/candidates.csv`: Master USDA database with 1,051 verified candidate foods.
- `nutrition_mapping.csv`: Standardized nutrition schema.
- `reports/`: Audit logs and validation reports.

### Nutrition Mapping Schema:
Nutritional data is maintained strictly decoupled from image classification labels in `ml/data/nutrition_mapping.csv`:
```csv
food_name,USDA_ID,calories,protein,carbs,fat,fiber
kale raw,323505,35.0,2.92,4.42,1.49,4.1
garlic raw,1104647,143.0,6.62,28.2,0.38,2.7
broccoli raw,321900,32.0,2.57,6.29,0.34,2.4
nectarines raw,327357,39.0,1.06,9.18,0.28,1.5
```

---

## 3. Dataset Collection & Ingestion

The system supports expanding the training catalog up to 600--700 food classes from multiple sources:

### A. Food-101 Dataset
Import 101 common meal and dish categories mapped directly to normalized labels:
```powershell
# From local Food-101 directory:
python ml/src/collectors/food101_collector.py --source-dir path/to/food-101 --max-per-class 150

# Or automatically download and extract:
python ml/src/collectors/food101_collector.py --download --max-per-class 150
```

### B. Open Food Facts
Collect category-specific branded or generic product images using rate-limited API queries (1 req/sec):
```powershell
python ml/src/collectors/openfoodfacts_collector.py --category pizzas --label pizza --max-images 50
```

### C. USDA FoodData Central
Initialize class folders corresponding to verified USDA foods:
```powershell
python ml/src/collectors/usda_collector.py --limit 600
```

### D. Custom Collected Images (Camera / User Uploads / Web)
Sanitize, deduplicate via SHA-256, and import custom photos:
```powershell
# If folder has subdirectories named by class:
python ml/src/collectors/custom_collector.py --source-dir path/to/raw_camera_photos/

# If folder contains flat images for a single food item:
python ml/src/collectors/custom_collector.py --source-dir path/to/banana_photos/ --default-label "banana raw"
```

---

## 4. How to Add New Food Classes

1. **Verify or Add Nutrition**:
   Ensure the food name exists in `ml/data/nutrition_mapping.csv`. If adding a custom or regional food, append a row:
   ```csv
   food_name,USDA_ID,calories,protein,carbs,fat,fiber
   paneer tikka,CUSTOM_001,260.0,18.0,6.0,19.0,1.5
   ```
2. **Create Image Folder**:
   Create a subfolder in `ml/data/images/<class_name>/` (e.g. `ml/data/images/paneer_tikka/`).
3. **Add Photos**:
   Place at least 15--30 representative photos in the class folder.
4. **Regenerate Splits & Validate**:
   Run `python ml/generate_splits.py` followed by `python ml/validate_dataset.py`.

---

## 5. CSV Annotation Generation

To generate stratified `train.csv`, `val.csv`, and `test.csv` (70% / 15% / 15% split):
```powershell
python ml/generate_splits.py --images-dir ml/data/images --output-dir ml/data/annotations
```

### Generated CSV Format:
```csv
image_path,label
apple_raw/apple_001.jpg,apple raw
kale_raw/kale_014.jpg,kale raw
hummus_commercial/hummus_022.jpg,hummus commercial
```

---

## 6. Dataset Validation

Run validation to verify file integrity, detect duplicates, check missing images, and examine class distributions:
```powershell
python ml/validate_dataset.py
```

### Standardized Output:
```text
============================================================
Dataset Report:
- Total images                : 15420
- Total classes               : 102
- Images per class            : min=80, max=150, avg=151.2
- Missing files               : 0
- Corrupted images            : 0
- Duplicate image pairs       : 0
- Splits                      : {'train': 10794, 'val': 2313, 'test': 2313}
- Training readiness status   : READY
============================================================
```
Detailed reports are saved to:
- `ml/data/reports/dataset_validation_report.json`
- `ml/data/reports/dataset_validation_report.md`

---

## 7. Training Readiness Check

Before training, execute the master readiness command:
```powershell
python prepare_dataset.py
```

This verifies:
1. Directory structure exists (`images/`, `annotations/`, `raw/`, `reports/`, `nutrition_mapping.csv`).
2. Annotation CSVs are present, non-empty, and correctly formatted (`image_path,label`).
3. Every image file referenced in the CSVs exists and is valid on disk.
4. Labels match the nutrition catalog.
5. At least 2 classes and required sample counts are met.

Outputs:
```text
============================================================
READY FOR TRAINING: YES
============================================================
```
*(Or `READY FOR TRAINING: NO` with exact itemized blockers to fix).*

---

## 8. Model Training

Once the readiness check reports `READY FOR TRAINING: YES`, launch training:
```powershell
python ml/train.py `
  --train-csv ml/data/annotations/train.csv `
  --val-csv ml/data/annotations/val.csv `
  --images-dir ml/data/images `
  --epochs 25 `
  --batch-size 32 `
  --lr 0.001 `
  --optimizer adamw `
  --scheduler cosine `
  --device auto
```

---

## 9. Model Evaluation & Testing

```powershell
# Evaluate on test set
python ml/evaluate.py `
  --checkpoint ml/models/checkpoints/best_model.pt `
  --test-csv ml/data/annotations/test.csv

# Single image test inference with nutrition mapping
python ml/inference.py `
  --image sample_food.jpg `
  --checkpoint ml/models/checkpoints/best_model.pt `
  --top-k 5
```
