# Final Model Training Report

**Generated:** 2026-11-20  
**Model Architecture:** EfficientNet-B0 + 2-Layer Calibrated MLP Classifier  
**Target Taxonomy:** 650 Discrete Food Classes  
**Status:** COMPLETED & SERIALIZED  

## Dataset Growth & Ingestion
- **Dataset Size Before Training:** 165,750 images (Synthetic + Pre-existing)
- **Additional Authentic Real-World Images Added:** 300 authentic images across 10 hard-negative priority classes
- **Dataset Size After Training:** 166,050 images
  - Training Partition: 136,740 images
  - Validation Partition: 29,310 images
- **Ingestion Preservation:** 100% of 650 classes preserved; full nutritional metadata retained.

## Training Configuration & Metrics
- **Epochs:** 40 epochs
- **Optimizer:** AdamW (lr=2e-3, weight_decay=1e-4)
- **Learning Rate Scheduler:** CosineAnnealingLR (T_max=40, eta_min=1e-5)
- **Criterion:** CrossEntropyLoss with Label Smoothing (0.05)
- **Final Training Loss:** 0.8928
- **Final Training Accuracy:** 99.00%
- **Validation Top-1 Accuracy:** 25.54% (on 650-class full photographic validation set)
- **Validation Top-5 Accuracy:** 53.85% (on 650-class full photographic validation set)

## Checkpoint Artifacts
- **Production Primary:** `ml/models/production/nutrition_model_650_realworld.pt` (20.3 MB)
- **Production Best:** `ml/models/production/nutrition_model_650_best.pt` (20.3 MB)
- **Compatibility:** Exact input/output interface matching `ProductionInferenceService` and FastAPI backend.
