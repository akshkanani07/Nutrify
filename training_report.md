# Nutrition AI — Phase 7 Real Model Training Report

- **Execution Date:** 2026-11-03 09:52:17 UTC
- **Dataset:** Cleaned Real-World Photographic Food Dataset (5,286 images)
- **Architecture:** EfficientNet-B0 + 2-Layer Enhanced MLP Head (512-d hidden, Dropout 0.3)
- **Classes:** 650
- **Optimizer:** AdamW (lr=2e-3, weight_decay=1e-4) with CosineAnnealingLR
- **Loss Function:** CrossEntropy with Label Smoothing (0.05)
- **Total Epochs:** 6 (Best Epoch: 1)
- **Best Validation Loss:** 5.9004
- **Checkpoints Saved:**
  - `ml/models/checkpoints/real_train_best.pt`
  - `ml/models/checkpoints/real_train_last.pt`

## Empirical Training History

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Top-1 (%) | Val Top-5 (%) | Learning Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 6.0730 | 2.39% | 5.9004 ★ | 0.61% | 4.60% | 0.001988 |
| 2 | 4.3683 | 12.83% | 6.2952 | 0.77% | 4.45% | 0.001951 |
| 3 | 3.2346 | 35.69% | 6.7869 | 0.92% | 4.14% | 0.001892 |
| 4 | 2.2332 | 62.13% | 7.5418 | 1.53% | 3.53% | 0.001810 |
| 5 | 1.5503 | 80.17% | 8.0506 | 0.46% | 3.22% | 0.001709 |
| 6 | 1.1861 | 90.15% | 8.3617 | 1.07% | 3.07% | 0.001590 |
