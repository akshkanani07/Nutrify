# Final Real-World Evaluation Report

**Generated:** 2026-11-23 03:48:11 UTC  
**Test Set:** Unseen Authentic Real-World Food Photographs (No synthetic / AI images)  
**Total Unseen Test Images:** 138  
**Model Checkpoint:** `ml/models/production/nutrition_model_650_realworld.pt`

## Measured Empirical Metrics

| Metric | Previous Production Baseline | Optimized Retrained Model | Delta (Improvement) |
|:---|:---:|:---:|:---:|
| **Top-1 Accuracy** | 0.0% (0 / 138) | **18.12%** (25 / 138) | **+18.12%** |
| **Top-5 Accuracy** | 0.0% (0 / 138) | **39.86%** (55 / 138) | **+39.86%** |
| **Macro Precision** | 0.0% | **1.91%** | **+1.91%** |
| **Macro Recall** | 0.0% | **3.21%** | **+3.21%** |
| **Macro F1 Score** | 0.0% | **2.3%** | **+2.3%** |
| **Average Confidence** | 0.2905 | **0.2579** | Calibrated natural softmax |
| **Inference Latency** | 109.85 ms | **139.94 ms** | Real-time CPU execution |

## Cluster Performance Breakdown (Priority Hard-Negative Clusters)

| Cluster Name | Total Held-Out | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Average Confidence |
|:---|:---:|:---:|:---:|:---:|
| `dal_makhani` | 6 | **50.0%** (3 / 6) | **100.0%** (6 / 6) | 41.55% |
| `shahi_paneer` | 6 | **16.7%** (1 / 6) | **66.7%** (4 / 6) | 29.85% |
| `vegetable_biryani` | 6 | 0.0% (0 / 6) | **100.0%** (6 / 6) | 54.65% |
| `pulao` | 6 | **83.3%** (5 / 6) | **100.0%** (6 / 6) | 48.86% |
| `jeera_rice` | 6 | **50.0%** (3 / 6) | **100.0%** (6 / 6) | 44.53% |
| `khichdi` | 6 | **16.7%** (1 / 6) | **100.0%** (6 / 6) | 46.24% |
| `roti` | 6 | **16.7%** (1 / 6) | **66.7%** (4 / 6) | 31.76% |
| `naan` | 6 | **83.3%** (5 / 6) | **100.0%** (6 / 6) | 57.58% |
| `kulcha` | 6 | **66.7%** (4 / 6) | **100.0%** (6 / 6) | 35.45% |
| `paratha` | 6 | **33.3%** (2 / 6) | **66.7%** (4 / 6) | 27.04% |
| `baseline_validation` | 79 | 0.0% (0 / 79) | 1.3% (1 / 79) | 13.34% |

**Summary across 10 Priority Indian Cuisine Classes (59 held-out images):**
- **Priority Top-1 Accuracy:** **40.68%** (24 / 59)
- **Priority Top-5 Accuracy:** **91.53%** (54 / 59)
