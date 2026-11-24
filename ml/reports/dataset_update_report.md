# Dataset Ingestion & Update Report

**Generated:** 2026-11-20  
**Status:** SUCCESS  

## Overview
The hard-negative authentic real-world food dataset was ingested into the production pipeline.  
650-class taxonomy and nutrition metadata were strictly preserved.

## Ingestion Metrics

| Metric | Before Ingestion | Additional Added | After Ingestion |
|:---|:---:|:---:|:---:|
| **Training Samples** | 136,500 | +240 | 136,740 |
| **Validation Samples** | 29,250 | +60 | 29,310 |
| **Total Dataset Size** | 165,750 | +300 | 166,050 |
| **Active Food Classes** | 650 | 0 (preserves taxonomy) | 650 |

## Class-Wise Additions

| Raw Priority Class | Mapped Production Label | Total Raw Images | Added to Train | Added to Val |
|:---|:---|:---:|:---:|:---:|
| `dal_makhani` | `dal makhani` | 30 | 24 | 6 |
| `shahi_paneer` | `paneer butter masala` | 30 | 24 | 6 |
| `vegetable_biryani` | `biryani chicken` | 30 | 24 | 6 |
| `pulao` | `fried rice` | 30 | 24 | 6 |
| `jeera_rice` | `fried rice` | 30 | 24 | 6 |
| `khichdi` | `dal tadka` | 30 | 24 | 6 |
| `roti` | `roti chapati` | 30 | 24 | 6 |
| `naan` | `naan plain` | 30 | 24 | 6 |
| `kulcha` | `naan plain` | 30 | 24 | 6 |
| `paratha` | `paratha aloo` | 30 | 24 | 6 |

## File Artifacts Generated
- `ml/data/processed/train.csv` (136,740 rows with nutrition columns: `[image_path, label, calories, protein, carbs, fat, fiber]`)
- `ml/data/processed/val.csv` (29,310 rows with nutrition columns: `[image_path, label, calories, protein, carbs, fat, fiber]`)
- `ml/data/annotations/train.csv` (updated with ingested real paths)
- `ml/data/annotations/val.csv` (updated with ingested real paths)
