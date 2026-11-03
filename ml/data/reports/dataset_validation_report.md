# Nutrition AI: 650-Class Dataset Validation Report

- **Validation Timestamp**: `2026-11-03T04:46:02Z`
- **Total Target Classes**: `650`
- **Total Images**: `195000`
- **READY FOR TRAINING**: **YES**

## 1. Quality & Integrity Audit Summary

| Audit Parameter | Verified Requirement | Result | Status |
|:---|:---|:---:|:---:|
| **Class Count** | Exactly 650 classes | 650 classes | **PASSED** |
| **Total Images** | 195,000 images | 195000 images | **PASSED** |
| **Distribution Balance** | 300 per class (210/45/45) | Balanced (0% variance) | **PASSED** |
| **Missing Files** | 0 missing | 0 | **PASSED** |
| **Corrupted Files** | 0 corrupt | 0 | **PASSED** |
| **SHA-256 Duplicates** | 0 duplicates | 0 | **PASSED** |
| **Image Resolution** | 224 x 224 RGB | 100% compliant | **PASSED** |
| **Aspect Ratio** | 1:1 square | 100% compliant | **PASSED** |
| **Sharpness / Blur** | Laplacian Var > 35.0 | 100% compliant | **PASSED** |
| **Split Correctness** | 70% Train / 15% Val / 15% Test | 136.5k / 29.25k / 29.25k | **PASSED** |

## 2. Category Distribution Overview

| Category | Classes | Total Images |
|:---|:---:|:---:|
| Vegetable | 212 | 63600 |
| Fruit | 168 | 50400 |
| Meat | 85 | 25500 |
| Dairy | 53 | 15900 |
| Poultry | 39 | 11700 |
| Mixed_Dish | 20 | 6000 |
| Seafood | 17 | 5100 |
| Bakery | 12 | 3600 |
| Sweet | 12 | 3600 |
| Legume | 9 | 2700 |
| Grain | 7 | 2100 |
| Snack | 7 | 2100 |
| Soup | 4 | 1200 |
| Egg | 3 | 900 |
| Nut_Seed | 1 | 300 |
| Condiment_Sauce | 1 | 300 |
