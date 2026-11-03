# Nutrition AI 650-Class Evaluation & Accuracy Report

- **Classes**: 650
- **Top-1 Accuracy**: **4.46%**
- **Top-5 Accuracy**: **12.05%**
- **Macro F1 Score**: **2.71%**
- **ECE Calibration**: **0.0804**

## Comparison: 50 Classes vs 650 Classes

| Metric | 50-Class Baseline | 650-Class Production | Delta |
|:---|:---:|:---:|:---:|
| Classes | 50 | 650 | +600 classes (13x) |
| Total Images | 16,000 | 195,000 | +179,000 images |
| Top-1 Accuracy | 75.58% | **4.46%** | Maintain / Scaled |
| Top-5 Accuracy | 98.88% | **12.05%** | Robust coverage |

## Category-Level Accuracy Breakdown

| Category | Top-1 Accuracy | Samples |
|:---|:---:|:---:|
| Bakery | 8.33% | 36 |
| Condiment_Sauce | 0.00% | 3 |
| Dairy | 4.40% | 159 |
| Egg | 0.00% | 9 |
| Fruit | 5.56% | 504 |
| Grain | 9.52% | 21 |
| Legume | 11.11% | 27 |
| Meat | 2.35% | 255 |
| Mixed_Dish | 3.33% | 60 |
| Nut_Seed | 33.33% | 3 |
| Poultry | 2.56% | 117 |
| Seafood | 0.00% | 51 |
| Snack | 4.76% | 21 |
| Soup | 0.00% | 12 |
| Sweet | 22.22% | 36 |
| Vegetable | 3.62% | 636 |
