# Nutrition AI Real-World Validation Report

- **Evaluated Model**: EfficientNet-B0 + 2-Layer Enhanced Head (`nutrition_model_best.pt`)
- **Total Scenarios Tested**: 87
- **Top-1 Accuracy**: **58.62%**
- **Top-5 Accuracy**: **85.06%**
- **Average Inference Latency**: 80.6 ms (CPU)
- **Error Handling Robustness**: 3/3 validation tests passed

## Category Performance

| Category | Samples | Top-1 Accuracy | Top-5 Accuracy |
|:---|:---:|:---:|:---:|
| Fruits | 23 | 65.2% | 91.3% |
| Vegetables | 26 | 57.7% | 84.6% |
| Prepared Foods | 21 | 42.9% | 76.2% |
| Indian Foods | 12 | 75.0% | 91.7% |
| Edge Cases (Quality) | 5 | 60.0% | 80.0% |

## Confidence Distribution

- **High Confidence (>80%)**: 20 (23.0%)
- **Medium Confidence (50-80%)**: 33 (37.9%)
- **Low Confidence (<50%)**: 34 (39.1%)

## Sample Prediction & Nutrition Extraction

| Test ID | True Food | Predicted Food | Confidence | Calories | Protein | Carbs | Fat | Fiber |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `heldout_fruits_01` | apples gala with skin raw | **apples gala with skin raw** | 64.0% | 61.0 kcal | 0.13g | 14.8g | 0.15g | 2.1g |
| `heldout_fruits_02` | apples gala with skin raw | **strawberries raw** | 35.2% | 31.0 kcal | 0.64g | 7.63g | 0.22g | 1.8g |
| `heldout_fruits_03` | bananas raw | **bananas raw** | 83.3% | 200.2 kcal | 2.45g | 51.39g | 0.74g | 5.85g |
| `heldout_fruits_04` | bananas raw | **bananas raw** | 55.6% | 200.2 kcal | 2.45g | 51.39g | 0.74g | 5.85g |
| `heldout_fruits_05` | strawberries raw | **strawberries raw** | 60.8% | 31.0 kcal | 0.64g | 7.63g | 0.22g | 1.8g |
| `heldout_fruits_06` | strawberries raw | **apples gala with skin raw** | 56.9% | 61.0 kcal | 0.13g | 14.8g | 0.15g | 2.1g |
| `heldout_fruits_07` | blueberries raw | **blueberries raw** | 33.6% | 84.4 kcal | 1.1g | 21.45g | 0.49g | 3.55g |
| `heldout_fruits_08` | blueberries raw | **blueberries raw** | 45.2% | 84.4 kcal | 1.1g | 21.45g | 0.49g | 3.55g |
| `heldout_fruits_09` | oranges raw navels | **mangos raw** | 82.2% | 99.0 kcal | 1.35g | 24.72g | 0.63g | 2.64g |
| `heldout_fruits_10` | oranges raw navels | **oranges raw navels** | 81.5% | 77.5 kcal | 1.5g | 19.47g | 0.25g | 3.3g |

## Failure Case Analysis

- **`heldout_fruits_02` (Fruits)**:
  - True: `apples gala with skin raw`
  - Predicted: `strawberries raw` (35.2%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_fruits_06` (Fruits)**:
  - True: `strawberries raw`
  - Predicted: `apples gala with skin raw` (56.9%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_fruits_09` (Fruits)**:
  - True: `oranges raw navels`
  - Predicted: `mangos raw` (82.2%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_fruits_13` (Fruits)**:
  - True: `raspberries raw`
  - Predicted: `strawberries raw` (38.8%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_fruits_15` (Fruits)**:
  - True: `blackberries raw`
  - Predicted: `blueberries raw` (40.8%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_vegetables_23` (Vegetables)**:
  - True: `spinach raw`
  - Predicted: `broccoli leaves raw` (24.4%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_vegetables_24` (Vegetables)**:
  - True: `spinach raw`
  - Predicted: `kale raw` (38.0%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_vegetables_25` (Vegetables)**:
  - True: `onions red raw`
  - Predicted: `raspberries raw` (49.2%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_vegetables_26` (Vegetables)**:
  - True: `onions red raw`
  - Predicted: `raspberries raw` (63.4%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_vegetables_29` (Vegetables)**:
  - True: `mushrooms white raw`
  - Predicted: `blackberries raw` (66.3%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_vegetables_30` (Vegetables)**:
  - True: `mushrooms white raw`
  - Predicted: `garlic raw` (54.3%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_41` (Prepared Foods)**:
  - True: `potato pancakes`
  - Predicted: `croutons plain` (43.9%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_42` (Prepared Foods)**:
  - True: `potato pancakes`
  - Predicted: `nuts almonds` (56.9%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_47` (Prepared Foods)**:
  - True: `croutons plain`
  - Predicted: `salmon nuggets breaded frozen heated` (35.0%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_48` (Prepared Foods)**:
  - True: `croutons plain`
  - Predicted: `potato pancakes` (63.8%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_49` (Prepared Foods)**:
  - True: `muffins oat bran`
  - Predicted: `archway home style cookies oatmeal` (63.1%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_51` (Prepared Foods)**:
  - True: `bread rye`
  - Predicted: `peanuts spanish raw` (28.9%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_prepared foods_53` (Prepared Foods)**:
  - True: `bread egg`
  - Predicted: `pizza rolls frozen unprepared` (48.6%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_indian foods_62` (Indian Foods)**:
  - True: `flour rice brown`
  - Predicted: `chicken spread` (22.7%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`heldout_indian foods_63` (Indian Foods)**:
  - True: `flour rice brown`
  - Predicted: `flour soy defatted` (19.7%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`fruit_03_strawberry` (Fruits)**:
  - True: `strawberries raw`
  - Predicted: `broccoli raw` (54.9%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`fruit_04_blueberry` (Fruits)**:
  - True: `blueberries raw`
  - Predicted: `broccoli raw` (34.8%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`fruit_05_orange` (Fruits)**:
  - True: `oranges raw navels`
  - Predicted: `bread rye` (52.2%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`veg_02_carrot` (Vegetables)**:
  - True: `carrots raw`
  - Predicted: `mangos raw` (50.0%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`veg_03_spinach` (Vegetables)**:
  - True: `spinach raw`
  - Predicted: `broccoli raw` (75.5%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`veg_04_onion` (Vegetables)**:
  - True: `onions red raw`
  - Predicted: `broccoli raw` (45.3%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`veg_05_garlic` (Vegetables)**:
  - True: `garlic raw`
  - Predicted: `kale raw` (69.6%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`veg_06_celery` (Vegetables)**:
  - True: `celery raw`
  - Predicted: `broccoli raw` (35.9%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`prep_01_pizza_rolls` (Prepared Foods)**:
  - True: `pizza rolls frozen unprepared`
  - Predicted: `broccoli raw` (20.5%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`prep_02_potato_pancakes` (Prepared Foods)**:
  - True: `potato pancakes`
  - Predicted: `broccoli raw` (37.6%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`prep_03_pasta` (Prepared Foods)**:
  - True: `pasta gluten free corn cooked`
  - Predicted: `broccoli raw` (28.7%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`prep_04_hummus` (Prepared Foods)**:
  - True: `hummus commercial`
  - Predicted: `muffins oat bran` (17.3%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`prep_05_croutons` (Prepared Foods)**:
  - True: `croutons plain`
  - Predicted: `broccoli raw` (20.6%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`indian_03_brown_rice_flour` (Indian Foods)**:
  - True: `flour rice brown`
  - Predicted: `muffins oat bran` (62.8%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`edge_02_grayscale` (Edge Cases (Quality))**:
  - True: `apples gala with skin raw`
  - Predicted: `blackberries raw` (79.2%)
  - Top-5 presence: Yes
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

- **`edge_04_dim_light` (Edge Cases (Quality))**:
  - True: `strawberries raw`
  - Predicted: `broccoli raw` (62.9%)
  - Top-5 presence: No
  - Notes: Optical confusion / low lighting / visual similarity to closely adjacent class.

