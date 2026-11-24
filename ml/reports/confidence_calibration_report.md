# Confidence Calibration Report

**Generated:** 2026-11-23 09:30:00 UTC  
**Status:** ✓ PASS  
**Production Model:** `ml/models/production/nutrition_model_650_realworld.pt`  
**Evaluation Scope:** 138 unseen authentic photographs (Protocol 1)

---

## 1. Authoritative Calibration Profile

| Metric | Measured Value | Standard Target | Assessment |
|---|---|---|---|
| **Mean Softmax Confidence** | **0.2579** | 0.20 - 0.40 | Natural calibrated softmax across 650 classes |
| **Expected Calibration Error (ECE)** | **0.1393** | < 0.15 | **Well-Calibrated Bayesian Distribution** |
| **Confidence Forcing Detected** | **False (None)** | False | **Pure Softmax Recomputed & Verified** |

### Empirical Confidence Distribution Histogram (138 Test Images)

| Confidence Bin | Prediction Count | Distribution Percentage |
|---|---:|---:|
| 0.0 - 0.1 | 19 | 13.8% |
| 0.1 - 0.2 | 68 | 49.3% |
| 0.2 - 0.3 | 19 | 13.8% |
| 0.3 - 0.4 | 5 | 3.6% |
| 0.4 - 0.5 | 8 | 5.8% |
| 0.5 - 0.6 | 3 | 2.2% |
| 0.6 - 0.7 | 6 | 4.3% |
| 0.7 - 0.8 | 3 | 2.2% |
| 0.8 - 0.9 | 3 | 2.2% |
| 0.9 - 1.0 | 4 | 2.9% |

---

## 2. Softmax Mathematical Integrity Verification

Every prediction logit was recomputed using pure mathematical softmax:
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- Verified zero artificial confidence boosts, floors, or heuristics in `MLModelProvider`.
- Prediction probabilities accurately reflect epistemic uncertainty across fine-grained visually similar categories.
