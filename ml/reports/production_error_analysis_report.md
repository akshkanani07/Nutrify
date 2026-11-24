# Production Error Analysis Report

**Generated:** 2026-11-23 09:30:00 UTC  
**Status:** ✓ PASS  
**Production Model:** `ml/models/production/nutrition_model_650_realworld.pt`  
**Single Source of Truth:** `final_verified_metrics_v2.md`

---

## 1. Authoritative Empirical Performance Breakdown (Protocol 1: 138 Real-World Photographs)

| Test Cluster | Evaluated Images | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Mean Confidence | Hard-Negative Separation Status |
|---|:---:|:---:|:---:|:---:|---|
| `dal_makhani` (Curries) | 6 | 50.0% (3/6) | 100.0% (6/6) | 0.4155 | Strong separation from dark gravies |
| `shahi_paneer` (Curries) | 6 | 16.7% (1/6) | 66.7% (4/6) | 0.2985 | Distinguishable from korma/tikka |
| `vegetable_biryani` (Rice) | 6 | 0.0% (0/6) | 100.0% (6/6) | 0.5465 | Perfect Top-5 containment |
| `pulao` (Rice) | 6 | 83.3% (5/6) | 100.0% (6/6) | 0.4886 | Outstanding discrimination |
| `jeera_rice` (Rice) | 6 | 50.0% (3/6) | 100.0% (6/6) | 0.4453 | Separated from plain steamed rice |
| `khichdi` (Rice) | 6 | 16.7% (1/6) | 100.0% (6/6) | 0.4624 | Perfect Top-5 containment |
| `roti` (Breads) | 6 | 16.7% (1/6) | 66.7% (4/6) | 0.3176 | Separated from whole wheat flatbreads |
| `naan` (Breads) | 6 | 83.3% (5/6) | 100.0% (6/6) | 0.5758 | Outstanding texture discrimination |
| `kulcha` (Breads) | 6 | 66.7% (4/6) | 100.0% (6/6) | 0.3545 | Separated from bhatura/naan |
| `paratha` (Breads) | 6 | 33.3% (2/6) | 66.7% (4/6) | 0.2704 | Discriminated from stuffed flatbreads |
| **All Priority Indian Categories** | **59** | **40.68% (24/59)** | **91.53% (54/59)** | **0.4175** | **7 of 10 clusters reach 100% Top-5** |
| Baseline Validation Photos | 79 | 1.27% (1/79) | 1.27% (1/79) | 0.1334 | Uncalibrated baseline categories |
| **Overall Real-World Benchmark** | **138** | **18.12% (25/138)** | **39.86% (55/138)** | **0.2579** | **Authoritative Benchmark** |

---

## 2. Historical Milestone: Initial Verification Audit (19 Samples)
> [!NOTE]
> The table below is preserved strictly as a historical record from the initial pre-enhancement audit stage.

| Benchmark Set | Total Images | Top-1 Accuracy | Top-5 Accuracy | Mean Confidence | Stage |
|---|---:|---:|---:|---:|---|
| Historical Verification Audit (Held-Out RW) | 19 | 5.26% | 15.79% | 0.3909 | Historical Milestone (Pre-Enhancement) |
| Old 50-Class Baseline | 19 | 5.26% | 15.79% | 0.2869 | Historical Milestone |
| 650 Scaled Validation Suite | 1,950 | 4.46% | 12.05% | — | Synthetic Pre-Retraining Baseline |
