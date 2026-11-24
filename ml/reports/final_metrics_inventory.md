# Final Metrics Inventory

**Generated:** 2026-11-23 09:30:00 UTC  
**Scope:** Repository-wide scan of all metric-bearing documents, logs, and artifacts across the Nutrify AI project.  
**Single Source of Truth:** `final_verified_metrics_v2.md`

---

## 1. Authoritative Benchmark (v2.0) — Single Source of Truth

The authoritative benchmark for the Nutrify AI platform, empirically verified on the production checkpoint `nutrition_model_650_realworld.pt`:

| Parameter | Authoritative Value | Protocol | Status |
|---|:---:|---|:---:|
| **Real-World Top-1 Accuracy** | **18.12%** (25 / 138) | Protocol 1 (138 unseen authentic photographs) | **AUTHORITATIVE** |
| **Real-World Top-5 Accuracy** | **39.86%** (55 / 138) | Protocol 1 (138 unseen authentic photographs) | **AUTHORITATIVE** |
| **Priority Indian Cuisine Top-1** | **40.68%** (24 / 59) | Protocol 1 (59 priority authentic photographs) | **AUTHORITATIVE** |
| **Priority Indian Cuisine Top-5** | **91.53%** (54 / 59) | Protocol 1 (59 priority authentic photographs) | **AUTHORITATIVE** |
| **Macro Precision** | **1.91%** | Protocol 1 | **AUTHORITATIVE** |
| **Macro Recall** | **3.21%** | Protocol 1 | **AUTHORITATIVE** |
| **Macro F1 Score** | **2.30%** | Protocol 1 | **AUTHORITATIVE** |
| **Expected Calibration Error (ECE)** | **0.1393** | Protocol 1 | **AUTHORITATIVE** |
| **Average Softmax Confidence** | **0.2579** | Protocol 1 | **AUTHORITATIVE** |
| **CPU Forward Latency** | **139.94 ms** | Protocol 1 | **AUTHORITATIVE** |
| **FastAPI End-to-End Latency** | **127.49 ms** | Protocol 4 | **AUTHORITATIVE** |
| **Total Dataset Size** | **166,050 records** | Processed Dataset (136,740 train / 29,310 val) | **AUTHORITATIVE** |

---

## 2. Inventory of Historical & Intermediate Documents

The following documents contain intermediate historical values generated during earlier development milestones:

| Report File | Location | Historical Metrics Reported | Status / Role |
|---|---|---|---|
| `final_accuracy_comparison_report.md` | `ml/reports/` | Top-1: 18.12%, Top-5: 39.86% | **Authoritative Comparison** |
| `final_realworld_evaluation_report.md` | `ml/reports/` | Top-1: 18.12%, Top-5: 39.86% | **Authoritative Breakdown** |
| `final_verified_metrics_v2.md` | `ml/reports/` | Top-1: 18.12%, Top-5: 39.86% | **Single Source of Truth** |
| `final_consistency_certificate_v2.md` | `ml/reports/` | Top-1: 18.12%, Top-5: 39.86% | **Official Certificate** |
| `final_project_handover_v2.md` | `ml/reports/` | Top-1: 18.12%, Top-5: 39.86% | **Official Handover** |
| `final_training_report.md` | `ml/reports/` | Train Loss: 0.8911, Val Top-1: 25.54%, Val Top-5: 53.85% | Historical Training Partition Run |
| `verification_audit_results.json` | `ml/data/reports/` | Top-1: 5.26%, Top-5: 15.79% | Historical 19-sample audit |
| `evaluation_650_report.json` | `ml/data/reports/` | Top-1: 4.46%, Top-5: 12.05% | Historical Synthetic 650 baseline |
| `dataset_update_report.md` | `ml/reports/` | Total: 166,050 records (+300 authentic images) | Verified Dataset Ingestion |

---
*Cataloged and synchronized by automated consistency auditor.*
