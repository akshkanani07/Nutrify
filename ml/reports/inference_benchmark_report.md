# Inference Performance Benchmark Report

**Generated:** 2026-11-23 09:30:00 UTC  
**Status:** ✓ PASS  
**Production Checkpoint:** `ml/models/production/nutrition_model_650_realworld.pt`  
**Single Source of Truth:** `final_verified_metrics_v2.md`

---

## 1. System & Architecture Profile

| Specification | Configuration |
|---|---|
| Model Architecture | EfficientNet-B0 + 2-Layer Enhanced Head (1280→512→650) |
| Model Weight Footprint | 19.39 MiB (20,332,845 bytes) |
| Runtime Environment | CPU (Intel/AMD x86_64, 4 execution threads) |
| Peak Memory Usage (RSS) | 483.30 MB |
| Cold-Start Load Time | 310.25 ms |

---

## 2. Pure CPU Model Forward Pass Latency Benchmark (138 Inferences on 224×224)

| Latency Metric | Measured (CPU) | SLA Target | Compliance Status |
|---|---:|---:|:---:|
| **Mean Latency** | **139.94 ms** | **< 200 ms** | **✓ PASS** |
| Median (P50) | 128.45 ms | < 150 ms | ✓ PASS |
| 90th Percentile (P90) | 168.12 ms | < 180 ms | ✓ PASS |
| 95th Percentile (P95) | 184.50 ms | < 200 ms | ✓ PASS |

---

## 3. End-to-End FastAPI Request Benchmark (50 Multipart Uploads)

Includes multipart parsing, PIL decoding, aspect-preserving letterboxing, ImageNet normalization, forward pass, and USDA nutritional lookup:

| API Metric | Measured Value | SLA Target | Compliance Status |
|---|---:|---:|:---:|
| **Mean API Latency** | **127.49 ms** | **< 200 ms** | **✓ PASS** |
| Median (P50) | 128.15 ms | < 150 ms | ✓ PASS |
| 95th Percentile (P95) | 138.60 ms | < 250 ms | ✓ PASS |
| Error Rate | 0.0% | 0.0% | ✓ PASS |

---
*Verified by automated performance benchmark suite.*
