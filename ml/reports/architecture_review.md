# System Architecture Review — Nutrify AI

**Generated:** 2026-11-20 17:20:00 UTC  
**Scope:** Architectural evaluation of Frontend, Backend, Machine Learning Pipeline, and Data Flow.

---

## 1. System Architecture Overview

Nutrify AI implements a **Backend-For-Frontend (BFF)** micro-architecture separating client rendering from inference:

```
┌────────────────────────────────────────────────────────┐
│               Frontend Tier (Next.js 14)               │
│  - App Router (`/app/page.tsx`, `/app/analyze/page.tsx`)│
│  - Client Camera / Upload Components (`ImageUploadZone`)│
│  - Presentation & Nutrition Scoring Display            │
│  - Next.js API Proxy (`/app/api/analyze-food/route.ts`) │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST (Multipart)
                            ▼
┌────────────────────────────────────────────────────────┐
│               Backend Tier (FastAPI)                   │
│  - Security & Request Tracing Middleware               │
│  - IP Sliding-Window Rate Limiter                      │
│  - Binary Magic-Byte Image Preprocessor & Sanitizer    │
│  - Analytics & User Correction Feedback Loop           │
│  - Model Provider Switcher (`MLModelProvider`)         │
└───────────────────────────┬────────────────────────────┘
                            │ In-Memory Tensor
                            ▼
┌────────────────────────────────────────────────────────┐
│               ML Inference Engine                      │
│  - ProductionInferenceService                          │
│  - EfficientNet-B0 (1280-dim feature extraction)       │
│  - ScalableClassifierHead (1280 → 512 → 650 classes)   │
│  - Pure Softmax Probabilities                          │
│  - USDA Nutrition Database Resolver (650 Profiles)     │
└────────────────────────────────────────────────────────┘
```

---

## 2. Tier-by-Tier Architectural Evaluation

### 2.1 Frontend Tier (`frontend/`)
- **Strengths:**
  - Modern Next.js 14 App Router with clean component hierarchy (`components/ImageUploadZone.tsx`, `components/NutritionResultView.tsx`, `components/ConfidenceBadge.tsx`).
  - Next.js API Route acts as a robust BFF proxy (`frontend/lib/proxy.ts`), isolating client browsers from internal backend topologies.
  - AbortController forwarding cancels pending backend requests when a user navigates away or cancels an upload, preventing wasted compute.
  - Client-side error mapping translates backend HTTP failure codes (503, 504, 429) into intuitive user error cards.
- **Architectural Debt:**
  - Multiple fallback candidate URLs hardcoded in `proxy.ts`.

### 2.2 Backend Tier (`backend/`)
- **Strengths:**
  - FastAPI ASGI application factory pattern (`create_app()`) enables isolated testing via `TestClient`.
  - Non-blocking pure ASGI middleware (`SecurityAndTracingMiddleware`) injects security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `HSTS`) and monitors request time without stream buffering.
  - Decoupled Provider pattern (`MLModelProvider` vs `MockProvider`) permits instant mocking for frontend development without GPU/PyTorch overhead.
  - Feedback collection loop (`/api/v1/feedback`) streams user corrections to disk for active learning.
- **Architectural Debt:**
  - Rate limiter is currently in-memory (requires Redis for distributed multi-instance deployment).

### 2.3 ML Pipeline Tier (`ml/`)
- **Strengths:**
  - Standardized `ProductionInferenceService` wraps preprocessing, forward pass, candidate ranking, and nutritional scaling into a single deterministic class.
  - 100% pure softmax Bayesian probabilities with zero artificial forcing or floors.
  - Aspect-preserving letterbox canvas with ImageNet neutral gray padding prevents food distortion during resize.
- **Architectural Debt:**
  - High script count in `ml/src/` (40+ scripts). Active production code should be separated into a distinct `nutrify_ml/` package.

---

## 3. Data Pipeline & Taxonomy Integration

- **650-Class USDA Alignment:** Every class is mapped bijectively (`class_to_idx` and `idx_to_class`) with nutritional profiles loaded from `nutrition_mapping_650.json`.
- **Hard-Negative Enhancement:** 300 authentic real-world images integrated across 10 confusion clusters (curries, rice, breads) directly resolved domain collapse.

---

## 4. Architectural Readiness Score

| Subsystem | Architectural Soundness | Decoupling | Scalability | Maintainability | Overall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Frontend BFF** | 9 / 10 | 10 / 10 | 9 / 10 | 9 / 10 | **9.25 / 10** |
| **Backend API** | 9 / 10 | 9 / 10 | 8 / 10 | 9 / 10 | **8.75 / 10** |
| **Inference Service**| 9 / 10 | 9 / 10 | 9 / 10 | 9 / 10 | **9.00 / 10** |
| **ML Training Scripts**| 7 / 10 | 6 / 10 | 7 / 10 | 6 / 10 | **6.50 / 10** |
