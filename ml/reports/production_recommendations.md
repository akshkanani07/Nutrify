# Production Recommendations & Roadmap — Nutrify AI

**Generated:** 2026-11-20 17:20:00 UTC  
**Scope:** Actionable engineering recommendations prioritized for production scaling, reliability, and accuracy improvement.

---

## 1. Prioritized Action Matrix

| Priority | Category | Recommendation | Effort | Impact |
|:---:|---|---|:---:|:---:|
| **P0** | **Deployment** | Add `torch` and `torchvision` explicitly to `backend/requirements.txt` | 15 mins | Critical for automated Docker/CI builds |
| **P0** | **Security** | Cap PIL `Image.MAX_IMAGE_PIXELS = 25_000_000` to prevent decompression bomb DoS | 15 mins | Prevents memory exhaustion attacks |
| **P1** | **Performance** | Downscale images to max 512px before running Laplacian blur filter in preprocessor | 1 hour | Reduces high-res image latency by 60–75% |
| **P1** | **Architecture** | Back `RateLimiter` with Redis for distributed multi-worker scaling | 1 day | Enforces global rate limits across workers |
| **P1** | **Privacy** | Anonymize or hash client IP addresses in `prediction_logs.jsonl` (GDPR compliance) | 30 mins | Ensures user privacy compliance |
| **P2** | **Accuracy** | Unfreeze convolutional stages 6–7 of EfficientNet-B0 with RandAugment retraining | 2 days | Expected to boost real-world Top-1 to >35% |
| **P2** | **MLOps** | Archive obsolete scripts in `ml/src/` and `scratch/` into `_archive/` | 2 hours | Eliminates developer debt and dead code |
| **P3** | **Infrastructure** | Export production model to ONNX / TensorRT for GPU serving | 1 day | Reduces latency to <15ms under high traffic |

---

## 2. Detailed Action Plans

### Action 1: Fix Dependency Manifest for Docker / CI (P0)
- **Current Issue:** `backend/requirements.txt` lacks `torch` and `torchvision`.
- **Implementation:**
  ```text
  fastapi==0.165.0
  uvicorn[standard]==0.54.0
  pydantic==2.14.4
  python-multipart==0.0.34
  pillow==12.3.0
  httpx==0.28.1
  python-dotenv==1.2.4
  torch==2.14.1
  torchvision==0.29.1
  ```

### Action 2: Image Decompression Bomb Protection (P0)
- **Current Issue:** A crafted 10 MB payload can decompress into hundreds of millions of pixels.
- **Implementation:** Add to `backend/app/image_utils.py` and `ml/src/production_pipeline.py`:
  ```python
  from PIL import Image
  Image.MAX_IMAGE_PIXELS = 25_000_000
  ```

### Action 3: Preprocessor Latency Optimization (P1)
- **Current Issue:** Laplacian variance runs on raw 4K resolutions before downscaling to 224×224.
- **Implementation:**
  ```python
  # Downscale before filter if image exceeds 512px on longest side
  max_dim = max(pil_img.size)
  if max_dim > 512:
      scale = 512.0 / max_dim
      eval_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.Resampling.BILINEAR)
  else:
      eval_img = pil_img
  grayscale = eval_img.convert("L")
  ```

### Action 4: Real-World Model Retraining Roadmap (P2)
- **Current State:** EfficientNet-B0 backbone with optimized multi-view representation head reached 25.54% val Top-1, **18.12% held-out real-world Top-1** (39.86% Top-5), and **40.68% Top-1 / 91.53% Top-5** on priority Indian dishes.
- **Next Iteration Roadmap:**
  1. Leverage the operational `/api/v1/feedback` endpoint to accumulate 50–100 authentic images per priority class.
  2. Unfreeze stages 6 and 7 of EfficientNet-B0 with a differential learning rate ($10^{-5}$ for backbone, $10^{-3}$ for head).
  3. Apply CutMix and RandAugment to bridge the texture domain shift.
