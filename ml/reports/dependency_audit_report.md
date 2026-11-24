# Dependency & Environment Audit Report — Nutrify AI

**Generated:** 2026-11-20 17:20:00 UTC  
**Scope:** Python environment (`backend/requirements.txt`, system packages), Node.js environment (`frontend/package.json`), Docker container configurations, and dependency security.

---

## 1. Executive Summary

The dependency architecture of Nutrify AI is cleanly decoupled between the Python backend service and the Next.js frontend client:
- **Backend:** Light core footprint (`fastapi`, `uvicorn`, `pydantic`, `pillow`, `httpx`, `python-dotenv`, `python-multipart`, `torch`, `torchvision`).
- **Frontend:** Modern TypeScript stack (`next 14.2.5`, `react 18`, `tailwindcss 3.4`, `@radix-ui/react-icons`, `lucide-react`).
- Zero high-severity CVEs or deprecated abandoned libraries in active production paths.

---

## 2. Dependency Audit Findings

### Finding 1: Dependency Version Pinning vs Drift in Python Environment
- **Severity:** **Medium**
- **File Path:** `backend/requirements.txt`
- **Current File Contents:**
  ```
  fastapi==0.115.8
  uvicorn[standard]==0.34.0
  pydantic==2.10.6
  python-multipart==0.0.20
  pillow==11.1.0
  httpx==0.28.1
  python-dotenv==1.0.1
  ```
- **Root Cause:** In the local development environment, newer minor versions (`fastapi 0.165.0`, `pydantic 2.14.4`, `uvicorn 0.54.0`) were installed during validation, creating minor version drift between `backend/requirements.txt` and the verified local lock.
- **Recommended Fix:** Update `backend/requirements.txt` to reflect the exact verified packages or generate a deterministic `requirements.lock` file via `pip freeze`.
- **Expected Impact:** Guarantees bit-for-bit reproducible container builds in CI/CD pipelines.

### Finding 2: Missing Explicit PyTorch Dependencies in `requirements.txt`
- **Severity:** **High**
- **File Path:** `backend/requirements.txt`
- **Root Cause:** `torch` and `torchvision` are imported in `backend/app/providers/ml_provider.py` and `ml/src/production_pipeline.py`, but are not listed in `backend/requirements.txt` (they were installed in the global Python environment). A fresh deployment using only `pip install -r backend/requirements.txt` will fail with `ModuleNotFoundError: No module named 'torch'`.
- **Recommended Fix:** Add explicit torch dependencies to `backend/requirements.txt`:
  ```
  torch==2.14.1
  torchvision==0.29.1
  ```
- **Expected Impact:** Critical for automated Docker and Kubernetes deployment pipelines.

### Finding 3: Starlette Deprecation Warning with `httpx`
- **Severity:** **Low**
- **File Path:** `backend/tests/conftest.py` & `fastapi.testclient`
- **Root Cause:** FastAPI `TestClient` emitted a `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.`
- **Recommended Fix:** Suppress or upgrade test dependencies when updating to Starlette 1.10+.
- **Expected Impact:** Eliminates test warning output.

### Finding 4: Frontend Node Engine Specification
- **Severity:** **Low**
- **File Path:** `frontend/package.json` & `frontend/.nvmrc`
- **Current State:** `.nvmrc` specifies `20.18.0`. `package.json` lacks an explicit `"engines"` block.
- **Recommended Fix:** Add `"engines": { "node": ">=20.18.0", "npm": ">=10.0.0" }` to `frontend/package.json`.
- **Expected Impact:** Prevents builds from executing on unsupported Node.js versions in Vercel or cloud hosting.

---

## 3. Dependency Inventory & License Compliance

| Library | Version Verified | License | Purpose | Compliance |
|---|---|---|---|:---:|
| `fastapi` | 0.165.0 | MIT | Asynchronous REST API framework | Clean |
| `uvicorn` | 0.54.0 | BSD-3-Clause | ASGI web server | Clean |
| `pydantic` | 2.14.4 | MIT | Data validation and JSON schemas | Clean |
| `pillow` | 12.3.0 | HPND | Image decoding and EXIF handling | Clean |
| `torch` | 2.14.1 | BSD-3-Clause | Deep learning inference framework | Clean |
| `torchvision` | 0.29.1 | BSD-3-Clause | Vision backbone & tensor transforms | Clean |
| `python-dotenv` | 1.2.4 | BSD-3-Clause | Environment variable management | Clean |
| `next` | 14.2.5 | MIT | React full-stack framework & BFF | Clean |
| `react` | 18.3.1 | MIT | User interface rendering | Clean |
| `tailwindcss` | 3.4.1 | MIT | Utility-first CSS framework | Clean |
| `lucide-react` | 0.428.0 | ISC | UI iconography | Clean |
