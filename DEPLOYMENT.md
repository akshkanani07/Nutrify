# Nutrify AI — Enterprise Production Deployment Guide

This guide details the deployment architecture, configuration, security hardening, and operational steps for launching the Nutrify AI application in production.

---

## 1. Architecture Overview

Nutrify AI operates as a decoupled, multi-tier microservice architecture:

```
[ Public Users / Mobile Browsers ]
                 │
                 ▼ HTTPS (443)
       ┌──────────────────┐
       │   Edge Proxy     │  (Cloudflare / AWS CloudFront / Vercel Edge)
       │ Rate Limiting,   │
       │ SSL Termination  │
       └────────┬─────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌──────────────┐ ┌───────────────────────────┐
│ Next.js SSR  │ │ FastAPI ML Inference API  │
│ (Port 3000)  │ │ (Port 8000)               │
│ - UI/UX      │ │ - EfficientNet-B0 (650)   │
│ - BFF Proxy  │ │ - Nutrition Accuracy DB   │
│ - SSR        │ │ - Analytics & Feedback    │
└──────────────┘ └───────────────────────────┘
```

---

## 2. Hardware & Infrastructure Requirements

### Minimum Requirements (Production 50 concurrent users)
- **CPU**: 4 vCPUs (Intel Xeon / AMD EPYC / Graviton2)
- **RAM**: 4 GB RAM (ML weights & pipeline footprint: ~340 MB RSS)
- **Disk**: 20 GB SSD
- **OS**: Linux (Ubuntu 22.04 LTS / Debian 12 / Amazon Linux 2023) or Docker host

### Recommended Requirements (Production 200+ concurrent users)
- **CPU**: 8 vCPUs
- **RAM**: 8 GB RAM
- **GPU (Optional)**: NVIDIA T4 / A10G (for <20 ms batch throughput; CPU latency is already ~78 ms)

---

## 3. Environment Variables Reference

### Backend (`backend/.env`)

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `APP_ENV` | string | `production` | Set to `production` to enforce security checks |
| `AI_PROVIDER` | string | `ml_model` | Core provider: `ml_model` |
| `ALLOW_MOCK` | boolean | `false` | Disallow mock demo provider in production |
| `ENABLE_ML_INTEGRATION` | boolean | `true` | Enables real model weights inference |
| `ALLOWED_ORIGINS` | string | `https://nutrify.yourdomain.com` | Comma-separated CORS allowed domains |
| `RATE_LIMIT_MAX_REQUESTS`| int | `60` | Sliding window request limit per client |
| `RATE_LIMIT_WINDOW_SECONDS`| int | `60` | Rate limit window in seconds |
| `TRUST_EDGE_PROXY` | boolean | `true` | Evaluates rightmost `X-Forwarded-For` from reverse proxy |

### Frontend (`frontend/.env`)

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `BACKEND_API_URL` | string | `http://127.0.0.1:8000` | Target URL for backend FastAPI service |
| `TRUST_EDGE_PROXY` | boolean | `true` | Forwards client IP to backend |
| `NODE_ENV` | string | `production` | Production Next.js build |

---

## 4. Deployment Methods

### Option A: Docker Compose (Single Host / VPS)

1. Clone repository to server:
   ```bash
   git clone https://github.com/your-org/Nutrify.git /opt/nutrify
   cd /opt/nutrify
   ```

2. Configure environment files:
   ```bash
   cp backend/.env.production.example backend/.env
   cp frontend/.env.production.example frontend/.env
   # Edit with your production domain
   ```

3. Start services with Docker Compose:
   ```bash
   docker compose up -d --build
   ```

4. Verify health:
   ```bash
   curl http://localhost:8000/api/v1/health
   # Expected: {"status":"ok","version":"2.0.0","provider":"ml_model","provider_ready":true}
   ```

---

### Option B: Hybrid Cloud Deployment (Vercel Frontend + AWS/GCP Backend)

1. **Deploy Backend to AWS ECS / GCP Cloud Run / Ubuntu VM**:
   ```bash
   docker build -t nutrify-backend:latest -f backend/Dockerfile .
   docker run -d -p 8000:8000 \
     -e APP_ENV=production \
     -e AI_PROVIDER=ml_model \
     -e ENABLE_ML_INTEGRATION=true \
     -e ALLOWED_ORIGINS=https://nutrify.vercel.app \
     --name nutrify-backend nutrify-backend:latest
   ```

2. **Deploy Frontend to Vercel**:
   - Link repository in Vercel Dashboard.
   - Set Root Directory to `frontend`.
   - Add Environment Variable:
     - `BACKEND_API_URL`: `https://api.yourdomain.com`
     - `TRUST_EDGE_PROXY`: `true`
   - Deploy.

---

## 5. Security Checklist

- [x] **Non-Root Execution**: Backend runs under unprivileged user `nutrify` (UID 1001).
- [x] **Security Headers**: HSTS, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`.
- [x] **Rate Limiting**: Per-IP sliding-window limiter (60 req/min).
- [x] **Magic-Byte Image Validation**: Strict validation for JPEG, PNG, WEBP preventing MIME spoofing.
- [x] **Payload Safeguards**: Strict 10 MB payload ceiling and dimension validation ($64\times 64$ px min to $1024\times 1024$ px max).
- [x] **Offline Guard**: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` prevent arbitrary external connections.

---

## 6. Monitoring & Maintenance

- **Health Endpoint**: `GET /api/v1/health`
- **Real-time Analytics**: `GET /api/v1/metrics`
- **Prediction Logs**: `ml/data/reports/prediction_logs.json` and `ml/data/reports/prediction_logs.jsonl`
- **User Feedback Storage**: `ml/data/feedback/user_feedback.jsonl`

