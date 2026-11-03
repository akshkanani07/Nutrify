# ==============================================================================
# Nutrify AI — Hardened Production FastAPI Container
# Multi-stage, non-root user, offline-ready, CPU-optimized inference
# ==============================================================================

FROM python:3.12-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to minimize image size and compile overhead
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Final Production Runtime Stage
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged user for security hardening
RUN groupadd -g 1001 nutrify && \
    useradd -u 1001 -g nutrify -s /bin/bash -m nutrify

# Copy backend application source
COPY backend /app/backend
COPY ml /app/ml

# Ensure permissions
RUN chown -R nutrify:nutrify /app

USER nutrify

ENV PYTHONPATH=/app/backend:/app \
    APP_ENV=production \
    AI_PROVIDER=ml_model \
    ENABLE_ML_INTEGRATION=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
