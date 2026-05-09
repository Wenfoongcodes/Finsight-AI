# ─────────────────────────────────────────────────────────────────────────────
# FinSight AI — Production Dockerfile (multi-stage)
# Stage 1: builder  — installs Python dependencies into a virtual env
# Stage 2: runtime  — slim image copying only the venv and app source
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for compiling wheels (numpy, faiss, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (ml.txt includes base.txt)
COPY requirements/ requirements/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements/ml.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN addgroup --system finsight && adduser --system --ingroup finsight finsight

WORKDIR /app

# Runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY app/         app/
COPY configs/     configs/
COPY main.py      main.py

# Create runtime directories and set ownership
RUN mkdir -p data/raw data/processed data/embeddings data/models logs && \
    chown -R finsight:finsight /app

USER finsight

# ── Environment defaults (override via docker-compose or env file) ────────────
ENV ENVIRONMENT=production \
    DEBUG=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

# Healthcheck — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
