# ==============================================================================
# GeoDemand AI — Production Container Image
# ==============================================================================
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:/app/src/data:/app/src/features:/app/src/api"

WORKDIR /app

# Install system dependencies (build-essential, libgomp1 for LightGBM, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, data artifacts, configuration, and dashboard
COPY config/ ./config/
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/
COPY .env.example ./.env

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/system/ping || exit 1

# Start production uvicorn server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
