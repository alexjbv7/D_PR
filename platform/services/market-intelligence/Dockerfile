FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Shared lib first (better layer caching)
COPY shared/ /tmp/quant_shared/
RUN pip install --no-cache-dir /tmp/quant_shared/
COPY platform/libs/shared /app/libs/shared
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir /app/libs/shared

# Service python deps
COPY platform/services/market-intelligence/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code
COPY platform/services/market-intelligence/app/ ./app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
