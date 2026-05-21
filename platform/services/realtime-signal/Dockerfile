FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && \
    rm -rf /var/lib/apt/lists/*

COPY shared/ /tmp/quant_shared/
RUN pip install --no-cache-dir /tmp/quant_shared/
COPY platform/libs/shared /app/libs/shared
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir /app/libs/shared

COPY platform/services/realtime-signal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY platform/services/realtime-signal/app/ ./app/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
# Single worker for stateful WebSocket connections
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]
