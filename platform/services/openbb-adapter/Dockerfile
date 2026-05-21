# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /build

# Build context = raíz del monorepo (ver platform-ci.yml)
COPY shared/ /build/shared/
COPY platform/services/openbb-adapter/requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
    -e /build/shared/ \
    -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local

# shared lib
COPY shared/ /shared/
RUN pip install --no-cache-dir -e /shared/

# app code
COPY platform/services/openbb-adapter/app/ ./app/

# OpenBB config
ENV OPENBB_DISABLE_TELEMETRY=true
ENV OPENBB_API_MODE=1

EXPOSE 8009
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8009/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8009", "--log-level", "info"]
