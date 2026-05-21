# =============================================================================
# execution-engine — multi-stage build
# =============================================================================
# Stage 1 (builder):  installs every Python dep into /install using gcc +
#                     libpq-dev (compilation tools).
# Stage 2 (runtime):  copies only /install into a clean slim image with the
#                     runtime libs (libpq5 + curl for healthcheck).  No gcc,
#                     no headers — typically saves ~300 MB vs single-stage.
# =============================================================================

# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder
# -----------------------------------------------------------------------------

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Shared libraries from monorepo root
COPY shared/ /tmp/quant_shared/
COPY platform/libs/shared /tmp/libs_shared/
COPY platform/services/execution-engine/requirements.txt .

# Install everything into a relocatable prefix that we can copy into the
# runtime image.  --prefix=/install pairs with `COPY --from=builder /install
# /usr/local` in stage 2.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefix=/install /tmp/quant_shared/ \
 && pip install --no-cache-dir --prefix=/install /tmp/libs_shared/ \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime
# -----------------------------------------------------------------------------

WORKDIR /app

# Runtime-only system libs (libpq5 << libpq-dev; curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 curl \
 && rm -rf /var/lib/apt/lists/*

# Pull installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# App code
COPY platform/services/execution-engine/app/ ./app/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
