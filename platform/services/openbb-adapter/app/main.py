"""
OpenBB Adapter — FastAPI entrypoint.

Servicio centralizado de acceso a OpenBB Platform 4.7.1.
Actúa como proxy inteligente con cache Redis, rate limiting
por provider y fallback automático.

Puertos:
  8009 — HTTP API

Routers:
  /macro        — Series FRED, yield curve, snapshot macro
  /crypto       — OHLCV histórico, funding rates, noticias
  /derivatives  — Opciones (Deribit), futuros basis, PCR
  /regulators   — SEC filings, RSS litigation, CFTC COT

Background:
  APScheduler   — polling automático a intervalos configurados
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .cache import ResponseCache
from .client import OpenBBClient
from .config import get_settings
from .models import HealthResponse
from .producers.crypto_producer import CryptoProducer
from .producers.macro_producer import MacroProducer
from .routers import crypto, derivatives, macro, regulators
from .scheduler import setup_scheduler

logger = structlog.get_logger(__name__)

# ── Singletons (inyectados en Depends de los routers) ─────────────────────────
_obb_client: Optional[OpenBBClient] = None
_scheduler  = None


def get_obb_client() -> OpenBBClient:
    """Dependency injection para routers."""
    if _obb_client is None:
        raise RuntimeError("OpenBBClient no inicializado — ¿olvidaste llamar connect()?")
    return _obb_client


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _obb_client, _scheduler

    settings = get_settings()
    logger.info("openbb_adapter.startup")

    # Redis
    redis = None
    try:
        import aioredis
        redis = await aioredis.from_url(
            settings.redis_url,
            db=settings.redis_db,
            decode_responses=True,
        )
        logger.info("openbb_adapter.redis_connected")
    except Exception as exc:
        logger.warning("openbb_adapter.redis_unavailable", error=str(exc))

    # OpenBB client
    cache = ResponseCache(redis)
    _obb_client = OpenBBClient(settings, cache)
    await _obb_client.configure()

    # Kafka producer
    producer = None
    try:
        from aiokafka import AIOKafkaProducer
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_servers,
            value_serializer=lambda v: v if isinstance(v, bytes) else v.encode("utf-8"),
        )
        await producer.start()
        logger.info("openbb_adapter.kafka_connected")
    except Exception as exc:
        logger.warning("openbb_adapter.kafka_unavailable", error=str(exc))

    # Producers
    macro_prod  = MacroProducer(_obb_client, producer, redis)
    crypto_prod = CryptoProducer(_obb_client, producer, redis)

    # Scheduler
    _scheduler = setup_scheduler(settings, macro_prod, crypto_prod)
    _scheduler.start()
    logger.info("openbb_adapter.scheduler_started")

    # Initial warmup — ejecutar macro poll inmediatamente para poblar cache
    try:
        await macro_prod.poll_all_series()
        logger.info("openbb_adapter.warmup_complete")
    except Exception as exc:
        logger.warning("openbb_adapter.warmup_error", error=str(exc))

    logger.info("openbb_adapter.ready port=8009")
    yield

    # Shutdown
    if _scheduler:
        _scheduler.shutdown(wait=False)

    if producer:
        await producer.stop()

    if redis:
        await redis.close()

    logger.info("openbb_adapter.shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Los Ojos — OpenBB Adapter",
    version="1.0.0",
    description=(
        "Proxy centralizado de OpenBB Platform 4.7.1. "
        "Provee datos macro (FRED), crypto OHLCV, derivados (Deribit) "
        "y regulatorios (SEC/CFTC) con cache Redis y fallback de providers."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(macro.router)
app.include_router(crypto.router)
app.include_router(derivatives.router)
app.include_router(regulators.router)


# ── Endpoints base ────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"], response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        openbb_ready=_obb_client is not None and _obb_client._obb is not None,
        cache_ready=_obb_client is not None and _obb_client._cache.available,
        ts=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", tags=["ops"])
async def root() -> dict:
    return {
        "service": "openbb-adapter",
        "version": "1.0.0",
        "docs":    "/docs",
        "routes": [
            "/macro/fred?series_id=UNRATE",
            "/macro/snapshot",
            "/macro/yield_curve",
            "/crypto/ohlcv/BTC",
            "/crypto/funding/BTC",
            "/derivatives/options/BTC",
            "/derivatives/options/BTC/pcr",
            "/regulators/sec/filings/IBIT",
            "/regulators/cftc/cot",
        ],
    }
