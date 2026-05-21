"""
Drift cron startup — wires DriftCron into the ml-feature-store lifespan.
"""
from __future__ import annotations

import os
from typing import Any

import asyncpg
import structlog
from aiokafka import AIOKafkaProducer
from prometheus_client import make_asgi_app

logger = structlog.get_logger(__name__)

_drift_cron: Any = None
_kafka_producer: Any = None
_pg_pool: asyncpg.Pool | None = None


def prometheus_asgi():
    """ASGI mount for ``/metrics`` (drift + default collectors)."""
    return make_asgi_app()


async def start_drift_cron(
    *,
    redis: Any,
    postgres_dsn: str,
    kafka_servers: str,
) -> Any | None:
    """
    Start the daily drift detection loop if ``DRIFT_CRON_ENABLED`` is true.

    Returns the ``DriftCron`` instance, or None when disabled.
    """
    global _drift_cron, _kafka_producer, _pg_pool

    enabled = os.getenv("DRIFT_CRON_ENABLED", "true").lower() not in (
        "0", "false", "no",
    )
    if not enabled:
        logger.info("drift_cron.disabled", reason="DRIFT_CRON_ENABLED=false")
        return None

    if not postgres_dsn:
        logger.warning("drift_cron.skipped", reason="no POSTGRES_DSN")
        return None

    from .drift_cron import DriftCron
    from .macro_event_filter import MacroEventFilter

    _pg_pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=3)
    _kafka_producer = AIOKafkaProducer(bootstrap_servers=kafka_servers)
    await _kafka_producer.start()

    _drift_cron = DriftCron(
        pool=_pg_pool,
        redis=redis,
        producer=_kafka_producer,
        mac_filter=MacroEventFilter(),
    )
    logger.info("drift_cron.scheduled", hour_utc=3)
    return _drift_cron


async def stop_drift_cron() -> None:
    """Stop Kafka producer and close the drift DB pool."""
    global _kafka_producer, _pg_pool

    if _kafka_producer is not None:
        await _kafka_producer.stop()
        _kafka_producer = None
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None
