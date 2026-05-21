"""
MacroProducer — Publica series FRED a Kafka (los_ojos.macro.series).

Ciclo:
  1. Descarga todas las series FRED via OpenBBClient
  2. Calcula z-score rolling y cambio MoM
  3. Publica MacroDataEvent a Kafka
  4. Actualiza Redis con snapshot actualizado

Emite también MacroRegimeEvent si el régimen cambia.
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..client import OpenBBClient
from ..routers.macro import FRED_SERIES_META

logger = structlog.get_logger(__name__)


class MacroProducer:
    """
    Produce eventos macro a Kafka y Redis.

    Parameters
    ----------
    producer   : aiokafka.AIOKafkaProducer | None
    redis      : aioredis.Redis | None
    client     : OpenBBClient
    """

    TOPIC = "los_ojos.macro.series"

    def __init__(
        self,
        client: OpenBBClient,
        producer: Optional[Any] = None,
        redis: Optional[Any] = None,
    ):
        self._client   = client
        self._producer = producer
        self._redis    = redis
        self._last_values: dict[str, float] = {}

    async def poll_all_series(self) -> None:
        """Descarga todas las series FRED y publica eventos."""
        ok = 0
        for series_id, meta in FRED_SERIES_META.items():
            try:
                await self._poll_series(series_id, meta)
                ok += 1
            except Exception as exc:
                logger.warning(
                    "macro_producer.series_error",
                    series_id=series_id,
                    error=str(exc),
                )
        logger.info("macro_producer.poll_complete", ok=ok, total=len(FRED_SERIES_META))

    async def _poll_series(self, series_id: str, meta: dict) -> None:
        data = await self._client.get_fred_series(series_id, start_date="2010-01-01")
        if not data:
            return

        values = [d["value"] for d in data]
        last_value  = values[-1]
        prior_value = values[-2] if len(values) >= 2 else None

        # MoM %
        mom_pct = None
        if prior_value and prior_value != 0:
            mom_pct = round((last_value - prior_value) / abs(prior_value) * 100, 4)

        # Z-score rolling (ventana 36 períodos)
        window  = min(36, len(values))
        tail    = values[-window:]
        mu      = statistics.mean(tail)
        sd      = statistics.stdev(tail) if len(tail) > 1 else 1e-9
        z_score = round((last_value - mu) / max(sd, 1e-9), 4)

        event = {
            "event_type":   "MacroDataEvent",
            "source":       "openbb-adapter",
            "series_id":    series_id,
            "series_name":  meta["name"],
            "value":        last_value,
            "prior_value":  prior_value,
            "mom_pct":      mom_pct,
            "z_score":      z_score,
            "frequency":    meta["freq"],
            "category":     meta["category"],
            "date":         data[-1]["date"],
            "ts":           datetime.now(timezone.utc).isoformat(),
        }

        # Kafka
        if self._producer:
            try:
                await self._producer.send(
                    self.TOPIC,
                    value=json.dumps(event).encode("utf-8"),
                    key=series_id.encode("utf-8"),
                )
            except Exception as exc:
                logger.warning("macro_producer.kafka_error", series_id=series_id, error=str(exc))

        # Redis — key: macro:fred:{series_id}
        if self._redis:
            try:
                await self._redis.setex(
                    f"macro:fred:{series_id}",
                    86_400,  # TTL 24h
                    json.dumps(event, default=str),
                )
            except Exception as exc:
                logger.debug("macro_producer.redis_error", series_id=series_id, error=str(exc))

        self._last_values[series_id] = last_value

    async def poll_yield_curve(self) -> None:
        """Publica snapshot de la yield curve a Redis."""
        data = await self._client.get_yield_curve()
        if not data or not self._redis:
            return
        try:
            await self._redis.setex(
                "macro:yield_curve",
                14_400,  # TTL 4h
                json.dumps(data, default=str),
            )
            logger.debug("macro_producer.yield_curve_updated", points=len(data))
        except Exception as exc:
            logger.warning("macro_producer.yield_curve_error", error=str(exc))
