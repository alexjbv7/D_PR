"""
CryptoProducer — Publica OHLCV y funding rates a Kafka.

Topics:
  - los_ojos.market.normalized  — ticks OHLCV normalizados
  - los_ojos.derivatives.events — funding rates y datos de derivados
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..client import OpenBBClient

logger = structlog.get_logger(__name__)


class CryptoProducer:
    """
    Produce eventos de mercado crypto a Kafka y Redis.

    Diseñado para complementar (NO reemplazar) el feed Binance WS
    de market-intelligence. Provee datos históricos y de menor frecuencia.
    """

    TOPIC_MARKET    = "los_ojos.market.normalized"
    TOPIC_DERIVS    = "los_ojos.derivatives.events"

    def __init__(
        self,
        client: OpenBBClient,
        producer: Optional[Any] = None,
        redis: Optional[Any] = None,
    ):
        self._client   = client
        self._producer = producer
        self._redis    = redis

    async def poll_ohlcv(
        self,
        symbols: list[str],
        interval: str = "1d",
    ) -> None:
        """
        Descarga OHLCV para todos los símbolos y publica el último bar a Kafka.

        Solo publica el bar más reciente (no backfill) — los históricos
        van directamente a TimescaleDB via el endpoint REST.
        """
        for symbol in symbols:
            try:
                data = await self._client.get_crypto_ohlcv(
                    symbol, interval=interval, start_date="2025-01-01"
                )
                if not data:
                    continue

                last_bar = data[-1]
                event = {
                    "event_type": "OHLCVEvent",
                    "source":     "openbb-adapter",
                    "symbol":     f"{symbol}USDT",
                    "interval":   interval,
                    "open":       last_bar.get("open"),
                    "high":       last_bar.get("high"),
                    "low":        last_bar.get("low"),
                    "close":      last_bar.get("close"),
                    "volume":     last_bar.get("volume"),
                    "date":       str(last_bar.get("date", "")),
                    "ts":         datetime.now(timezone.utc).isoformat(),
                }

                if self._producer:
                    await self._producer.send(
                        self.TOPIC_MARKET,
                        value=json.dumps(event, default=str).encode("utf-8"),
                        key=f"{symbol}USDT".encode("utf-8"),
                    )

                # Redis: último bar para feature-store de baja latencia
                if self._redis:
                    await self._redis.setex(
                        f"ohlcv:{symbol}:{interval}:last",
                        300,
                        json.dumps(event, default=str),
                    )

            except Exception as exc:
                logger.warning(
                    "crypto_producer.ohlcv_error",
                    symbol=symbol, interval=interval, error=str(exc),
                )

        logger.debug("crypto_producer.ohlcv_poll_done", symbols=symbols, interval=interval)

    async def poll_funding_rates(self, symbols: list[str]) -> None:
        """Descarga funding rates y los publica a los_ojos.derivatives.events."""
        for symbol in symbols:
            try:
                data = await self._client.get_crypto_funding_rate(symbol)
                if not data:
                    continue

                last = data[-1]
                event = {
                    "event_type":   "FundingRateEvent",
                    "source":       "openbb-adapter",
                    "symbol":       f"{symbol}USDT",
                    "funding_rate": last.get("funding_rate"),
                    "date":         str(last.get("date", "")),
                    "ts":           datetime.now(timezone.utc).isoformat(),
                }

                if self._producer:
                    await self._producer.send(
                        self.TOPIC_DERIVS,
                        value=json.dumps(event, default=str).encode("utf-8"),
                        key=f"{symbol}USDT".encode("utf-8"),
                    )

                if self._redis:
                    await self._redis.setex(
                        f"funding:{symbol}:last",
                        28_800,
                        json.dumps(event, default=str),
                    )

            except Exception as exc:
                logger.warning(
                    "crypto_producer.funding_error",
                    symbol=symbol, error=str(exc),
                )
