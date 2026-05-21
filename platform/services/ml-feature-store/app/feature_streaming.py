"""
FeatureStreaming — Cómputo en tiempo real de features técnicas desde OHLCV.

Transforma ticks / velas normalizadas en el vector de 19 features canónicos
definidos en quant_shared.features. Opera como consumidor Kafka de
`los_ojos.market.normalized` y publica a `los_ojos.features.vector`.

MIGRACIÓN 2026-05-14:
  Los indicadores técnicos (_rsi, _macd_hist, _atr, _bollinger_width, _adx)
  fueron eliminados de esta clase. Ahora viven en:

      quant_shared.features.compute  (shared/quant_shared/features/compute.py)

  Esta es la fuente canónica de verdad — idéntica a la que usa research/
  en backtesting. Cambios en los indicadores se hacen UNA SOLA VEZ en shared/.

Nota sobre cambios semánticos vs versión anterior:
  - atr_14: ahora normalizado por precio (ATR/close). Antes era ATR absoluto.
  - sma_cross: ahora SMA20 vs SMA50 (corto plazo). Antes era SMA50 vs SMA200.
  Ambas definiciones canónicas en: shared/quant_shared/features/definitions.py
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import structlog

# ── Importar desde la fuente canónica ───────────────────────────────────────
from quant_shared.features.compute import (
    compute_features,
    FeatureVector,          # re-exportado para que los imports existentes funcionen
)
from quant_shared.features.definitions import FEATURE_COUNT

logger = structlog.get_logger(__name__)

# Re-exportar FeatureVector para que `from app.feature_streaming import FeatureVector`
# siga funcionando sin cambios en código que lo importa.
__all__ = ["FeatureStreaming", "FeatureVector"]


class FeatureStreaming:
    """
    Mantiene buffers de precios por símbolo y computa features en streaming.

    El cómputo real de indicadores está delegado a quant_shared.features.compute_features().
    Esta clase sólo se ocupa de:
      1. Mantener los deques de OHLCV por símbolo.
      2. Leer el contexto externo (regime_id, macro_leverage, etc.) desde Redis.
      3. Llamar a compute_features() y publicar el resultado a Kafka.

    Usage
    -----
    fs = FeatureStreaming(kafka_servers="...", redis_url="...")
    await fs.connect()
    await fs.run()  # blocking consumer loop
    """

    _CLOSE_WINDOW = 250   # bars retenidos por símbolo
    _MIN_BARS     = 14    # mínimo para que compute_features produzca valores no-default

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        redis_url:     str = "redis://localhost:6379/0",
        topic_in:      str = "los_ojos.market.normalized",
        topic_out:     str = "los_ojos.features.vector",
        group_id:      str = "feature-streaming",
    ):
        self._kafka_servers = kafka_servers
        self._redis_url     = redis_url
        self._topic_in      = topic_in
        self._topic_out     = topic_out
        self._group_id      = group_id

        # Buffers por símbolo
        self._closes:  dict[str, deque[float]] = {}
        self._highs:   dict[str, deque[float]] = {}
        self._lows:    dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._vwaps:   dict[str, deque[float]] = {}
        self._oi:      dict[str, deque[float]] = {}

        self._consumer = None
        self._producer = None
        self._redis    = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
            self._consumer = AIOKafkaConsumer(
                self._topic_in,
                bootstrap_servers=self._kafka_servers,
                group_id=self._group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
            )
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._consumer.start()
            await self._producer.start()
        except Exception as e:
            logger.warning("feature_streaming.kafka_unavailable", error=str(e))

        try:
            import aioredis
            self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
            logger.info("feature_streaming.connected")
        except Exception as e:
            logger.warning("feature_streaming.redis_unavailable", error=str(e))

    async def close(self) -> None:
        if self._consumer: await self._consumer.stop()
        if self._producer: await self._producer.stop()
        if self._redis:    await self._redis.close()

    async def run(self) -> None:
        """Main consumer loop."""
        if not self._consumer:
            logger.error("feature_streaming.not_connected")
            return
        async for msg in self._consumer:
            try:
                await self._handle_message(msg.value)
            except Exception as e:
                logger.error("feature_streaming.handle_error", error=str(e))

    async def compute_for(self, symbol: str, tick: dict) -> Optional[FeatureVector]:
        """
        Computa FeatureVector desde un tick dict.
        Uso directo desde market-intelligence para path de baja latencia.
        """
        self._update_buffers(symbol, tick)
        if len(self._closes.get(symbol, [])) < self._MIN_BARS:
            return None
        return await self._build_vector(symbol, tick)

    # ── Handlers ───────────────────────────────────────────────────────────

    async def _handle_message(self, data: dict) -> None:
        symbol = data.get("symbol", "")
        if not symbol:
            return
        self._update_buffers(symbol, data)
        fv = await self._build_vector(symbol, data)
        if fv and self._producer:
            await self._producer.send(self._topic_out, value=fv.to_dict())

    def _update_buffers(self, symbol: str, tick: dict) -> None:
        """Inicializa y actualiza los deques OHLCV por símbolo."""
        w = self._CLOSE_WINDOW
        for buf in (self._closes, self._highs, self._lows,
                    self._volumes, self._vwaps, self._oi):
            if symbol not in buf:
                buf[symbol] = deque(maxlen=w)

        close  = float(tick.get("price", tick.get("close", 0)) or 0)
        high   = float(tick.get("high",  close) or close)
        low    = float(tick.get("low",   close) or close)
        volume = float(tick.get("volume", 0) or 0)
        vwap   = float(tick.get("vwap",  close) or close)
        oi     = float(tick.get("open_interest", 0) or 0)

        if close > 0:
            self._closes[symbol].append(close)
            self._highs[symbol].append(high)
            self._lows[symbol].append(low)
            self._volumes[symbol].append(volume)
            self._vwaps[symbol].append(vwap)
            self._oi[symbol].append(oi)

    async def _build_vector(self, symbol: str, tick: dict) -> Optional[FeatureVector]:
        """
        Construye un FeatureVector canónico usando quant_shared.features.compute_features().

        No contiene lógica de indicadores — toda la computación está en shared/.
        """
        closes  = np.array(self._closes.get(symbol, []))
        highs   = np.array(self._highs.get(symbol, []))
        lows    = np.array(self._lows.get(symbol, []))
        volumes = np.array(self._volumes.get(symbol, []))
        vwaps   = np.array(self._vwaps.get(symbol, []))
        oi_arr  = np.array(self._oi.get(symbol, []))

        if len(closes) < self._MIN_BARS:
            return None

        # Microestructura del tick actual
        ob_imbalance = float(tick.get("ob_imbalance", 0) or 0)
        spread_bps   = float(tick.get("spread_bps",   0) or 0)
        funding_rate = float(tick.get("funding_rate",  0) or 0)
        vwap_current = float(vwaps[-1]) if len(vwaps) > 0 else float(closes[-1])
        oi_current   = float(oi_arr[-1]) if len(oi_arr) > 0 else 0.0
        oi_1h_ago    = float(oi_arr[-2]) if len(oi_arr) >= 2 else 0.0

        # Contexto externo desde Redis
        regime_id, macro_lev, reserve_z, whale_sent = await self._get_context(symbol)

        # Delegar todo el cómputo de indicadores a quant_shared ──────────────
        fv = compute_features(
            symbol=symbol,
            ts=datetime.now(timezone.utc).isoformat(),
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            ob_imbalance=ob_imbalance,
            spread_bps=spread_bps,
            vwap=vwap_current,
            funding_rate=funding_rate,
            oi_current=oi_current,
            oi_1h_ago=oi_1h_ago,
            regime_id=regime_id,
            macro_leverage=macro_lev,
            reserve_z=reserve_z,
            whale_sentiment=whale_sent,
        )

        # Sanity check: no NaN/Inf en el vector final
        arr = fv.to_array()
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            logger.warning(
                "feature_streaming.nan_in_vector",
                symbol=symbol,
                bad_features=[
                    name for name, val in fv.to_dict().items()
                    if isinstance(val, float) and (np.isnan(val) or np.isinf(val))
                ],
            )
            return None

        return fv

    async def _get_context(self, symbol: str) -> tuple[float, float, float, float]:
        """Lee regime_id, macro_leverage, reserve_z, whale_sentiment desde Redis."""
        if not self._redis:
            return 0.0, 1.0, 0.0, 0.0
        try:
            pipe = self._redis.pipeline()
            pipe.get("context:regime_id")
            pipe.get("macro:leverage_adj")
            pipe.get(f"onchain:reserve_z:{symbol[:3]}")
            pipe.get(f"onchain:whale_sentiment:{symbol[:3]}")
            results = await pipe.execute()
            return (
                float(results[0] or 0),
                float(results[1] or 1.0),
                float(results[2] or 0),
                float(results[3] or 0),
            )
        except Exception as e:
            logger.debug("feature_streaming.redis_miss", error=str(e))
            return 0.0, 1.0, 0.0, 0.0
