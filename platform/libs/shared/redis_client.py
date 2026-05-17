"""
Redis Client — Async cache + pub/sub para todos los servicios.
==============================================================
Wrapper sobre redis.asyncio con:
- Serialización JSON automática para Pydantic models
- TTL estándar por categoría
- Pub/Sub para signals realtime
- Pipeline helper para batch ops
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional, Type, TypeVar

import redis.asyncio as aioredis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_PUBSUB = os.getenv("REDIS_PUBSUB_URL", "redis://localhost:6379/1")

T = TypeVar("T", bound=BaseModel)

# TTLs estándar (segundos)
TTL = {
    "tick":         10,
    "ohlcv":        300,
    "feature":      600,
    "regime":       300,
    "macro":        3600 * 6,
    "signal":       60,
    "orderbook":    5,
    "funding":      60,
    "whale":        3600,
    "session":      3600 * 24,
    "default":      300,
}

# Canales pub/sub
CHANNELS = {
    "signals":    "ch:signals:final",
    "whale":      "ch:onchain:whale",
    "macro":      "ch:macro:event",
    "regime":     "ch:regime:update",
    "anomaly":    "ch:system:anomaly",
    "kill":       "ch:system:kill",
}


class RedisCache:
    """
    Cache asíncrono con helpers para objetos Pydantic.

    pool = RedisCache()
    await pool.connect()
    await pool.set_model("features:BTCUSDT", feature_event, ttl="feature")
    event = await pool.get_model("features:BTCUSDT", FeatureUpdateEvent)
    """

    def __init__(self, url: str = REDIS_URL):
        self._url = url
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = await aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        logger.info("Redis connected → %s", self._url)

    async def close(self):
        if self._client:
            await self._client.aclose()

    # ── Generic ────────────────────────────────────────────────────────

    async def set(self, key: str, value: Any, ttl: int = TTL["default"]) -> None:
        await self._client.set(key, json.dumps(value), ex=ttl)

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._client.get(key)
        return json.loads(raw) if raw else None

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))

    # ── Pydantic models ────────────────────────────────────────────────

    async def set_model(
        self,
        key: str,
        model: BaseModel,
        ttl: int | str = TTL["default"],
    ) -> None:
        ttl_val = TTL.get(ttl, ttl) if isinstance(ttl, str) else ttl
        await self._client.set(key, model.model_dump_json(), ex=ttl_val)

    async def get_model(self, key: str, model_cls: Type[T]) -> Optional[T]:
        raw = await self._client.get(key)
        if raw is None:
            return None
        return model_cls.model_validate_json(raw)

    # ── Hash (feature vectors) ─────────────────────────────────────────

    async def hset_features(
        self, key: str, features: dict[str, float], ttl: int = TTL["feature"]
    ) -> None:
        pipe = self._client.pipeline()
        pipe.hset(key, mapping={k: str(v) for k, v in features.items()})
        pipe.expire(key, ttl)
        await pipe.execute()

    async def hget_features(self, key: str) -> dict[str, float]:
        raw = await self._client.hgetall(key)
        return {k: float(v) for k, v in raw.items()} if raw else {}

    # ── Sorted set (time-series buffer) ───────────────────────────────

    async def zadd_ts(
        self, key: str, ts_epoch: float, payload: str, max_size: int = 1000
    ) -> None:
        pipe = self._client.pipeline()
        pipe.zadd(key, {payload: ts_epoch})
        pipe.zremrangebyrank(key, 0, -(max_size + 1))
        await pipe.execute()

    async def zrange_ts(
        self, key: str, start: float = 0, end: float = float("inf")
    ) -> list[str]:
        return await self._client.zrangebyscore(key, start, end)

    # ── Atomic counter ────────────────────────────────────────────────

    async def incr(self, key: str, ttl: int = 3600) -> int:
        val = await self._client.incr(key)
        if val == 1:
            await self._client.expire(key, ttl)
        return val

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()


class RedisPubSub:
    """
    Pub/Sub para realtime signals entre servicios.

    Productor:
        async with RedisPubSub() as ps:
            await ps.publish(CHANNELS["signals"], event.model_dump_json())

    Consumidor:
        async with RedisPubSub() as ps:
            async for message in ps.subscribe(CHANNELS["signals"]):
                event = FinalSignalEvent.model_validate_json(message)
    """

    def __init__(self, url: str = REDIS_PUBSUB):
        self._url = url
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        self._client = await aioredis.from_url(
            self._url, encoding="utf-8", decode_responses=True
        )

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def publish(self, channel: str, message: str) -> int:
        """Returns number of subscribers that received the message."""
        return await self._client.publish(channel, message)

    async def subscribe(self, *channels: str) -> AsyncIterator[str]:
        """Yields raw string messages from the given channels."""
        pubsub = self._client.pubsub()
        await pubsub.subscribe(*channels)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield message["data"]
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()
