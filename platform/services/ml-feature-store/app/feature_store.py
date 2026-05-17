"""
FeatureStore — Persistencia y serving de features para ML inference.

Arquitectura de dos capas:
  - ONLINE  : Redis hash por símbolo, TTL corto (5 min). Path crítico.
  - OFFLINE : PostgreSQL (TimescaleDB) tabla features.feature_values. Entrenamiento.

Garantías:
  - Lecturas del online store son O(1) via hgetall.
  - Escrituras batched hacia offline cada 60s para reducir I/O.
  - Versión de feature set incluida para detectar schema drift.
  - NaN policy declarada por feature: drop | forward_fill | zero.

Decisiones de diseño:
  - No usar Redis como DB principal — solo cache con TTL.
  - El offline store es la fuente de verdad para training.
  - Si el online store expira, el servicio vuelve al offline (degraded mode).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

@dataclass
class FeatureDef:
    name:         str
    group:        str          # market | macro | onchain | derived
    dtype:        str = "float64"
    nan_policy:   str = "forward_fill"  # drop | forward_fill | zero
    ttl_seconds:  int = 300
    version:      int = 1


# Default feature catalog
FEATURE_CATALOG: dict[str, FeatureDef] = {
    "rsi_14":          FeatureDef("rsi_14",          "market",  nan_policy="forward_fill"),
    "rsi_7":           FeatureDef("rsi_7",            "market",  nan_policy="forward_fill"),
    "ema_9":           FeatureDef("ema_9",            "market",  nan_policy="forward_fill"),
    "ema_21":          FeatureDef("ema_21",           "market",  nan_policy="forward_fill"),
    "ema_50":          FeatureDef("ema_50",           "market",  nan_policy="forward_fill"),
    "bb_pct":          FeatureDef("bb_pct",           "market",  nan_policy="zero"),
    "atr_14":          FeatureDef("atr_14",           "market",  nan_policy="forward_fill"),
    "volume_ratio":    FeatureDef("volume_ratio",     "market",  nan_policy="zero"),
    "spread_bps":      FeatureDef("spread_bps",       "market",  nan_policy="zero"),
    "ob_imbalance":    FeatureDef("ob_imbalance",     "market",  nan_policy="zero"),
    "funding_z":       FeatureDef("funding_z",        "market",  nan_policy="zero"),
    "recession_prob":  FeatureDef("recession_prob",   "macro",   nan_policy="forward_fill"),
    "yield_inv":       FeatureDef("yield_inv",        "macro",   nan_policy="forward_fill"),
    "sahm_value":      FeatureDef("sahm_value",       "macro",   nan_policy="forward_fill"),
    "whale_sentiment": FeatureDef("whale_sentiment",  "onchain", nan_policy="zero"),
    "whale_net_flow":  FeatureDef("whale_net_flow",   "onchain", nan_policy="zero"),
    "regime_id":       FeatureDef("regime_id",        "derived", dtype="int32", nan_policy="zero"),
    "p_win_ml":        FeatureDef("p_win_ml",         "derived", nan_policy="zero"),
    "p_win_bayesian":  FeatureDef("p_win_bayesian",   "derived", nan_policy="zero"),
}


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Online + offline feature store backed by Redis and TimescaleDB.

    Parameters
    ----------
    redis_url : str
    postgres_dsn : str
    feature_version : int
        Schema version — increment on breaking changes.
    flush_interval : int
        Seconds between offline batch flushes.
    """

    REDIS_KEY_PREFIX = "features:"
    REDIS_SNAPSHOT_TTL = 300   # 5 minutes

    def __init__(
        self,
        redis_url:       str = "redis://localhost:6379/0",
        postgres_dsn:    str = "",
        feature_version: int = 1,
        flush_interval:  int = 60,
    ):
        self._redis_url       = redis_url
        self._postgres_dsn    = postgres_dsn
        self._version         = feature_version
        self._flush_interval  = flush_interval

        self._redis: Optional[aioredis.Redis] = None
        self._pg:    Optional[asyncpg.Pool]   = None

        # Pending offline writes buffer  {symbol → {feat → value, ts}}
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_ts: dict[str, datetime]    = {}
        self._flush_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            self._redis_url, decode_responses=True
        )
        if self._postgres_dsn:
            self._pg = await asyncpg.create_pool(self._postgres_dsn, min_size=2, max_size=5)
        self._flush_task = asyncio.create_task(self._flush_loop(), name="feature-store-flush")
        logger.info("feature_store.connected")

    async def close(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)
        # Final flush
        await self._flush_to_offline()
        if self._redis:
            await self._redis.aclose()
        if self._pg:
            await self._pg.close()
        logger.info("feature_store.closed")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def update(
        self,
        symbol:   str,
        features: dict[str, float | int | None],
        ts:       Optional[datetime] = None,
    ) -> None:
        """Write feature values for a symbol to the online store."""
        now = ts or datetime.now(timezone.utc)
        redis_key = f"{self.REDIS_KEY_PREFIX}{symbol}"

        # Serialize + add metadata
        payload = {k: str(v) if v is not None else "" for k, v in features.items()}
        payload["__version__"]    = str(self._version)
        payload["__updated_at__"] = now.isoformat()

        if self._redis:
            pipe = self._redis.pipeline()
            pipe.hset(redis_key, mapping=payload)
            pipe.expire(redis_key, self.REDIS_SNAPSHOT_TTL)
            await pipe.execute()

        # Buffer for offline
        if symbol not in self._pending:
            self._pending[symbol] = {}
        self._pending[symbol].update(features)
        self._pending_ts[symbol] = now

    # ------------------------------------------------------------------
    # Read (online)
    # ------------------------------------------------------------------

    async def get(
        self,
        symbol:        str,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, float | int | None]:
        """
        Read features from online Redis store.
        Falls back to offline store if Redis miss.
        """
        redis_key = f"{self.REDIS_KEY_PREFIX}{symbol}"

        if self._redis:
            raw = await self._redis.hgetall(redis_key)
            if raw:
                return self._deserialize(raw, feature_names)

        # Fallback to offline
        return await self._get_offline(symbol, feature_names)

    async def get_vector(
        self,
        symbol:        str,
        feature_names: list[str],
    ) -> list[float | None]:
        """Return ordered feature vector — None where missing."""
        feats = await self.get(symbol, feature_names)
        return [feats.get(f) for f in feature_names]

    async def get_all_symbols(self) -> list[str]:
        if not self._redis:
            return []
        keys = await self._redis.keys(f"{self.REDIS_KEY_PREFIX}*")
        prefix_len = len(self.REDIS_KEY_PREFIX)
        return [k[prefix_len:] for k in keys]

    # ------------------------------------------------------------------
    # Read (offline / historical)
    # ------------------------------------------------------------------

    async def get_history(
        self,
        symbol:       str,
        feature_name: str,
        since:        datetime,
        limit:        int = 1000,
    ) -> list[dict]:
        """Read historical feature values from TimescaleDB."""
        if not self._pg:
            return []
        rows = await self._pg.fetch(
            """
            SELECT time, value
            FROM features.feature_values
            WHERE symbol = $1 AND feature_name = $2 AND time >= $3
            ORDER BY time DESC
            LIMIT $4
            """,
            symbol, feature_name, since, limit,
        )
        return [{"time": r["time"].isoformat(), "value": float(r["value"])} for r in rows]

    # ------------------------------------------------------------------
    # Offline flush
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_to_offline()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("feature_store.flush.error", error=str(e))

    async def _flush_to_offline(self) -> None:
        if not self._pg or not self._pending:
            return

        rows = []
        snapshot = dict(self._pending)
        self._pending.clear()

        for symbol, feats in snapshot.items():
            ts = self._pending_ts.get(symbol, datetime.now(timezone.utc))
            for feat_name, value in feats.items():
                if value is None or feat_name.startswith("__"):
                    continue
                rows.append((ts, symbol, feat_name, float(value), self._version))

        if not rows:
            return

        try:
            async with self._pg.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO features.feature_values (time, symbol, feature_name, value, version)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (time, symbol, feature_name) DO UPDATE
                        SET value = EXCLUDED.value, version = EXCLUDED.version
                    """,
                    rows,
                )
            logger.debug("feature_store.flushed", rows=len(rows))
        except Exception as e:
            logger.error("feature_store.flush.db_error", error=str(e))
            # Restore failed rows to pending
            for ts, symbol, feat_name, value, _ in rows:
                if symbol not in self._pending:
                    self._pending[symbol] = {}
                self._pending[symbol][feat_name] = value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialize(
        raw:           dict[str, str],
        feature_names: Optional[list[str]],
    ) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {}
        keys = feature_names or [k for k in raw if not k.startswith("__")]
        for k in keys:
            v = raw.get(k)
            if v is None or v == "":
                result[k] = None
            else:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = None
        return result

    async def _get_offline(
        self,
        symbol:        str,
        feature_names: Optional[list[str]],
    ) -> dict[str, float | int | None]:
        if not self._pg:
            return {}
        where_clause = ""
        params: list = [symbol]
        if feature_names:
            where_clause = "AND feature_name = ANY($2)"
            params.append(feature_names)
        rows = await self._pg.fetch(
            f"""
            SELECT DISTINCT ON (feature_name) feature_name, value
            FROM features.feature_values
            WHERE symbol = $1 {where_clause}
            ORDER BY feature_name, time DESC
            """,
            *params,
        )
        return {r["feature_name"]: float(r["value"]) for r in rows}
