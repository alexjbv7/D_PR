"""
AllocatorRepository — persistence + per-process cache + per-horizon lock.

Schema (see infra/sql/migrations/009_allocator_state.sql)
---------------------------------------------------------
risk.allocator_state    — one row per horizon (α, β, last_update_ts)
risk.allocator_updates  — append-only audit log; PK = update_id (UUID v7)

Hot-path semantics
------------------
load(horizon) hits an in-memory cache; cache invalidates on save().  This keeps
ThompsonAllocator.choose() free of database round-trips.

Idempotency
-----------
was_update_applied(trade_id) consults the unique index on
risk.allocator_updates.trade_id.  The consumer must call this BEFORE updating
the posterior.  A re-applied trade_id is a no-op.

Concurrency
-----------
lock_horizon(horizon) returns an asyncio.Lock per horizon.  The consumer
holds it across (load → mutate → save → record_update) to prevent two
in-flight trade closes from racing each other's posterior write.
choose() never takes this lock — it only reads (via cache).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

from .posterior import BetaPosterior, MIN_ALPHA, MIN_BETA

logger = structlog.get_logger(__name__)


# Warm-start priors — matches the SQL seed in migration 009.
_WARMSTART_ALPHA: Decimal = Decimal("20.0")
_WARMSTART_BETA:  Decimal = Decimal("20.0")


class AllocatorRepository:
    """Postgres-backed state with per-process cache.

    Parameters
    ----------
    pool : asyncpg.Pool
        Connected pool, lifecycle owned by the caller.
    """

    def __init__(self, pool: Any) -> None:
        self._pool: Any = pool
        self._cache: dict[str, BetaPosterior] = {}
        self._horizon_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Per-horizon mutex (used by update_consumer)
    # ------------------------------------------------------------------

    def lock_horizon(self, horizon: str) -> asyncio.Lock:
        """Return the asyncio.Lock for *horizon*, creating it on first call."""
        lock = self._horizon_locks.get(horizon)
        if lock is None:
            lock = asyncio.Lock()
            self._horizon_locks[horizon] = lock
        return lock

    # ------------------------------------------------------------------
    # State (allocator_state)
    # ------------------------------------------------------------------

    async def load(self, horizon: str) -> BetaPosterior:
        """Return the cached posterior for *horizon* (load from DB on miss).

        If the row is absent (cold start before migration seed), returns the
        warm-start prior Beta(20, 20) anchored at now().
        """
        cached = self._cache.get(horizon)
        if cached is not None:
            return cached

        sql = """
            SELECT alpha, beta, last_update_ts
            FROM risk.allocator_state
            WHERE horizon = $1
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, horizon)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "allocator_repo.load.error",
                horizon=horizon, error=str(exc),
            )
            row = None

        if row is None:
            posterior = BetaPosterior(
                alpha=_WARMSTART_ALPHA,
                beta=_WARMSTART_BETA,
                last_update_ts=datetime.now(tz=timezone.utc),
            )
        else:
            posterior = BetaPosterior(
                alpha=Decimal(str(row["alpha"])),
                beta=Decimal(str(row["beta"])),
                last_update_ts=row["last_update_ts"],
            )
        self._cache[horizon] = posterior
        return posterior

    async def save(self, horizon: str, posterior: BetaPosterior) -> None:
        """Upsert state row and refresh the cache."""
        sql = """
            INSERT INTO risk.allocator_state (horizon, alpha, beta, last_update_ts)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (horizon) DO UPDATE
              SET alpha           = EXCLUDED.alpha,
                  beta            = EXCLUDED.beta,
                  last_update_ts  = EXCLUDED.last_update_ts
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql,
                    horizon,
                    posterior.alpha,
                    posterior.beta,
                    posterior.last_update_ts,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "allocator_repo.save.error",
                horizon=horizon, error=str(exc),
            )
            return
        self._cache[horizon] = posterior

    # ------------------------------------------------------------------
    # Audit log (allocator_updates)
    # ------------------------------------------------------------------

    async def was_update_applied(self, trade_id: str) -> bool:
        """True if *trade_id* has already been written to allocator_updates."""
        sql = "SELECT 1 FROM risk.allocator_updates WHERE trade_id = $1 LIMIT 1"
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, trade_id)
            return row is not None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "allocator_repo.idempotency_check.error",
                trade_id=trade_id, error=str(exc),
            )
            # Conservative: assume already applied to avoid double-update.
            return True

    async def record_update(
        self,
        update_id:    str,
        horizon:      str,
        trade_id:     str,
        outcome:      str,
        realized_pnl: Decimal,
        alpha_delta:  Decimal,
        beta_delta:   Decimal,
        alpha_after:  Decimal,
        beta_after:   Decimal,
        ts:           datetime,
    ) -> None:
        """Insert an audit row.  ON CONFLICT (trade_id) DO NOTHING."""
        sql = """
            INSERT INTO risk.allocator_updates (
                update_id, horizon, trade_id, outcome, realized_pnl,
                alpha_delta, beta_delta, alpha_after, beta_after, ts_utc
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (trade_id) DO NOTHING
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql,
                    update_id, horizon, trade_id, outcome, realized_pnl,
                    alpha_delta, beta_delta, alpha_after, beta_after, ts,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "allocator_repo.record_update.error",
                trade_id=trade_id, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Cache control (test helper)
    # ------------------------------------------------------------------

    def invalidate_cache(self, horizon: str | None = None) -> None:
        """Drop one or all cached posteriors.  Intended for tests."""
        if horizon is None:
            self._cache.clear()
        else:
            self._cache.pop(horizon, None)


__all__ = ["AllocatorRepository"]
