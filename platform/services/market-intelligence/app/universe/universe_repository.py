"""Persistence layer for data.universe_historical and delisting_candidates."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

UTC = timezone.utc


class UniverseRepository:
    """
    Reads and writes ``data.universe_historical`` and
    ``data.delisting_candidates`` via an asyncpg pool.

    Parameters
    ----------
    pool : asyncpg.Pool
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # universe_historical
    # ------------------------------------------------------------------

    async def upsert_asset(self, asset: dict[str, Any]) -> str:
        """
        Insert or update one asset row.

        Returns ``"inserted"`` or ``"updated"``.
        """
        sql = """
            INSERT INTO data.universe_historical (
                symbol, asset_class, exchange, name,
                is_tradable, fractionable, shortable,
                last_updated_ts, raw
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8::jsonb)
            ON CONFLICT (symbol, asset_class) DO UPDATE SET
                exchange        = EXCLUDED.exchange,
                name            = EXCLUDED.name,
                is_tradable     = EXCLUDED.is_tradable,
                fractionable    = EXCLUDED.fractionable,
                shortable       = EXCLUDED.shortable,
                last_updated_ts = NOW(),
                raw             = EXCLUDED.raw
            RETURNING (xmax = 0) AS inserted
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                asset.get("symbol", ""),
                asset.get("class", "us_equity"),
                asset.get("exchange", ""),
                asset.get("name"),
                asset.get("tradable", True),
                asset.get("fractionable", False),
                asset.get("shortable", False),
                json.dumps(asset),
            )
        return "inserted" if row and row["inserted"] else "updated"

    async def mark_delisted(self, symbol: str, asset_class: str, ts: datetime) -> None:
        sql = """
            UPDATE data.universe_historical
               SET delisted_ts = $3, last_updated_ts = NOW()
             WHERE symbol = $1 AND asset_class = $2 AND delisted_ts IS NULL
        """
        async with self._pool.acquire() as conn:
            await conn.execute(sql, symbol, asset_class, ts)

    async def get_active_symbols(self) -> set[str]:
        sql = """
            SELECT symbol FROM data.universe_historical
             WHERE delisted_ts IS NULL AND asset_class = 'us_equity'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return {r["symbol"] for r in rows}

    async def get_asset(self, symbol: str, asset_class: str = "us_equity") -> Optional[dict[str, Any]]:
        sql = "SELECT * FROM data.universe_historical WHERE symbol=$1 AND asset_class=$2"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, symbol, asset_class)
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # delisting_candidates  (3-day halt buffer)
    # ------------------------------------------------------------------

    async def add_delisting_candidate(self, symbol: str) -> None:
        sql = """
            INSERT INTO data.delisting_candidates (symbol, first_seen_inactive_ts, last_checked_ts)
            VALUES ($1, NOW(), NOW())
            ON CONFLICT (symbol) DO UPDATE SET last_checked_ts = NOW()
        """
        async with self._pool.acquire() as conn:
            await conn.execute(sql, symbol)

    async def remove_delisting_candidate(self, symbol: str) -> None:
        sql = "DELETE FROM data.delisting_candidates WHERE symbol = $1"
        async with self._pool.acquire() as conn:
            await conn.execute(sql, symbol)

    async def confirm_delisting_candidates_older_than_days(self, days: int = 3) -> list[str]:
        """Return symbols that have been inactive for more than ``days`` days."""
        sql = """
            UPDATE data.delisting_candidates
               SET confirmed = TRUE
             WHERE confirmed = FALSE
               AND first_seen_inactive_ts < NOW() - ($1 || ' days')::INTERVAL
            RETURNING symbol
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, str(days))
        return [r["symbol"] for r in rows]
