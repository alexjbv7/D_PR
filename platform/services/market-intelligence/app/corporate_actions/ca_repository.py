"""Persistence for market.corporate_actions and market.corporate_actions_applied."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import asyncpg

from quant_shared.schemas.orders import _uuid7 as uuid7

logger = logging.getLogger(__name__)
UTC = timezone.utc


def _parse_decimal(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    return Decimal(str(v))


class CARepository:
    """
    Manages corporate action rows and the idempotency audit log.

    Parameters
    ----------
    pool : asyncpg.Pool
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_ca(self, announcement: dict[str, Any]) -> tuple[str, bool]:
        """
        Upsert a CA from an Alpaca announcement dict.

        Returns
        -------
        (ca_id, is_new) : tuple[str, bool]
        """
        alpaca_id = str(announcement.get("id", ""))
        existing = await self._get_by_alpaca_id(alpaca_id) if alpaca_id else None

        if existing:
            ca_id    = existing["ca_id"]
            is_new   = False
        else:
            ca_id    = uuid7()
            is_new   = True

        ca_type_raw = str(announcement.get("ca_type", "")).lower().replace(" ", "_")
        sql = """
            INSERT INTO market.corporate_actions (
                ca_id, alpaca_id, symbol, ca_type,
                declared_ts, ex_ts, record_ts, payable_ts,
                split_from, split_to, cash_amount, stock_amount,
                new_symbol, is_provisional, raw
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10, $11, $12,
                $13, $14, $15::jsonb
            )
            ON CONFLICT (alpaca_id) DO UPDATE SET
                ca_type      = EXCLUDED.ca_type,
                ex_ts        = EXCLUDED.ex_ts,
                split_from   = EXCLUDED.split_from,
                split_to     = EXCLUDED.split_to,
                cash_amount  = EXCLUDED.cash_amount,
                stock_amount = EXCLUDED.stock_amount,
                new_symbol   = EXCLUDED.new_symbol,
                raw          = EXCLUDED.raw
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                sql,
                ca_id,
                alpaca_id or None,
                announcement.get("symbol", ""),
                ca_type_raw,
                _parse_ts(announcement.get("declaration_date")),
                _parse_ts(announcement.get("ex_date")),
                _parse_ts(announcement.get("record_date")),
                _parse_ts(announcement.get("payable_date")),
                _parse_decimal(announcement.get("old_rate")),
                _parse_decimal(announcement.get("new_rate")),
                _parse_decimal(announcement.get("cash")),
                _parse_decimal(announcement.get("new_rate")),  # stock dividend ratio
                announcement.get("new_symbol"),
                True,
                json.dumps(announcement),
            )

        return ca_id, is_new

    async def was_applied(self, ca_id: str, target: str) -> bool:
        sql = """
            SELECT 1 FROM market.corporate_actions_applied
             WHERE ca_id = $1 AND target = $2 AND success = TRUE
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, ca_id, target)
        return row is not None

    async def record_application(
        self,
        ca_id: str,
        target: str,
        rows_affected: int,
        success: bool,
        error_msg: Optional[str] = None,
    ) -> None:
        sql = """
            INSERT INTO market.corporate_actions_applied
                (ca_id, target, rows_affected, success, error_msg)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (ca_id, target) DO UPDATE SET
                ts_applied    = NOW(),
                rows_affected = EXCLUDED.rows_affected,
                success       = EXCLUDED.success,
                error_msg     = EXCLUDED.error_msg
        """
        async with self._pool.acquire() as conn:
            await conn.execute(sql, ca_id, target, rows_affected, success, error_msg)

    async def get_pending_provisional(self, older_than_hours: int = 48) -> list[dict[str, Any]]:
        """Return provisional CAs whose ex_ts is > older_than_hours ago."""
        sql = """
            SELECT * FROM market.corporate_actions
             WHERE is_provisional = TRUE
               AND ex_ts < NOW() - ($1 || ' hours')::INTERVAL
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, str(older_than_hours))
        return [dict(r) for r in rows]

    async def mark_confirmed(self, ca_id: str) -> None:
        sql = "UPDATE market.corporate_actions SET is_provisional = FALSE WHERE ca_id = $1"
        async with self._pool.acquire() as conn:
            await conn.execute(sql, ca_id)

    async def _get_by_alpaca_id(self, alpaca_id: str) -> Optional[dict[str, Any]]:
        sql = "SELECT * FROM market.corporate_actions WHERE alpaca_id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, alpaca_id)
        return dict(row) if row else None


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    # Alpaca returns "YYYY-MM-DD" date strings for CA dates
    try:
        from datetime import date as _date
        d = _date.fromisoformat(str(value))
        return datetime(d.year, d.month, d.day, tzinfo=UTC)
    except ValueError:
        return None
