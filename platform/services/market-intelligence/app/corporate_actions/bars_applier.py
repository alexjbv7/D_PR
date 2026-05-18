"""Apply corporate actions to market.bars_1m_adjusted.

Design decisions (see docs/adr/024-corporate-actions-architecture.md):
- Raw market.ohlcv is NEVER modified.
- bars_1m_adjusted stores price/volume already adjusted for all CAs up to now.
- Idempotence: check market.corporate_actions_applied before running.
- Cash dividends do NOT adjust price (see ADR-025).
- Spinoffs: log WARN, leave as TODO (two-symbol events).

Adjustment formulas
-------------------
ratio = split_to / split_from  (>1 for forward splits, <1 for reverse splits)

Forward split (ratio > 1):
    adj_price  = raw_price  / ratio
    adj_volume = raw_volume * ratio

Reverse split (ratio < 1):  same formula, ratio < 1 → price increases, volume decreases
    adj_price  = raw_price  / ratio   (dividing by <1 multiplies)
    adj_volume = raw_volume * ratio   (multiplying by <1 divides)

Stock dividend (ratio = 1 + stock_amount):
    same as forward split with above ratio

Cash dividend: no price adjustment (ADR-025).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any

import asyncpg
import structlog

from .ca_repository import CARepository

logger = structlog.get_logger(__name__)
UTC = timezone.utc

_QUANTIZE = Decimal("0.00000001")   # 8dp — same as raw bars


class BarsApplier:
    """
    Apply a single corporate action to ``market.bars_1m_adjusted``.

    Parameters
    ----------
    pool : asyncpg.Pool
    ca_repo : CARepository
    """

    def __init__(self, pool: asyncpg.Pool, ca_repo: CARepository) -> None:
        self._pool = pool
        self._repo = ca_repo

    async def apply(self, ca: dict[str, Any]) -> int:
        """
        Apply *ca* to adjusted bars.  Idempotent via corporate_actions_applied.

        Returns number of rows updated (0 if already applied).
        """
        ca_id   = ca["ca_id"]
        ca_type = ca["ca_type"]
        symbol  = ca["symbol"]
        ex_ts   = ca["ex_ts"]
        log     = logger.bind(ca_id=ca_id, ca_type=ca_type, symbol=symbol)

        if await self._repo.was_applied(ca_id, "bars"):
            log.debug("bars_applier.already_applied")
            return 0

        if ca_type == "cash_dividend":
            # ADR-025: cash dividends do not adjust price. Record no-op.
            await self._repo.record_application(ca_id, "bars", 0, True)
            log.info("bars_applier.cash_dividend_noop")
            return 0

        if ca_type == "spinoff":
            # TODO(@alex 2026-07-01): two-symbol event; requires parent + spinoff handling.
            log.warning("bars_applier.spinoff_not_implemented", symbol=symbol)
            await self._repo.record_application(ca_id, "bars", 0, True, "spinoff_manual_resolution_required")
            return 0

        if ca_type in ("merger", "name_change"):
            # Mark bars with last_ca_id_applied; symbol stays unchanged (audit trail).
            rows = await self._tag_bars(symbol, ex_ts, ca_id)
            await self._repo.record_application(ca_id, "bars", rows, True)
            log.info("bars_applier.tagged_for_rename", rows=rows)
            return rows

        # --- Compute ratio for splits / stock dividends ---
        ratio = _compute_ratio(ca)
        if ratio is None or ratio <= 0:
            log.error("bars_applier.invalid_ratio", ca_id=ca_id, ca=ca)
            await self._repo.record_application(ca_id, "bars", 0, False, "invalid_ratio")
            return 0

        try:
            rows = await self._adjust_bars(symbol, ex_ts, ca_id, ratio)
            await self._repo.record_application(ca_id, "bars", rows, True)
            log.info("bars_applier.applied", rows=rows, ratio=str(ratio))
            return rows
        except Exception as exc:
            await self._repo.record_application(ca_id, "bars", 0, False, str(exc))
            raise

    async def _adjust_bars(
        self,
        symbol: str,
        ex_ts: datetime,
        ca_id: str,
        ratio: Decimal,
    ) -> int:
        """
        Copy raw bars from market.ohlcv into bars_1m_adjusted applying ratio.
        Only rows with ``time < ex_ts`` are affected (pre-ex-date prices).

        Uses INSERT ... ON CONFLICT DO UPDATE so re-runs are safe.
        """
        sql = """
            INSERT INTO market.bars_1m_adjusted (
                time, symbol, timeframe,
                open, high, low, close, volume,
                quote_volume, trade_count, taker_buy_vol,
                source, is_provisional, last_ca_id_applied
            )
            SELECT
                time, symbol, timeframe,
                open  / $3 AS open,
                high  / $3 AS high,
                low   / $3 AS low,
                close / $3 AS close,
                volume * $3 AS volume,
                quote_volume,
                trade_count,
                taker_buy_vol * $3,
                source,
                FALSE,
                $4
            FROM market.ohlcv
            WHERE symbol = $1 AND time < $2
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                open                = EXCLUDED.open,
                high                = EXCLUDED.high,
                low                 = EXCLUDED.low,
                close               = EXCLUDED.close,
                volume              = EXCLUDED.volume,
                taker_buy_vol       = EXCLUDED.taker_buy_vol,
                is_provisional      = FALSE,
                last_ca_id_applied  = EXCLUDED.last_ca_id_applied
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, symbol, ex_ts, ratio, ca_id)
        # asyncpg returns "INSERT 0 N" or "UPDATE N"
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    async def _tag_bars(self, symbol: str, ex_ts: datetime, ca_id: str) -> int:
        sql = """
            UPDATE market.bars_1m_adjusted
               SET last_ca_id_applied = $3
             WHERE symbol = $1 AND time < $2
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, symbol, ex_ts, ca_id)
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0


def _compute_ratio(ca: dict[str, Any]) -> Decimal | None:
    """
    Return the adjustment ratio for price.
    ratio = split_to / split_from  for splits.
    ratio = 1 + stock_amount       for stock dividends.
    """
    ca_type = ca.get("ca_type", "")
    if ca_type in ("forward_split", "reverse_split"):
        from_val = ca.get("split_from")
        to_val   = ca.get("split_to")
        if from_val is None or to_val is None:
            return None
        split_from = Decimal(str(from_val))
        split_to   = Decimal(str(to_val))
        if split_from == 0:
            return None
        return split_to / split_from
    if ca_type == "stock_dividend":
        amount = ca.get("stock_amount")
        if amount is None:
            return None
        return Decimal("1") + Decimal(str(amount))
    return None
