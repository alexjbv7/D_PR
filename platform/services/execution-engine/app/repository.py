"""
Repository — persistence layer for the execution-engine.
========================================================

Two implementations are provided behind a common :class:`Repository` ABC:

* :class:`MemoryRepository`
    In-memory dict-based.  Used for tests and for paper-trading bootstrap
    (process-local state, no external DB required).

* :class:`PostgresRepository`
    ``asyncpg``-backed implementation that writes to the ``orders.*`` schema
    defined in ``migrations/001_execution_engine.sql``.

The risk-gate, the reconciler, and the (future) FastAPI service all depend
on the abstract :class:`Repository`, never on the Postgres class directly.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

from quant_shared.schemas.orders import _uuid7

from quant_shared.schemas.orders import (
    Fill,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class _PositionActionRow:
    account_id:    str
    symbol:        str
    side:          str
    trade_date_et: date


# ---------------------------------------------------------------------------
# Repository ABC
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskDecision:
    """
    Persisted alongside an :class:`OrderIntent`.

    Parameters
    ----------
    approved : bool
        True if the intent passed every risk check.
    reason : str
        Human-readable explanation of the decision.
    breach : str, optional
        Tag identifying which limit was breached
        (``"per_symbol_cap"``, ``"daily_dd"``, ``"cash_buffer"``, …).
    """
    approved: bool
    reason:   str = ""
    breach:   Optional[str] = None


class Repository(ABC):
    """Abstract persistence interface used by risk-gate, reconciler, REST."""

    # ---- intents ----

    @abstractmethod
    async def save_intent(self, intent: OrderIntent, decision: RiskDecision) -> None:
        """Persist an :class:`OrderIntent` together with its risk decision."""

    @abstractmethod
    async def get_intent(self, intent_id: str) -> Optional[OrderIntent]:
        ...

    # ---- results ----

    @abstractmethod
    async def save_result(self, result: OrderResult) -> None:
        """Persist (insert or update) an :class:`OrderResult` by ``result_id``."""

    @abstractmethod
    async def list_recent_results(self, limit: int = 50) -> list[OrderResult]:
        ...

    # ---- fills ----

    @abstractmethod
    async def save_fill(self, fill: Fill) -> None:
        ...

    # ---- positions ----

    @abstractmethod
    async def upsert_position(self, pos: Position) -> None:
        """Insert or update by ``(venue, symbol)`` composite key."""

    @abstractmethod
    async def remove_position(self, venue: str, symbol: str) -> None:
        ...

    @abstractmethod
    async def get_open_positions(self, venue: Optional[str] = None) -> list[Position]:
        ...

    @abstractmethod
    async def get_open_positions_for_symbol(self, symbol: str) -> list[Position]:
        """Return all open positions for a specific symbol across all venues."""
        ...

    # ---- corporate actions (idempotency) ----

    @abstractmethod
    async def was_ca_applied(self, ca_id: str, target: str) -> bool:
        """True if this CA was already applied to ``target`` (e.g. "positions")."""
        ...

    @abstractmethod
    async def record_ca_application(
        self,
        ca_id: str,
        target: str,
        rows_affected: int,
        success: bool = True,
        error_msg: Optional[str] = None,
    ) -> None:
        """Persist that a CA was applied to ``target``."""
        ...

    # ---- PDT day-trade ledger ----

    @abstractmethod
    async def count_day_trades(
        self,
        account_id: str,
        since_date_et: date,
        until_date_et: date,
    ) -> int:
        """
        Count distinct (symbol, trade_date_et) with both buy and sell actions
        in the inclusive ET date range.
        """

    @abstractmethod
    async def record_position_action(
        self,
        account_id: str,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        notional: Optional[Decimal],
        fill_id: str,
        ts_utc: datetime,
    ) -> None:
        """Append one fill-derived action for PDT counting."""


# ---------------------------------------------------------------------------
# MemoryRepository
# ---------------------------------------------------------------------------

class MemoryRepository(Repository):
    """In-memory implementation; suitable for tests and paper-mode bootstrap."""

    def __init__(self) -> None:
        self._intents:    dict[str, tuple[OrderIntent, RiskDecision]] = {}
        self._results:    dict[str, OrderResult] = {}
        self._fills:      list[Fill] = []
        self._positions:  dict[tuple[str, str], Position] = {}
        self._ca_applied: dict[tuple[str, str], bool] = {}
        self._position_actions: list[_PositionActionRow] = []

    # ---- intents ----

    async def save_intent(self, intent: OrderIntent, decision: RiskDecision) -> None:
        self._intents[intent.intent_id] = (intent, decision)

    async def get_intent(self, intent_id: str) -> Optional[OrderIntent]:
        entry = self._intents.get(intent_id)
        return entry[0] if entry else None

    async def get_intent_decision(self, intent_id: str) -> Optional[RiskDecision]:
        """Convenience for tests; not on the ABC."""
        entry = self._intents.get(intent_id)
        return entry[1] if entry else None

    # ---- results ----

    async def save_result(self, result: OrderResult) -> None:
        self._results[result.result_id] = result

    async def list_recent_results(self, limit: int = 50) -> list[OrderResult]:
        ordered = sorted(
            self._results.values(),
            key=lambda r: r.ts_updated,
            reverse=True,
        )
        return ordered[:limit]

    # ---- fills ----

    async def save_fill(self, fill: Fill) -> None:
        self._fills.append(fill)
        if fill.account_id:
            await self.record_position_action(
                account_id=fill.account_id,
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                notional=fill.notional,
                fill_id=fill.fill_id,
                ts_utc=fill.ts,
            )

    # ---- positions ----

    async def upsert_position(self, pos: Position) -> None:
        self._positions[(pos.venue, pos.symbol)] = pos

    async def remove_position(self, venue: str, symbol: str) -> None:
        self._positions.pop((venue, symbol), None)

    async def get_open_positions(self, venue: Optional[str] = None) -> list[Position]:
        if venue is None:
            return list(self._positions.values())
        return [p for (v, _s), p in self._positions.items() if v == venue]

    async def get_open_positions_for_symbol(self, symbol: str) -> list[Position]:
        return [p for (_v, s), p in self._positions.items() if s == symbol]

    # ---- corporate actions ----

    async def was_ca_applied(self, ca_id: str, target: str) -> bool:
        return self._ca_applied.get((ca_id, target), False)

    async def record_ca_application(
        self,
        ca_id: str,
        target: str,
        rows_affected: int,
        success: bool = True,
        error_msg: Optional[str] = None,
    ) -> None:
        if success:
            self._ca_applied[(ca_id, target)] = True

    async def count_day_trades(
        self,
        account_id: str,
        since_date_et: date,
        until_date_et: date,
    ) -> int:
        by_key: dict[tuple[str, date], set[str]] = {}
        for row in self._position_actions:
            if row.account_id != account_id:
                continue
            if row.trade_date_et < since_date_et or row.trade_date_et > until_date_et:
                continue
            key = (row.symbol, row.trade_date_et)
            by_key.setdefault(key, set()).add(row.side)
        return sum(1 for sides in by_key.values() if "buy" in sides and "sell" in sides)

    async def record_position_action(
        self,
        account_id: str,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        notional: Optional[Decimal],
        fill_id: str,
        ts_utc: datetime,
    ) -> None:
        trade_date_et = ts_utc.astimezone(_ET).date()
        self._position_actions.append(
            _PositionActionRow(
                account_id=account_id,
                symbol=symbol,
                side=side.value,
                trade_date_et=trade_date_et,
            )
        )


# ---------------------------------------------------------------------------
# PostgresRepository
# ---------------------------------------------------------------------------

class PostgresRepository(Repository):
    """
    ``asyncpg``-backed implementation.

    Parameters
    ----------
    pool : asyncpg.Pool
        A connected asyncpg pool.  The caller owns the pool's lifecycle.

    Notes
    -----
    asyncpg returns ``Decimal`` for ``NUMERIC`` columns and ``datetime`` for
    ``TIMESTAMPTZ`` columns, so no extra coercion is required.  UUIDs are
    represented as ``str`` on the Python side (UUID v7 from
    :func:`quant_shared.schemas.orders._uuid7`); we cast with ``::uuid`` in
    the SQL where needed.
    """

    def __init__(self, pool: Any):
        if pool is None:
            raise ValueError("pool must not be None")
        self._pool = pool

    # ---- helpers ----

    def _conn(self) -> Any:  # asyncpg PoolConnectionProxy — no public stub
        return self._pool.acquire()

    # ---- intents ----

    async def save_intent(self, intent: OrderIntent, decision: RiskDecision) -> None:
        sql = """
            INSERT INTO orders.intents (
                intent_id, signal_id, strategy, symbol, side, qty, order_type,
                limit_price, sl_price, tp_price, tif, venue,
                kelly_fraction, target_risk_pct, p_win,
                risk_decision, risk_reason, risk_breach, ts
            )
            VALUES (
                $1::uuid, NULLIF($2,'')::uuid, $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12,
                $13, $14, $15,
                $16, $17, $18, $19
            )
            ON CONFLICT (intent_id) DO UPDATE SET
                risk_decision = EXCLUDED.risk_decision,
                risk_reason   = EXCLUDED.risk_reason,
                risk_breach   = EXCLUDED.risk_breach
        """
        async with self._conn() as conn:
            await conn.execute(
                sql,
                intent.intent_id, intent.signal_id, intent.strategy,
                intent.symbol, intent.side.value, intent.qty,
                intent.order_type.value, intent.limit_price, intent.sl_price,
                intent.tp_price, intent.tif.value, intent.venue,
                Decimal(str(intent.kelly_fraction)),
                Decimal(str(intent.target_risk_pct)),
                Decimal(str(intent.p_win)),
                "approved" if decision.approved else "rejected",
                decision.reason, decision.breach, intent.ts,
            )

    async def get_intent(self, intent_id: str) -> Optional[OrderIntent]:
        sql = "SELECT * FROM orders.intents WHERE intent_id = $1::uuid"
        async with self._conn() as conn:
            row = await conn.fetchrow(sql, intent_id)
        return _row_to_intent(row) if row else None

    # ---- results ----

    async def save_result(self, result: OrderResult) -> None:
        sql = """
            INSERT INTO orders.results (
                result_id, intent_id, broker_id, symbol, side, status,
                qty, filled_qty, avg_price, venue, reject_reason,
                ts_submitted, ts_updated
            )
            VALUES (
                $1::uuid, $2::uuid, $3, $4, $5, $6,
                $7, $8, $9, $10, $11, $12, $13
            )
            ON CONFLICT (result_id) DO UPDATE SET
                broker_id     = EXCLUDED.broker_id,
                status        = EXCLUDED.status,
                filled_qty    = EXCLUDED.filled_qty,
                avg_price     = EXCLUDED.avg_price,
                reject_reason = EXCLUDED.reject_reason,
                ts_updated    = EXCLUDED.ts_updated
        """
        async with self._conn() as conn:
            await conn.execute(
                sql,
                result.result_id, result.intent_id, result.broker_id or None,
                result.symbol, result.side.value, result.status.value,
                result.qty, result.filled_qty, result.avg_price,
                result.venue, result.reject_reason,
                result.ts_submitted, result.ts_updated,
            )

    async def list_recent_results(self, limit: int = 50) -> list[OrderResult]:
        sql = "SELECT * FROM orders.results ORDER BY ts_updated DESC LIMIT $1"
        async with self._conn() as conn:
            rows = await conn.fetch(sql, limit)
        return [_row_to_result(r) for r in rows]

    # ---- fills ----

    async def save_fill(self, fill: Fill) -> None:
        sql = """
            INSERT INTO orders.fills (
                fill_id, order_id, symbol, side, qty, price,
                fee, fee_asset, venue, ts
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        async with self._conn() as conn:
            await conn.execute(
                sql,
                fill.fill_id, fill.order_id, fill.symbol, fill.side.value,
                fill.qty, fill.price, fill.fee, fill.fee_asset,
                fill.venue, fill.ts,
            )
        if fill.account_id:
            await self.record_position_action(
                account_id=fill.account_id,
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                notional=fill.notional,
                fill_id=fill.fill_id,
                ts_utc=fill.ts,
            )

    # ---- positions ----

    async def upsert_position(self, pos: Position) -> None:
        sql = """
            INSERT INTO orders.positions (
                venue, symbol, side, qty, avg_entry,
                current_price, unrealized_pnl, margin_used, ts_opened
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (venue, symbol) DO UPDATE SET
                side           = EXCLUDED.side,
                qty            = EXCLUDED.qty,
                avg_entry      = EXCLUDED.avg_entry,
                current_price  = EXCLUDED.current_price,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                margin_used    = EXCLUDED.margin_used
        """
        async with self._conn() as conn:
            await conn.execute(
                sql,
                pos.venue, pos.symbol, pos.side.value,
                pos.qty, pos.avg_entry,
                pos.current_price, pos.unrealized_pnl, pos.margin_used,
                pos.ts_opened,
            )

    async def remove_position(self, venue: str, symbol: str) -> None:
        sql = "DELETE FROM orders.positions WHERE venue = $1 AND symbol = $2"
        async with self._conn() as conn:
            await conn.execute(sql, venue, symbol)

    async def get_open_positions(self, venue: Optional[str] = None) -> list[Position]:
        if venue is None:
            sql = "SELECT * FROM orders.positions"
            args: tuple[object, ...] = ()
        else:
            sql = "SELECT * FROM orders.positions WHERE venue = $1"
            args = (venue,)
        async with self._conn() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_position(r) for r in rows]

    async def get_open_positions_for_symbol(self, symbol: str) -> list[Position]:
        sql = "SELECT * FROM orders.positions WHERE symbol = $1"
        async with self._conn() as conn:
            rows = await conn.fetch(sql, symbol)
        return [_row_to_position(r) for r in rows]

    # ---- corporate actions (idempotency) ----

    async def was_ca_applied(self, ca_id: str, target: str) -> bool:
        sql = """
            SELECT 1 FROM market.corporate_actions_applied
             WHERE ca_id = $1 AND target = $2 AND success = TRUE
        """
        async with self._conn() as conn:
            row = await conn.fetchrow(sql, ca_id, target)
        return row is not None

    async def record_ca_application(
        self,
        ca_id: str,
        target: str,
        rows_affected: int,
        success: bool = True,
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
        async with self._conn() as conn:
            await conn.execute(sql, ca_id, target, rows_affected, success, error_msg)

    async def count_day_trades(
        self,
        account_id: str,
        since_date_et: date,
        until_date_et: date,
    ) -> int:
        sql = """
            SELECT COUNT(*) FROM (
                SELECT symbol, trade_date_et
                  FROM risk.position_actions
                 WHERE account_id = $1
                   AND trade_date_et BETWEEN $2 AND $3
                 GROUP BY symbol, trade_date_et
                HAVING SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) > 0
                   AND SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) > 0
            ) day_trades
        """
        async with self._conn() as conn:
            row = await conn.fetchrow(sql, account_id, since_date_et, until_date_et)
        return int(row[0]) if row else 0

    async def record_position_action(
        self,
        account_id: str,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        notional: Optional[Decimal],
        fill_id: str,
        ts_utc: datetime,
    ) -> None:
        trade_date_et = ts_utc.astimezone(_ET).date()
        sql = """
            INSERT INTO risk.position_actions
                (action_id, account_id, symbol, side, qty, notional,
                 fill_id, ts_utc, trade_date_et)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        async with self._conn() as conn:
            await conn.execute(
                sql,
                _uuid7(),
                account_id,
                symbol,
                side.value,
                qty,
                notional,
                fill_id,
                ts_utc,
                trade_date_et,
            )


# ---------------------------------------------------------------------------
# Row → domain object helpers
# ---------------------------------------------------------------------------

def _row_to_intent(row: Any) -> OrderIntent:
    return OrderIntent(
        intent_id=str(row["intent_id"]),
        signal_id=str(row["signal_id"]) if row["signal_id"] else "",
        strategy=row["strategy"] or "",
        symbol=row["symbol"],
        side=OrderSide(row["side"]),
        qty=row["qty"],
        order_type=OrderType(row["order_type"]),
        limit_price=row["limit_price"],
        sl_price=row["sl_price"],
        tp_price=row["tp_price"],
        tif=TimeInForce(row["tif"]),
        venue=row["venue"] or "",
        kelly_fraction=float(row["kelly_fraction"] or 0),
        target_risk_pct=float(row["target_risk_pct"] or 0),
        p_win=float(row["p_win"] or 0),
        ts=_ensure_utc(row["ts"]),
    )


def _row_to_result(row: Any) -> OrderResult:
    return OrderResult(
        result_id=str(row["result_id"]),
        intent_id=str(row["intent_id"]),
        broker_id=row["broker_id"] or "",
        symbol=row["symbol"],
        side=OrderSide(row["side"]),
        status=OrderStatus(row["status"]),
        qty=row["qty"],
        filled_qty=row["filled_qty"],
        avg_price=row["avg_price"],
        venue=row["venue"] or "",
        reject_reason=row["reject_reason"],
        ts_submitted=_ensure_utc(row["ts_submitted"]),
        ts_updated=_ensure_utc(row["ts_updated"]),
    )


def _row_to_position(row: Any) -> Position:
    return Position(
        symbol=row["symbol"],
        side=OrderSide(row["side"]),
        qty=row["qty"],
        avg_entry=row["avg_entry"],
        current_price=row["current_price"],
        unrealized_pnl=row["unrealized_pnl"],
        margin_used=row["margin_used"],
        venue=row["venue"],
        ts_opened=_ensure_utc(row["ts_opened"]) if row["ts_opened"] else None,
        ts_updated=_ensure_utc(row["ts_updated"]),
    )


def _ensure_utc(value: Any) -> datetime:
    """asyncpg returns tz-aware UTC for TIMESTAMPTZ, but be defensive."""
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc) if isinstance(value, datetime) else value
