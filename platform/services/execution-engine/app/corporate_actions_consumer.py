"""Kafka consumer for los_ojos.corporate_actions.

Adjusts open Position.qty and Position.avg_entry atomically when a
CorporateActionEvent arrives.

Architecture note (see ADR-024):
  market-intelligence emits the event → execution-engine adjusts positions.
  market-intelligence NEVER touches Position objects directly.

Idempotence: every application is recorded in market.corporate_actions_applied
  (via Repository.record_ca_application). If the consumer is restarted and
  processes the same ca_id again, the second run is a no-op.

Fractional shares from reverse splits:
  If the adjusted qty is not an integer and the symbol is not fractionable,
  a WARNING is logged and the fractional part is stored in the Position's
  qty as-is (broker will pay out the residual in cash on next settlement).
  Documented with TODO for manual resolution path.
"""
from __future__ import annotations

import logging
from decimal import Decimal, ROUND_DOWN
from typing import Optional

import structlog

# Prometheus metrics (optional — graceful no-op if prometheus_client absent)
try:
    from prometheus_client import Counter  # type: ignore[import-untyped,unused-ignore]
    _positions_adjusted_by_ca_total = Counter(
        "positions_adjusted_by_ca_total",
        "Positions adjusted by corporate action consumer",
        ["ca_type"],
    )
    _ca_consumer_errors_total = Counter(
        "ca_consumer_errors_total",
        "Errors in the corporate actions Kafka consumer",
    )
except ImportError:  # pragma: no cover
    class _FakeCounter:
        def labels(self, **kw: object) -> "_FakeCounter":
            return self
        def inc(self, n: int = 1) -> None:
            pass
    _positions_adjusted_by_ca_total = _FakeCounter()  # type: ignore[assignment]
    _ca_consumer_errors_total = _FakeCounter()         # type: ignore[assignment]

from quant_shared.schemas.events import CorporateActionEvent
from quant_shared.schemas.orders import Position

from .repository import Repository

logger = structlog.get_logger(__name__)

_ZERO = Decimal("0")
_ONE  = Decimal("1")


async def consume_corporate_actions(
    repo: Repository,
    kafka_consumer: object,
) -> None:
    """
    Listen to ``los_ojos.corporate_actions`` and adjust open positions.

    Parameters
    ----------
    repo : Repository
    kafka_consumer : KafkaConsumerClient (duck-typed to avoid circular import)
    """
    async for msg in kafka_consumer.consume():  # type: ignore[attr-defined]
        event = CorporateActionEvent.model_validate(msg if isinstance(msg, dict) else msg.model_dump())
        try:
            await apply_to_positions(event, repo)
        except Exception as exc:
            _ca_consumer_errors_total.inc()
            logger.exception(
                "ca_consumer.apply_failed",
                ca_id=event.ca_id,
                symbol=event.symbol,
                error=str(exc),
            )
            # No commit — offset stays so message can be reprocessed after fix.


async def apply_to_positions(
    event: CorporateActionEvent,
    repo: Repository,
) -> None:
    """
    Apply a CorporateActionEvent to all open positions for event.symbol.

    Idempotent: if already applied, exits immediately.
    """
    if await repo.was_ca_applied(event.ca_id, "positions"):
        logger.debug("ca_consumer.already_applied", ca_id=event.ca_id)
        return

    positions = await repo.get_open_positions_for_symbol(event.symbol)

    for pos in positions:
        adjusted = _adjust_position(pos, event)
        if adjusted is not pos:
            await repo.upsert_position(adjusted)
            # For mergers / name_changes, remove the old symbol key
            if event.ca_type in ("merger", "name_change") and event.new_symbol and event.new_symbol != pos.symbol:
                await repo.remove_position(pos.venue, pos.symbol)

    await repo.record_ca_application(
        event.ca_id,
        target="positions",
        rows_affected=len(positions),
        success=True,
    )
    _positions_adjusted_by_ca_total.labels(ca_type=event.ca_type).inc(len(positions))
    logger.info(
        "ca_consumer.applied",
        ca_id=event.ca_id,
        ca_type=event.ca_type,
        symbol=event.symbol,
        positions_updated=len(positions),
    )


def _adjust_position(pos: Position, event: CorporateActionEvent) -> Position:
    """
    Pure function — returns an adjusted Position (or the original if no change).

    Forward split (ratio > 1):
        new_qty       = qty * ratio
        new_avg_entry = avg_entry / ratio

    Reverse split (ratio < 1):
        new_qty       = qty * ratio  (decreases)
        new_avg_entry = avg_entry / ratio  (increases)
        If new_qty is not an integer, emit WARN and keep fractional.

    Stock dividend (ratio = 1 + stock_amount):
        Same as forward split.

    Cash dividend, merger, spinoff, name_change:
        Handled explicitly below.
    """
    ca_type = event.ca_type

    if ca_type in ("forward_split", "reverse_split"):
        ratio = event.split_ratio
        if ratio is None or ratio <= _ZERO:
            logger.error(
                "ca_consumer.invalid_ratio",
                ca_id=event.ca_id,
                ratio=str(ratio),
            )
            return pos

        new_qty   = pos.qty * ratio
        new_avg   = pos.avg_entry / ratio

        if ca_type == "reverse_split" and not _is_integer(new_qty):
            residual = new_qty - Decimal(int(new_qty))
            logger.warning(
                "ca.reverse_split_fractional_residual",
                symbol=pos.symbol,
                original_qty=str(pos.qty),
                new_qty=str(new_qty),
                residual=str(residual),
                ca_id=event.ca_id,
                # TODO(@alex 2026-07-01): emit AnomalyEvent for broker cash settlement.
            )

        return pos.model_copy(update={"qty": new_qty, "avg_entry": new_avg})

    if ca_type == "stock_dividend":
        amount = event.stock_amount
        if amount is None:
            return pos
        ratio   = _ONE + amount
        new_qty = pos.qty * ratio
        new_avg = pos.avg_entry / ratio
        return pos.model_copy(update={"qty": new_qty, "avg_entry": new_avg})

    if ca_type == "cash_dividend":
        # No qty / avg_entry adjustment.  P&L attribution downstream.
        return pos

    if ca_type in ("merger", "name_change"):
        # Rename symbol; qty / avg_entry unchanged.
        if event.new_symbol:
            return pos.model_copy(update={"symbol": event.new_symbol})
        return pos

    if ca_type == "spinoff":
        # TODO(@alex 2026-07-01): spinoffs are two-symbol events; requires
        # parent + spinoff allocation logic. Leaving as WARN + no-op.
        logger.warning(
            "ca_consumer.spinoff_not_implemented",
            symbol=pos.symbol,
            ca_id=event.ca_id,
        )
        return pos

    raise ValueError(f"Unknown ca_type: {ca_type}")


def _is_integer(value: Decimal) -> bool:
    """Return True if Decimal value has no fractional part."""
    return value == value.to_integral_value(rounding=ROUND_DOWN)
