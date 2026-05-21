"""
Tests for MemoryRepository (the Repository ABC's reference implementation).

The PostgresRepository is exercised via integration tests against a real
Timescale instance — not covered here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_shared.schemas.orders import (
    Fill,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

from app.repository import MemoryRepository, RiskDecision


@pytest.fixture
def repo():
    return MemoryRepository()


def _make_intent(symbol="BTCUSDT", venue="binance") -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT_MAKER,
        limit_price=Decimal("65000"),
        venue=venue,
    )


def _make_result(intent: OrderIntent, status=OrderStatus.SUBMITTED) -> OrderResult:
    return OrderResult(
        intent_id=intent.intent_id,
        broker_id="b-1",
        symbol=intent.symbol,
        side=intent.side,
        status=status,
        qty=intent.qty,
        venue=intent.venue,
    )


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

async def test_save_and_get_intent(repo):
    intent   = _make_intent()
    decision = RiskDecision(approved=True, reason="ok")

    await repo.save_intent(intent, decision)
    loaded = await repo.get_intent(intent.intent_id)
    assert loaded is not None
    assert loaded.intent_id == intent.intent_id


async def test_save_and_get_notional_intent(repo):
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        notional=Decimal("50"),
        order_type=OrderType.MARKET,
        extended_hours=False,
        venue="alpaca",
    )

    await repo.save_intent(intent, RiskDecision(approved=True, reason="ok"))
    loaded = await repo.get_intent(intent.intent_id)

    assert loaded is not None
    assert loaded.qty is None
    assert loaded.notional == Decimal("50")


async def test_intent_decision_persisted(repo):
    intent   = _make_intent()
    decision = RiskDecision(approved=False, reason="cap", breach="per_symbol_cap")

    await repo.save_intent(intent, decision)
    stored = await repo.get_intent_decision(intent.intent_id)
    assert stored is not None
    assert stored.approved is False
    assert stored.breach   == "per_symbol_cap"


async def test_get_intent_returns_none_for_unknown(repo):
    assert await repo.get_intent("unknown-id") is None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

async def test_save_and_list_results(repo):
    intent = _make_intent()
    await repo.save_intent(intent, RiskDecision(approved=True))

    r1 = _make_result(intent, status=OrderStatus.SUBMITTED)
    r2 = _make_result(intent, status=OrderStatus.FILLED)

    await repo.save_result(r1)
    await repo.save_result(r2)

    recent = await repo.list_recent_results(limit=10)
    assert len(recent) == 2


async def test_save_result_updates_in_place(repo):
    intent = _make_intent()
    await repo.save_intent(intent, RiskDecision(approved=True))

    r = _make_result(intent, status=OrderStatus.SUBMITTED)
    await repo.save_result(r)

    r.status     = OrderStatus.FILLED
    r.filled_qty = intent.qty
    await repo.save_result(r)

    recent = await repo.list_recent_results()
    assert len(recent) == 1
    assert recent[0].status == OrderStatus.FILLED


async def test_list_recent_results_respects_limit(repo):
    intent = _make_intent()
    await repo.save_intent(intent, RiskDecision(approved=True))
    for _ in range(5):
        await repo.save_result(_make_result(intent))
    assert len(await repo.list_recent_results(limit=3)) == 3


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------

async def test_save_fill(repo):
    fill = Fill(
        order_id="b-1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        price=Decimal("65000"),
        fee=Decimal("0.65"),
    )
    await repo.save_fill(fill)
    assert len(repo._fills) == 1


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

async def test_upsert_position(repo):
    pos = Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"),
        venue="binance",
    )
    await repo.upsert_position(pos)
    positions = await repo.get_open_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"


async def test_upsert_position_updates_in_place(repo):
    pos = Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"),
        venue="binance",
    )
    await repo.upsert_position(pos)

    pos.qty = Decimal("0.02")
    await repo.upsert_position(pos)

    positions = await repo.get_open_positions()
    assert len(positions) == 1
    assert positions[0].qty == Decimal("0.02")


async def test_remove_position(repo):
    pos = Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"),
        venue="binance",
    )
    await repo.upsert_position(pos)
    await repo.remove_position("binance", "BTCUSDT")
    assert await repo.get_open_positions() == []


async def test_get_open_positions_filters_by_venue(repo):
    await repo.upsert_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"), venue="binance",
    ))
    await repo.upsert_position(Position(
        symbol="AAPL", side=OrderSide.BUY,
        qty=Decimal("10"), avg_entry=Decimal("170"), venue="alpaca",
    ))

    assert len(await repo.get_open_positions()) == 2
    assert len(await repo.get_open_positions(venue="alpaca")) == 1
    assert len(await repo.get_open_positions(venue="kraken")) == 0


async def test_positions_keyed_by_venue_and_symbol(repo):
    """Same symbol on two venues must coexist."""
    await repo.upsert_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"), venue="binance",
    ))
    await repo.upsert_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.005"), avg_entry=Decimal("64500"), venue="bybit",
    ))
    positions = await repo.get_open_positions()
    assert len(positions) == 2
