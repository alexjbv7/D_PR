"""
Integration test: PostgresRepository against a live Timescale/Postgres.

Skipped automatically when ``EXECUTION_PG_DSN`` is not set, so unit-test
runs (which never set it) stay fast and dependency-free.

Run locally::

    set EXECUTION_PG_DSN=postgresql://trading:trading@localhost:5432/trading_db
    pytest tests/integration/test_postgres_repository.py -v

The test inserts, reads, and deletes every domain object inside a unique
``test-<uuid>`` namespace and cleans up after itself.  It does NOT truncate
tables or touch other rows, so it is safe to run against the dev database.
"""
from __future__ import annotations

import os
import uuid
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
    TimeInForce,
)

from app.repository import PostgresRepository, RiskDecision

PG_DSN = os.getenv("EXECUTION_PG_DSN")
pytestmark = pytest.mark.skipif(
    not PG_DSN, reason="EXECUTION_PG_DSN not set; integration test skipped"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def pool():
    import asyncpg
    p = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=2)
    yield p
    await p.close()


@pytest.fixture
async def repo(pool):
    return PostgresRepository(pool)


@pytest.fixture
def unique_tag():
    """Unique per-test marker baked into test data so cleanup is targeted."""
    return f"itest-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def cleanup(pool, unique_tag):
    """Sweeps test rows after each test by tag (intents.strategy = unique_tag)."""
    yield
    async with pool.acquire() as conn:
        # Delete in FK-friendly order: fills → results → intents; positions by venue tag
        await conn.execute(
            "DELETE FROM orders.fills WHERE order_id LIKE $1", f"{unique_tag}%"
        )
        await conn.execute(
            "DELETE FROM orders.results WHERE intent_id IN "
            "(SELECT intent_id FROM orders.intents WHERE strategy = $1)",
            unique_tag,
        )
        await conn.execute(
            "DELETE FROM orders.intents WHERE strategy = $1", unique_tag
        )
        await conn.execute(
            "DELETE FROM orders.positions WHERE venue = $1", unique_tag
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_save_and_get_intent_roundtrip(repo, unique_tag, cleanup):
    intent = OrderIntent(
        signal_id=str(uuid.uuid4()),
        strategy=unique_tag,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT_MAKER,
        limit_price=Decimal("65000.50"),
        tif=TimeInForce.GTC,
        venue="binance",
        kelly_fraction=0.10,
        p_win=0.58,
    )
    await repo.save_intent(intent, RiskDecision(approved=True, reason="ok"))

    loaded = await repo.get_intent(intent.intent_id)
    assert loaded is not None
    assert loaded.symbol      == "BTCUSDT"
    assert loaded.side        == OrderSide.BUY
    assert loaded.qty         == Decimal("0.01")
    assert loaded.limit_price == Decimal("65000.500000000000")   # NUMERIC(28,12)
    assert loaded.strategy    == unique_tag


async def test_save_and_get_notional_intent_roundtrip(repo, unique_tag, cleanup):
    intent = OrderIntent(
        signal_id=str(uuid.uuid4()),
        strategy=unique_tag,
        symbol="AAPL",
        side=OrderSide.BUY,
        notional=Decimal("50"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        venue="alpaca",
        extended_hours=False,
    )
    await repo.save_intent(intent, RiskDecision(approved=True, reason="ok"))

    loaded = await repo.get_intent(intent.intent_id)
    assert loaded is not None
    assert loaded.qty is None
    assert loaded.notional == Decimal("50.000000000000")
    assert loaded.extended_hours is False


async def test_save_result_upserts(repo, unique_tag, cleanup):
    intent = OrderIntent(
        strategy=unique_tag, symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), order_type=OrderType.LIMIT,
        limit_price=Decimal("65000"), venue="binance",
    )
    await repo.save_intent(intent, RiskDecision(approved=True))

    result = OrderResult(
        intent_id=intent.intent_id,
        broker_id="b-1",
        symbol="BTCUSDT", side=OrderSide.BUY,
        status=OrderStatus.SUBMITTED, qty=Decimal("0.01"),
        venue="binance",
    )
    await repo.save_result(result)

    # Update: same result_id, status flips to FILLED
    result.status     = OrderStatus.FILLED
    result.filled_qty = Decimal("0.01")
    result.avg_price  = Decimal("65010")
    result.ts_updated = datetime.now(tz=timezone.utc)
    await repo.save_result(result)

    recent = await repo.list_recent_results(limit=5)
    matching = [r for r in recent if r.result_id == result.result_id]
    assert len(matching) == 1
    assert matching[0].status     == OrderStatus.FILLED
    assert matching[0].filled_qty == Decimal("0.010000000000")
    assert matching[0].avg_price  == Decimal("65010.000000000000")


async def test_position_upsert_and_remove(repo, unique_tag, cleanup):
    pos = Position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        avg_entry=Decimal("65000"),
        venue=unique_tag,                  # tag goes here for sweep
    )
    await repo.upsert_position(pos)

    positions = await repo.get_open_positions(venue=unique_tag)
    assert len(positions) == 1
    assert positions[0].symbol == "BTCUSDT"

    # Upsert same key, new qty
    pos.qty = Decimal("0.02")
    await repo.upsert_position(pos)
    positions = await repo.get_open_positions(venue=unique_tag)
    assert len(positions) == 1
    assert positions[0].qty == Decimal("0.020000000000")

    await repo.remove_position(unique_tag, "BTCUSDT")
    assert await repo.get_open_positions(venue=unique_tag) == []


async def test_save_fill(repo, unique_tag, cleanup):
    fill = Fill(
        order_id=f"{unique_tag}-ord",
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), price=Decimal("65000.5"),
        fee=Decimal("0.65"), fee_asset="USDT",
        venue="binance",
    )
    await repo.save_fill(fill)
    # Verify via a raw query (the repo has no list_fills method by design)
    async with repo._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM orders.fills WHERE order_id = $1",
            f"{unique_tag}-ord",
        )
    assert len(rows) == 1
    assert rows[0]["price"] == Decimal("65000.500000000000")


async def test_intent_decision_persisted_correctly(repo, unique_tag, cleanup):
    intent = OrderIntent(
        strategy=unique_tag, symbol="ETHUSDT", side=OrderSide.SELL,
        qty=Decimal("0.5"), order_type=OrderType.LIMIT,
        limit_price=Decimal("3200"), venue="binance",
    )
    await repo.save_intent(intent, RiskDecision(
        approved=False, reason="cap breach", breach="per_symbol_cap",
    ))

    async with repo._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT risk_decision, risk_reason, risk_breach "
            "FROM orders.intents WHERE intent_id = $1::uuid",
            intent.intent_id,
        )
    assert row["risk_decision"] == "rejected"
    assert row["risk_reason"]   == "cap breach"
    assert row["risk_breach"]   == "per_symbol_cap"
