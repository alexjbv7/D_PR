"""
Tests for BrokerAdapter ABC contract.
======================================
Two goals:
  1. Verify the ABC itself cannot be instantiated.
  2. Verify a correct MockBroker satisfies every abstract method and that
     the domain objects (OrderIntent, OrderResult, Fill, Position, AccountInfo)
     behave correctly.

Run from the service root::

    cd platform/services/execution-engine
    pytest
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional

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
from app.brokers.base import BrokerAdapter, AccountInfo, BrokerError


# ---------------------------------------------------------------------------
# Minimal mock broker — implements every abstract method
# ---------------------------------------------------------------------------

class MockBroker(BrokerAdapter):
    """Minimal in-memory broker for contract verification."""

    venue = "mock"

    def __init__(self):
        self._connected = False
        self._orders: dict[str, OrderResult] = {}
        self._positions: list[Position] = []

    # lifecycle

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    # order management

    async def submit(self, intent: OrderIntent) -> OrderResult:
        result = OrderResult(
            intent_id=intent.intent_id,
            broker_id=f"MOCK-{intent.intent_id[:8]}",
            symbol=intent.symbol,
            side=intent.side,
            status=OrderStatus.SUBMITTED,
            qty=intent.qty,
            venue=self.venue,
        )
        self._orders[result.broker_id] = result
        return result

    async def cancel(self, broker_id: str) -> bool:
        order = self._orders.get(broker_id)
        if order is None or order.is_complete:
            return False
        order.status = OrderStatus.CANCELLED
        return True

    async def get_order(self, broker_id: str) -> OrderResult:
        if broker_id not in self._orders:
            raise BrokerError(f"Order not found: {broker_id}")
        return self._orders[broker_id]

    # positions / account

    async def get_positions(self) -> list[Position]:
        return list(self._positions)

    async def get_account(self) -> AccountInfo:
        return AccountInfo(
            account_id="MOCK-ACCOUNT",
            venue=self.venue,
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            is_paper=True,
        )

    async def reconcile(self, internal_positions: list[Position]) -> list[str]:
        broker_symbols  = {p.symbol for p in self._positions}
        internal_symbols = {p.symbol for p in internal_positions}
        discrepancies: list[str] = []
        for sym in broker_symbols - internal_symbols:
            discrepancies.append(f"PHANTOM: broker has {sym}, internal does not")
        for sym in internal_symbols - broker_symbols:
            discrepancies.append(f"MISSING: internal has {sym}, broker does not")
        return discrepancies


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def broker():
    return MockBroker()


@pytest.fixture
def btc_intent():
    return OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT_MAKER,
        limit_price=Decimal("65000"),
        venue="mock",
        kelly_fraction=0.1,
        target_risk_pct=0.02,
        p_win=0.58,
    )


# ---------------------------------------------------------------------------
# ABC instantiation guard
# ---------------------------------------------------------------------------

def test_cannot_instantiate_abstract():
    """BrokerAdapter must not be instantiable directly."""
    with pytest.raises(TypeError):
        BrokerAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def test_context_manager_connect_close():
    async with MockBroker() as b:
        assert b._connected is True
    assert b._connected is False


async def test_explicit_connect_close():
    b = MockBroker()
    assert b._connected is False
    await b.connect()
    assert b._connected is True
    await b.close()
    assert b._connected is False


# ---------------------------------------------------------------------------
# OrderIntent — schema correctness
# ---------------------------------------------------------------------------

def test_order_intent_uuid7_is_string(btc_intent):
    assert isinstance(btc_intent.intent_id, str)
    # UUID v7 string is 36 chars: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx
    assert len(btc_intent.intent_id) == 36
    assert btc_intent.intent_id[14] == "7"   # version nibble


def test_order_intent_qty_is_decimal(btc_intent):
    assert isinstance(btc_intent.qty, Decimal)


def test_order_intent_coerces_string_price():
    intent = OrderIntent(
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        qty="0.5",          # str → Decimal
        limit_price="3200", # str → Decimal
    )
    assert intent.qty       == Decimal("0.5")
    assert intent.limit_price == Decimal("3200")


def test_order_intent_coerces_float_price():
    intent = OrderIntent(
        symbol="SOLUSDT",
        side=OrderSide.BUY,
        qty=0.1,
        limit_price=150.25,
    )
    assert isinstance(intent.qty, Decimal)
    assert isinstance(intent.limit_price, Decimal)


def test_order_intent_optional_prices_default_none(btc_intent):
    assert btc_intent.sl_price is None
    assert btc_intent.tp_price is None


def test_order_intent_ts_is_utc(btc_intent):
    assert btc_intent.ts.tzinfo is not None
    assert btc_intent.ts.tzinfo == timezone.utc


def test_order_intent_rejects_zero_qty():
    with pytest.raises(Exception):   # ValidationError subclasses Exception
        OrderIntent(symbol="X", side=OrderSide.BUY, qty=Decimal("0"))


def test_order_intent_rejects_negative_qty():
    with pytest.raises(Exception):
        OrderIntent(symbol="X", side=OrderSide.BUY, qty=Decimal("-1"))


# ---------------------------------------------------------------------------
# Submit & OrderResult
# ---------------------------------------------------------------------------

async def test_submit_returns_order_result(broker, btc_intent):
    result = await broker.submit(btc_intent)

    assert isinstance(result, OrderResult)
    assert result.intent_id  == btc_intent.intent_id
    assert result.symbol     == btc_intent.symbol
    assert result.side       == btc_intent.side
    assert result.status     == OrderStatus.SUBMITTED
    assert result.qty        == btc_intent.qty
    assert result.venue      == "mock"


async def test_submit_broker_id_assigned(broker, btc_intent):
    result = await broker.submit(btc_intent)
    assert result.broker_id != ""


async def test_order_result_not_complete_when_submitted(broker, btc_intent):
    result = await broker.submit(btc_intent)
    assert result.is_complete is False


async def test_order_result_remaining_qty_full_when_submitted(broker, btc_intent):
    result = await broker.submit(btc_intent)
    assert result.remaining_qty == btc_intent.qty


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

async def test_cancel_returns_true_for_open_order(broker, btc_intent):
    result = await broker.submit(btc_intent)
    cancelled = await broker.cancel(result.broker_id)
    assert cancelled is True


async def test_cancel_returns_false_for_unknown_order(broker):
    assert await broker.cancel("NONEXISTENT") is False


async def test_cancel_idempotent(broker, btc_intent):
    result = await broker.submit(btc_intent)
    assert await broker.cancel(result.broker_id) is True
    assert await broker.cancel(result.broker_id) is False   # already terminal


# ---------------------------------------------------------------------------
# get_order
# ---------------------------------------------------------------------------

async def test_get_order_returns_result(broker, btc_intent):
    submitted = await broker.submit(btc_intent)
    fetched   = await broker.get_order(submitted.broker_id)
    assert fetched.broker_id == submitted.broker_id


async def test_get_order_raises_for_unknown(broker):
    with pytest.raises(BrokerError):
        await broker.get_order("UNKNOWN-ID")


# ---------------------------------------------------------------------------
# get_account
# ---------------------------------------------------------------------------

async def test_get_account_returns_account_info(broker):
    info = await broker.get_account()
    assert isinstance(info, AccountInfo)
    assert info.equity   == Decimal("100000")
    assert info.is_paper is True
    assert info.currency == "USD"


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------

async def test_get_positions_initially_empty(broker):
    positions = await broker.get_positions()
    assert positions == []


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

async def test_reconcile_clean_when_both_empty(broker):
    discrepancies = await broker.reconcile([])
    assert discrepancies == []


async def test_reconcile_detects_missing_position(broker):
    """Execution engine tracks a position the broker does not have."""
    internal = [
        Position(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            qty=Decimal("0.01"),
            avg_entry=Decimal("65000"),
            venue="mock",
        )
    ]
    discrepancies = await broker.reconcile(internal)
    assert any("MISSING" in d for d in discrepancies)


async def test_reconcile_detects_phantom_position(broker):
    """Broker reports a position the execution engine does not track."""
    broker._positions.append(
        Position(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            qty=Decimal("1"),
            avg_entry=Decimal("3200"),
            venue="mock",
        )
    )
    discrepancies = await broker.reconcile([])
    assert any("PHANTOM" in d for d in discrepancies)


async def test_reconcile_no_discrepancy_when_in_sync(broker):
    pos = Position(
        symbol="SOLUSDT",
        side=OrderSide.BUY,
        qty=Decimal("10"),
        avg_entry=Decimal("150"),
        venue="mock",
    )
    broker._positions.append(pos)
    discrepancies = await broker.reconcile([pos])
    assert discrepancies == []


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def test_fill_notional():
    fill = Fill(
        order_id="ORD-001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        price=Decimal("65000"),
        fee=Decimal("0.65"),
    )
    assert fill.notional == Decimal("650.00")


def test_fill_uuid7_version(btc_intent):
    fill = Fill(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        price=Decimal("65000"),
    )
    assert fill.fill_id[14] == "7"


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

def test_position_notional():
    pos = Position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("1"),
        avg_entry=Decimal("65000"),
    )
    assert pos.notional == Decimal("65000")


def test_position_pnl_pct_long_profit():
    pos = Position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("1"),
        avg_entry=Decimal("60000"),
        current_price=Decimal("66000"),
    )
    assert pytest.approx(pos.pnl_pct, abs=1e-6) == 0.1


def test_position_pnl_pct_short_profit():
    pos = Position(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        qty=Decimal("1"),
        avg_entry=Decimal("66000"),
        current_price=Decimal("60000"),
    )
    # short profits when price falls
    assert pos.pnl_pct is not None
    assert pos.pnl_pct > 0


def test_position_pnl_pct_none_without_price():
    pos = Position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("1"),
        avg_entry=Decimal("65000"),
    )
    assert pos.pnl_pct is None


# ---------------------------------------------------------------------------
# get_last_price default
# ---------------------------------------------------------------------------

async def test_get_last_price_default_returns_none(broker):
    price = await broker.get_last_price("BTCUSDT")
    assert price is None
