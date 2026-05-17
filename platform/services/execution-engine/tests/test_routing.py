"""
Tests for the venue Router.
============================
Verifies:
  * is_equity heuristic
  * adapter registry (register / get / venues / unknown)
  * routing rules:
      - explicit intent.venue takes precedence
      - empty venue → equity → default_equity
      - empty venue → crypto → default_crypto
  * delegate methods (submit / cancel / get_order / get_positions_all)
  * close_all is best-effort
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
)

from app.brokers.base import BrokerError
from app.routing import Router, is_equity


# ---------------------------------------------------------------------------
# is_equity heuristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sym", ["AAPL", "MSFT", "SPY", "QQQ", "F"])
def test_is_equity_true(sym):
    assert is_equity(sym) is True


@pytest.mark.parametrize("sym", [
    "BTCUSDT", "ETHUSDT", "BTC/USDT", "BTC/USDT:USDT", "BTCUSDT.P",
])
def test_is_equity_false_for_crypto(sym):
    assert is_equity(sym) is False


def test_is_equity_rejects_too_long():
    assert is_equity("VERYLONGTICKER") is False


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _make_adapter(venue: str) -> MagicMock:
    a = MagicMock()
    a.venue           = venue
    a.submit          = AsyncMock()
    a.cancel          = AsyncMock(return_value=True)
    a.get_order       = AsyncMock()
    a.get_positions   = AsyncMock(return_value=[])
    a.close           = AsyncMock()
    return a


@pytest.fixture
def alpaca():
    return _make_adapter("alpaca")


@pytest.fixture
def binance():
    return _make_adapter("binance")


@pytest.fixture
def router(alpaca, binance):
    r = Router(default_equity="alpaca", default_crypto="binance")
    r.register(alpaca)
    r.register(binance)
    return r


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_register_rejects_unknown_venue_tag(router):
    bad = _make_adapter("unknown")
    with pytest.raises(BrokerError, match="no venue tag"):
        router.register(bad)


def test_get_unknown_venue_raises(router):
    with pytest.raises(BrokerError, match="No adapter registered"):
        router.get("kraken")


def test_venues_list(router):
    assert router.venues() == ["alpaca", "binance"]


# ---------------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------------

def test_route_explicit_venue_wins(router, binance):
    intent = OrderIntent(
        symbol="AAPL",                       # would default to alpaca
        side=OrderSide.BUY,
        qty=Decimal("1"),
        venue="binance",                     # but explicit venue beats it
    )
    assert router.route(intent) is binance


def test_route_equity_defaults_to_alpaca(router, alpaca):
    intent = OrderIntent(symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"))
    assert router.route(intent) is alpaca


def test_route_crypto_defaults_to_binance(router, binance):
    intent = OrderIntent(symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"))
    assert router.route(intent) is binance


def test_route_unknown_venue_raises(router):
    intent = OrderIntent(
        symbol="X", side=OrderSide.BUY, qty=Decimal("1"), venue="kraken",
    )
    with pytest.raises(BrokerError):
        router.route(intent)


# ---------------------------------------------------------------------------
# Delegates
# ---------------------------------------------------------------------------

async def test_submit_delegates_to_routed_adapter(router, binance):
    intent = OrderIntent(symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"))
    binance.submit.return_value = OrderResult(
        symbol="BTCUSDT", side=OrderSide.BUY, status=OrderStatus.SUBMITTED,
        qty=Decimal("0.01"), venue="binance",
    )
    result = await router.submit(intent)
    binance.submit.assert_awaited_once_with(intent)
    assert result.venue == "binance"


async def test_cancel_delegates(router, alpaca):
    await router.cancel("alpaca", "ord-1")
    alpaca.cancel.assert_awaited_once_with("ord-1")


async def test_get_order_delegates(router, binance):
    binance.get_order.return_value = OrderResult(
        symbol="BTCUSDT", side=OrderSide.BUY, status=OrderStatus.FILLED,
        qty=Decimal("0.01"),
    )
    await router.get_order("binance", "ccxt-1")
    binance.get_order.assert_awaited_once_with("ccxt-1")


async def test_get_positions_all_aggregates(router, alpaca, binance):
    alpaca.get_positions.return_value = [Position(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        avg_entry=Decimal("170"),
    )]
    binance.get_positions.return_value = [Position(
        symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"),
        avg_entry=Decimal("65000"),
    )]
    all_pos = await router.get_positions_all()
    assert len(all_pos) == 2
    assert {p.symbol for p in all_pos} == {"AAPL", "BTCUSDT"}


async def test_get_positions_all_tolerates_partial_failure(router, alpaca, binance):
    alpaca.get_positions.side_effect = BrokerError("alpaca down")
    binance.get_positions.return_value = [Position(
        symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"),
        avg_entry=Decimal("65000"),
    )]
    all_pos = await router.get_positions_all()
    assert len(all_pos) == 1
    assert all_pos[0].symbol == "BTCUSDT"


async def test_close_all_calls_every_adapter(router, alpaca, binance):
    await router.close_all()
    alpaca.close.assert_awaited_once()
    binance.close.assert_awaited_once()


async def test_close_all_tolerates_errors(router, alpaca, binance):
    alpaca.close.side_effect = RuntimeError("boom")
    await router.close_all()                # must not raise
    binance.close.assert_awaited_once()
