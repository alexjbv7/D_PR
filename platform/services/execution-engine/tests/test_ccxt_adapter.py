"""
Tests for CCXTAdapter.
=======================
The ccxt library is mocked at the boundary via the ``exchange_factory`` DI
hook so these tests run without ccxt installed.

We verify:
  * config validation (unknown venue / unknown market_type)
  * lifecycle (connect/close, sandbox honoured, load_markets called)
  * OrderIntent → ccxt.create_order argument translation
  * ccxt order dict → OrderResult translation (incl. partial fill detection)
  * cancel / get_order / get_positions (spot vs swap) / get_account / reconcile
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

import app.brokers.ccxt_adapter as ccxt_mod
from app.brokers.ccxt_adapter import CCXTAdapter, CCXTConfig
from app.brokers.base import BrokerError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_exchange_mock(**fetch_balance_overrides) -> MagicMock:
    """Build a ccxt-shaped exchange with all async methods mocked."""
    ex = MagicMock()
    ex.load_markets    = AsyncMock(return_value={})
    ex.create_order    = AsyncMock(return_value={
        "id":        "ccxt-001",
        "symbol":    "BTC/USDT",
        "side":      "buy",
        "status":    "open",
        "type":      "limit",
        "amount":    0.01,
        "filled":    0.0,
        "remaining": 0.01,
        "average":   None,
        "price":     65000,
        "timestamp": 1747490400000,
    })
    ex.cancel_order    = AsyncMock(return_value={})
    ex.fetch_order     = AsyncMock(return_value={
        "id": "ccxt-001", "symbol": "BTC/USDT", "side": "buy",
        "status": "closed", "amount": 0.01, "filled": 0.01,
        "average": 65010, "timestamp": 1747490400000,
    })
    default_balance = {
        "BTC":  {"free": 0.5, "used": 0.0, "total": 0.5},
        "USDT": {"free": 50000, "used": 0, "total": 50000},
    }
    default_balance.update(fetch_balance_overrides)
    ex.fetch_balance   = AsyncMock(return_value=default_balance)
    ex.fetch_positions = AsyncMock(return_value=[])
    ex.fetch_ticker    = AsyncMock(return_value={"last": 66000.0})
    ex.close           = AsyncMock(return_value=None)
    ex.set_sandbox_mode = MagicMock()
    return ex


@pytest.fixture
def exchange():
    return _make_exchange_mock()


@pytest.fixture
async def adapter(exchange):
    cfg = CCXTConfig(
        exchange="binance", api_key="k", api_secret="s",
        testnet=True, market_type="spot",
    )
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    yield a
    await a.close()


@pytest.fixture
def btc_intent():
    return OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=Decimal("0.01"),
        order_type=OrderType.LIMIT_MAKER,
        limit_price=Decimal("65000"),
        tif=TimeInForce.GTC,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_unknown_venue():
    with pytest.raises(BrokerError, match="Unsupported CCXT venue"):
        CCXTConfig(exchange="unknown_venue", api_key="k", api_secret="s")


def test_config_rejects_bad_market_type():
    with pytest.raises(BrokerError, match="market_type"):
        CCXTConfig(exchange="binance", api_key="k", api_secret="s",
                   market_type="margin")


def test_config_repr_masks_api_key():
    cfg = CCXTConfig(exchange="binance", api_key="ABCDEFGHIJ", api_secret="x")
    r = repr(cfg)
    assert "ABCD" in r
    assert "EFGHIJ" not in r


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def test_connect_calls_load_markets(exchange):
    cfg = CCXTConfig(exchange="binance", api_key="k", api_secret="s", testnet=True)
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    exchange.load_markets.assert_awaited_once()


async def test_connect_enables_sandbox_when_testnet(exchange):
    cfg = CCXTConfig(exchange="binance", api_key="k", api_secret="s", testnet=True)
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    exchange.set_sandbox_mode.assert_called_once_with(True)


async def test_connect_does_not_enable_sandbox_when_live(exchange):
    cfg = CCXTConfig(exchange="binance", api_key="k", api_secret="s", testnet=False)
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    exchange.set_sandbox_mode.assert_not_called()


async def test_close_calls_exchange_close(exchange):
    cfg = CCXTConfig(exchange="binance", api_key="k", api_secret="s")
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    await a.close()
    exchange.close.assert_awaited_once()


async def test_connect_without_sdk_raises(monkeypatch):
    monkeypatch.setattr(ccxt_mod, "_HAS_CCXT", False, raising=True)
    a = CCXTAdapter(config=CCXTConfig(api_key="k", api_secret="s"))
    with pytest.raises(BrokerError, match="ccxt is not installed"):
        await a.connect()


async def test_connect_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(ccxt_mod, "_HAS_CCXT", True, raising=True)
    a = CCXTAdapter(config=CCXTConfig(exchange="binance"))   # empty creds
    with pytest.raises(BrokerError, match="credentials missing"):
        await a.connect()


async def test_venue_overridden_per_instance(exchange):
    cfg = CCXTConfig(exchange="bybit", api_key="k", api_secret="s")
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    assert a.venue == "bybit"


# ---------------------------------------------------------------------------
# Submit — argument translation
# ---------------------------------------------------------------------------

async def test_submit_translates_symbol(adapter, exchange, btc_intent):
    await adapter.submit(btc_intent)
    kwargs = exchange.create_order.call_args.kwargs
    assert kwargs["symbol"] == "BTC/USDT"


async def test_submit_translates_side_and_amount(adapter, exchange, btc_intent):
    await adapter.submit(btc_intent)
    kwargs = exchange.create_order.call_args.kwargs
    assert kwargs["side"]   == "buy"
    assert kwargs["amount"] == 0.01
    assert kwargs["type"]   == "limit"


async def test_submit_limit_maker_sets_post_only(adapter, exchange, btc_intent):
    await adapter.submit(btc_intent)
    params = exchange.create_order.call_args.kwargs["params"]
    assert params["postOnly"] is True


async def test_submit_market_order_omits_price(adapter, exchange):
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.SELL, qty=Decimal("0.01"),
        order_type=OrderType.MARKET, tif=TimeInForce.IOC,
    )
    await adapter.submit(intent)
    kwargs = exchange.create_order.call_args.kwargs
    assert kwargs["type"]  == "market"
    assert kwargs["price"] is None


async def test_submit_tif_translated(adapter, exchange):
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"),
        order_type=OrderType.LIMIT, limit_price=Decimal("65000"),
        tif=TimeInForce.IOC,
    )
    await adapter.submit(intent)
    params = exchange.create_order.call_args.kwargs["params"]
    assert params["timeInForce"] == "IOC"


async def test_submit_propagates_error(adapter, exchange, btc_intent):
    exchange.create_order.side_effect = RuntimeError("insufficient funds")
    with pytest.raises(BrokerError, match="submit failed"):
        await adapter.submit(btc_intent)


# ---------------------------------------------------------------------------
# Submit — response translation
# ---------------------------------------------------------------------------

async def test_submit_returns_order_result(adapter, exchange, btc_intent):
    result = await adapter.submit(btc_intent)
    assert result.broker_id  == "ccxt-001"
    assert result.symbol     == "BTCUSDT"           # canonical, not BTC/USDT
    assert result.side       == OrderSide.BUY
    assert result.status     == OrderStatus.SUBMITTED
    assert result.intent_id  == btc_intent.intent_id
    assert result.venue      == "binance"


async def test_submit_status_filled(adapter, exchange, btc_intent):
    exchange.create_order.return_value = {
        "id": "x", "symbol": "BTC/USDT", "side": "buy",
        "status": "closed", "amount": 0.01, "filled": 0.01,
        "average": 65010, "timestamp": 1747490400000,
    }
    result = await adapter.submit(btc_intent)
    assert result.status     == OrderStatus.FILLED
    assert result.filled_qty == Decimal("0.01")
    assert result.avg_price  == Decimal("65010")


async def test_submit_status_partial_when_open_with_filled(adapter, exchange, btc_intent):
    exchange.create_order.return_value = {
        "id": "x", "symbol": "BTC/USDT", "side": "buy",
        "status": "open", "amount": 0.01, "filled": 0.005,
        "average": 65000, "timestamp": 1747490400000,
    }
    result = await adapter.submit(btc_intent)
    assert result.status == OrderStatus.PARTIAL


async def test_submit_status_canceled(adapter, exchange, btc_intent):
    exchange.create_order.return_value = {
        "id": "x", "symbol": "BTC/USDT", "side": "buy",
        "status": "canceled", "amount": 0.01, "filled": 0,
    }
    result = await adapter.submit(btc_intent)
    assert result.status == OrderStatus.CANCELLED


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

async def test_cancel_returns_true_on_success(adapter, exchange):
    assert await adapter.cancel("ccxt-001") is True
    exchange.cancel_order.assert_awaited_once_with("ccxt-001")


async def test_cancel_returns_false_when_already_done(adapter, exchange):
    exchange.cancel_order.side_effect = RuntimeError("OrderNotFound: id=x")
    assert await adapter.cancel("ccxt-001") is False


async def test_cancel_raises_on_unknown_error(adapter, exchange):
    exchange.cancel_order.side_effect = RuntimeError("rate limit")
    with pytest.raises(BrokerError):
        await adapter.cancel("ccxt-001")


# ---------------------------------------------------------------------------
# get_order
# ---------------------------------------------------------------------------

async def test_get_order_returns_result(adapter, exchange):
    result = await adapter.get_order("ccxt-001")
    assert result.broker_id == "ccxt-001"
    assert result.status    == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Positions — spot
# ---------------------------------------------------------------------------

async def test_get_positions_spot_synthesises_from_balance(adapter, exchange):
    exchange.fetch_balance.return_value = {
        "BTC":  {"free": 0.5, "used": 0.0, "total": 0.5},
        "ETH":  {"free": 2.0, "used": 0.0, "total": 2.0},
        "USDT": {"free": 50000, "used": 0, "total": 50000},     # ignored (quote)
    }
    positions = await adapter.get_positions()
    by_sym = {p.symbol: p for p in positions}
    assert "BTCUSDT" in by_sym
    assert "ETHUSDT" in by_sym
    assert "USDTUSDT" not in by_sym                              # filtered
    assert by_sym["BTCUSDT"].qty == Decimal("0.5")
    assert by_sym["BTCUSDT"].side == OrderSide.BUY


async def test_get_positions_spot_skips_zero_balance(adapter, exchange):
    exchange.fetch_balance.return_value = {
        "BTC":  {"free": 0, "used": 0, "total": 0},
        "ETH":  {"free": 2.0, "used": 0, "total": 2.0},
    }
    positions = await adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# Positions — swap
# ---------------------------------------------------------------------------

async def test_get_positions_swap_uses_fetch_positions(exchange):
    exchange.fetch_positions.return_value = [{
        "symbol":         "BTC/USDT:USDT",
        "side":           "long",
        "contracts":      0.01,
        "entryPrice":     65000,
        "markPrice":      66000,
        "unrealizedPnl":  10.0,
        "initialMargin":  65,
    }]
    cfg = CCXTConfig(exchange="binance", api_key="k", api_secret="s",
                     market_type="swap", testnet=True)
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    positions = await a.get_positions()
    await a.close()

    assert len(positions) == 1
    assert positions[0].symbol         == "BTCUSDT.P"
    assert positions[0].side           == OrderSide.BUY
    assert positions[0].avg_entry      == Decimal("65000")
    assert positions[0].unrealized_pnl == Decimal("10.0")


async def test_get_positions_swap_filters_zero_contracts(exchange):
    exchange.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "side": "long",  "contracts": 0,
         "entryPrice": 0},
        {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1,
         "entryPrice": 3200, "markPrice": 3100},
    ]
    cfg = CCXTConfig(exchange="binance", api_key="k", api_secret="s",
                     market_type="swap", testnet=True)
    a = CCXTAdapter(config=cfg, exchange_factory=lambda: exchange)
    await a.connect()
    positions = await a.get_positions()
    await a.close()

    assert len(positions) == 1
    assert positions[0].symbol == "ETHUSDT.P"
    assert positions[0].side   == OrderSide.SELL


# ---------------------------------------------------------------------------
# get_account
# ---------------------------------------------------------------------------

async def test_get_account_uses_usdt(adapter, exchange):
    info = await adapter.get_account()
    assert info.equity   == Decimal("50000")
    assert info.cash     == Decimal("50000")
    assert info.currency == "USDT"
    assert info.is_paper is True       # testnet
    assert info.venue    == "binance"


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

async def test_reconcile_detects_phantom(adapter, exchange):
    exchange.fetch_balance.return_value = {
        "BTC":  {"free": 0.01, "used": 0, "total": 0.01},
        "USDT": {"free": 1000, "used": 0, "total": 1000},
    }
    disc = await adapter.reconcile([])
    assert any("PHANTOM" in d and "BTCUSDT" in d for d in disc)


async def test_reconcile_detects_qty_mismatch(adapter, exchange):
    exchange.fetch_balance.return_value = {
        "BTC":  {"free": 0.01, "used": 0, "total": 0.01},
        "USDT": {"free": 1000, "used": 0, "total": 1000},
    }
    internal = [Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.02"), avg_entry=Decimal("65000"),
    )]
    disc = await adapter.reconcile(internal)
    assert any("QTY_MISMATCH" in d for d in disc)


# ---------------------------------------------------------------------------
# get_last_price
# ---------------------------------------------------------------------------

async def test_get_last_price_returns_decimal(adapter, exchange):
    price = await adapter.get_last_price("BTCUSDT")
    assert price == Decimal("66000.0")


async def test_get_last_price_returns_none_on_error(adapter, exchange):
    exchange.fetch_ticker.side_effect = RuntimeError("symbol not supported")
    assert await adapter.get_last_price("XXXUSDT") is None


# ---------------------------------------------------------------------------
# Pre-connect guards
# ---------------------------------------------------------------------------

async def test_methods_require_connect_first():
    a = CCXTAdapter(config=CCXTConfig(exchange="binance", api_key="k",
                                       api_secret="s"))
    with pytest.raises(BrokerError, match="connect"):
        await a.get_positions()
