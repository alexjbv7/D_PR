"""
Tests for AlpacaAdapter.
=========================

The official ``alpaca-py`` SDK is mocked at the boundary; we never make a
real HTTP request.  The adapter is instantiated via the ``client_factory``
dependency-injection hook so these tests run without ``alpaca-py`` installed.

We verify:
  * connect / close lifecycle
  * OrderIntent → Alpaca request translation (symbol, side, TIF, type)
  * Alpaca order → OrderResult translation (status, qty, avg_price)
  * cancel / get_order / get_positions / get_account / reconcile
  * status enum mapping (filled / canceled / partially_filled / rejected)
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

import app.brokers.alpaca as alpaca_mod
from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig
from app.brokers.base import BrokerError


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _AlpacaStatusStub:
    """Mimics ``alpaca.trading.enums.OrderStatus``: an enum-like with .value."""

    def __init__(self, value: str):
        self.value = value


def _make_raw_order(
    *,
    order_id:    str = "alpaca-001",
    symbol:      str = "BTC/USD",
    side:        str = "buy",
    status:      str = "new",
    qty:         str = "0.01",
    filled_qty:  str = "0",
    filled_avg:  object | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace that looks like an alpaca-py Order response."""
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side=_AlpacaStatusStub(side),
        status=_AlpacaStatusStub(status),
        qty=qty,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg,
        submitted_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 17, 12, 1, tzinfo=timezone.utc),
    )


def _make_raw_position(
    *, symbol: str, qty: str, avg_entry: str, current_price: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg_entry,
        current_price=current_price,
        unrealized_pl="0",
        initial_margin=None,
    )


def _make_raw_account() -> SimpleNamespace:
    return SimpleNamespace(
        id="acc-123",
        equity="100000.00",
        last_equity="99500.00",
        cash="100000.00",
        initial_margin="0",
        currency="USD",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def alpaca_sdk_installed(monkeypatch):
    """Force-enable the SDK guards and the enum maps for tests."""
    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", True, raising=True)

    # Minimal stand-ins for alpaca-py enum classes (only .BUY/.SELL/etc. needed)
    class _Side:
        BUY  = SimpleNamespace(value="buy")
        SELL = SimpleNamespace(value="sell")

    class _TIF:
        GTC = SimpleNamespace(value="gtc")
        IOC = SimpleNamespace(value="ioc")
        FOK = SimpleNamespace(value="fok")
        DAY = SimpleNamespace(value="day")

    # Request classes are just constructors that store kwargs for assertion.
    class _Req:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(alpaca_mod, "_AC_Side", _Side, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_TIF",  _TIF,  raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_Market",    _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_Limit",     _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_StopLimit", _Req, raising=False)

    # Reset enum-lookup tables so the next _init_enum_maps() rebuilds them.
    alpaca_mod._SIDE_TO_ALPACA.clear()
    alpaca_mod._TIF_TO_ALPACA.clear()
    yield


@pytest.fixture
def mock_client():
    """Stand-in TradingClient with MagicMock methods."""
    client = MagicMock()
    client.submit_order.return_value      = _make_raw_order()
    client.get_order_by_id.return_value   = _make_raw_order()
    client.cancel_order_by_id.return_value = None
    client.get_all_positions.return_value = []
    client.get_account.return_value       = _make_raw_account()
    return client


@pytest.fixture
async def adapter(alpaca_sdk_installed, mock_client):
    cfg = AlpacaConfig(api_key="test", api_secret="test", paper=True)
    a = AlpacaAdapter(config=cfg, client_factory=lambda: mock_client)
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
# Config
# ---------------------------------------------------------------------------

def test_alpaca_config_defaults_to_paper(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    cfg = AlpacaConfig(api_key="k", api_secret="s")
    assert cfg.paper is True


def test_alpaca_config_repr_masks_api_key():
    cfg = AlpacaConfig(api_key="ABCDEFGHIJ", api_secret="x")
    r = repr(cfg)
    assert "ABCD" in r
    assert "EFGHIJ" not in r        # masked


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def test_connect_without_sdk_raises(monkeypatch):
    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", False, raising=True)
    a = AlpacaAdapter(config=AlpacaConfig(api_key="k", api_secret="s"))
    with pytest.raises(BrokerError, match="alpaca-py is not installed"):
        await a.connect()


async def test_connect_without_credentials_raises(alpaca_sdk_installed):
    a = AlpacaAdapter(config=AlpacaConfig(api_key="", api_secret=""))
    with pytest.raises(BrokerError, match="credentials missing"):
        await a.connect()


async def test_connect_with_factory_succeeds(alpaca_sdk_installed, mock_client):
    a = AlpacaAdapter(client_factory=lambda: mock_client)
    await a.connect()
    assert a._client is mock_client
    await a.close()
    assert a._client is None


async def test_connect_is_idempotent(alpaca_sdk_installed, mock_client):
    factory_calls = [0]
    def factory():
        factory_calls[0] += 1
        return mock_client
    a = AlpacaAdapter(client_factory=factory)
    await a.connect()
    await a.connect()
    assert factory_calls[0] == 1


# ---------------------------------------------------------------------------
# Submit — request translation
# ---------------------------------------------------------------------------

async def test_submit_translates_symbol_and_side(adapter, mock_client, btc_intent):
    await adapter.submit(btc_intent)
    request = mock_client.submit_order.call_args.kwargs["order_data"]
    assert request.symbol  == "BTC/USD"
    assert request.side    is alpaca_mod._AC_Side.BUY
    assert request.time_in_force is alpaca_mod._AC_TIF.GTC
    assert request.limit_price == 65000.0


async def test_submit_market_order_omits_price(adapter, mock_client):
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.MARKET, tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)
    request = mock_client.submit_order.call_args.kwargs["order_data"]
    assert request.symbol == "AAPL"
    assert not hasattr(request, "limit_price")


async def test_submit_limit_without_price_raises(adapter):
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"),
        order_type=OrderType.LIMIT,        # no limit_price
    )
    with pytest.raises(BrokerError, match="limit_price required"):
        await adapter.submit(intent)


async def test_submit_crypto_tif_falls_back_to_gtc(adapter, mock_client):
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.SELL, qty=Decimal("0.01"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,                # not valid for crypto on Alpaca
    )
    await adapter.submit(intent)
    request = mock_client.submit_order.call_args.kwargs["order_data"]
    assert request.time_in_force is alpaca_mod._AC_TIF.GTC


async def test_submit_propagates_broker_error(adapter, mock_client, btc_intent):
    mock_client.submit_order.side_effect = RuntimeError("rate limit")
    with pytest.raises(BrokerError, match="Alpaca submit failed"):
        await adapter.submit(btc_intent)


# ---------------------------------------------------------------------------
# Submit — result translation
# ---------------------------------------------------------------------------

async def test_submit_returns_order_result(adapter, mock_client, btc_intent):
    mock_client.submit_order.return_value = _make_raw_order(
        order_id="ord-42", symbol="BTC/USD", status="accepted",
    )
    result = await adapter.submit(btc_intent)
    assert result.broker_id == "ord-42"
    assert result.symbol    == "BTCUSDT"
    assert result.side      == OrderSide.BUY
    assert result.status    == OrderStatus.SUBMITTED
    assert result.intent_id == btc_intent.intent_id
    assert result.venue     == "alpaca"


async def test_submit_filled_status_maps_correctly(adapter, mock_client, btc_intent):
    mock_client.submit_order.return_value = _make_raw_order(
        status="filled", filled_qty="0.01", filled_avg="65000.50",
    )
    result = await adapter.submit(btc_intent)
    assert result.status     == OrderStatus.FILLED
    assert result.filled_qty == Decimal("0.01")
    assert result.avg_price  == Decimal("65000.50")


async def test_submit_partial_fill_status(adapter, mock_client, btc_intent):
    mock_client.submit_order.return_value = _make_raw_order(
        status="partially_filled", filled_qty="0.005", filled_avg="65000",
    )
    result = await adapter.submit(btc_intent)
    assert result.status     == OrderStatus.PARTIAL
    assert result.filled_qty == Decimal("0.005")


async def test_submit_rejected_status(adapter, mock_client, btc_intent):
    mock_client.submit_order.return_value = _make_raw_order(status="rejected")
    result = await adapter.submit(btc_intent)
    assert result.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

async def test_cancel_returns_true_on_success(adapter, mock_client):
    assert await adapter.cancel("ord-1") is True
    mock_client.cancel_order_by_id.assert_called_once_with("ord-1")


async def test_cancel_returns_false_when_already_terminal(adapter, mock_client):
    mock_client.cancel_order_by_id.side_effect = RuntimeError("order already filled")
    assert await adapter.cancel("ord-9") is False


async def test_cancel_raises_on_unknown_error(adapter, mock_client):
    mock_client.cancel_order_by_id.side_effect = RuntimeError("internal server error")
    with pytest.raises(BrokerError):
        await adapter.cancel("ord-99")


# ---------------------------------------------------------------------------
# get_order / get_account / get_positions
# ---------------------------------------------------------------------------

async def test_get_order_returns_result(adapter, mock_client):
    mock_client.get_order_by_id.return_value = _make_raw_order(
        order_id="ord-1", status="filled", filled_qty="0.01", filled_avg="65000",
    )
    result = await adapter.get_order("ord-1")
    assert result.broker_id == "ord-1"
    assert result.status    == OrderStatus.FILLED


async def test_get_account_returns_account_info(adapter, mock_client):
    info = await adapter.get_account()
    assert info.account_id == "acc-123"
    assert info.equity     == Decimal("100000.00")
    assert info.cash       == Decimal("100000.00")
    assert info.pnl_day    == Decimal("500.00")     # 100000 - 99500
    assert info.is_paper   is True
    assert info.venue      == "alpaca"


async def test_get_positions_empty(adapter, mock_client):
    assert await adapter.get_positions() == []


async def test_get_positions_translates_symbols_and_sides(adapter, mock_client):
    mock_client.get_all_positions.return_value = [
        _make_raw_position(symbol="BTC/USD", qty="0.5", avg_entry="60000",
                           current_price="66000"),
        _make_raw_position(symbol="AAPL", qty="-100", avg_entry="170"),
    ]
    positions = await adapter.get_positions()

    assert len(positions) == 2
    btc, aapl = positions
    assert btc.symbol == "BTCUSDT"
    assert btc.side   == OrderSide.BUY
    assert btc.qty    == Decimal("0.5")

    assert aapl.symbol == "AAPL"
    assert aapl.side   == OrderSide.SELL   # negative qty → short
    assert aapl.qty    == Decimal("100")   # absolute value


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

async def test_reconcile_clean(adapter, mock_client):
    assert await adapter.reconcile([]) == []


async def test_reconcile_detects_phantom(adapter, mock_client):
    mock_client.get_all_positions.return_value = [
        _make_raw_position(symbol="BTC/USD", qty="0.01", avg_entry="65000"),
    ]
    disc = await adapter.reconcile([])
    assert any("PHANTOM" in d and "BTCUSDT" in d for d in disc)


async def test_reconcile_detects_qty_mismatch(adapter, mock_client):
    mock_client.get_all_positions.return_value = [
        _make_raw_position(symbol="BTC/USD", qty="0.01", avg_entry="65000"),
    ]
    internal = [Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.02"), avg_entry=Decimal("65000"),
    )]
    disc = await adapter.reconcile(internal)
    assert any("QTY_MISMATCH" in d and "BTCUSDT" in d for d in disc)


async def test_reconcile_detects_side_mismatch(adapter, mock_client):
    mock_client.get_all_positions.return_value = [
        _make_raw_position(symbol="AAPL", qty="-50", avg_entry="170"),
    ]
    internal = [Position(
        symbol="AAPL", side=OrderSide.BUY,
        qty=Decimal("50"), avg_entry=Decimal("170"),
    )]
    disc = await adapter.reconcile(internal)
    assert any("SIDE_MISMATCH" in d for d in disc)


# ---------------------------------------------------------------------------
# Pre-connect guards
# ---------------------------------------------------------------------------

async def test_methods_require_connect_first(alpaca_sdk_installed):
    a = AlpacaAdapter(client_factory=lambda: MagicMock())
    # not connected yet
    with pytest.raises(BrokerError, match="connect"):
        await a.get_positions()
