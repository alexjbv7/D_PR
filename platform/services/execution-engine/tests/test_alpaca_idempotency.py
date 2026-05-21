"""
Tests — Alpaca idempotency via client_order_id.
=================================================

Verifies:
  * Every intent → request includes client_order_id == intent.intent_id.
  * Resending the same intent produces the same client_order_id.
  * Two different intents produce distinct client_order_ids.
  * An HTTP 422 "duplicate" response raises BrokerError (future: DuplicateOrderError).
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
    OrderType,
    TimeInForce,
)

import app.brokers.alpaca as alpaca_mod
from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig
from app.brokers.base import BrokerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AlpacaStatusStub:
    def __init__(self, value: str):
        self.value = value


def _make_raw_order(order_id: str = "alpaca-001") -> SimpleNamespace:
    return SimpleNamespace(
        id=order_id,
        symbol="BTC/USD",
        side=_AlpacaStatusStub("buy"),
        status=_AlpacaStatusStub("new"),
        qty="0.01",
        filled_qty="0",
        filled_avg_price=None,
        submitted_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 17, 12, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def alpaca_sdk_installed(monkeypatch):
    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", True, raising=True)

    class _Side:
        BUY  = SimpleNamespace(value="buy")
        SELL = SimpleNamespace(value="sell")

    class _TIF:
        GTC = SimpleNamespace(value="gtc")
        IOC = SimpleNamespace(value="ioc")
        FOK = SimpleNamespace(value="fok")
        DAY = SimpleNamespace(value="day")

    class _Req:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(alpaca_mod, "_AC_Side", _Side, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_TIF",  _TIF,  raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_Market",    _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_Limit",     _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_StopLimit", _Req, raising=False)

    alpaca_mod._SIDE_TO_ALPACA.clear()
    alpaca_mod._TIF_TO_ALPACA.clear()
    yield


class _TrackingClient:
    """Records every client_order_id submitted."""

    def __init__(self):
        self.submitted_coids: list[str] = []

    def submit_order(self, *, order_data):
        coid = getattr(order_data, "client_order_id", None)
        self.submitted_coids.append(coid)
        return _make_raw_order()

    def get_order_by_id(self, _):
        return _make_raw_order()

    def cancel_order_by_id(self, _):
        pass

    def get_all_positions(self):
        return []

    def get_account(self):
        return SimpleNamespace(
            id="acc-1", equity="10000", cash="10000",
            last_equity="10000", initial_margin="0", currency="USD",
        )


@pytest.fixture
def tracking_client():
    return _TrackingClient()


@pytest.fixture
async def adapter(alpaca_sdk_installed, tracking_client):
    cfg = AlpacaConfig(api_key="k", api_secret="s", paper=True)
    a = AlpacaAdapter(config=cfg, client_factory=lambda: tracking_client)
    await a.connect()
    yield a
    await a.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_submit_stamps_client_order_id(adapter, tracking_client):
    """Every submit_order call must carry client_order_id == intent.intent_id."""
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"),
        order_type=OrderType.MARKET, tif=TimeInForce.GTC,
    )
    await adapter.submit(intent)

    assert len(tracking_client.submitted_coids) == 1
    assert tracking_client.submitted_coids[0] == intent.intent_id


async def test_same_intent_produces_same_client_order_id(adapter, tracking_client):
    """Resending the same intent must produce the same client_order_id."""
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.MARKET, tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)
    await adapter.submit(intent)

    assert tracking_client.submitted_coids[0] == tracking_client.submitted_coids[1]
    assert tracking_client.submitted_coids[0] == intent.intent_id


async def test_different_intents_produce_different_client_order_ids(adapter, tracking_client):
    """Two distinct intents must have different client_order_ids."""
    i1 = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.BUY, qty=Decimal("0.01"),
        order_type=OrderType.MARKET,
    )
    i2 = OrderIntent(
        symbol="ETHUSDT", side=OrderSide.SELL, qty=Decimal("1"),
        order_type=OrderType.MARKET,
    )
    await adapter.submit(i1)
    await adapter.submit(i2)

    assert tracking_client.submitted_coids[0] != tracking_client.submitted_coids[1]


async def test_limit_order_includes_client_order_id(adapter, tracking_client):
    """Limit orders must also carry client_order_id."""
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("5"),
        order_type=OrderType.LIMIT, limit_price=Decimal("150"),
        tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)

    assert tracking_client.submitted_coids[0] == intent.intent_id


async def test_stop_limit_includes_client_order_id(adapter, tracking_client):
    """StopLimit orders must also carry client_order_id."""
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.SELL, qty=Decimal("0.01"),
        order_type=OrderType.STOP_LIMIT,
        limit_price=Decimal("60000"),
        sl_price=Decimal("59000"),
        tif=TimeInForce.GTC,
    )
    await adapter.submit(intent)

    assert tracking_client.submitted_coids[0] == intent.intent_id


async def test_422_duplicate_raises_broker_error(alpaca_sdk_installed):
    """HTTP 422 duplicate order → BrokerError (future: DuplicateOrderError)."""

    class _RejectClient:
        def submit_order(self, *, order_data):
            raise RuntimeError("422 Unprocessable Entity: duplicate client_order_id")

        def get_all_positions(self):
            return []

        def get_account(self):
            return SimpleNamespace(
                id="acc-1", equity="10000", cash="10000",
                last_equity="10000", initial_margin="0", currency="USD",
            )

    cfg = AlpacaConfig(api_key="k", api_secret="s", paper=True)
    a = AlpacaAdapter(config=cfg, client_factory=lambda: _RejectClient())
    await a.connect()

    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("1"),
        order_type=OrderType.MARKET,
    )
    with pytest.raises(BrokerError, match="Alpaca submit failed"):
        await a.submit(intent)
    await a.close()
