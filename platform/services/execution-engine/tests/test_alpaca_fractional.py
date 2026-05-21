"""Alpaca fractional notional orders (Semana 6)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from quant_shared.schemas.orders import OrderIntent, OrderSide, OrderType, TimeInForce

import app.brokers.alpaca as alpaca_mod
from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig
from app.brokers.alpaca_errors import FractionalShareNotAllowedError


class _Req:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


@pytest.fixture
def fractional_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", True)
    monkeypatch.setattr(alpaca_mod, "_AC_Market", _Req)
    monkeypatch.setattr(alpaca_mod, "_AC_Limit", _Req)
    monkeypatch.setattr(alpaca_mod, "_SIDE_TO_ALPACA", {OrderSide.BUY: "buy", OrderSide.SELL: "sell"})
    monkeypatch.setattr(alpaca_mod, "_TIF_TO_ALPACA", {TimeInForce.DAY: "day"})
    alpaca_mod._init_enum_maps()


@pytest.fixture
def adapter(fractional_sdk: None) -> AlpacaAdapter:
    mock_client = MagicMock()
    a = AlpacaAdapter(config=AlpacaConfig(), client_factory=lambda: mock_client)
    a._client = mock_client
    return a


def test_fractional_notional_path(adapter: AlpacaAdapter) -> None:
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        notional=Decimal("50"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
    )
    req = adapter._build_request(intent)
    assert isinstance(req, _Req)
    assert req.notional == 50.0
    assert not hasattr(req, "qty") or getattr(req, "qty", None) is None


def test_qty_path_unchanged(adapter: AlpacaAdapter) -> None:
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=Decimal("10"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
    )
    req = adapter._build_request(intent)
    assert isinstance(req, _Req)
    assert req.qty == 10.0


def test_fractional_limit_rejected(adapter: AlpacaAdapter) -> None:
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.BUY,
        notional=Decimal("50"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
        tif=TimeInForce.DAY,
    )
    with pytest.raises(FractionalShareNotAllowedError, match="MARKET"):
        adapter._build_request(intent)


def test_fractional_sell_rejected(adapter: AlpacaAdapter) -> None:
    intent = OrderIntent(
        symbol="AAPL",
        side=OrderSide.SELL,
        notional=Decimal("50"),
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
    )
    with pytest.raises(FractionalShareNotAllowedError, match="SELL"):
        adapter._build_request(intent)
