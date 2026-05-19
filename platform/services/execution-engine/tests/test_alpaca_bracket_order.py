"""
Tests for Alpaca bracket / OCO / trailing stop order building (Semana 5).

Mocks ``alpaca-py`` at the boundary via ``client_factory`` — no live HTTP.
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
    TimeInForce,
)

import app.brokers.alpaca as alpaca_mod
from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig
from app.brokers.alpaca_errors import (
    BracketIncompatibleOrderTypeError,
    BracketRejectedError,
    FractionalShareNotAllowedError,
    TrailingStopRejectedError,
)
from app.brokers.base import BrokerError


# ---------------------------------------------------------------------------
# SDK stubs (extends test_alpaca_adapter pattern)
# ---------------------------------------------------------------------------

class _AlpacaStatusStub:
    def __init__(self, value: str) -> None:
        self.value = value


class _OrderClassStub:
    BRACKET = "bracket"
    OTO     = "oto"


class _Req:
    """Captures kwargs like alpaca-py request constructors."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _TakeProfitStub(_Req):
    pass


class _StopLossStub(_Req):
    pass


def _make_raw_order(**kwargs: object) -> SimpleNamespace:
    defaults = dict(
        id="alpaca-bracket-001",
        symbol="AAPL",
        side=_AlpacaStatusStub("buy"),
        status=_AlpacaStatusStub("accepted"),
        qty="10",
        filled_qty="0",
        filled_avg_price=None,
        submitted_at=datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def bracket_sdk_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", True, raising=True)

    class _Side:
        BUY  = SimpleNamespace(value="buy")
        SELL = SimpleNamespace(value="sell")

    class _TIF:
        GTC = SimpleNamespace(value="gtc")
        IOC = SimpleNamespace(value="ioc")
        FOK = SimpleNamespace(value="fok")
        DAY = SimpleNamespace(value="day")

    monkeypatch.setattr(alpaca_mod, "_AC_Side", _Side, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_TIF", _TIF, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_OrderClass", _OrderClassStub, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_Market", _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_Limit", _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_StopLimit", _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_TrailingStop", _Req, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_TakeProfit", _TakeProfitStub, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_StopLoss", _StopLossStub, raising=False)

    alpaca_mod._SIDE_TO_ALPACA.clear()
    alpaca_mod._TIF_TO_ALPACA.clear()
    yield


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.submit_order.return_value = _make_raw_order()
    return client


@pytest.fixture
async def adapter(bracket_sdk_installed: None, mock_client: MagicMock) -> AlpacaAdapter:
    cfg = AlpacaConfig(api_key="test", api_secret="test", paper=True)
    a = AlpacaAdapter(config=cfg, client_factory=lambda: mock_client)
    await a.connect()
    yield a
    await a.close()


def _bracket_intent(
    *,
    side: OrderSide = OrderSide.BUY,
    entry: str = "150",
    tp: str = "160",
    sl: str = "140",
    qty: str = "10",
    symbol: str = "AAPL",
    order_type: OrderType = OrderType.LIMIT,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=Decimal(qty),
        order_type=order_type,
        limit_price=Decimal(entry),
        tp_price=Decimal(tp),
        sl_price=Decimal(sl),
        tif=TimeInForce.DAY,
    )


# ---------------------------------------------------------------------------
# Group A — Bracket LONG
# ---------------------------------------------------------------------------

async def test_long_bracket_basic(adapter: AlpacaAdapter, mock_client: MagicMock) -> None:
    intent = _bracket_intent()
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]

    assert req.symbol == "AAPL"
    assert req.order_class == _OrderClassStub.BRACKET
    assert req.take_profit.limit_price == 160.0
    assert req.stop_loss.stop_price == 140.0
    assert req.limit_price == 150.0


async def test_long_bracket_carries_client_order_id(
    adapter: AlpacaAdapter, mock_client: MagicMock,
) -> None:
    intent = _bracket_intent()
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]
    assert req.client_order_id == intent.intent_id


async def test_long_bracket_inverted_pricing_rejected(adapter: AlpacaAdapter) -> None:
    intent = _bracket_intent(tp="140", sl="160")
    with pytest.raises(BracketRejectedError, match="LONG bracket invalid"):
        await adapter.submit(intent)


# ---------------------------------------------------------------------------
# Group B — Bracket SHORT
# ---------------------------------------------------------------------------

async def test_short_bracket_basic(adapter: AlpacaAdapter, mock_client: MagicMock) -> None:
    intent = _bracket_intent(side=OrderSide.SELL, entry="150", tp="140", sl="160")
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]

    assert req.side is alpaca_mod._AC_Side.SELL
    assert req.take_profit.limit_price == 140.0
    assert req.stop_loss.stop_price == 160.0


async def test_short_bracket_inverted_pricing_rejected(adapter: AlpacaAdapter) -> None:
    intent = _bracket_intent(side=OrderSide.SELL, tp="160", sl="140")
    with pytest.raises(BracketRejectedError, match="SHORT bracket invalid"):
        await adapter.submit(intent)


async def test_short_bracket_with_limit_maker_rejected(adapter: AlpacaAdapter) -> None:
    # model_construct bypasses Pydantic model_validator so we test adapter/builder path
    intent = OrderIntent.model_construct(
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=Decimal("10"),
        order_type=OrderType.LIMIT_MAKER,
        limit_price=Decimal("150"),
        tp_price=Decimal("140"),
        sl_price=Decimal("160"),
        tif=TimeInForce.DAY,
        intent_id="test-limit-maker-bracket",
    )
    with pytest.raises(BracketIncompatibleOrderTypeError):
        await adapter.submit(intent)


# ---------------------------------------------------------------------------
# Group C — OCO
# ---------------------------------------------------------------------------

async def test_oco_only_stop_loss(adapter: AlpacaAdapter, mock_client: MagicMock) -> None:
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.LIMIT, limit_price=Decimal("150"),
        sl_price=Decimal("140"), tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]

    assert req.order_class == _OrderClassStub.OTO
    assert req.stop_loss.stop_price == 140.0
    assert not hasattr(req, "take_profit") or req.__dict__.get("take_profit") is None


async def test_oco_only_take_profit(adapter: AlpacaAdapter, mock_client: MagicMock) -> None:
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.LIMIT, limit_price=Decimal("150"),
        tp_price=Decimal("160"), tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]

    assert req.order_class == _OrderClassStub.OTO
    assert req.take_profit.limit_price == 160.0
    assert not hasattr(req, "stop_loss") or req.__dict__.get("stop_loss") is None


# ---------------------------------------------------------------------------
# Group D — Trailing Stop
# ---------------------------------------------------------------------------

async def test_trailing_stop_with_percent(
    adapter: AlpacaAdapter, mock_client: MagicMock,
) -> None:
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.TRAILING_STOP,
        trail_percent=Decimal("1.5"),
        tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]
    assert req.trail_percent == 1.5
    assert not hasattr(req, "trail_price") or req.__dict__.get("trail_price") is None


async def test_trailing_stop_with_price(
    adapter: AlpacaAdapter, mock_client: MagicMock,
) -> None:
    intent = OrderIntent(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.TRAILING_STOP,
        trail_price=Decimal("2.50"),
        tif=TimeInForce.DAY,
    )
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]
    assert req.trail_price == 2.5


async def test_trailing_stop_both_set_rejected(adapter: AlpacaAdapter) -> None:
    intent = OrderIntent.model_construct(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.TRAILING_STOP,
        trail_percent=Decimal("1.0"),
        trail_price=Decimal("2.0"),
        intent_id="trail-both",
    )
    with pytest.raises(TrailingStopRejectedError, match="mutually exclusive"):
        await adapter.submit(intent)


async def test_trailing_stop_neither_set_rejected(adapter: AlpacaAdapter) -> None:
    intent = OrderIntent.model_construct(
        symbol="AAPL", side=OrderSide.BUY, qty=Decimal("10"),
        order_type=OrderType.TRAILING_STOP,
        intent_id="trail-none",
    )
    with pytest.raises(TrailingStopRejectedError, match="requires trail_percent"):
        await adapter.submit(intent)


# ---------------------------------------------------------------------------
# Group E — Crypto
# ---------------------------------------------------------------------------

async def test_bracket_crypto_btc_allowed(
    adapter: AlpacaAdapter, mock_client: MagicMock,
) -> None:
    intent = _bracket_intent(symbol="BTCUSDT", entry="65000", tp="70000", sl="60000", qty="0.01")
    await adapter.submit(intent)
    req = mock_client.submit_order.call_args.kwargs["order_data"]
    assert req.symbol == "BTC/USD"
    assert req.order_class == _OrderClassStub.BRACKET


async def test_bracket_crypto_obscure_altcoin_rejected(adapter: AlpacaAdapter) -> None:
    intent = _bracket_intent(symbol="SHIBUSDT", entry="0.00002", tp="0.00003", sl="0.00001", qty="1000")
    with pytest.raises(BracketRejectedError, match="Bracket orders for crypto"):
        await adapter.submit(intent)


async def test_bracket_fractional_equity_rejected(adapter: AlpacaAdapter) -> None:
    intent = _bracket_intent(qty="10.5")
    with pytest.raises(FractionalShareNotAllowedError, match="whole-share"):
        await adapter.submit(intent)


# ---------------------------------------------------------------------------
# Group F — E2E submit
# ---------------------------------------------------------------------------

async def test_submit_bracket_returns_order_result(
    adapter: AlpacaAdapter, mock_client: MagicMock,
) -> None:
    mock_client.submit_order.return_value = _make_raw_order(
        id="brk-99", status="accepted",
    )
    intent = _bracket_intent()
    result = await adapter.submit(intent)

    assert result.broker_id == "brk-99"
    assert result.status == OrderStatus.SUBMITTED
    assert result.intent_id == intent.intent_id
    mock_client.submit_order.assert_called_once()
