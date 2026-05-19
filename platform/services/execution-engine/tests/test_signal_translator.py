"""Tests for FinalSignalEvent → OrderIntent translation."""
from __future__ import annotations

from decimal import Decimal

import pytest

from quant_shared.schemas.orders import OrderSide, OrderType, TimeInForce

from app.signal_translator import translate_signal


def _signal(**overrides):
    base = {
        "event_id":      "sig-1",
        "strategy":      "regime_adaptive",
        "symbol":        "BTCUSDT",
        "direction":     1,
        "p_win":         0.6,
        "position_size": 0.02,
        "target_risk_pct": 0.01,
        "ts":            "2026-05-17T17:00:00+00:00",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_long_signal_produces_buy_intent():
    intent = translate_signal(
        _signal(direction=1),
        equity=Decimal("100000"),
        current_price=Decimal("1000"),
        default_venue="binance",
    )
    assert intent is not None
    assert intent.side       == OrderSide.BUY
    assert intent.symbol     == "BTCUSDT"
    assert intent.signal_id  == "sig-1"
    assert intent.venue      == "binance"
    assert intent.order_type == OrderType.LIMIT_MAKER
    assert intent.tif        == TimeInForce.GTC


def test_short_signal_produces_sell_intent():
    intent = translate_signal(
        _signal(direction=-1),
        equity=Decimal("100000"),
        current_price=Decimal("65000"),
        default_venue="binance",
    )
    assert intent is not None
    assert intent.side == OrderSide.SELL


def test_qty_sized_from_kelly_equity_and_price():
    # kelly 0.02 × 100k = 2000 notional ; / 1000 = 2 shares
    intent = translate_signal(
        _signal(position_size=0.02),
        equity=Decimal("100000"),
        current_price=Decimal("1000"),
        default_venue="binance",
    )
    assert intent is not None
    expected = Decimal("2000") / Decimal("1000")
    assert intent.qty == expected


def test_limit_price_equals_current_price():
    intent = translate_signal(
        _signal(), equity=Decimal("100000"),
        current_price=Decimal("1000"), default_venue="binance",
    )
    assert intent is not None
    assert intent.limit_price == Decimal("1000")


def test_fractional_notional_when_below_one_share():
    # 2000 notional cannot buy 1 unit at 65000 → MARKET notional order
    intent = translate_signal(
        _signal(position_size=0.02),
        equity=Decimal("100000"),
        current_price=Decimal("65000"),
        default_venue="binance",
    )
    assert intent is not None
    assert intent.notional == Decimal("2000")
    assert intent.qty is None
    assert intent.order_type == OrderType.MARKET


def test_p_win_carried_as_metadata():
    intent = translate_signal(
        _signal(p_win=0.65), equity=Decimal("100000"),
        current_price=Decimal("65000"), default_venue="binance",
    )
    assert intent is not None
    assert intent.p_win == 0.65


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def test_flat_signal_returns_none():
    assert translate_signal(
        _signal(direction=0), equity=Decimal("100000"),
        current_price=Decimal("65000"),
    ) is None


def test_missing_price_returns_none():
    assert translate_signal(
        _signal(), equity=Decimal("100000"), current_price=None,
    ) is None


def test_zero_price_returns_none():
    assert translate_signal(
        _signal(), equity=Decimal("100000"), current_price=Decimal("0"),
    ) is None


def test_zero_kelly_returns_none():
    assert translate_signal(
        _signal(position_size=0), equity=Decimal("100000"),
        current_price=Decimal("65000"),
    ) is None


def test_below_min_notional_returns_none():
    # kelly 0.0001 × 50 = 0.005 notional ; below the 10-USD floor
    result = translate_signal(
        _signal(position_size=0.0001),
        equity=Decimal("50"),
        current_price=Decimal("65000"),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Venue handling
# ---------------------------------------------------------------------------

def test_signal_venue_overrides_default():
    intent = translate_signal(
        _signal(venue="kraken"),
        equity=Decimal("100000"),
        current_price=Decimal("65000"),
        default_venue="binance",
    )
    assert intent is not None
    assert intent.venue == "kraken"


def test_default_venue_used_when_missing():
    sig = _signal()
    sig.pop("venue", None)
    intent = translate_signal(
        sig,
        equity=Decimal("100000"),
        current_price=Decimal("65000"),
        default_venue="bybit",
    )
    assert intent is not None
    assert intent.venue == "bybit"
