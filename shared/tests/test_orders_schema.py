"""OrderIntent schema validators — trailing stop and bracket constraints."""
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_shared.schemas.orders import OrderIntent, OrderSide, OrderType, TimeInForce


def _intent(**kwargs: object) -> OrderIntent:
    base = dict(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=Decimal("10"),
        order_type=OrderType.LIMIT,
    )
    base.update(kwargs)
    return OrderIntent(**base)  # type: ignore[arg-type]


def test_trailing_stop_requires_xor_trail_fields() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _intent(
            order_type=OrderType.TRAILING_STOP,
            trail_percent=Decimal("1.5"),
            trail_price=Decimal("2"),
        )
    with pytest.raises(ValidationError, match="exactly one"):
        _intent(order_type=OrderType.TRAILING_STOP)


def test_trailing_stop_percent_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="trail_percent"):
        _intent(
            order_type=OrderType.TRAILING_STOP,
            trail_percent=Decimal("0"),
        )


def test_trailing_stop_valid_percent() -> None:
    o = _intent(
        order_type=OrderType.TRAILING_STOP,
        trail_percent=Decimal("1.5"),
    )
    assert o.trail_percent == Decimal("1.5")


def test_bracket_rejects_limit_maker_with_sl_tp() -> None:
    with pytest.raises(ValidationError, match="LIMIT_MAKER"):
        _intent(
            order_type=OrderType.LIMIT_MAKER,
            limit_price=Decimal("100"),
            sl_price=Decimal("90"),
            tp_price=Decimal("110"),
        )


def test_bracket_limit_with_sl_tp_ok() -> None:
    o = _intent(
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
        sl_price=Decimal("90"),
        tp_price=Decimal("110"),
    )
    assert o.sl_price == Decimal("90")
    assert o.tp_price == Decimal("110")
