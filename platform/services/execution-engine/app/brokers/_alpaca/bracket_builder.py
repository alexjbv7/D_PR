"""
Bracket / OCO / Trailing builders for AlpacaAdapter.

Pure functions: input = (OrderIntent + resolved Alpaca symbol/side/tif),
output = Alpaca request object. No I/O, no state.

Decision tree (used by AlpacaAdapter._build_request):
  - TRAILING_STOP           → build_trailing_request
  - sl_price AND tp_price   → build_bracket_request
  - sl_price XOR tp_price   → build_oco_request
  - else                    → _build_simple_request in alpaca.py
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from quant_shared.schemas.orders import OrderIntent, OrderSide, OrderType

from .. import _symbol_mapping as sym_map
from .errors import (
    BracketIncompatibleOrderTypeError,
    BracketRejectedError,
    FractionalShareNotAllowedError,
    TrailingStopRejectedError,
)

# Bracket on crypto — Alpaca whitelist (TODO(@alex 2026-09): expand when Alpaca adds symbols)
BRACKET_CRYPTO_ALLOWED: frozenset[str] = frozenset({"BTC/USD", "ETH/USD"})


def should_use_trailing(intent: OrderIntent) -> bool:
    return bool(intent.order_type == OrderType.TRAILING_STOP)


def should_use_bracket(intent: OrderIntent) -> bool:
    return intent.sl_price is not None and intent.tp_price is not None


def should_use_oco(intent: OrderIntent) -> bool:
    return (intent.sl_price is None) != (intent.tp_price is None)


def _is_whole_share(qty: Decimal) -> bool:
    return qty == qty.to_integral_value()


def _validate_fractional_bracket(intent: OrderIntent) -> None:
    """Bracket orders do not support fractional share qty on equities."""
    if not _is_whole_share(intent.qty):
        raise FractionalShareNotAllowedError(
            f"Bracket orders require whole-share qty; got {intent.qty}"
        )


def _validate_bracket_pricing(intent: OrderIntent) -> None:
    """Raise if pricing is incoherent for the trade direction."""
    if intent.order_type == OrderType.LIMIT_MAKER:
        raise BracketIncompatibleOrderTypeError(
            "Bracket orders require LIMIT (not post-only) or MARKET; "
            "LIMIT_MAKER would cause Alpaca to reject."
        )

    if intent.sl_price is None or intent.tp_price is None:
        return

    if intent.limit_price is None:
        raise BracketRejectedError("Bracket requires limit_price as entry.")

    entry = intent.limit_price
    if intent.side == OrderSide.BUY:
        if not (intent.tp_price > entry > intent.sl_price):
            raise BracketRejectedError(
                f"LONG bracket invalid: tp={intent.tp_price} entry={entry} "
                f"sl={intent.sl_price}; expected tp > entry > sl"
            )
    else:
        if not (intent.tp_price < entry < intent.sl_price):
            raise BracketRejectedError(
                f"SHORT bracket invalid: tp={intent.tp_price} entry={entry} "
                f"sl={intent.sl_price}; expected tp < entry < sl"
            )


def _validate_crypto_compatibility(intent: OrderIntent, has_bracket: bool) -> None:
    if not sym_map.is_crypto(intent.symbol):
        return
    alpaca_sym = sym_map.to_alpaca(intent.symbol)
    if has_bracket and alpaca_sym not in BRACKET_CRYPTO_ALLOWED:
        raise BracketRejectedError(
            f"Bracket orders for crypto only supported on {set(BRACKET_CRYPTO_ALLOWED)}, "
            f"got {alpaca_sym}"
        )


def build_bracket_request(
    intent: OrderIntent,
    alpaca_sym: str,
    side: Any,
    tif: Any,
    ac_classes: dict[str, Any],
) -> Any:
    """LimitOrderRequest with OrderClass.BRACKET + take_profit + stop_loss."""
    _validate_bracket_pricing(intent)
    _validate_crypto_compatibility(intent, has_bracket=True)
    if not sym_map.is_crypto(intent.symbol):
        _validate_fractional_bracket(intent)

    LimitOrderRequest = ac_classes["LimitOrderRequest"]
    TakeProfitRequest = ac_classes["TakeProfitRequest"]
    StopLossRequest   = ac_classes["StopLossRequest"]
    OrderClass        = ac_classes["OrderClass"]

    assert intent.limit_price is not None
    assert intent.tp_price is not None
    assert intent.sl_price is not None

    return LimitOrderRequest(
        symbol=alpaca_sym,
        qty=float(intent.qty),
        side=side,
        time_in_force=tif,
        limit_price=float(intent.limit_price),
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=float(intent.tp_price)),
        stop_loss=StopLossRequest(stop_price=float(intent.sl_price)),
        extended_hours=intent.extended_hours,
        client_order_id=intent.intent_id,
    )


def build_oco_request(
    intent: OrderIntent,
    alpaca_sym: str,
    side: Any,
    tif: Any,
    ac_classes: dict[str, Any],
) -> Any:
    """OTO with stop_loss and/or take_profit leg (single exit leg)."""
    if intent.limit_price is None:
        raise BracketRejectedError("OTO requires limit_price as entry.")

    LimitOrderRequest = ac_classes["LimitOrderRequest"]
    TakeProfitRequest = ac_classes["TakeProfitRequest"]
    StopLossRequest   = ac_classes["StopLossRequest"]
    OrderClass        = ac_classes["OrderClass"]

    kwargs: dict[str, Any] = dict(
        symbol=alpaca_sym,
        qty=float(intent.qty),
        side=side,
        time_in_force=tif,
        limit_price=float(intent.limit_price),
        order_class=OrderClass.OTO,
        extended_hours=intent.extended_hours,
        client_order_id=intent.intent_id,
    )
    if intent.tp_price is not None:
        kwargs["take_profit"] = TakeProfitRequest(limit_price=float(intent.tp_price))
    if intent.sl_price is not None:
        kwargs["stop_loss"] = StopLossRequest(stop_price=float(intent.sl_price))
    return LimitOrderRequest(**kwargs)


def build_trailing_request(
    intent: OrderIntent,
    alpaca_sym: str,
    side: Any,
    tif: Any,
    ac_classes: dict[str, Any],
) -> Any:
    """TrailingStopOrderRequest — XOR trail_percent or trail_price."""
    TrailingStopOrderRequest = ac_classes["TrailingStopOrderRequest"]

    if intent.trail_percent is not None and intent.trail_price is not None:
        raise TrailingStopRejectedError(
            "trail_percent and trail_price are mutually exclusive."
        )
    if intent.trail_percent is None and intent.trail_price is None:
        raise TrailingStopRejectedError(
            "TRAILING_STOP requires trail_percent or trail_price."
        )

    kwargs: dict[str, Any] = dict(
        symbol=alpaca_sym,
        qty=float(intent.qty),
        side=side,
        time_in_force=tif,
        extended_hours=intent.extended_hours,
        client_order_id=intent.intent_id,
    )
    if intent.trail_percent is not None:
        kwargs["trail_percent"] = float(intent.trail_percent)
    else:
        assert intent.trail_price is not None
        kwargs["trail_price"] = float(intent.trail_price)
    return TrailingStopOrderRequest(**kwargs)
