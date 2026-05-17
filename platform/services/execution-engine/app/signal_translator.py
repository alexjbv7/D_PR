"""
Translate strategy-orchestrator FinalSignalEvent → OrderIntent.

The strategy-orchestrator publishes signals on ``los_ojos.signals.trading``
with a shape like::

    {
      "event_id":          "...",
      "strategy":          "regime_adaptive",
      "symbol":            "BTCUSDT",
      "direction":         1,            // 1 = long, -1 = short, 0 = flat
      "p_win":             0.6,
      "position_size":     0.02,         // kelly fraction
      "target_risk_pct":   0.01,         // optional
      "rr_ratio":          2.0,          // optional
      "confidence":        0.5,
      "regime":            "ranging",
      "source_service":    "strategy-orchestrator",
      "ts":                "2026-05-17T..."
    }

This module converts that payload to :class:`OrderIntent`, sizing qty from
``kelly_fraction × equity / current_price`` when a current price is known.
If no price can be obtained, the translator returns ``None`` so the caller
can skip the signal cleanly.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger(__name__)


# Minimum tradable notional in account currency.  Below this we skip
# (most exchanges have notional minimums; 10 is a safe floor for spot crypto).
_MIN_NOTIONAL = Decimal("10")


def translate_signal(
    signal:        dict,
    equity:        Decimal,
    current_price: Optional[Decimal],
    default_venue: str = "",
) -> Optional[OrderIntent]:
    """
    Convert a FinalSignalEvent dict to an :class:`OrderIntent`.

    Parameters
    ----------
    signal : dict
        Raw FinalSignalEvent payload.
    equity : Decimal
        Current account equity, used to size qty from ``kelly_fraction``.
    current_price : Decimal, optional
        Last known mid / mark price for ``signal["symbol"]``.  Required to
        compute qty; if ``None`` the function returns ``None``.
    default_venue : str
        Venue tag to attach when the signal does not specify one.

    Returns
    -------
    OrderIntent or None
        ``None`` is returned for:
          * direction == 0 (flat)
          * current_price missing or non-positive
          * resulting notional < _MIN_NOTIONAL
    """
    direction = int(signal.get("direction", 0) or 0)
    if direction == 0:
        logger.debug("translator.flat_signal symbol=%s", signal.get("symbol"))
        return None

    if current_price is None or current_price <= 0:
        logger.warning(
            "translator.no_price symbol=%s — skipping signal",
            signal.get("symbol"),
        )
        return None

    kelly = Decimal(str(signal.get("position_size", 0) or 0))
    if kelly <= 0:
        logger.warning(
            "translator.zero_kelly symbol=%s — skipping signal",
            signal.get("symbol"),
        )
        return None

    # Notional = kelly × equity ; qty = notional / price
    notional = kelly * equity
    if notional < _MIN_NOTIONAL:
        logger.info(
            "translator.below_min_notional symbol=%s notional=%s",
            signal.get("symbol"), notional,
        )
        return None

    qty = notional / current_price

    side = OrderSide.BUY if direction > 0 else OrderSide.SELL

    return OrderIntent(
        signal_id=str(signal.get("event_id", "")),
        strategy=str(signal.get("strategy", "")),
        symbol=str(signal.get("symbol", "")),
        side=side,
        qty=qty,
        order_type=OrderType.LIMIT_MAKER,
        limit_price=current_price,
        tif=TimeInForce.GTC,
        venue=str(signal.get("venue", "") or default_venue),
        kelly_fraction=float(kelly),
        target_risk_pct=float(signal.get("target_risk_pct", 0) or 0),
        p_win=float(signal.get("p_win", 0) or 0),
    )
