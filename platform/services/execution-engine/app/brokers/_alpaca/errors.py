"""Typed errors for Alpaca bracket / trailing — re-exports from alpaca_errors."""
from __future__ import annotations

from ..alpaca_errors import (
    AlpacaError,
    AlpacaFractionalNotSupportedError,
    AssetNotTradableError,
    BracketIncompatibleOrderTypeError,
    BracketRejectedError,
    FractionalShareNotAllowedError,
    InsufficientBuyingPowerError,
    MarketClosedError,
    TrailingStopRejectedError,
)

__all__ = [
    "AlpacaError",
    "BracketRejectedError",
    "TrailingStopRejectedError",
    "BracketIncompatibleOrderTypeError",
    "FractionalShareNotAllowedError",
    "InsufficientBuyingPowerError",
    "MarketClosedError",
    "AssetNotTradableError",
    "AlpacaFractionalNotSupportedError",
]
