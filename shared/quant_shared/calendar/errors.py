"""Calendar-related domain errors. Neutral al broker — usable cross-service."""
from __future__ import annotations

from datetime import datetime


class MarketClosedError(Exception):
    """Raised when an order is attempted on a closed market for the symbol's venue."""

    def __init__(
        self,
        symbol: str,
        venue: str,
        ts_utc: datetime,
        reason: str = "",
    ) -> None:
        self.symbol = symbol
        self.venue = venue
        self.ts_utc = ts_utc
        self.reason = reason
        super().__init__(
            f"Market closed for {symbol} @ {venue} at {ts_utc.isoformat()}: {reason}"
        )
