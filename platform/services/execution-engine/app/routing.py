"""
Venue routing — choose which broker adapter handles a given OrderIntent.
========================================================================

Design
------
The router owns a registry of :class:`BrokerAdapter` instances (one per
venue) and dispatches each :class:`OrderIntent` to the appropriate adapter.

Routing rules, in order of precedence:
  1. ``intent.venue`` set  →  use that adapter (raise if unknown).
  2. Symbol is an equity (uppercase letters, no quote suffix, ≤ 5 chars)
     →  use the default equity venue (``"alpaca"``).
  3. Otherwise (crypto pair) →  use the default crypto venue
     (``"binance"`` by default).

The router does NOT own adapter lifecycle: the caller registers already-
connected adapters and calls ``close_all()`` on shutdown.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from quant_shared.schemas.orders import OrderIntent, OrderResult, Position

from .brokers.base import BrokerAdapter, BrokerError

logger = logging.getLogger(__name__)


# Heuristic for identifying equities: 1-5 uppercase letters, no digits,
# and no recognised crypto-quote suffix.
_EQUITY_RE = re.compile(r"^[A-Z]{1,5}$")
_CRYPTO_QUOTES = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP")


def is_equity(symbol: str) -> bool:
    """
    Heuristic: True if ``symbol`` looks like a US equity / ETF ticker.

    >>> is_equity("AAPL")
    True
    >>> is_equity("BTCUSDT")
    False
    >>> is_equity("BTC/USDT")
    False
    """
    sym = symbol.upper()
    if "/" in sym or ":" in sym:
        return False
    for q in _CRYPTO_QUOTES:
        if sym.endswith(q) and len(sym) > len(q):
            return False
    return bool(_EQUITY_RE.match(sym))


class Router:
    """
    Maintains a registry of :class:`BrokerAdapter` instances and routes
    each :class:`OrderIntent` to the correct one.

    Parameters
    ----------
    default_equity : str
        Venue tag to use for equity symbols when ``intent.venue`` is empty.
    default_crypto : str
        Venue tag to use for crypto symbols when ``intent.venue`` is empty.

    Examples
    --------
    >>> router = Router(default_equity="alpaca", default_crypto="binance")
    >>> router.register(alpaca_adapter)
    >>> router.register(binance_adapter)
    >>> result = await router.submit(intent)
    """

    def __init__(
        self,
        default_equity: str = "alpaca",
        default_crypto: str = "binance",
    ):
        self._adapters: dict[str, BrokerAdapter] = {}
        self.default_equity = default_equity
        self.default_crypto = default_crypto

    # -----------------------------------------------------------------------
    # Registry
    # -----------------------------------------------------------------------

    def register(self, adapter: BrokerAdapter) -> None:
        """Add an adapter to the registry (key = ``adapter.venue``)."""
        if not adapter.venue or adapter.venue == "unknown":
            raise BrokerError(
                f"Adapter has no venue tag: {type(adapter).__name__}"
            )
        self._adapters[adapter.venue] = adapter
        logger.info("router.registered venue=%s", adapter.venue)

    def get(self, venue: str) -> BrokerAdapter:
        """Lookup an adapter by venue tag.  Raises if not registered."""
        if venue not in self._adapters:
            raise BrokerError(
                f"No adapter registered for venue={venue!r}.  "
                f"Registered: {sorted(self._adapters)}"
            )
        return self._adapters[venue]

    def venues(self) -> list[str]:
        return sorted(self._adapters)

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    def route(self, intent: OrderIntent) -> BrokerAdapter:
        """
        Choose the adapter that should handle ``intent``.

        Raises
        ------
        BrokerError
            If the chosen venue is not registered.
        """
        if intent.venue:
            return self.get(intent.venue)
        venue = self.default_equity if is_equity(intent.symbol) else self.default_crypto
        return self.get(venue)

    # -----------------------------------------------------------------------
    # Convenience delegates
    # -----------------------------------------------------------------------

    async def submit(self, intent: OrderIntent) -> OrderResult:
        adapter = self.route(intent)
        return await adapter.submit(intent)

    async def cancel(self, venue: str, broker_id: str) -> bool:
        return await self.get(venue).cancel(broker_id)

    async def get_order(self, venue: str, broker_id: str) -> OrderResult:
        return await self.get(venue).get_order(broker_id)

    async def get_positions_all(self) -> list[Position]:
        """Aggregate positions from every registered adapter."""
        result: list[Position] = []
        for venue, adapter in self._adapters.items():
            try:
                result.extend(await adapter.get_positions())
            except BrokerError as exc:
                logger.error("router.get_positions venue=%s error=%s", venue, exc)
        return result

    async def close_all(self) -> None:
        """Close every registered adapter (best-effort)."""
        for venue, adapter in list(self._adapters.items()):
            try:
                await adapter.close()
            except Exception as exc:                                # noqa: BLE001
                logger.warning("router.close venue=%s error=%s", venue, exc)
