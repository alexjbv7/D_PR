"""
BrokerAdapter — Abstract base class for all broker integrations.
================================================================

Design decisions
----------------
* All methods are async-native (asyncio).  Brokers have network I/O; there is
  no reason for a synchronous API.
* ``submit`` returns an ``OrderResult`` immediately with status SUBMITTED
  (or FILLED for market orders that ack synchronously).
* ``cancel`` returns True / False — no exception on "already done" states.
* ``reconcile`` is fire-and-forget diagnostics: the caller decides how to act
  on discrepancies.
* Context-manager protocol (``async with``) handles connect/close lifecycle.
* All monetary values are ``Decimal``; IDs are ``str`` (UUID v7 internally,
  broker-native externally).

Concrete implementations (PASO C/D):
  app/brokers/alpaca.py  — Alpaca Markets (equities + crypto)
  app/brokers/ccxt.py    — ccxt multi-exchange (Binance, Bybit, Kraken)

References
----------
* CLAUDE.md §11 (Motor de ejecución — pseudocode)
* ADR-010: UTC + Decimal + UUID v7 on the whole platform
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from quant_shared.schemas.orders import OrderIntent, OrderResult, Position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class BrokerError(Exception):
    """Unrecoverable error from the broker (rejected order, auth failure, etc.)."""


class BrokerTimeoutError(BrokerError):
    """Order ack was not received within the configured timeout."""


# ---------------------------------------------------------------------------
# AccountInfo value object
# ---------------------------------------------------------------------------

@dataclass
class AccountInfo:
    """
    Snapshot of broker account state.

    All monetary values use ``Decimal``.  ``is_paper`` distinguishes paper
    trading (safe to test with) from live accounts.

    Parameters
    ----------
    account_id : str
        Broker's native account identifier.
    venue : str
        Source adapter name ("alpaca", "binance", …).
    equity : Decimal
        Total account value in ``currency``.
    cash : Decimal
        Available buying power / free cash.
    margin_used : Decimal
        Currently locked margin (0 for cash accounts).
    pnl_day : Decimal
        Intraday unrealized + realized P&L.
    currency : str
        Base currency (default "USD").
    is_paper : bool
        True when connected to a paper / sandbox environment.
    raw : dict
        Verbatim broker response for debugging.
    """

    account_id:  str
    venue:       str
    equity:      Decimal
    cash:        Decimal
    margin_used: Decimal = field(default_factory=lambda: Decimal("0"))
    pnl_day:     Decimal = field(default_factory=lambda: Decimal("0"))
    currency:    str     = "USD"
    is_paper:    bool    = True
    raw:         dict    = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BrokerAdapter ABC
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """
    Abstract interface for broker / exchange integrations.

    All subclasses MUST:
      * Implement every ``@abstractmethod``.
      * Use ``Decimal`` for all monetary values.
      * Raise ``BrokerError`` (or subclass) on unrecoverable failures.
      * Support ``async with`` via the inherited ``__aenter__`` / ``__aexit__``.

    Thread safety:  NOT guaranteed.  Each adapter instance should be used
    from a single asyncio event loop.

    Usage
    -----
    ::

        async with AlpacaAdapter(config) as broker:
            result = await broker.submit(intent)
            positions = await broker.get_positions()

    Attributes
    ----------
    venue : str
        Human-readable venue tag.  Overridden by each subclass.
        Used in logs, ``OrderResult.venue``, and routing decisions.
    """

    venue: str = "unknown"

    # -----------------------------------------------------------------------
    # Context manager — lifecycle
    # -----------------------------------------------------------------------

    async def __aenter__(self) -> "BrokerAdapter":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialize the connection / session to the broker.

        Called automatically by ``__aenter__``.  Must be idempotent (safe to
        call again if already connected).

        Raises
        ------
        BrokerError
            If authentication or network setup fails.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Gracefully close the connection / session.

        Called automatically by ``__aexit__``.  Must be safe to call even if
        ``connect`` was never called.
        """

    # -----------------------------------------------------------------------
    # Order management
    # -----------------------------------------------------------------------

    @abstractmethod
    async def submit(self, intent: OrderIntent) -> OrderResult:
        """
        Submit an order to the broker.

        The returned ``OrderResult`` may have status ``SUBMITTED`` (async
        confirmation pending) or ``FILLED`` (for IOC/market orders that fill
        synchronously).

        Parameters
        ----------
        intent : OrderIntent
            Risk-approved order intent.  ``intent.venue`` may be used as a
            routing hint; adapters may ignore it if they are venue-specific.

        Returns
        -------
        OrderResult
            Initial broker response.  The caller is responsible for polling
            ``get_order`` for asynchronous confirmation.

        Raises
        ------
        BrokerError
            On rejection, auth failure, or unrecoverable network error.
        BrokerTimeoutError
            If the ack is not received within the adapter's timeout.
        """

    @abstractmethod
    async def cancel(self, broker_id: str) -> bool:
        """
        Cancel an open order.

        Parameters
        ----------
        broker_id : str
            Broker-native order ID (from ``OrderResult.broker_id``).

        Returns
        -------
        bool
            True  — cancellation was accepted.
            False — order was already in a terminal state (FILLED, EXPIRED, …).

        Raises
        ------
        BrokerError
            On auth failure or unexpected broker response.
        """

    @abstractmethod
    async def get_order(self, broker_id: str) -> OrderResult:
        """
        Fetch the current state of an order from the broker.

        Used by the reconciler / status-polling loop to detect fills and
        unexpected terminal states.

        Parameters
        ----------
        broker_id : str
            Broker-native order ID.

        Returns
        -------
        OrderResult
            Latest broker view of the order.

        Raises
        ------
        BrokerError
            If the broker does not recognise the order ID.
        """

    # -----------------------------------------------------------------------
    # Position & account queries
    # -----------------------------------------------------------------------

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Return all currently open positions.

        Called periodically by the reconciler (§11.4 of CLAUDE.md).

        Returns
        -------
        list[Position]
            One ``Position`` per open symbol.  Empty list when flat.
        """

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """
        Return a snapshot of the account (equity, cash, buying power).

        Used by the risk-gate to check margin / buying power before
        submitting a new order.

        Returns
        -------
        AccountInfo
        """

    # -----------------------------------------------------------------------
    # Reconciliation
    # -----------------------------------------------------------------------

    @abstractmethod
    async def reconcile(self, internal_positions: list[Position]) -> list[str]:
        """
        Compare broker positions with the execution-engine's internal state.

        Detects: phantom positions, missing positions, quantity mismatches,
        side mismatches.  Does NOT fix anything — returns a list of
        human-readable discrepancy strings.  The caller decides how to react.

        Parameters
        ----------
        internal_positions : list[Position]
            What the execution-engine believes is currently open.

        Returns
        -------
        list[str]
            Discrepancy messages.  Empty list ⟹ all clear.

        Notes
        -----
        A non-empty return value should trigger an alert and halt new order
        submission until resolved.  See ``risk_gate.py`` (PASO E).
        """

    # -----------------------------------------------------------------------
    # Optional helpers
    # -----------------------------------------------------------------------

    async def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """
        Best-effort current price for a symbol.

        Not all brokers expose a lightweight price endpoint.  Default returns
        ``None``; override when available.

        Parameters
        ----------
        symbol : str
            Ticker / trading pair.

        Returns
        -------
        Decimal or None
        """
        return None
