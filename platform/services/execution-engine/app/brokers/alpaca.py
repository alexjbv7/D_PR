"""
AlpacaAdapter — Alpaca Markets broker integration.
====================================================

Supports both equities (US stocks + ETFs) and crypto via Alpaca's unified
Trading API.  Paper vs live is selected via :class:`AlpacaConfig`.

Design
------
* The official ``alpaca-py`` SDK is synchronous; every network call is offloaded
  to a thread pool via :func:`asyncio.to_thread` so the asyncio loop stays
  responsive.
* Symbol translation lives in :mod:`_symbol_mapping`.  The adapter accepts
  canonical symbols (``BTCUSDT``, ``AAPL``) and converts to Alpaca form
  (``BTC/USD``, ``AAPL``) before dispatch.
* ``alpaca-py`` is an optional dependency.  Importing this module never fails;
  instantiating :class:`AlpacaAdapter` raises :class:`BrokerError` when the
  SDK is missing.
* For SL/TP, the current implementation places the primary order only and
  carries SL/TP as metadata.  Bracket orders will be added in PASO E once the
  risk-gate's order-lifecycle state machine is in place.

References
----------
* Alpaca docs:        https://docs.alpaca.markets/
* Alpaca paper URL:   https://paper-api.alpaca.markets
* Alpaca live URL:    https://api.alpaca.markets
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from quant_shared.schemas.orders import (
    Fill,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

from .base import AccountInfo, BrokerAdapter, BrokerError
from . import _symbol_mapping as sym_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    from alpaca.trading.client import TradingClient                       # type: ignore
    from alpaca.trading.enums import (                                    # type: ignore
        OrderClass as _AC_OrderClass,
        OrderSide as _AC_Side,
        OrderStatus as _AC_Status,
        TimeInForce as _AC_TIF,
    )
    from alpaca.trading.requests import (                                 # type: ignore
        LimitOrderRequest as _AC_Limit,
        MarketOrderRequest as _AC_Market,
        StopLimitOrderRequest as _AC_StopLimit,
    )
    _HAS_ALPACA = True
except ImportError:  # pragma: no cover  (covered in tests via patching)
    _HAS_ALPACA = False
    TradingClient   = None       # type: ignore[assignment]
    _AC_Side        = None       # type: ignore[assignment]
    _AC_Status      = None       # type: ignore[assignment]
    _AC_TIF         = None       # type: ignore[assignment]
    _AC_OrderClass  = None       # type: ignore[assignment]
    _AC_Market      = None       # type: ignore[assignment]
    _AC_Limit       = None       # type: ignore[assignment]
    _AC_StopLimit   = None       # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class AlpacaConfig:
    """
    Connection configuration for :class:`AlpacaAdapter`.

    Reads from env vars by default:
      * ``ALPACA_API_KEY``
      * ``ALPACA_API_SECRET``
      * ``ALPACA_PAPER``  (``"true"`` / ``"false"``, default ``"true"``)
    """

    def __init__(
        self,
        api_key:    Optional[str]  = None,
        api_secret: Optional[str]  = None,
        paper:      Optional[bool] = None,
    ):
        self.api_key    = api_key    or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET", "")
        if paper is None:
            paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self.paper = paper

    def __repr__(self) -> str:
        env = "paper" if self.paper else "LIVE"
        masked = (self.api_key[:4] + "***") if self.api_key else "<missing>"
        return f"AlpacaConfig(env={env}, key={masked})"


# ---------------------------------------------------------------------------
# Enum translation
# ---------------------------------------------------------------------------

# Internal → Alpaca
_SIDE_TO_ALPACA: dict[OrderSide, Any] = {}
_TIF_TO_ALPACA:  dict[TimeInForce, Any] = {}

# Alpaca status name → internal
_ALPACA_STATUS_NAMES: dict[str, OrderStatus] = {
    "new":              OrderStatus.SUBMITTED,
    "accepted":         OrderStatus.SUBMITTED,
    "pending_new":      OrderStatus.PENDING,
    "pending_replace":  OrderStatus.SUBMITTED,
    "pending_cancel":   OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled":           OrderStatus.FILLED,
    "done_for_day":     OrderStatus.EXPIRED,
    "canceled":         OrderStatus.CANCELLED,
    "expired":          OrderStatus.EXPIRED,
    "replaced":         OrderStatus.CANCELLED,
    "rejected":         OrderStatus.REJECTED,
    "suspended":        OrderStatus.REJECTED,
    "stopped":          OrderStatus.REJECTED,
    "calculated":       OrderStatus.SUBMITTED,
    "held":             OrderStatus.PENDING,
}


def _init_enum_maps() -> None:
    """Populate enum lookup tables once the SDK is known to be present."""
    if not _HAS_ALPACA or _SIDE_TO_ALPACA:
        return
    _SIDE_TO_ALPACA[OrderSide.BUY]  = _AC_Side.BUY
    _SIDE_TO_ALPACA[OrderSide.SELL] = _AC_Side.SELL
    _TIF_TO_ALPACA[TimeInForce.GTC] = _AC_TIF.GTC
    _TIF_TO_ALPACA[TimeInForce.IOC] = _AC_TIF.IOC
    _TIF_TO_ALPACA[TimeInForce.FOK] = _AC_TIF.FOK
    _TIF_TO_ALPACA[TimeInForce.DAY] = _AC_TIF.DAY


def _status_to_internal(raw_status: Any) -> OrderStatus:
    """Translate an Alpaca order status (enum or str) to our enum."""
    name = getattr(raw_status, "value", raw_status)
    name = str(name).lower()
    return _ALPACA_STATUS_NAMES.get(name, OrderStatus.SUBMITTED)


# ---------------------------------------------------------------------------
# AlpacaAdapter
# ---------------------------------------------------------------------------

class AlpacaAdapter(BrokerAdapter):
    """
    Alpaca Markets broker adapter.

    Parameters
    ----------
    config : AlpacaConfig, optional
        Connection config.  If ``None`` an instance is built from env vars.
    client_factory : callable, optional
        Hook for dependency injection (used by tests).  Must return an object
        with the ``TradingClient`` interface.

    Raises
    ------
    BrokerError
        On :meth:`connect` if ``alpaca-py`` is not installed or credentials
        are missing.

    Notes
    -----
    Thread safety: a single instance is bound to one asyncio loop.  Network
    calls go through :func:`asyncio.to_thread` to keep the loop responsive.
    """

    venue = "alpaca"

    def __init__(
        self,
        config: Optional[AlpacaConfig] = None,
        client_factory: Optional[Any] = None,
    ):
        self.config = config or AlpacaConfig()
        self._client_factory = client_factory
        self._client: Any = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        if self._client is not None:
            return                          # idempotent

        if self._client_factory is not None:
            self._client = self._client_factory()
            _init_enum_maps()
            return

        if not _HAS_ALPACA:
            raise BrokerError(
                "alpaca-py is not installed.  Run: pip install alpaca-py"
            )
        if not self.config.api_key or not self.config.api_secret:
            raise BrokerError(
                "Alpaca credentials missing.  Set ALPACA_API_KEY / "
                "ALPACA_API_SECRET or pass AlpacaConfig."
            )

        _init_enum_maps()
        self._client = TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
            paper=self.config.paper,
        )
        logger.info(
            "alpaca.connected env=%s paper=%s",
            "paper" if self.config.paper else "live",
            self.config.paper,
        )

    async def close(self) -> None:
        self._client = None                 # alpaca-py has no explicit close

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _require_client(self) -> Any:
        if self._client is None:
            raise BrokerError("AlpacaAdapter.connect() must be called first.")
        return self._client

    def _build_request(self, intent: OrderIntent) -> Any:
        """Translate :class:`OrderIntent` → Alpaca request object."""
        alpaca_sym  = sym_map.to_alpaca(intent.symbol)
        side        = _SIDE_TO_ALPACA[intent.side]
        # Crypto only supports GTC / IOC on Alpaca; fall back to GTC if needed.
        is_crypto   = sym_map.is_crypto(intent.symbol)
        tif_choice  = intent.tif
        if is_crypto and tif_choice not in (TimeInForce.GTC, TimeInForce.IOC):
            tif_choice = TimeInForce.GTC
        tif = _TIF_TO_ALPACA[tif_choice]

        qty = float(intent.qty)             # alpaca-py accepts float or str

        if intent.order_type == OrderType.MARKET:
            return _AC_Market(
                symbol=alpaca_sym, qty=qty, side=side, time_in_force=tif,
            )
        if intent.order_type in (OrderType.LIMIT, OrderType.LIMIT_MAKER):
            if intent.limit_price is None:
                raise BrokerError(
                    f"limit_price required for order_type={intent.order_type}"
                )
            return _AC_Limit(
                symbol=alpaca_sym,
                qty=qty,
                side=side,
                time_in_force=tif,
                limit_price=float(intent.limit_price),
            )
        if intent.order_type == OrderType.STOP_LIMIT:
            if intent.limit_price is None or intent.sl_price is None:
                raise BrokerError(
                    "STOP_LIMIT requires both limit_price and sl_price"
                )
            return _AC_StopLimit(
                symbol=alpaca_sym,
                qty=qty,
                side=side,
                time_in_force=tif,
                limit_price=float(intent.limit_price),
                stop_price=float(intent.sl_price),
            )
        raise BrokerError(f"Unsupported order_type for Alpaca: {intent.order_type}")

    def _alpaca_order_to_result(self, raw: Any, intent_id: str = "") -> OrderResult:
        """Translate Alpaca order response → :class:`OrderResult`."""
        side = OrderSide(str(getattr(raw.side, "value", raw.side)).lower())
        status = _status_to_internal(raw.status)

        filled_qty = Decimal(str(getattr(raw, "filled_qty", 0) or 0))
        qty        = Decimal(str(raw.qty))
        avg_price  = (
            Decimal(str(raw.filled_avg_price))
            if getattr(raw, "filled_avg_price", None) is not None
            else None
        )

        ts_submitted = _to_utc(getattr(raw, "submitted_at", None)) or datetime.now(
            tz=timezone.utc
        )
        ts_updated   = _to_utc(getattr(raw, "updated_at", None)) or ts_submitted

        return OrderResult(
            intent_id   = intent_id,
            broker_id   = str(raw.id),
            symbol      = sym_map.from_alpaca(str(raw.symbol)),
            side        = side,
            status      = status,
            qty         = qty,
            filled_qty  = filled_qty,
            avg_price   = avg_price,
            ts_submitted= ts_submitted,
            ts_updated  = ts_updated,
            venue       = self.venue,
        )

    def _alpaca_position_to_internal(self, raw: Any) -> Position:
        """Translate Alpaca position → :class:`Position`."""
        qty_raw = Decimal(str(raw.qty))
        side    = OrderSide.BUY if qty_raw >= 0 else OrderSide.SELL
        return Position(
            symbol         = sym_map.from_alpaca(str(raw.symbol)),
            side           = side,
            qty            = abs(qty_raw),
            avg_entry      = Decimal(str(raw.avg_entry_price)),
            current_price  = _opt_decimal(getattr(raw, "current_price", None)),
            unrealized_pnl = _opt_decimal(getattr(raw, "unrealized_pl", None)),
            margin_used    = _opt_decimal(getattr(raw, "initial_margin", None)),
            venue          = self.venue,
        )

    # -----------------------------------------------------------------------
    # BrokerAdapter implementation
    # -----------------------------------------------------------------------

    async def submit(self, intent: OrderIntent) -> OrderResult:
        client = self._require_client()
        request = self._build_request(intent)
        try:
            raw = await asyncio.to_thread(client.submit_order, order_data=request)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca submit failed: {exc}") from exc
        return self._alpaca_order_to_result(raw, intent_id=intent.intent_id)

    async def cancel(self, broker_id: str) -> bool:
        client = self._require_client()
        try:
            await asyncio.to_thread(client.cancel_order_by_id, broker_id)
            return True
        except Exception as exc:                                     # noqa: BLE001
            msg = str(exc).lower()
            if any(kw in msg for kw in ("not found", "already", "422", "404")):
                return False
            raise BrokerError(f"Alpaca cancel failed: {exc}") from exc

    async def get_order(self, broker_id: str) -> OrderResult:
        client = self._require_client()
        try:
            raw = await asyncio.to_thread(client.get_order_by_id, broker_id)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca get_order failed: {exc}") from exc
        return self._alpaca_order_to_result(raw)

    async def get_positions(self) -> list[Position]:
        client = self._require_client()
        try:
            raws = await asyncio.to_thread(client.get_all_positions)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca get_positions failed: {exc}") from exc
        return [self._alpaca_position_to_internal(r) for r in raws]

    async def get_account(self) -> AccountInfo:
        client = self._require_client()
        try:
            acc = await asyncio.to_thread(client.get_account)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca get_account failed: {exc}") from exc

        return AccountInfo(
            account_id  = str(getattr(acc, "id", "")),
            venue       = self.venue,
            equity      = Decimal(str(acc.equity)),
            cash        = Decimal(str(acc.cash)),
            margin_used = _opt_decimal(getattr(acc, "initial_margin", None))
                          or Decimal("0"),
            pnl_day     = _opt_decimal(
                Decimal(str(acc.equity)) - Decimal(str(acc.last_equity))
            ) if getattr(acc, "last_equity", None) is not None else Decimal("0"),
            currency    = getattr(acc, "currency", "USD"),
            is_paper    = self.config.paper,
        )

    async def reconcile(self, internal_positions: list[Position]) -> list[str]:
        broker_positions = await self.get_positions()

        broker_by_sym   = {p.symbol: p for p in broker_positions}
        internal_by_sym = {p.symbol: p for p in internal_positions}

        discrepancies: list[str] = []

        for sym in broker_by_sym.keys() - internal_by_sym.keys():
            discrepancies.append(
                f"PHANTOM: broker has {sym} qty={broker_by_sym[sym].qty}, "
                f"internal has none"
            )
        for sym in internal_by_sym.keys() - broker_by_sym.keys():
            discrepancies.append(
                f"MISSING: internal has {sym} qty={internal_by_sym[sym].qty}, "
                f"broker has none"
            )
        for sym in broker_by_sym.keys() & internal_by_sym.keys():
            b, i = broker_by_sym[sym], internal_by_sym[sym]
            if b.side != i.side:
                discrepancies.append(
                    f"SIDE_MISMATCH {sym}: broker={b.side.value} "
                    f"internal={i.side.value}"
                )
            elif b.qty != i.qty:
                discrepancies.append(
                    f"QTY_MISMATCH {sym}: broker={b.qty} internal={i.qty}"
                )

        return discrepancies


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_utc(value: Any) -> Optional[datetime]:
    """Coerce a datetime-ish value (datetime / str / None) → UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:                                                # noqa: BLE001
        return None


def _opt_decimal(value: Any) -> Optional[Decimal]:
    """Coerce a numeric / None value → ``Decimal`` (or ``None``)."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:                                                # noqa: BLE001
        return None
