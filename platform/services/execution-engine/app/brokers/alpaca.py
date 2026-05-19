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
* Bracket / OTO / trailing-stop requests are built in
  :mod:`_alpaca.bracket_builder` and submitted atomically via Alpaca
  ``order_class`` (server-side cross-leg cancellation).

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
import time as _time
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
from ._alpaca import bracket_builder
from ._alpaca.rate_limiter import AlpacaRateLimiter
from ._alpaca.retry import retry_with_jitter
from ._alpaca.market_data import AlpacaMarketData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prometheus metrics (optional — graceful no-op if prometheus_client absent)
# ---------------------------------------------------------------------------

_HAS_PROMETHEUS = False
SUBMIT_ATTEMPTS: Any = None
RATE_LIMITED: Any = None
SUBMIT_LATENCY: Any = None

try:
    from prometheus_client import Counter, Histogram

    SUBMIT_ATTEMPTS = Counter(
        "alpaca_submit_attempts_total",
        "Alpaca submit attempts by result category",
        ["result"],  # success | 429 | 5xx | 4xx
    )
    RATE_LIMITED = Counter(
        "alpaca_429_total",
        "Number of 429 Too Many Requests responses from Alpaca",
    )
    SUBMIT_LATENCY = Histogram(
        "alpaca_submit_latency_seconds",
        "Latency of Alpaca submit_order calls (seconds)",
        buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    pass


def _record_submit(result_label: str) -> None:
    """Bump the appropriate Prometheus counter."""
    if _HAS_PROMETHEUS:
        SUBMIT_ATTEMPTS.labels(result=result_label).inc()
        if result_label == "429":
            RATE_LIMITED.inc()


# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

_HAS_ALPACA = False
TradingClient: Any = None
_AC_Side: Any = None
_AC_Status: Any = None
_AC_TIF: Any = None
_AC_OrderClass: Any = None
_AC_Market: Any = None
_AC_Limit: Any = None
_AC_StopLimit: Any = None
_AC_StopLoss: Any = None
_AC_TakeProfit: Any = None
_AC_TrailingStop: Any = None

try:
    from alpaca.trading.client import TradingClient as _TradingClient
    from alpaca.trading.enums import (
        OrderClass as _ImportedOrderClass,
        OrderSide as _ImportedSide,
        OrderStatus as _ImportedStatus,
        TimeInForce as _ImportedTIF,
    )
    from alpaca.trading.requests import (
        LimitOrderRequest as _ImportedLimit,
        MarketOrderRequest as _ImportedMarket,
        StopLimitOrderRequest as _ImportedStopLimit,
        StopLossRequest as _ImportedStopLoss,
        TakeProfitRequest as _ImportedTakeProfit,
        TrailingStopOrderRequest as _ImportedTrailingStop,
    )

    TradingClient = _TradingClient
    _AC_Side = _ImportedSide
    _AC_Status = _ImportedStatus
    _AC_TIF = _ImportedTIF
    _AC_OrderClass = _ImportedOrderClass
    _AC_Market = _ImportedMarket
    _AC_Limit = _ImportedLimit
    _AC_StopLimit = _ImportedStopLimit
    _AC_StopLoss = _ImportedStopLoss
    _AC_TakeProfit = _ImportedTakeProfit
    _AC_TrailingStop = _ImportedTrailingStop
    _HAS_ALPACA = True
except ImportError:  # pragma: no cover  (covered in tests via patching)
    pass


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
        rate_limiter: Optional[AlpacaRateLimiter] = None,
        market_data: Optional[AlpacaMarketData] = None,
    ):
        self.config = config or AlpacaConfig()
        self._client_factory = client_factory
        self._client: Any = None
        self._rate_limiter = rate_limiter or AlpacaRateLimiter()
        self._market_data = market_data

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

    def _ac_classes(self) -> dict[str, Any]:
        """Bundle of alpaca-py request classes for bracket_builder."""
        return {
            "LimitOrderRequest":        _AC_Limit,
            "MarketOrderRequest":       _AC_Market,
            "StopLimitOrderRequest":    _AC_StopLimit,
            "TrailingStopOrderRequest": _AC_TrailingStop,
            "TakeProfitRequest":        _AC_TakeProfit,
            "StopLossRequest":          _AC_StopLoss,
            "OrderClass":               _AC_OrderClass,
        }

    def _build_request(self, intent: OrderIntent) -> Any:
        """Translate :class:`OrderIntent` → Alpaca request object.

        Decision tree: trailing → bracket → OTO → simple (market/limit/stop).
        Every request includes ``client_order_id = intent.intent_id``.
        """
        alpaca_sym = sym_map.to_alpaca(intent.symbol)
        side       = _SIDE_TO_ALPACA[intent.side]
        is_crypto  = sym_map.is_crypto(intent.symbol)
        tif_choice = intent.tif
        if is_crypto and tif_choice not in (TimeInForce.GTC, TimeInForce.IOC):
            tif_choice = TimeInForce.GTC
        tif = _TIF_TO_ALPACA[tif_choice]
        ac  = self._ac_classes()

        if bracket_builder.should_use_trailing(intent):
            return bracket_builder.build_trailing_request(
                intent, alpaca_sym, side, tif, ac,
            )
        if bracket_builder.should_use_bracket(intent):
            return bracket_builder.build_bracket_request(
                intent, alpaca_sym, side, tif, ac,
            )
        if bracket_builder.should_use_oco(intent):
            return bracket_builder.build_oco_request(
                intent, alpaca_sym, side, tif, ac,
            )
        return self._build_simple_request(intent, alpaca_sym, side, tif)

    def _build_simple_request(
        self,
        intent: OrderIntent,
        alpaca_sym: str,
        side: Any,
        tif: Any,
    ) -> Any:
        """MARKET / LIMIT / LIMIT_MAKER / STOP_LIMIT — pre-S5 behaviour."""
        qty  = float(intent.qty)
        coid = intent.intent_id

        if intent.order_type == OrderType.MARKET:
            return _AC_Market(
                symbol=alpaca_sym, qty=qty, side=side, time_in_force=tif,
                client_order_id=coid,
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
                client_order_id=coid,
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
                client_order_id=coid,
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
        await self._rate_limiter.acquire("trading")
        t0 = _time.monotonic()
        try:
            raw = await self._submit_with_retry(client, request)
            _record_submit("success")
            return self._alpaca_order_to_result(raw, intent_id=intent.intent_id)
        except Exception as exc:                                     # noqa: BLE001
            self._classify_and_record(exc)
            raise BrokerError(f"Alpaca submit failed: {exc}") from exc
        finally:
            if _HAS_PROMETHEUS:
                SUBMIT_LATENCY.observe(_time.monotonic() - t0)

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _submit_with_retry(self, client: Any, request: Any) -> Any:
        return await asyncio.to_thread(client.submit_order, order_data=request)

    async def cancel(self, broker_id: str) -> bool:
        client = self._require_client()
        await self._rate_limiter.acquire("trading")
        try:
            await self._cancel_with_retry(client, broker_id)
            return True
        except Exception as exc:                                     # noqa: BLE001
            msg = str(exc).lower()
            if any(kw in msg for kw in ("not found", "already", "422", "404")):
                return False
            raise BrokerError(f"Alpaca cancel failed: {exc}") from exc

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _cancel_with_retry(self, client: Any, broker_id: str) -> None:
        await asyncio.to_thread(client.cancel_order_by_id, broker_id)

    async def get_order(self, broker_id: str) -> OrderResult:
        client = self._require_client()
        await self._rate_limiter.acquire("trading")
        try:
            raw = await self._get_order_with_retry(client, broker_id)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca get_order failed: {exc}") from exc
        return self._alpaca_order_to_result(raw)

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _get_order_with_retry(self, client: Any, broker_id: str) -> Any:
        return await asyncio.to_thread(client.get_order_by_id, broker_id)

    async def get_positions(self) -> list[Position]:
        client = self._require_client()
        await self._rate_limiter.acquire("trading")
        try:
            raws = await self._get_positions_with_retry(client)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca get_positions failed: {exc}") from exc
        return [self._alpaca_position_to_internal(r) for r in raws]

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _get_positions_with_retry(self, client: Any) -> Any:
        return await asyncio.to_thread(client.get_all_positions)

    async def get_account(self) -> AccountInfo:
        client = self._require_client()
        await self._rate_limiter.acquire("trading")
        try:
            acc = await self._get_account_with_retry(client)
        except Exception as exc:                                     # noqa: BLE001
            raise BrokerError(f"Alpaca get_account failed: {exc}") from exc

        last_eq = getattr(acc, "last_equity", None)
        if last_eq is not None:
            pnl_day = Decimal(str(acc.equity)) - Decimal(str(last_eq))
        else:
            pnl_day = Decimal("0")

        return AccountInfo(
            account_id  = str(getattr(acc, "id", "")),
            venue       = self.venue,
            equity      = Decimal(str(acc.equity)),
            cash        = Decimal(str(acc.cash)),
            margin_used = _opt_decimal(getattr(acc, "initial_margin", None))
                          or Decimal("0"),
            pnl_day     = pnl_day,
            currency    = getattr(acc, "currency", "USD"),
            is_paper    = self.config.paper,
        )

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _get_account_with_retry(self, client: Any) -> Any:
        return await asyncio.to_thread(client.get_account)

    # -----------------------------------------------------------------------
    # Market data — price awareness
    # -----------------------------------------------------------------------

    async def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """
        Best-effort current price using Alpaca market data.

        Falls back to ``None`` if no market data client is configured
        (preserves the BrokerAdapter default).
        """
        if self._market_data is None:
            return None
        alpaca_sym = sym_map.to_alpaca(symbol)
        price = await self._market_data.get_last_price(alpaca_sym)
        if price is None:
            return None
        return Decimal(str(price))

    @property
    def market_data(self) -> Optional[AlpacaMarketData]:
        """Expose the market data client for direct use by REST endpoints."""
        return self._market_data

    # -----------------------------------------------------------------------
    # Prometheus helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _classify_and_record(exc: BaseException) -> None:
        """Classify an exception as 429 / 5xx / 4xx and bump the counter."""
        import re
        msg = str(exc)
        match = re.search(r"\b(4\d{2}|5\d{2})\b", msg)
        if match:
            code = int(match.group(1))
            if code == 429:
                _record_submit("429")
            elif code >= 500:
                _record_submit("5xx")
            else:
                _record_submit("4xx")
        else:
            _record_submit("5xx")   # default: treat unknown as server error

    # -----------------------------------------------------------------------
    # Reconciliation
    # -----------------------------------------------------------------------

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
