"""
CCXTAdapter — Multi-exchange broker integration via ccxt.
=========================================================

Supports Binance, Bybit, Kraken (and any other ccxt-supported venue).
Defaults to **spot** trading; perpetual swaps are enabled by setting
``CCXTConfig.market_type = "swap"``.

Design
------
* Uses ``ccxt.async_support`` — native asyncio, no thread-pool offloading.
* One adapter instance == one exchange.  The routing layer holds a registry
  of adapters and dispatches by venue.
* Testnet/sandbox is enabled via ``CCXTConfig.testnet`` and only honoured
  for venues whose ccxt class supports ``set_sandbox_mode``.
* ``get_positions`` semantics:
    - spot: returns non-zero asset balances as long-only ``Position`` objects
      with ``avg_entry == 0`` (true entry is tracked in our DB, PASO E).
    - swap: returns true positions with broker-reported entry / margin / PnL.

References
----------
* ccxt docs:  https://docs.ccxt.com/
* ccxt async: https://github.com/ccxt/ccxt/blob/master/python/README.md
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

from .base import AccountInfo, BrokerAdapter, BrokerError
from . import _ccxt_symbol as sym_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    import ccxt.async_support as _ccxt           # type: ignore
    _HAS_CCXT = True
except ImportError:                              # pragma: no cover
    _HAS_CCXT = False
    _ccxt = None                                  # type: ignore[assignment]


# Supported venue → ccxt class name (must match ccxt module attribute)
_SUPPORTED_VENUES: dict[str, str] = {
    "binance": "binance",
    "bybit":   "bybit",
    "kraken":  "kraken",
    "okx":     "okx",
    "coinbase": "coinbase",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class CCXTConfig:
    """
    Connection configuration for :class:`CCXTAdapter`.

    Parameters
    ----------
    exchange : str
        Venue identifier (``"binance"``, ``"bybit"``, ``"kraken"``, …).
        Must be a key in :data:`_SUPPORTED_VENUES`.
    api_key, api_secret : str
        Exchange credentials.  Read-only keys are insufficient for trading.
    testnet : bool
        Use the venue's testnet / sandbox endpoint when ``True``.
    market_type : str
        ``"spot"`` (default) or ``"swap"`` for perpetual futures.

    Env defaults (used when args are omitted):
      * ``CCXT_EXCHANGE``
      * ``CCXT_API_KEY``
      * ``CCXT_API_SECRET``
      * ``CCXT_TESTNET``      ("true" / "false")
      * ``CCXT_MARKET_TYPE``  ("spot" / "swap")
    """

    def __init__(
        self,
        exchange:    Optional[str]  = None,
        api_key:     Optional[str]  = None,
        api_secret:  Optional[str]  = None,
        testnet:     Optional[bool] = None,
        market_type: Optional[str]  = None,
    ):
        self.exchange   = (exchange   or os.getenv("CCXT_EXCHANGE",   "binance")).lower()
        self.api_key    =  api_key    or os.getenv("CCXT_API_KEY",    "")
        self.api_secret =  api_secret or os.getenv("CCXT_API_SECRET", "")
        if testnet is None:
            testnet = os.getenv("CCXT_TESTNET", "true").lower() == "true"
        self.testnet = testnet
        self.market_type = (market_type or os.getenv("CCXT_MARKET_TYPE", "spot")).lower()

        if self.exchange not in _SUPPORTED_VENUES:
            raise BrokerError(
                f"Unsupported CCXT venue: {self.exchange!r}.  "
                f"Supported: {sorted(_SUPPORTED_VENUES)}"
            )
        if self.market_type not in ("spot", "swap"):
            raise BrokerError(
                f"market_type must be 'spot' or 'swap', got {self.market_type!r}"
            )

    def __repr__(self) -> str:
        env = "testnet" if self.testnet else "LIVE"
        masked = (self.api_key[:4] + "***") if self.api_key else "<missing>"
        return (
            f"CCXTConfig(venue={self.exchange}, env={env}, "
            f"type={self.market_type}, key={masked})"
        )


# ---------------------------------------------------------------------------
# CCXT status mapping
# ---------------------------------------------------------------------------
# ccxt's unified statuses are 'open' | 'closed' | 'canceled' | 'expired'.
# We refine 'open' → PARTIAL when filled > 0, else SUBMITTED.

_CCXT_TIF: dict[TimeInForce, str] = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.DAY: "GTC",       # ccxt has no DAY; map to GTC
}


# ---------------------------------------------------------------------------
# CCXTAdapter
# ---------------------------------------------------------------------------

class CCXTAdapter(BrokerAdapter):
    """
    Multi-exchange broker adapter using ``ccxt.async_support``.

    Parameters
    ----------
    config : CCXTConfig, optional
        Connection config.  Defaults to env-driven instance.
    exchange_factory : callable, optional
        DI hook for tests; must return an object with the ccxt exchange
        interface (``create_order``, ``cancel_order``, ``fetch_order``,
        ``fetch_balance``, ``fetch_positions``, ``fetch_ticker``,
        ``load_markets``, ``set_sandbox_mode``, ``close``).

    Raises
    ------
    BrokerError
        On :meth:`connect` if ``ccxt`` is not installed or credentials missing.
    """

    def __init__(
        self,
        config: Optional[CCXTConfig] = None,
        exchange_factory: Optional[Any] = None,
    ):
        self.config = config or CCXTConfig()
        self._factory = exchange_factory
        self._exchange: Any = None
        self.venue = self.config.exchange    # override class default

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        if self._exchange is not None:
            return                          # idempotent

        if self._factory is not None:
            self._exchange = self._factory()
        else:
            if not _HAS_CCXT:
                raise BrokerError("ccxt is not installed.  Run: pip install ccxt")
            if not self.config.api_key or not self.config.api_secret:
                raise BrokerError(
                    "CCXT credentials missing.  Set CCXT_API_KEY / "
                    "CCXT_API_SECRET or pass CCXTConfig."
                )
            klass = getattr(_ccxt, _SUPPORTED_VENUES[self.config.exchange])
            self._exchange = klass({
                "apiKey":          self.config.api_key,
                "secret":          self.config.api_secret,
                "enableRateLimit": True,
                "options":         {"defaultType": self.config.market_type},
            })

        # sandbox / testnet (best-effort — not all venues expose it)
        if self.config.testnet and hasattr(self._exchange, "set_sandbox_mode"):
            try:
                self._exchange.set_sandbox_mode(True)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("ccxt sandbox mode unavailable: %s", exc)

        # load market metadata once
        try:
            await self._exchange.load_markets()
        except Exception as exc:                                    # noqa: BLE001
            raise BrokerError(f"ccxt load_markets failed: {exc}") from exc

        logger.info(
            "ccxt.connected venue=%s testnet=%s type=%s",
            self.config.exchange, self.config.testnet, self.config.market_type,
        )

    async def close(self) -> None:
        if self._exchange is None:
            return
        try:
            await self._exchange.close()
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("ccxt close: %s", exc)
        self._exchange = None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _require_exchange(self) -> Any:
        if self._exchange is None:
            raise BrokerError("CCXTAdapter.connect() must be called first.")
        return self._exchange

    @staticmethod
    def _ccxt_order_type(intent: OrderIntent) -> str:
        """Translate :class:`OrderType` → ccxt order type string."""
        mapping = {
            OrderType.MARKET:      "market",
            OrderType.LIMIT:       "limit",
            OrderType.LIMIT_MAKER: "limit",   # post-only set via params
            OrderType.STOP_LIMIT:  "stop_limit",
        }
        if intent.order_type not in mapping:
            raise BrokerError(f"Unsupported order_type for CCXT: {intent.order_type}")
        return mapping[intent.order_type]

    def _ccxt_params(self, intent: OrderIntent) -> dict:
        """Build the ``params`` dict for ccxt.create_order."""
        params: dict[str, Any] = {}
        if intent.order_type == OrderType.LIMIT_MAKER:
            params["postOnly"] = True
        if intent.tif in _CCXT_TIF:
            params["timeInForce"] = _CCXT_TIF[intent.tif]
        if intent.order_type == OrderType.STOP_LIMIT and intent.sl_price is not None:
            params["stopPrice"] = float(intent.sl_price)
        return params

    @staticmethod
    def _status_to_internal(raw: dict) -> OrderStatus:
        """Translate ccxt unified status (+ filled qty) → our enum."""
        status = (raw.get("status") or "").lower()
        filled = float(raw.get("filled") or 0)
        amount = float(raw.get("amount") or 0)

        if status == "closed":
            return OrderStatus.FILLED
        if status == "canceled":
            return OrderStatus.CANCELLED
        if status == "expired":
            return OrderStatus.EXPIRED
        if status == "rejected":
            return OrderStatus.REJECTED
        if status == "open":
            if filled > 0 and filled < amount:
                return OrderStatus.PARTIAL
            return OrderStatus.SUBMITTED
        return OrderStatus.SUBMITTED        # safe default

    def _ccxt_order_to_result(self, raw: dict, intent_id: str = "") -> OrderResult:
        """Translate a ccxt unified order dict → :class:`OrderResult`."""
        side       = OrderSide(str(raw.get("side", "buy")).lower())
        status     = self._status_to_internal(raw)
        qty        = Decimal(str(raw.get("amount", 0)))
        filled_qty = Decimal(str(raw.get("filled", 0)))
        avg_price  = (
            Decimal(str(raw["average"]))
            if raw.get("average") is not None else None
        )
        ts = _ms_to_utc(raw.get("timestamp")) or datetime.now(tz=timezone.utc)

        return OrderResult(
            intent_id    = intent_id,
            broker_id    = str(raw.get("id", "")),
            symbol       = sym_map.from_ccxt(str(raw.get("symbol", ""))),
            side         = side,
            status       = status,
            qty          = qty,
            filled_qty   = filled_qty,
            avg_price    = avg_price,
            ts_submitted = ts,
            ts_updated   = ts,
            venue        = self.venue,
        )

    def _spot_balance_to_positions(self, balance: dict) -> list[Position]:
        """Spot mode: synthesise long positions from non-zero asset balances."""
        positions: list[Position] = []
        for asset, info in balance.items():
            if not isinstance(info, dict):
                continue
            total = info.get("total")
            if total is None or float(total) <= 0:
                continue
            # skip stablecoins / quote currencies
            if asset.upper() in ("USDT", "USDC", "USD", "DAI", "BUSD", "FDUSD"):
                continue
            positions.append(Position(
                symbol=f"{asset.upper()}USDT",   # canonical
                side=OrderSide.BUY,
                qty=Decimal(str(total)),
                avg_entry=Decimal("0"),          # not retrievable from spot balance
                venue=self.venue,
            ))
        return positions

    def _swap_position_to_internal(self, raw: dict) -> Position:
        """Swap mode: map a ccxt position dict → :class:`Position`."""
        contracts = float(raw.get("contracts", 0))
        side = OrderSide.BUY if str(raw.get("side", "long")).lower() == "long" else OrderSide.SELL
        return Position(
            symbol         = sym_map.from_ccxt(str(raw.get("symbol", ""))),
            side           = side,
            qty            = Decimal(str(abs(contracts))),
            avg_entry      = Decimal(str(raw.get("entryPrice", 0) or 0)),
            current_price  = _opt_decimal(raw.get("markPrice")),
            unrealized_pnl = _opt_decimal(raw.get("unrealizedPnl")),
            margin_used    = _opt_decimal(raw.get("initialMargin")),
            venue          = self.venue,
        )

    # -----------------------------------------------------------------------
    # BrokerAdapter implementation
    # -----------------------------------------------------------------------

    async def submit(self, intent: OrderIntent) -> OrderResult:
        exchange = self._require_exchange()
        ccxt_sym = sym_map.to_ccxt(intent.symbol)
        ord_type = self._ccxt_order_type(intent)
        side     = intent.side.value
        amount   = float(intent.qty)
        price    = float(intent.limit_price) if intent.limit_price is not None else None
        params   = self._ccxt_params(intent)

        try:
            raw = await exchange.create_order(
                symbol=ccxt_sym, type=ord_type, side=side,
                amount=amount, price=price, params=params,
            )
        except Exception as exc:                                    # noqa: BLE001
            raise BrokerError(f"{self.venue} submit failed: {exc}") from exc

        return self._ccxt_order_to_result(raw, intent_id=intent.intent_id)

    async def cancel(self, broker_id: str) -> bool:
        exchange = self._require_exchange()
        try:
            await exchange.cancel_order(broker_id)
            return True
        except Exception as exc:                                    # noqa: BLE001
            msg = str(exc).lower()
            if any(kw in msg for kw in ("not found", "already", "unknown order", "ordernotfound")):
                return False
            raise BrokerError(f"{self.venue} cancel failed: {exc}") from exc

    async def get_order(self, broker_id: str) -> OrderResult:
        exchange = self._require_exchange()
        try:
            raw = await exchange.fetch_order(broker_id)
        except Exception as exc:                                    # noqa: BLE001
            raise BrokerError(f"{self.venue} fetch_order failed: {exc}") from exc
        return self._ccxt_order_to_result(raw)

    async def get_positions(self) -> list[Position]:
        exchange = self._require_exchange()
        try:
            if self.config.market_type == "swap":
                raws = await exchange.fetch_positions()
                return [self._swap_position_to_internal(r) for r in raws
                        if float(r.get("contracts", 0) or 0) != 0]
            balance = await exchange.fetch_balance()
            return self._spot_balance_to_positions(balance)
        except Exception as exc:                                    # noqa: BLE001
            raise BrokerError(f"{self.venue} get_positions failed: {exc}") from exc

    async def get_account(self) -> AccountInfo:
        exchange = self._require_exchange()
        try:
            balance = await exchange.fetch_balance()
        except Exception as exc:                                    # noqa: BLE001
            raise BrokerError(f"{self.venue} get_account failed: {exc}") from exc

        # Quote-currency equity / cash.  Default to USDT for crypto-only venues.
        quote = "USDT"
        info_q = balance.get(quote, {}) if isinstance(balance.get(quote), dict) else {}
        equity = Decimal(str(info_q.get("total", 0) or 0))
        cash   = Decimal(str(info_q.get("free",  0) or 0))

        return AccountInfo(
            account_id  = "",                          # ccxt does not expose
            venue       = self.venue,
            equity      = equity,
            cash        = cash,
            margin_used = equity - cash if equity >= cash else Decimal("0"),
            pnl_day     = Decimal("0"),                # ccxt does not expose
            currency    = quote,
            is_paper    = self.config.testnet,
        )

    async def reconcile(self, internal_positions: list[Position]) -> list[str]:
        broker_positions = await self.get_positions()

        broker_by   = {p.symbol: p for p in broker_positions}
        internal_by = {p.symbol: p for p in internal_positions}

        discrepancies: list[str] = []

        for sym in broker_by.keys() - internal_by.keys():
            discrepancies.append(
                f"PHANTOM: broker has {sym} qty={broker_by[sym].qty}, internal has none"
            )
        for sym in internal_by.keys() - broker_by.keys():
            discrepancies.append(
                f"MISSING: internal has {sym} qty={internal_by[sym].qty}, broker has none"
            )
        for sym in broker_by.keys() & internal_by.keys():
            b, i = broker_by[sym], internal_by[sym]
            if b.side != i.side:
                discrepancies.append(
                    f"SIDE_MISMATCH {sym}: broker={b.side.value} internal={i.side.value}"
                )
            elif b.qty != i.qty:
                discrepancies.append(
                    f"QTY_MISMATCH {sym}: broker={b.qty} internal={i.qty}"
                )
        return discrepancies

    async def get_last_price(self, symbol: str) -> Optional[Decimal]:
        exchange = self._require_exchange()
        try:
            ticker = await exchange.fetch_ticker(sym_map.to_ccxt(symbol))
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("ccxt fetch_ticker(%s): %s", symbol, exc)
            return None
        last = ticker.get("last") if isinstance(ticker, dict) else None
        return Decimal(str(last)) if last is not None else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms_to_utc(ms: Any) -> Optional[datetime]:
    """Coerce a Unix-ms timestamp → UTC datetime."""
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
    except Exception:                                                # noqa: BLE001
        return None


def _opt_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:                                                # noqa: BLE001
        return None
