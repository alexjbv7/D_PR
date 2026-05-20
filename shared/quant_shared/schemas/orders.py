"""
Order domain schemas — Pydantic v2.
=====================================
Tipos de valor usados internamente por el execution-engine.
NO son eventos Kafka (esos están en events.py / ExecutionIntentEvent).
Son el dominio puro de órdenes / fills / posiciones.

Convenciones (ADR-010):
  - UTC timestamps en todas las fechas.
  - Decimal para todos los valores monetarios (nunca float).
  - UUID v7 para IDs (time-sortable, RFC 9562).
  - Pydantic v2 (model_config, no class Config).

Relación con otros tipos:
  - ExecutionIntentEvent (events.py) es el Kafka transport schema.
  - OrderIntent (este archivo) es el value object interno del execution-engine
    que el risk-engine crea a partir de un FinalSignalEvent.
"""
from __future__ import annotations

import random
import time
import uuid as _uuid_mod
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _uuid7() -> str:
    """Generate a UUID v7 (time-sortable, RFC 9562).

    Layout (128 bits, big-endian):
      [0-47]   unix_ts_ms  (48 bits)
      [48-51]  version=7   ( 4 bits)
      [52-63]  rand_a      (12 bits)
      [64-65]  variant=0b10 (2 bits)
      [66-127] rand_b      (62 bits)
    """
    ms     = int(time.time_ns() // 1_000_000)
    rand_a = random.getrandbits(12)
    rand_b = random.getrandbits(62)
    value  = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(_uuid_mod.UUID(int=value))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET         = "market"
    LIMIT          = "limit"
    LIMIT_MAKER    = "limit_maker"   # post-only
    STOP_LIMIT     = "stop_limit"
    TRAILING_STOP  = "trailing_stop"
    TWAP           = "twap"
    VWAP           = "vwap"


class OrderStatus(str, Enum):
    PENDING   = "pending"    # not yet submitted to broker
    SUBMITTED = "submitted"  # awaiting ack
    PARTIAL   = "partial"    # partially filled
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"
    EXPIRED   = "expired"


class TimeInForce(str, Enum):
    GTC = "gtc"   # good till cancelled
    IOC = "ioc"   # immediate or cancel
    FOK = "fok"   # fill or kill
    DAY = "day"


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------

class OrderIntent(BaseModel):
    """
    Risk-approved intent to place a single order.

    Created by the risk-engine after validating a FinalSignalEvent.
    Consumed by the execution-engine's BrokerAdapter.

    This is an internal value object, not a Kafka event schema.
    For the Kafka transport see ExecutionIntentEvent in events.py.

    Parameters
    ----------
    intent_id : str
        UUID v7 (auto-generated, time-sortable).
    signal_id : str
        Back-reference to the originating TradeSignal / FinalSignalEvent.
    symbol : str
        Ticker / trading pair (e.g. "BTCUSDT", "AAPL").
    side : OrderSide
        Direction of the trade.
    qty : Decimal
        Quantity to trade (always positive; side encodes direction).
    limit_price : Decimal, optional
        Required for LIMIT / LIMIT_MAKER order types.
    sl_price : Decimal, optional
        Stop-loss price. Execution engine may place a bracket order.
    tp_price : Decimal, optional
        Take-profit price.
    venue : str
        Routing hint (e.g. "alpaca", "binance"). Empty = auto-route.

    Examples
    --------
    >>> intent = OrderIntent(
    ...     symbol="BTCUSDT",
    ...     side=OrderSide.BUY,
    ...     qty=Decimal("0.01"),
    ...     order_type=OrderType.LIMIT_MAKER,
    ...     limit_price=Decimal("65000"),
    ...     venue="binance",
    ... )
    """

    # Pydantic v2 serialises datetime → ISO string and Decimal → str in JSON
    # mode by default; no explicit json_encoders needed.

    intent_id:   str          = Field(default_factory=_uuid7)
    signal_id:   str          = ""
    strategy:    str          = ""
    symbol:      str
    side:        OrderSide
    qty:         Optional[Decimal] = Field(default=None, gt=Decimal("0"))
    notional:    Optional[Decimal] = Field(default=None, gt=Decimal("0"))
    order_type:  OrderType    = OrderType.LIMIT_MAKER
    limit_price:    Optional[Decimal] = None
    sl_price:       Optional[Decimal] = None
    tp_price:       Optional[Decimal] = None
    trail_percent:  Optional[Decimal] = None   # e.g. Decimal("1.5") = 1.5%
    trail_price:    Optional[Decimal] = None   # absolute trail offset
    extended_hours: bool              = False
    tif:            TimeInForce       = TimeInForce.GTC
    venue:          str               = ""
    ts:             datetime          = Field(default_factory=_utcnow)

    # Risk metadata carried for audit trail
    kelly_fraction:  float = 0.0
    target_risk_pct: float = 0.0
    p_win:           float = 0.0

    @field_validator(
        "qty", "notional", "limit_price", "sl_price", "tp_price",
        "trail_percent", "trail_price",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: object) -> Optional[Decimal]:
        if v is None:
            return None
        return Decimal(str(v))

    @model_validator(mode="after")
    def _qty_xor_notional(self) -> "OrderIntent":
        if (self.qty is None) == (self.notional is None):
            raise ValueError(
                "OrderIntent requires exactly one of (qty, notional), "
                "not both nor neither"
            )
        return self

    @model_validator(mode="after")
    def _validate_trailing_and_bracket(self) -> "OrderIntent":
        if self.order_type == OrderType.TRAILING_STOP:
            has_pct = self.trail_percent is not None
            has_px  = self.trail_price is not None
            if has_pct == has_px:
                raise ValueError(
                    "TRAILING_STOP requires exactly one of trail_percent or trail_price"
                )
            if has_pct and self.trail_percent is not None and self.trail_percent <= 0:
                raise ValueError("trail_percent must be > 0")
            if has_px and self.trail_price is not None and self.trail_price <= 0:
                raise ValueError("trail_price must be > 0")

        if self.sl_price is not None and self.tp_price is not None:
            if self.order_type == OrderType.LIMIT_MAKER:
                raise ValueError(
                    "Bracket orders require LIMIT (not post-only) or MARKET; "
                    "LIMIT_MAKER is incompatible with sl_price+tp_price"
                )
        return self


class Fill(BaseModel):
    """
    A single execution fill returned by the broker.

    One order can produce multiple fills (partial fills on Binance, etc.).
    The `raw` dict stores the broker-native response for full audit trail.

    Parameters
    ----------
    fill_id : str
        UUID v7 (auto-generated).
    order_id : str
        Broker order ID this fill belongs to.
    fee : Decimal
        Commission paid.  Always ≥ 0.
    raw : dict
        Verbatim broker response for compliance / debugging.
    """

    # Pydantic v2 serialises datetime → ISO string and Decimal → str in JSON
    # mode by default; no explicit json_encoders needed.

    fill_id:   str       = Field(default_factory=_uuid7)
    order_id:  str       = ""
    symbol:    str
    side:      OrderSide
    qty:       Decimal
    price:     Decimal
    fee:       Decimal   = Decimal("0")
    fee_asset: str       = "USD"
    ts:        datetime  = Field(default_factory=_utcnow)
    venue:      str       = ""
    account_id: str       = ""
    raw:        dict[str, Any] = Field(default_factory=dict)

    @field_validator("qty", "price", "fee", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: object) -> Decimal:
        return Decimal(str(v))

    @property
    def notional(self) -> Decimal:
        """Gross notional value of this fill."""
        return self.qty * self.price


class OrderResult(BaseModel):
    """
    Result of submitting an OrderIntent to a broker.

    The `status` field tracks the order lifecycle:
      PENDING → SUBMITTED → PARTIAL → FILLED
                                    ↘ CANCELLED / REJECTED / EXPIRED

    Parameters
    ----------
    result_id : str
        UUID v7 (auto-generated).
    intent_id : str
        Matches the originating OrderIntent.intent_id.
    broker_id : str
        Exchange/broker native order ID (for cancel / query).
    fills : list[Fill]
        All fills received so far (may be empty for SUBMITTED status).
    reject_reason : str, optional
        Human-readable rejection cause from the broker.

    Examples
    --------
    >>> result.is_complete
    False
    >>> result.remaining_qty
    Decimal('0.005')
    """

    # Pydantic v2 serialises datetime → ISO string and Decimal → str in JSON
    # mode by default; no explicit json_encoders needed.

    result_id:    str          = Field(default_factory=_uuid7)
    intent_id:    str          = ""
    broker_id:    str          = ""
    symbol:       str
    side:         OrderSide
    status:       OrderStatus
    qty:          Decimal
    filled_qty:   Decimal      = Decimal("0")
    avg_price:    Optional[Decimal] = None
    fills:        list[Fill]   = Field(default_factory=list)
    reject_reason: Optional[str] = None
    ts_submitted: datetime     = Field(default_factory=_utcnow)
    ts_updated:   datetime     = Field(default_factory=_utcnow)
    venue:        str          = ""

    @field_validator("qty", "filled_qty", "avg_price", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: object) -> Optional[Decimal]:
        if v is None:
            return None
        return Decimal(str(v))

    @property
    def is_complete(self) -> bool:
        """True when no further state transitions are possible."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def remaining_qty(self) -> Decimal:
        """Quantity not yet filled."""
        return self.qty - self.filled_qty


class Position(BaseModel):
    """
    Open position as reported by the broker.

    Represents the broker's live view; the execution-engine's reconciler
    compares this against internal state periodically.

    Parameters
    ----------
    symbol : str
        Ticker / trading pair.
    side : OrderSide
        BUY = long, SELL = short.
    qty : Decimal
        Absolute size held.
    avg_entry : Decimal
        Volume-weighted average entry price.
    current_price : Decimal, optional
        Latest mark price (filled by broker response when available).
    unrealized_pnl : Decimal, optional
        Unrealized P&L in account currency (from broker).
    margin_used : Decimal, optional
        Required margin for leveraged positions.

    Examples
    --------
    >>> pos.notional
    Decimal('65000.00')
    >>> pos.pnl_pct
    0.015384...
    """

    # Pydantic v2 serialises datetime → ISO string and Decimal → str in JSON
    # mode by default; no explicit json_encoders needed.

    symbol:         str
    side:           OrderSide
    qty:            Decimal
    avg_entry:      Decimal
    current_price:  Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    margin_used:    Optional[Decimal] = None
    venue:          str      = ""
    ts_opened:      Optional[datetime] = None
    ts_updated:     datetime = Field(default_factory=_utcnow)

    @field_validator("qty", "avg_entry", "current_price",
                     "unrealized_pnl", "margin_used", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: object) -> Optional[Decimal]:
        if v is None:
            return None
        return Decimal(str(v))

    @property
    def notional(self) -> Decimal:
        """Gross notional (qty × avg_entry)."""
        return self.qty * self.avg_entry

    @property
    def pnl_pct(self) -> Optional[float]:
        """Unrealized return as a fraction (long-positive, short-positive)."""
        if self.current_price is None or self.avg_entry == 0:
            return None
        sign = Decimal("1") if self.side == OrderSide.BUY else Decimal("-1")
        return float(
            (self.current_price - self.avg_entry) / self.avg_entry * sign
        )
