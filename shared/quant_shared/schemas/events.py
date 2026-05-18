"""
Kafka event schemas — canonical source of truth for the monorepo.
==================================================================
Todos los eventos que viajan por Kafka están definidos aquí.
Cualquier cambio en este archivo es un breaking change: versionar con SemVer
y documentar en CHANGELOG antes de fusionar.

Convención de topic:
  <domain>.<entity>.<action>
  ej: market.tick.raw, macro.regime.update, signals.final

Todos los eventos heredan de BaseEvent:
  - event_id    : UUID v4 (str)
  - ts          : datetime UTC
  - source      : servicio origen
  - version     : versión del schema

MIGRATION STATUS (2026-05-14):
  Fuente de verdad: quant_shared.schemas.events  (este archivo)
  Compatibilidad:   platform/libs/shared/events.py re-exporta desde aquí.
  Los servicios deben migrar gradualmente de:
      from libs.shared.events import MarketDataEvent
  a:
      from quant_shared.schemas.events import MarketDataEvent
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _uuid4() -> str:
    return str(uuid.uuid4())


# ===========================================================================
# BASE
# ===========================================================================

class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=_uuid4)
    ts: datetime = Field(default_factory=_utcnow)
    source: str = "unknown"
    version: str = "1.0"

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ===========================================================================
# MARKET DATA
# ===========================================================================

class TickEvent(BaseEvent):
    """Raw trade tick from exchange."""
    source: str = "ingestion"
    symbol: str
    venue: str
    price: float
    qty: float
    side: str                    # "buy" | "sell"
    trade_id: str = ""


class OHLCVEvent(BaseEvent):
    """Completed OHLCV bar."""
    source: str = "normalizer"
    symbol: str
    venue: str
    timeframe: str               # "1m" | "5m" | "1h" | "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_ts: datetime             # timestamp de apertura del bar


class OrderBookEvent(BaseEvent):
    """Orderbook snapshot or diff."""
    source: str = "ingestion"
    symbol: str
    venue: str
    bids: list[list[float]]      # [[price, qty], ...]
    asks: list[list[float]]
    seq: int = 0
    is_snapshot: bool = True


class FundingRateEvent(BaseEvent):
    """Perpetual funding rate update."""
    source: str = "binance-collector"
    symbol: str
    venue: str
    funding_rate: float
    next_funding_ts: datetime
    mark_price: float
    index_price: float


class LiquidationEvent(BaseEvent):
    """Forced liquidation on perpetuals."""
    source: str = "binance-collector"
    symbol: str
    venue: str
    side: str                    # "long" | "short"
    qty: float
    price: float


# ===========================================================================
# DERIVED FEATURES
# ===========================================================================

class FeatureUpdateEvent(BaseEvent):
    """New feature vector computed for a symbol."""
    source: str = "feature-engine"
    symbol: str
    timeframe: str
    feature_set: str             # "technical_v1", "orderflow_v1", etc.
    feature_set_hash: str
    features: dict[str, float]
    bar_ts: datetime


class RegimeUpdateEvent(BaseEvent):
    """Market regime classification update."""
    source: str = "context-engine"
    symbol: str
    regime_label: int
    regime_probs: list[float]
    regime_name: str             # "bull_trend" | "bear_trend" | "range" | "high_vol"
    stability: float             # 0-1
    method: str = "gmm"


# ===========================================================================
# MACRO
# ===========================================================================

class MacroDataEvent(BaseEvent):
    """New macro data point from FRED or other source."""
    source: str = "macroeconomic"
    series_id: str               # "CPIAUCSL", "DFF", "T10Y2Y", etc.
    series_name: str
    value: float
    frequency: str               # "monthly" | "weekly" | "daily"
    release_date: datetime
    prior_value: Optional[float] = None
    surprise_pct: Optional[float] = None  # (actual - estimate) / |estimate|


class MacroRegimeEvent(BaseEvent):
    """Macro regime classification."""
    source: str = "macroeconomic"
    regime: str                  # "expansion" | "slowdown" | "recession" | "recovery"
    confidence: float
    dominant_signal: str
    rate_environment: str        # "hiking" | "cutting" | "hold"
    features: dict[str, float]


class RecessionAlertEvent(BaseEvent):
    """Recession probability spike."""
    source: str = "macroeconomic"
    probability: float
    model: str                   # "yield_curve" | "sahm_rule" | "ensemble"
    threshold_breached: float
    severity: str                # "watch" | "warning" | "alert"


# ===========================================================================
# ON-CHAIN
# ===========================================================================

class WhaleAlertEvent(BaseEvent):
    """Large on-chain transaction detected."""
    source: str = "onchain-analysis"
    blockchain: str
    tx_hash: str
    from_address: str
    to_address: str
    amount_usd: float
    amount_native: float
    token: str
    direction: str               # "exchange_inflow" | "exchange_outflow" | "wallet_to_wallet"
    from_label: Optional[str] = None
    to_label: Optional[str] = None


class SmartMoneyFlowEvent(BaseEvent):
    """Aggregate smart money flow signal."""
    source: str = "onchain-analysis"
    blockchain: str
    token: str
    net_flow_exchange_24h: float
    whale_accumulation_score: float
    dex_volume_24h: float
    signal: str                  # "accumulation" | "distribution" | "neutral"
    confidence: float


class OnChainSignalEvent(BaseEvent):
    """Derived on-chain trading signal."""
    source: str = "onchain-analysis"
    symbol: str
    signal_type: str
    direction: int               # -1 | 0 | 1
    strength: float
    metadata: dict[str, Any] = {}


# ===========================================================================
# SENTIMENT / SEC
# ===========================================================================

class NewsSentimentEvent(BaseEvent):
    """Processed news article with sentiment."""
    source: str = "sec-research"
    headline: str
    body_snippet: str
    url: str
    publisher: str
    sentiment_score: float       # -1 to 1
    sentiment_label: str         # "bearish" | "neutral" | "bullish"
    entities: list[str]
    relevance: float
    category: str                # "macro" | "crypto" | "fed" | "earnings"


class SECFilingEvent(BaseEvent):
    """New SEC filing processed."""
    source: str = "sec-research"
    cik: str
    company: str
    ticker: str
    form_type: str               # "8-K" | "10-Q" | "10-K" | "S-1"
    filed_date: datetime
    summary: str
    sentiment: float
    key_risks: list[str] = []
    key_opportunities: list[str] = []


class EarningsEvent(BaseEvent):
    """Earnings call / transcript processed."""
    source: str = "sec-research"
    ticker: str
    quarter: str                 # "Q1 2026"
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_actual: Optional[float] = None
    guidance_sentiment: float
    tone_ceo: float
    key_topics: list[str] = []


# ===========================================================================
# PREDICTION MARKETS
# ===========================================================================

class PredictionMarketEvent(BaseEvent):
    """Polymarket / Kalshi market update."""
    source: str = "market-intelligence"
    market_id: str
    question: str
    category: str
    yes_probability: float
    no_probability: float
    volume_24h: float
    liquidity: float
    event_date: Optional[datetime] = None
    tags: list[str] = []


# ===========================================================================
# SIGNALS
# ===========================================================================

class RawSignalEvent(BaseEvent):
    """Signal from a single model/strategy."""
    source: str = "ml-inference"
    strategy: str
    symbol: str
    timeframe: str
    direction: int               # -1 | 0 | 1
    p_win: float
    p_win_raw: float
    model_version: str
    feature_set_hash: str
    regime: int = 0
    confidence_tier: str = "low"


class FinalSignalEvent(BaseEvent):
    """Aggregated final signal after meta-labeling + Bayesian update."""
    source: str = "signal-router"
    strategy: str
    symbol: str
    timeframe: str
    direction: int
    p_win: float
    kelly_fraction: float
    rr_ratio: float
    target_risk_pct: float
    sl_atr_mult: float
    meta_filter_passed: bool
    bayesian_updated: bool


class ExecutionIntentEvent(BaseEvent):
    """Risk-approved order intent."""
    source: str = "risk-engine"
    strategy: str
    symbol: str
    side: str                    # "buy" | "sell"
    qty: float
    price: Optional[float] = None
    order_type: str = "LIMIT_MAKER"
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    risk_pct: float
    rationale: str = ""


# ===========================================================================
# SYSTEM
# ===========================================================================

class AnomalyEvent(BaseEvent):
    """System or market anomaly detected."""
    source: str = "context-engine"
    anomaly_type: str
    severity: str                # "info" | "warning" | "critical"
    symbol: Optional[str] = None
    description: str
    affected_services: list[str] = []
    auto_action: Optional[str] = None


class KillSwitchEvent(BaseEvent):
    """Emergency kill switch activation."""
    source: str = "risk-engine"
    triggered_by: str            # "drawdown" | "latency" | "manual" | "anomaly"
    scope: str                   # "all" | strategy name | symbol
    message: str
    resume_condition: Optional[str] = None


# ===========================================================================
# TOPIC REGISTRY
# ===========================================================================

# ===========================================================================
# CORPORATE ACTIONS & UNIVERSE (Semana 4)
# ===========================================================================

class CorporateActionEvent(BaseEvent):
    """Emitted by market-intelligence cron. Consumed by execution-engine and
    feature-engine to adjust positions and rolling features respectively.

    ``split_ratio = split_to / split_from`` for *_split events.
    For forward splits ratio > 1; for reverse splits ratio < 1.
    """

    source: str = "market-intelligence"
    ca_id: str                          # UUID v7
    alpaca_id: Optional[str] = None
    symbol: str
    ca_type: Literal[
        "forward_split",
        "reverse_split",
        "stock_dividend",
        "cash_dividend",
        "merger",
        "spinoff",
        "name_change",
    ]
    ex_ts: datetime                     # UTC — adjustment applies to bars with ts < ex_ts
    split_ratio: Optional[Decimal] = None   # split_to / split_from
    cash_amount: Optional[Decimal] = None
    stock_amount: Optional[Decimal] = None
    new_symbol: Optional[str] = None    # for mergers / name_changes / spinoffs
    is_provisional: bool = True
    emitted_ts: datetime = Field(default_factory=_utcnow)


class UniverseUpdateEvent(BaseEvent):
    """Emitted when a symbol changes listing status.

    ``change_type`` values:
    - ``new_listing``     : symbol newly active in /v2/assets
    - ``delisting``       : confirmed delisted after 3-day inactive buffer
    - ``metadata_update`` : fractionable / tradable / exchange flags changed
    """

    source: str = "market-intelligence"
    symbol: str
    asset_class: str = "us_equity"
    change_type: Literal["new_listing", "delisting", "metadata_update"]
    delisted_ts: Optional[datetime] = None
    emitted_ts: datetime = Field(default_factory=_utcnow)


class KafkaTopics:
    """Central registry of all Kafka topic names (authoritative).

    All topics use the los_ojos.* prefix, matching the broker configuration
    in infra/kafka/topics.yml.
    """

    # Raw market data
    RAW_TICK            = "los_ojos.market.data"
    CLEAN_OHLCV         = "los_ojos.market.normalized"
    ORDERBOOK           = "los_ojos.market.orderbook"
    FUNDING_RATE        = "los_ojos.market.funding"
    LIQUIDATION         = "los_ojos.derivatives.events"

    # Features
    FEATURE_UPDATE      = "los_ojos.features.vector"
    REGIME_UPDATE       = "los_ojos.context.regime"

    # Macro
    MACRO_DATA          = "los_ojos.macro.indicators"
    MACRO_REGIME        = "los_ojos.macro.regime"
    RECESSION_ALERT     = "los_ojos.macro.recession_alert"
    MACRO_SERIES        = "los_ojos.macro.series"
    MACRO_SIGNAL        = "los_ojos.macro.signal"

    # On-chain
    WHALE_ALERT         = "los_ojos.onchain.whale_alert"
    SMART_MONEY         = "los_ojos.onchain.smart_money"
    ONCHAIN_SIGNAL      = "los_ojos.macro.signal"

    # Sentiment
    NEWS_SENTIMENT      = "los_ojos.market.data"
    SEC_FILING          = "los_ojos.sec.signal"
    EARNINGS            = "los_ojos.market.data"

    # Prediction markets
    POLYMARKET_UPDATE   = "los_ojos.macro.signal"

    # Signals
    SIGNAL_RAW          = "los_ojos.signals.trading"
    SIGNAL_FINAL        = "los_ojos.signals.trading"
    EXECUTION_INTENT    = "los_ojos.signals.trading"

    # ML features
    ML_FEATURE_VECTOR   = "los_ojos.ml.feature_vector"

    # Universe & Corporate Actions (Semana 4)
    CORPORATE_ACTIONS   = "los_ojos.corporate_actions"
    UNIVERSE_UPDATES    = "los_ojos.universe.updates"

    # System
    ANOMALY             = "los_ojos.context.anomaly"
    KILL_SWITCH         = "los_ojos.bot.kill_switch"
    CONTEXT_STATE       = "los_ojos.context.state"

    @classmethod
    def all_topics(cls) -> list[str]:
        return [v for k, v in vars(cls).items()
                if not k.startswith("_") and isinstance(v, str)]
