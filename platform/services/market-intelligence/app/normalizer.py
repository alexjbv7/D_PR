"""
DataNormalizer — Normalización cross-source de datos de mercado.

Unifica datos de Binance, OpenBB, y fuentes on-chain en un esquema
canónico. Maneja:
  - OHLCV de diferentes exchanges con timestamps inconsistentes
  - Funding rates en distintos formatos (8h vs annualized)
  - Orderbook imbalance en distintas unidades
  - Cross-asset price normalization (USDT, USD, USDC base)

Salida: NormalizedMarketData publicado a Kafka `los_ojos.market.normalized`
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NormalizedOHLCV:
    symbol:      str
    ts:          str              # ISO-8601 UTC
    open:        float
    high:        float
    low:         float
    close:       float
    volume:      float            # base asset volume
    volume_usd:  float            # quote volume in USD
    vwap:        float            # volume-weighted avg price
    source:      str              # binance | coinbase | bybit | etc.

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class NormalizedTick:
    """Real-time tick with derived micro-structure fields."""
    event_id:       str
    event_type:     str = "NormalizedTick"
    ts:             str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    symbol:         str = ""
    price:          float = 0.0
    volume:         float = 0.0
    bid:            float = 0.0
    ask:            float = 0.0
    spread_bps:     float = 0.0
    ob_imbalance:   float = 0.0   # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    funding_rate:   float = 0.0   # 8h decimal
    open_interest:  float = 0.0   # USD notional
    source:         str = ""

    def to_dict(self) -> dict:
        return {
            "event_id":      self.event_id,
            "event_type":    self.event_type,
            "ts":            self.ts,
            "symbol":        self.symbol,
            "price":         self.price,
            "volume":        self.volume,
            "bid":           self.bid,
            "ask":           self.ask,
            "spread_bps":    self.spread_bps,
            "ob_imbalance":  self.ob_imbalance,
            "funding_rate":  self.funding_rate,
            "open_interest": self.open_interest,
            "source":        self.source,
        }


class DataNormalizer:
    """
    Normalizes heterogeneous market data into canonical schema.

    Usage
    -----
    norm = DataNormalizer(kafka_servers="...")
    await norm.connect()
    tick = norm.normalize_tick(raw_binance_event)
    await norm.publish(tick)
    """

    # Pairs that trade in USDC instead of USDT (map to canonical USDT equivalent)
    _USDC_ALIASES: dict[str, str] = {
        "BTCUSDC": "BTCUSDT",
        "ETHUSDC": "ETHUSDT",
        "SOLUSDC": "SOLUSDT",
    }

    # Funding rate multipliers to normalize to 8h decimal
    # (Binance already gives 8h; OKX gives 8h; Bybit gives 8h)
    _FUNDING_NORMALIZER: dict[str, float] = {
        "binance": 1.0,
        "okx":     1.0,
        "bybit":   1.0,
        "bitmex":  1.0 / 3,   # BitMEX pays 3× daily = 8h equivalent ÷3
    }

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        topic_out:     str = "los_ojos.market.normalized",
    ):
        self._kafka_servers = kafka_servers
        self._topic_out     = topic_out
        self._producer      = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()
            logger.info("normalizer.connected", topic=self._topic_out)
        except Exception as e:
            logger.warning("normalizer.kafka_unavailable", error=str(e))

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_tick(self, raw: dict, source: str = "binance") -> NormalizedTick:
        """
        Convert raw exchange WebSocket event to NormalizedTick.

        Handles Binance aggTrade, bookTicker, and markPrice formats.
        """
        symbol = self._canonical_symbol(raw.get("s", raw.get("symbol", "")))
        price  = self._safe_float(raw, ["p", "price", "c", "close"])
        volume = self._safe_float(raw, ["q", "volume", "v"])
        bid    = self._safe_float(raw, ["b", "bid", "B"])
        ask    = self._safe_float(raw, ["a", "ask", "A"])

        # Spread in basis points
        spread_bps = 0.0
        mid = (bid + ask) / 2
        if mid > 0 and bid > 0 and ask > 0:
            spread_bps = (ask - bid) / mid * 10_000

        # Orderbook imbalance from raw book levels if available
        ob_imbalance = self._compute_ob_imbalance(raw)

        # Funding rate — normalize to 8h decimal
        fr_raw = self._safe_float(raw, ["r", "fundingRate", "funding_rate"])
        funding_mult = self._FUNDING_NORMALIZER.get(source, 1.0)
        funding_rate = fr_raw * funding_mult

        # Open interest
        oi = self._safe_float(raw, ["openInterest", "open_interest", "oi"])

        tick = NormalizedTick(
            event_id=str(uuid.uuid4()),
            symbol=symbol,
            price=round(price, 8),
            volume=round(volume, 8),
            bid=round(bid, 8),
            ask=round(ask, 8),
            spread_bps=round(spread_bps, 4),
            ob_imbalance=round(ob_imbalance, 4),
            funding_rate=round(funding_rate, 8),
            open_interest=round(oi, 0),
            source=source,
        )
        return tick

    def normalize_ohlcv(self, raw: dict, source: str = "binance") -> NormalizedOHLCV:
        """
        Convert raw kline/candle to NormalizedOHLCV.

        Binance kline format: {"t": ts, "o": open, "h": high, "l": low,
                               "c": close, "v": volume, "q": quote_volume}
        """
        # Binance WebSocket kline wraps data under "k"
        k = raw.get("k", raw)

        symbol = self._canonical_symbol(k.get("s", raw.get("symbol", "")))
        ts_ms  = k.get("t", k.get("ts", k.get("open_time", 0)))
        ts_iso = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat() \
                 if ts_ms else datetime.now(timezone.utc).isoformat()

        o     = float(k.get("o", k.get("open",  0)))
        h     = float(k.get("h", k.get("high",  0)))
        low_  = float(k.get("l", k.get("low",   0)))
        c     = float(k.get("c", k.get("close", 0)))
        vol   = float(k.get("v", k.get("volume", 0)))
        q_vol = float(k.get("q", k.get("quote_volume", k.get("volume_usd", vol * c))))

        vwap = q_vol / vol if vol > 0 else c

        return NormalizedOHLCV(
            symbol=symbol,
            ts=ts_iso,
            open=round(o, 8),
            high=round(h, 8),
            low=round(low_, 8),
            close=round(c, 8),
            volume=round(vol, 8),
            volume_usd=round(q_vol, 2),
            vwap=round(vwap, 8),
            source=source,
        )

    async def publish(self, tick: NormalizedTick) -> None:
        if self._producer:
            try:
                await self._producer.send(self._topic_out, value=tick.to_dict())
            except Exception as e:
                logger.error("normalizer.publish.error", error=str(e))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _canonical_symbol(self, raw: str) -> str:
        """Map USDC variants → USDT canonical symbol."""
        upper = raw.upper().replace("-", "").replace("/", "")
        return self._USDC_ALIASES.get(upper, upper)

    @staticmethod
    def _safe_float(d: dict, keys: list[str]) -> float:
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return 0.0

    @staticmethod
    def _compute_ob_imbalance(raw: dict) -> float:
        """
        Compute (bid_vol - ask_vol) / total from top-of-book or
        level-2 data if available in raw message.
        """
        # Top-of-book quantities (Binance bookTicker: "B"=bestBidQty, "A"=bestAskQty)
        bq = raw.get("B") or raw.get("bid_qty") or raw.get("bids_qty")
        aq = raw.get("A") or raw.get("ask_qty") or raw.get("asks_qty")
        if bq is not None and aq is not None:
            try:
                b, a = float(bq), float(aq)
                total = b + a
                return (b - a) / (total + 1e-9) if total > 0 else 0.0
            except (TypeError, ValueError):
                pass

        # Level-2 snapshot: bids/asks lists of [price, qty]
        bids = raw.get("bids", [])
        asks = raw.get("asks", [])
        if bids and asks:
            try:
                bid_vol = sum(float(level[1]) for level in bids[:10])
                ask_vol = sum(float(level[1]) for level in asks[:10])
                total   = bid_vol + ask_vol
                return (bid_vol - ask_vol) / (total + 1e-9) if total > 0 else 0.0
            except (IndexError, TypeError, ValueError):
                pass

        return 0.0
