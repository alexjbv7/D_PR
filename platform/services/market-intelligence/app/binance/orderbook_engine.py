"""
Binance Orderbook Engine — LOB analysis + fair value + imbalance detection.
===========================================================================
Consume el WS diff stream de Binance, mantiene el LOB en memoria y calcula:

  - Imbalance top-N (bids vs asks por volumen)
  - Weighted mid-price (Stoikov)
  - Spread y profundidad
  - Queue pressure
  - Fair value estimado (mid + imbalance adj)

Anti-leakage: este módulo solo consume datos de mercado y emite features;
no toma decisiones de trading.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import websockets
from pydantic import BaseModel

from libs.shared.events import (
    FeatureUpdateEvent, OrderBookEvent, KafkaTopics
)
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache, TTL

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = os.getenv(
    "BINANCE_WS_BASE", "wss://stream.binance.com:9443/ws"
)
BINANCE_REST_BASE = os.getenv(
    "BINANCE_REST_BASE", "https://api.binance.com/api/v3"
)


class OrderBookLevel(BaseModel):
    price: float
    qty: float


class LOBSnapshot:
    """
    Libro de órdenes en memoria con actualización incremental.

    Mantiene bids (desc) y asks (asc) como dicts price→qty.
    Aplica diff updates del WS stream.
    """

    def __init__(self, symbol: str, depth: int = 50):
        self.symbol = symbol
        self.depth = depth
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.ts: float = time.time()

    def apply_snapshot(self, data: dict):
        self.bids = {float(p): float(q) for p, q in data["bids"]}
        self.asks = {float(p): float(q) for p, q in data["asks"]}
        self.last_update_id = data.get("lastUpdateId", 0)
        self.ts = time.time()

    def apply_diff(self, data: dict):
        """Apply incremental update from depth diff stream."""
        for p, q in data.get("b", []):  # bids update
            p, q = float(p), float(q)
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q

        for p, q in data.get("a", []):  # asks update
            p, q = float(p), float(q)
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

        self.last_update_id = data.get("u", self.last_update_id)
        self.ts = time.time()

    # ── Analytics ────────────────────────────────────────────────────────

    def top_n(self, n: int = 5) -> tuple[list, list]:
        """Returns top-n bids (desc) and top-n asks (asc)."""
        top_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:n]
        top_asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return top_bids, top_asks

    def best_bid(self) -> float:
        return max(self.bids.keys()) if self.bids else 0.0

    def best_ask(self) -> float:
        return min(self.asks.keys()) if self.asks else 0.0

    def mid_price(self) -> float:
        bb = self.best_bid()
        ba = self.best_ask()
        return (bb + ba) / 2 if bb > 0 and ba > 0 else 0.0

    def spread(self) -> float:
        bb = self.best_bid()
        ba = self.best_ask()
        return (ba - bb) if bb > 0 and ba > 0 else 0.0

    def spread_bps(self) -> float:
        mid = self.mid_price()
        return (self.spread() / mid * 10_000) if mid > 0 else 0.0

    def imbalance(self, n: int = 5) -> float:
        """
        Orderbook imbalance top-N.
        = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Range [-1, +1]. Positivo → más presión compradora.
        """
        top_bids, top_asks = self.top_n(n)
        bid_vol = sum(q for _, q in top_bids)
        ask_vol = sum(q for _, q in top_asks)
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def weighted_mid(self, n: int = 5) -> float:
        """
        Weighted mid-price (Stoikov): pondera mid por imbalance.
        WM = ask_price * V_bid/(V_bid+V_ask) + bid_price * V_ask/(V_bid+V_ask)
        """
        top_bids, top_asks = self.top_n(1)
        if not top_bids or not top_asks:
            return self.mid_price()

        bid_p, bid_q = top_bids[0]
        ask_p, ask_q = top_asks[0]
        total = bid_q + ask_q
        if total == 0:
            return self.mid_price()
        return ask_p * (bid_q / total) + bid_p * (ask_q / total)

    def depth_total(self, n: int = 10) -> float:
        """Total volume USD in top-N levels."""
        top_bids, top_asks = self.top_n(n)
        mid = self.mid_price()
        return sum(p * q for p, q in top_bids + top_asks) / (mid or 1)

    def compute_features(self) -> dict[str, float]:
        """Returns all LOB features as a flat dict."""
        top5_imb = self.imbalance(5)
        top10_imb = self.imbalance(10)
        return {
            "lob_best_bid":         self.best_bid(),
            "lob_best_ask":         self.best_ask(),
            "lob_mid_price":        self.mid_price(),
            "lob_spread_bps":       self.spread_bps(),
            "lob_imbalance_5":      top5_imb,
            "lob_imbalance_10":     top10_imb,
            "lob_weighted_mid":     self.weighted_mid(),
            "lob_depth_total_10":   self.depth_total(10),
            "lob_bid_ask_ratio":    (
                sum(q for _, q in self.top_n(5)[0]) /
                max(sum(q for _, q in self.top_n(5)[1]), 1e-9)
            ),
        }


class OrderbookEngine:
    """
    Suscribe al depth stream de Binance y mantiene el LOB actualizado.

    Emite:
      - OrderBookEvent a Kafka (raw snapshot periódico)
      - FeatureUpdateEvent a Kafka (LOB features cada N updates)
      - Feature vector a Redis (para ml-inference)
    """

    def __init__(
        self,
        symbols: list[str],
        producer: KafkaProducerClient,
        cache: RedisCache,
        feature_emit_every: int = 10,   # emitir features cada N updates
        depth_levels: int = 20,
    ):
        self._symbols = [s.lower() for s in symbols]
        self._producer = producer
        self._cache = cache
        self._emit_every = feature_emit_every
        self._depth = depth_levels
        self._lobs: dict[str, LOBSnapshot] = {
            s.upper(): LOBSnapshot(s.upper(), depth_levels)
            for s in symbols
        }
        self._update_counts: dict[str, int] = defaultdict(int)

    async def run(self):
        """Inicia streams para todos los símbolos."""
        tasks = [self._stream_symbol(s) for s in self._symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _stream_symbol(self, symbol: str):
        """WS diff depth stream para un símbolo."""
        stream = f"{symbol}@depth@100ms"
        uri = f"{BINANCE_WS_BASE}/{stream}"
        symbol_upper = symbol.upper()

        # 1. Fetch snapshot inicial via REST
        await self._fetch_initial_snapshot(symbol_upper)

        while True:
            try:
                async with websockets.connect(uri, ping_interval=20) as ws:
                    logger.info("LOB stream connected: %s", symbol_upper)
                    async for raw_msg in ws:
                        data = json.loads(raw_msg)
                        lob = self._lobs[symbol_upper]
                        lob.apply_diff(data)
                        self._update_counts[symbol_upper] += 1

                        # Emitir features periódicamente
                        if self._update_counts[symbol_upper] % self._emit_every == 0:
                            await self._emit_features(symbol_upper, lob)

            except (websockets.ConnectionClosed, ConnectionError) as exc:
                logger.warning("LOB stream disconnected %s: %s. Reconnecting...", symbol, exc)
                await asyncio.sleep(2)
            except Exception as exc:
                logger.error("LOB stream error %s: %s", symbol, exc)
                await asyncio.sleep(5)

    async def _fetch_initial_snapshot(self, symbol: str):
        """Carga snapshot inicial via REST de Binance."""
        import aiohttp
        url = f"{BINANCE_REST_BASE}/depth?symbol={symbol}&limit={self._depth}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    self._lobs[symbol].apply_snapshot(data)
                    logger.info("LOB snapshot loaded: %s depth=%d", symbol, self._depth)
        except Exception as exc:
            logger.error("LOB snapshot failed %s: %s", symbol, exc)

    async def _emit_features(self, symbol: str, lob: LOBSnapshot):
        """Emite LOB features a Kafka y Redis."""
        features = lob.compute_features()

        # Redis (online feature store)
        cache_key = f"features:lob:{symbol}"
        await self._cache.hset_features(cache_key, features, ttl=TTL["feature"])

        # Kafka
        event = FeatureUpdateEvent(
            source="orderbook-engine",
            symbol=symbol,
            timeframe="tick",
            feature_set="orderflow_v1",
            feature_set_hash="orderflow_v1_hash",
            features=features,
            bar_ts=datetime.now(tz=timezone.utc),
        )
        await self._producer.send(
            KafkaTopics.FEATURE_UPDATE, event, key=symbol
        )

    def get_snapshot(self, symbol: str) -> Optional[LOBSnapshot]:
        return self._lobs.get(symbol.upper())
