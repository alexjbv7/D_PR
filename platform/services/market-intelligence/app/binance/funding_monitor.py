"""
Funding Monitor — Perpetual funding rate analysis + fair value estimation.
=========================================================================
Binance perpetuals tienen funding rate cada 8h.

Señales derivadas:
  - funding_z_score: z-score vs rolling 60-período histórico
    → valores extremos (>2 o <-2) señalan mean-reversion de basis
  - futures_basis_bps: (mark_price - spot_price) / spot_price * 10000
  - fair_value_adj: ajuste de fair value por basis y funding

Emite:
  - FundingRateEvent a Kafka
  - Features a Redis

Referencia: Basis trading en crypto perpetuals.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Optional

import aiohttp

from libs.shared.events import FundingRateEvent, FeatureUpdateEvent, KafkaTopics
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache, TTL

logger = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = os.getenv(
    "BINANCE_FUTURES_BASE", "https://fapi.binance.com/fapi/v1"
)
POLL_INTERVAL_S = int(os.getenv("FUNDING_POLL_INTERVAL_S", "60"))
HISTORY_WINDOW  = int(os.getenv("FUNDING_HISTORY_WINDOW", "60"))  # períodos


class FundingMonitor:
    """
    Poll periódico de funding rates de Binance USDT-M.

    funding_history[symbol] = deque de últimos HISTORY_WINDOW funding rates.
    Calcula z-score para detectar extremos y emitir señales.
    """

    def __init__(
        self,
        symbols: list[str],
        producer: KafkaProducerClient,
        cache: RedisCache,
        spot_prices: Optional[dict] = None,
    ):
        self._symbols = [s.upper() for s in symbols]
        self._producer = producer
        self._cache = cache
        self._spot_prices = spot_prices or {}
        self._funding_history: dict[str, deque] = {
            s: deque(maxlen=HISTORY_WINDOW) for s in self._symbols
        }

    async def run(self):
        """Loop de polling cada POLL_INTERVAL_S segundos."""
        logger.info("FundingMonitor started for %s", self._symbols)
        while True:
            try:
                await self._poll_all()
            except Exception as exc:
                logger.error("FundingMonitor poll error: %s", exc)
            await asyncio.sleep(POLL_INTERVAL_S)

    async def _poll_all(self):
        """Fetch funding rate para todos los símbolos."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._poll_symbol(session, s) for s in self._symbols]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_symbol(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch y procesa funding rate para un símbolo."""
        try:
            # Premium index contiene funding rate, mark price, index price
            url = f"{BINANCE_FUTURES_BASE}/premiumIndex?symbol={symbol}"
            async with session.get(url) as resp:
                data = await resp.json()

            if "lastFundingRate" not in data:
                return

            funding_rate = float(data["lastFundingRate"])
            mark_price   = float(data["markPrice"])
            index_price  = float(data["indexPrice"])
            next_ts_ms   = int(data["nextFundingTime"])
            next_ts      = datetime.fromtimestamp(next_ts_ms / 1000, tz=timezone.utc)

            # Persistir historial
            self._funding_history[symbol].append(funding_rate)

            # Kafka event
            event = FundingRateEvent(
                source="funding-monitor",
                symbol=symbol,
                venue="binance_futures",
                funding_rate=funding_rate,
                next_funding_ts=next_ts,
                mark_price=mark_price,
                index_price=index_price,
            )
            await self._producer.send(KafkaTopics.FUNDING_RATE, event, key=symbol)

            # Features
            features = self._compute_features(symbol, funding_rate, mark_price, index_price)
            cache_key = f"features:funding:{symbol}"
            await self._cache.hset_features(cache_key, features, ttl=TTL["funding"])

            # Feature update event
            feat_event = FeatureUpdateEvent(
                source="funding-monitor",
                symbol=symbol,
                timeframe="8h",
                feature_set="funding_v1",
                feature_set_hash="funding_v1_hash",
                features=features,
                bar_ts=datetime.now(tz=timezone.utc),
            )
            await self._producer.send(KafkaTopics.FEATURE_UPDATE, feat_event, key=symbol)

        except Exception as exc:
            logger.warning("Funding poll error %s: %s", symbol, exc)

    def _compute_features(
        self,
        symbol: str,
        funding_rate: float,
        mark_price: float,
        index_price: float,
    ) -> dict[str, float]:
        """Calcula features derivadas del funding rate."""
        history = list(self._funding_history[symbol])

        # Z-score
        if len(history) >= 3:
            mu = mean(history)
            sd = stdev(history) if len(history) > 1 else 1e-9
            z = (funding_rate - mu) / max(sd, 1e-9)
        else:
            z = 0.0

        # Basis
        basis_bps = (mark_price - index_price) / index_price * 10_000 if index_price > 0 else 0.0

        # Spot-futures basis (si tenemos spot price)
        spot = self._spot_prices.get(symbol, index_price)
        spot_basis_bps = (mark_price - spot) / spot * 10_000 if spot > 0 else 0.0

        # Funding premium anualizado (funding_rate * 3 * 365)
        annual_funding_pct = funding_rate * 3 * 365 * 100

        # Rolling stats
        rolling_mean = mean(history) if history else 0.0
        rolling_abs_mean = mean(abs(f) for f in history) if history else 0.0

        return {
            "funding_rate":          funding_rate,
            "funding_z_score":       round(z, 4),
            "funding_basis_bps":     round(basis_bps, 2),
            "spot_basis_bps":        round(spot_basis_bps, 2),
            "annual_funding_pct":    round(annual_funding_pct, 4),
            "funding_rolling_mean":  round(rolling_mean, 6),
            "funding_rolling_abs":   round(rolling_abs_mean, 6),
            "mark_index_ratio":      round(mark_price / index_price, 6) if index_price > 0 else 1.0,
        }

    def update_spot_price(self, symbol: str, price: float):
        """Actualizar precio spot para cálculo de basis."""
        self._spot_prices[symbol] = price
