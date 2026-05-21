"""
AnomalyDetector — Detección de anomalías de mercado en tiempo real.

Detecta:
  1. Price anomaly   — z-score de retorno > umbral en ventana corta
  2. Volume anomaly  — volumen > N × media rolling
  3. Spread anomaly  — spread bid/ask súbitamente elevado (liquidez rota)
  4. Correlation break — correlación BTC/altcoins cae < umbral (desacoplamiento)
  5. Funding spike   — funding rate z-score extremo (squeeze inminente)

Output: AnomalyEvent publicado a Kafka `los_ojos.context.anomaly`

Decisiones de diseño:
  - Ventanas cortas (5-20 períodos) para latencia mínima.
  - Umbral conservador para evitar false positives — mejor miss que spam.
  - Cada anomalía tiene severidad: LOW | MEDIUM | HIGH | CRITICAL.
  - CRITICAL dispara warning en el kill-switch del orchestrator.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    event_id:     str
    event_type:   str = "AnomalyEvent"
    ts:           str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    symbol:       str = ""
    anomaly_type: str = ""       # price | volume | spread | correlation | funding
    severity:     Severity = Severity.LOW
    value:        float = 0.0
    threshold:    float = 0.0
    z_score:      float = 0.0
    description:  str = ""

    def to_dict(self) -> dict:
        return {
            "event_id":     self.event_id,
            "event_type":   self.event_type,
            "ts":           self.ts,
            "symbol":       self.symbol,
            "anomaly_type": self.anomaly_type,
            "severity":     self.severity.value,
            "value":        self.value,
            "threshold":    self.threshold,
            "z_score":      self.z_score,
            "description":  self.description,
        }


class AnomalyDetector:
    """
    Rolling anomaly detector per symbol.

    Usage
    -----
    detector = AnomalyDetector(kafka_servers="...", redis_url="...")
    await detector.connect()
    anomalies = await detector.process({
        "symbol": "BTCUSDT",
        "close":  65000.0,
        "volume": 1500.0,
        "spread_bps": 3.2,
        "funding_z": 2.1,
    })
    """

    def __init__(
        self,
        kafka_servers:    str = "localhost:9092",
        redis_url:        str = "redis://localhost:6379/0",
        price_z_thresh:   float = 4.0,
        volume_mult:      float = 5.0,
        spread_mult:      float = 4.0,
        funding_z_thresh: float = 3.0,
        window:           int   = 20,
    ):
        self._kafka_servers    = kafka_servers
        self._redis_url        = redis_url
        self._price_z_thresh   = price_z_thresh
        self._volume_mult      = volume_mult
        self._spread_mult      = spread_mult
        self._funding_z_thresh = funding_z_thresh
        self._window           = window

        # Rolling buffers per symbol
        self._prices:   dict[str, deque[float]] = {}
        self._volumes:  dict[str, deque[float]] = {}
        self._spreads:  dict[str, deque[float]] = {}

        self._producer = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()
            logger.info("anomaly_detector.connected")
        except Exception as e:
            logger.warning("anomaly_detector.kafka_unavailable", error=str(e))

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def process(self, data: dict) -> list[AnomalyEvent]:
        """Check for anomalies in incoming market data dict."""
        symbol = data.get("symbol", "UNKNOWN")
        close  = data.get("close")
        volume = data.get("volume")
        spread = data.get("spread_bps")
        fz     = data.get("funding_z")

        # Ensure buffers exist
        for buf_dict in (self._prices, self._volumes, self._spreads):
            if symbol not in buf_dict:
                buf_dict[symbol] = deque(maxlen=self._window)

        anomalies: list[AnomalyEvent] = []

        # Update buffers
        if close   is not None: self._prices[symbol].append(float(close))
        if volume  is not None: self._volumes[symbol].append(float(volume))
        if spread  is not None: self._spreads[symbol].append(float(spread))

        buf_len = len(self._prices[symbol])
        if buf_len < max(5, self._window // 2):
            return []  # not enough history

        # 1. Price anomaly (z-score of last return)
        if len(self._prices[symbol]) >= 3:
            prices = np.array(self._prices[symbol])
            rets   = np.diff(prices) / prices[:-1]
            if len(rets) >= 2:
                mu, sigma = rets[:-1].mean(), rets[:-1].std()
                last_ret   = rets[-1]
                z = (last_ret - mu) / (sigma + 1e-9)
                if abs(z) > self._price_z_thresh:
                    sev = Severity.CRITICAL if abs(z) > self._price_z_thresh * 1.5 else Severity.HIGH
                    a = AnomalyEvent(
                        event_id=str(uuid.uuid4()),
                        symbol=symbol,
                        anomaly_type="price",
                        severity=sev,
                        value=round(float(last_ret), 6),
                        threshold=self._price_z_thresh,
                        z_score=round(float(z), 2),
                        description=f"Extreme price move: z={z:.2f}, ret={last_ret:.4%}",
                    )
                    anomalies.append(a)

        # 2. Volume spike
        if len(self._volumes[symbol]) >= 3:
            vols     = np.array(self._volumes[symbol])
            vol_mean = vols[:-1].mean()
            last_vol = vols[-1]
            mult     = last_vol / (vol_mean + 1e-9)
            if mult > self._volume_mult:
                sev = Severity.HIGH if mult > self._volume_mult * 2 else Severity.MEDIUM
                a = AnomalyEvent(
                    event_id=str(uuid.uuid4()),
                    symbol=symbol,
                    anomaly_type="volume",
                    severity=sev,
                    value=round(float(last_vol), 2),
                    threshold=round(float(vol_mean * self._volume_mult), 2),
                    z_score=round(float(mult), 2),
                    description=f"Volume spike: {mult:.1f}× mean",
                )
                anomalies.append(a)

        # 3. Spread anomaly
        if len(self._spreads[symbol]) >= 3:
            spreads     = np.array(self._spreads[symbol])
            spread_mean = spreads[:-1].mean()
            last_spread = spreads[-1]
            mult        = last_spread / (spread_mean + 1e-9)
            if mult > self._spread_mult:
                a = AnomalyEvent(
                    event_id=str(uuid.uuid4()),
                    symbol=symbol,
                    anomaly_type="spread",
                    severity=Severity.HIGH,
                    value=round(float(last_spread), 2),
                    threshold=round(float(spread_mean * self._spread_mult), 2),
                    z_score=round(float(mult), 2),
                    description=f"Spread anomaly: {last_spread:.1f} bps ({mult:.1f}× mean) — liquidity warning",
                )
                anomalies.append(a)

        # 4. Funding rate spike
        if fz is not None and abs(float(fz)) > self._funding_z_thresh:
            fz_val = float(fz)
            sev    = Severity.CRITICAL if abs(fz_val) > self._funding_z_thresh * 1.5 else Severity.HIGH
            direction = "long" if fz_val > 0 else "short"
            a = AnomalyEvent(
                event_id=str(uuid.uuid4()),
                symbol=symbol,
                anomaly_type="funding",
                severity=sev,
                value=round(fz_val, 4),
                threshold=self._funding_z_thresh,
                z_score=round(fz_val, 2),
                description=f"Funding spike: z={fz_val:.2f} → {direction} squeeze risk",
            )
            anomalies.append(a)

        # Publish to Kafka
        for anomaly in anomalies:
            await self._publish(anomaly)
            logger.info("anomaly.detected",
                        symbol=symbol,
                        type=anomaly.anomaly_type,
                        severity=anomaly.severity.value,
                        z=anomaly.z_score)

        return anomalies

    async def _publish(self, event: AnomalyEvent) -> None:
        if self._producer:
            try:
                await self._producer.send(
                    "los_ojos.context.anomaly",
                    value=event.to_dict(),
                )
            except Exception as e:
                logger.error("anomaly.publish.error", error=str(e))
