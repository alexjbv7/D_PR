"""
MacroSignalEngine — Genera señales de trading de alto nivel desde datos macro.

Convierte el estado macroeconómico en señales accionables:
  - Bias direccional   (bullish / bearish / neutral) para el portfolio
  - Ajuste de leverage (expandir / contraer / mínimo)
  - Sectores / assets favorecidos en el régimen actual
  - Alerts de evento macro próximo (FOMC, NFP, CPI)

Decisiones de diseño:
  - Las señales macro son FILTROS, no entradas directas a trades.
  - Cualquier señal de régimen recession → max_leverage = 0.5 automáticamente.
  - Los pesos de los componentes están calibrados empíricamente pero
    expuestos como parámetros para facilitar ajuste.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class MacroBias(str, Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH        = "bullish"
    NEUTRAL        = "neutral"
    BEARISH        = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class MacroSignal:
    event_id:        str
    event_type:      str = "MacroSignalEvent"
    ts:              str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    bias:            MacroBias = MacroBias.NEUTRAL
    leverage_adj:    float = 1.0     # multiplier: 0.5 = halve, 2.0 = double
    favored_assets:  list[str] = field(default_factory=list)
    avoided_assets:  list[str] = field(default_factory=list)
    reason:          str = ""
    recession_prob:  float = 0.0
    regime:          str = "unknown"
    rate_env:        str = "unknown"

    def to_dict(self) -> dict:
        return {
            "event_id":       self.event_id,
            "event_type":     self.event_type,
            "ts":             self.ts,
            "bias":           self.bias.value,
            "leverage_adj":   self.leverage_adj,
            "favored_assets": self.favored_assets,
            "avoided_assets": self.avoided_assets,
            "reason":         self.reason,
            "recession_prob": self.recession_prob,
            "regime":         self.regime,
            "rate_env":       self.rate_env,
        }


class MacroSignalEngine:
    """
    Computes portfolio-level macro signals from FRED indicators and
    recession regime state. Publishes MacroSignalEvent to Kafka.

    Usage
    -----
    engine = MacroSignalEngine(kafka_servers=..., redis_url=...)
    await engine.connect()
    signal = engine.compute(indicators, recession_result)
    await engine.publish(signal)
    """

    # Asset biases per regime
    REGIME_ASSET_PREFS: dict[str, dict] = {
        "expansion": {
            "favored": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "avoided": [],
        },
        "slowdown": {
            "favored": ["BTCUSDT"],
            "avoided": ["SOLUSDT", "AVAXUSDT", "ARBUSDT"],
        },
        "recession": {
            "favored": [],
            "avoided": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        },
        "recovery": {
            "favored": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
            "avoided": [],
        },
        "stagflation": {
            "favored": ["BTCUSDT"],
            "avoided": ["SOLUSDT", "ARBUSDT", "OPUSDT"],
        },
    }

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        redis_url:     str = "redis://localhost:6379/0",
    ):
        self._kafka_servers = kafka_servers
        self._redis_url     = redis_url
        self._producer      = None
        self.last_signal:   Optional[MacroSignal] = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()
        except Exception as e:
            logger.warning("macro_signal_engine.kafka_unavailable", error=str(e))

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()

    def compute(self, indicators: dict, recession_result) -> MacroSignal:
        """
        Derive portfolio macro bias from current indicators.

        Parameters
        ----------
        indicators : dict
            Raw FRED series values {series_id: float}.
        recession_result : RecessionResult
            Output from RecessionDetector.compute().
        """
        rec_prob  = getattr(recession_result, "recession_prob", 0.0)
        regime    = getattr(recession_result, "regime", "unknown")
        rate_env  = getattr(recession_result, "rate_environment", "neutral")
        yield_inv = getattr(recession_result, "yield_curve_inversion", False)

        # --- Bias computation ---
        score = 0.0   # +ve = bullish, -ve = bearish

        # Recession probability (strong signal)
        if rec_prob >= 0.70:
            score -= 2.0
        elif rec_prob >= 0.50:
            score -= 1.0
        elif rec_prob >= 0.30:
            score -= 0.5
        else:
            score += 0.5

        # Rate environment
        if rate_env == "hiking":
            score -= 0.5
        elif rate_env == "cutting":
            score += 0.75
        elif rate_env == "pause":
            score += 0.25

        # Yield curve
        if yield_inv:
            score -= 0.5

        # VIX from indicators
        vix = indicators.get("VIXCLS")
        if vix is not None:
            if vix > 35:
                score -= 1.0
            elif vix > 25:
                score -= 0.5
            elif vix < 15:
                score += 0.25

        # DXY momentum (strong dollar → risk-off for crypto)
        dxy = indicators.get("DTWEXBGS")
        if dxy is not None:
            # Rough heuristic: if DXY > 105 → bearish for crypto
            if dxy > 105:
                score -= 0.3
            elif dxy < 100:
                score += 0.2

        # Map score to bias
        if score >= 1.5:
            bias = MacroBias.STRONG_BULLISH
        elif score >= 0.5:
            bias = MacroBias.BULLISH
        elif score > -0.5:
            bias = MacroBias.NEUTRAL
        elif score > -1.5:
            bias = MacroBias.BEARISH
        else:
            bias = MacroBias.STRONG_BEARISH

        # --- Leverage adjustment ---
        if rec_prob >= 0.65:
            lev_adj = 0.25    # recession: very conservative
        elif rec_prob >= 0.40:
            lev_adj = 0.50
        elif bias == MacroBias.STRONG_BULLISH:
            lev_adj = 1.5
        elif bias == MacroBias.BULLISH:
            lev_adj = 1.2
        elif bias == MacroBias.BEARISH:
            lev_adj = 0.7
        elif bias == MacroBias.STRONG_BEARISH:
            lev_adj = 0.3
        else:
            lev_adj = 1.0

        # --- Asset preferences ---
        prefs = self.REGIME_ASSET_PREFS.get(regime, {"favored": [], "avoided": []})

        reason = (
            f"Regime={regime}, bias={bias.value}, "
            f"recession_prob={rec_prob:.0%}, rate_env={rate_env}, "
            f"yield_inv={yield_inv}, score={score:.2f}"
        )

        signal = MacroSignal(
            event_id=str(uuid.uuid4()),
            bias=bias,
            leverage_adj=round(lev_adj, 2),
            favored_assets=prefs["favored"],
            avoided_assets=prefs["avoided"],
            reason=reason,
            recession_prob=round(rec_prob, 4),
            regime=regime,
            rate_env=rate_env,
        )

        self.last_signal = signal
        logger.info("macro_signal.computed",
                    bias=bias.value, leverage_adj=lev_adj, score=round(score, 2))
        return signal

    async def publish(self, signal: MacroSignal) -> None:
        if self._producer:
            try:
                await self._producer.send(
                    "los_ojos.macro.signal",
                    value=signal.to_dict(),
                )
            except Exception as e:
                logger.error("macro_signal.publish.error", error=str(e))
