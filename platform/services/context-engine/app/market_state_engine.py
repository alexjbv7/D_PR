"""
MarketStateEngine — Construye el estado consolidado del mercado.

Agrega señales de múltiples fuentes en un snapshot único que consume
el strategy-orchestrator para tomar decisiones de portfolio:

  - Regime (GMM 5-component desde context-engine)
  - Liquidez (LiquidityScanner — squeeze risk)
  - Anomalías activas (AnomalyDetector)
  - Sentimiento whale (WhaleDetector)
  - Macro bias (MacroSignalEngine)
  - Señales Polymarket (ProbabilityAnalyzer)

Output: MarketState publicado a Redis (key `market:state`) con TTL 60s
y a Kafka `los_ojos.context.state`.

Decisiones de diseño:
  - "Composite risk score" 0-100: agrega todas las señales de riesgo.
  - score > 70 → no nuevas posiciones largas
  - score > 85 → activar defensive mode en el orchestrator
  - Cada componente tiene peso fijo; recalibración manual trimestral.
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
class MarketState:
    event_id:         str
    event_type:       str = "MarketStateEvent"
    ts:               str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Regime
    regime_id:        int   = 0        # 0-4 (GMM label)
    regime_label:     str   = "unknown"
    regime_confidence: float = 0.0

    # Liquidity / squeeze
    squeeze_risk:     str   = "none"   # none | low | medium | high | critical
    oi_z_score:       float = 0.0
    funding_z:        float = 0.0

    # Anomalies (count by severity)
    anomalies_high:   int   = 0
    anomalies_critical: int = 0
    active_anomaly_types: list[str] = field(default_factory=list)

    # Whale sentiment: +1 = accumulating, -1 = distributing
    whale_sentiment:  float = 0.0
    whale_confidence: float = 0.0

    # Macro
    macro_bias:       str   = "neutral"
    macro_leverage:   float = 1.0
    recession_prob:   float = 0.0

    # Polymarket
    btc_up_prob:      Optional[float] = None
    recession_market_prob: Optional[float] = None

    # Composite
    composite_risk_score: float = 0.0  # 0-100
    allow_new_longs:  bool  = True
    defensive_mode:   bool  = False
    summary:          str   = ""

    def to_dict(self) -> dict:
        return {
            "event_id":             self.event_id,
            "event_type":           self.event_type,
            "ts":                   self.ts,
            "regime_id":            self.regime_id,
            "regime_label":         self.regime_label,
            "regime_confidence":    round(self.regime_confidence, 3),
            "squeeze_risk":         self.squeeze_risk,
            "oi_z_score":           round(self.oi_z_score, 3),
            "funding_z":            round(self.funding_z, 3),
            "anomalies_high":       self.anomalies_high,
            "anomalies_critical":   self.anomalies_critical,
            "active_anomaly_types": self.active_anomaly_types,
            "whale_sentiment":      round(self.whale_sentiment, 3),
            "whale_confidence":     round(self.whale_confidence, 3),
            "macro_bias":           self.macro_bias,
            "macro_leverage":       round(self.macro_leverage, 2),
            "recession_prob":       round(self.recession_prob, 4),
            "btc_up_prob":          round(self.btc_up_prob, 4) if self.btc_up_prob else None,
            "recession_market_prob": round(self.recession_market_prob, 4) if self.recession_market_prob else None,
            "composite_risk_score": round(self.composite_risk_score, 1),
            "allow_new_longs":      self.allow_new_longs,
            "defensive_mode":       self.defensive_mode,
            "summary":              self.summary,
        }


class MarketStateEngine:
    """
    Builds and publishes consolidated MarketState from sub-system signals.

    Usage
    -----
    engine = MarketStateEngine(kafka_servers="...", redis_url="...")
    await engine.connect()
    state = engine.build(
        regime=regime_result,
        liquidity=liquidity_snap,
        anomalies=anomaly_list,
        macro=macro_signal,
        whale_sentiment=0.4,
    )
    await engine.publish(state)
    """

    # Risk score weights (sum ~ 100 in worst case)
    _WEIGHTS = {
        "squeeze_critical": 30,
        "squeeze_high":     20,
        "squeeze_medium":   10,
        "squeeze_low":       5,
        "anomaly_critical": 20,
        "anomaly_high":     10,
        "macro_strong_bearish": 20,
        "macro_bearish":    10,
        "recession_high":   15,   # rec_prob >= 0.65
        "recession_mid":     8,   # rec_prob >= 0.40
        "regime_bearish":   10,   # regime = recession / crisis
    }

    # Regime label → semantic name
    _REGIME_LABELS: dict[int, str] = {
        0: "bull_trend",
        1: "bear_trend",
        2: "high_volatility",
        3: "low_volatility",
        4: "crisis",
    }

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        redis_url:     str = "redis://localhost:6379/0",
        topic_out:     str = "los_ojos.context.state",
    ):
        self._kafka_servers = kafka_servers
        self._redis_url     = redis_url
        self._topic_out     = topic_out
        self._producer      = None
        self._redis         = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()
        except Exception as e:
            logger.warning("market_state_engine.kafka_unavailable", error=str(e))

        try:
            import aioredis
            self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
            logger.info("market_state_engine.connected")
        except Exception as e:
            logger.warning("market_state_engine.redis_unavailable", error=str(e))

    async def close(self) -> None:
        if self._producer: await self._producer.stop()
        if self._redis:    await self._redis.close()

    def build(
        self,
        regime=None,
        liquidity=None,
        anomalies: list | None = None,
        macro=None,
        whale_sentiment: float = 0.0,
        whale_confidence: float = 0.0,
        btc_up_prob: Optional[float] = None,
        recession_market_prob: Optional[float] = None,
    ) -> MarketState:
        """
        Build MarketState from component signals.

        Parameters
        ----------
        regime    : RegimeResult from RegimeClassifier
        liquidity : LiquiditySnapshot from LiquidityScanner
        anomalies : list[AnomalyEvent] currently active
        macro     : MacroSignal from MacroSignalEngine
        whale_*   : aggregated whale flow score
        """
        anomalies = anomalies or []

        # --- Regime ---
        regime_id    = int(getattr(regime, "regime_id", 0))
        regime_label = self._REGIME_LABELS.get(regime_id, "unknown")
        regime_conf  = float(getattr(regime, "confidence", 0.0))

        # Override label if provided directly
        if hasattr(regime, "label"):
            regime_label = regime.label

        # --- Liquidity ---
        squeeze   = getattr(liquidity, "squeeze_risk", "none") if liquidity else "none"
        oi_z      = float(getattr(liquidity, "oi_z_score", 0.0))
        funding_z = float(getattr(liquidity, "funding_z", 0.0))

        # --- Anomalies ---
        high_anoms     = sum(1 for a in anomalies if getattr(a, "severity", "") in ("high", "HIGH"))
        critical_anoms = sum(1 for a in anomalies if getattr(a, "severity", "") in ("critical", "CRITICAL"))
        anom_types     = list({getattr(a, "anomaly_type", "") for a in anomalies})

        # --- Macro ---
        macro_bias  = getattr(macro, "bias", "neutral")
        if hasattr(macro_bias, "value"):
            macro_bias = macro_bias.value   # unwrap Enum
        macro_lev   = float(getattr(macro, "leverage_adj", 1.0))
        rec_prob    = float(getattr(macro, "recession_prob", 0.0))

        # --- Composite risk score ---
        score = 0.0
        w = self._WEIGHTS

        # Squeeze contribution
        score += {
            "critical": w["squeeze_critical"],
            "high":     w["squeeze_high"],
            "medium":   w["squeeze_medium"],
            "low":      w["squeeze_low"],
        }.get(squeeze, 0)

        # Anomaly contribution
        score += critical_anoms * w["anomaly_critical"]
        score += high_anoms     * w["anomaly_high"]

        # Macro contribution
        score += {
            "strong_bearish": w["macro_strong_bearish"],
            "bearish":        w["macro_bearish"],
        }.get(macro_bias, 0)

        # Recession
        if rec_prob >= 0.65:
            score += w["recession_high"]
        elif rec_prob >= 0.40:
            score += w["recession_mid"]

        # Bearish regime
        if regime_label in ("bear_trend", "crisis"):
            score += w["regime_bearish"]

        # Polymarket recession signal
        poly_rec = recession_market_prob or 0.0
        if poly_rec >= 0.60:
            score += 8
        elif poly_rec >= 0.40:
            score += 4

        # Clamp to 0-100
        score = min(100.0, max(0.0, score))

        # Thresholds
        allow_new_longs = score <= 70
        defensive_mode  = score > 85

        summary = self._build_summary(
            regime_label, squeeze, macro_bias, rec_prob, score,
            defensive_mode, critical_anoms, high_anoms
        )

        state = MarketState(
            event_id=str(uuid.uuid4()),
            regime_id=regime_id,
            regime_label=regime_label,
            regime_confidence=round(regime_conf, 3),
            squeeze_risk=squeeze,
            oi_z_score=round(oi_z, 3),
            funding_z=round(funding_z, 3),
            anomalies_high=high_anoms,
            anomalies_critical=critical_anoms,
            active_anomaly_types=anom_types,
            whale_sentiment=round(float(whale_sentiment), 3),
            whale_confidence=round(float(whale_confidence), 3),
            macro_bias=macro_bias,
            macro_leverage=round(macro_lev, 2),
            recession_prob=round(rec_prob, 4),
            btc_up_prob=btc_up_prob,
            recession_market_prob=recession_market_prob,
            composite_risk_score=round(score, 1),
            allow_new_longs=allow_new_longs,
            defensive_mode=defensive_mode,
            summary=summary,
        )

        logger.info("market_state.built",
                    risk_score=round(score, 1),
                    regime=regime_label,
                    squeeze=squeeze,
                    defensive=defensive_mode)
        return state

    async def publish(self, state: MarketState) -> None:
        """Publish to Kafka and cache in Redis."""
        d = state.to_dict()

        if self._producer:
            try:
                await self._producer.send(self._topic_out, value=d)
            except Exception as e:
                logger.error("market_state.kafka_error", error=str(e))

        if self._redis:
            try:
                await self._redis.setex(
                    "market:state",
                    60,   # TTL 60s
                    json.dumps(d),
                )
                # Also store leverage for feature-streaming consumption
                await self._redis.setex(
                    "macro:leverage_adj",
                    300,
                    str(state.macro_leverage),
                )
                await self._redis.setex(
                    "context:regime_id",
                    300,
                    str(state.regime_id),
                )
            except Exception as e:
                logger.error("market_state.redis_error", error=str(e))

    @staticmethod
    def _build_summary(
        regime: str, squeeze: str, macro_bias: str,
        rec_prob: float, score: float,
        defensive: bool, crit: int, high: int,
    ) -> str:
        parts = [
            f"Regime={regime}",
            f"squeeze={squeeze}",
            f"macro={macro_bias}",
            f"rec={rec_prob:.0%}",
            f"risk={score:.0f}/100",
        ]
        if defensive:
            parts.append("⚠ DEFENSIVE MODE")
        if crit > 0:
            parts.append(f"{crit} CRITICAL anomalies")
        elif high > 0:
            parts.append(f"{high} high anomalies")
        return " | ".join(parts)
