"""
Recession Detector — Ensemble de señales de recesión.
=====================================================
Implementa tres métodos canónicos de detección temprana:

  1. Yield Curve (10Y-2Y inversión)
     - Inversión sostenida > 3 meses → señal fuerte (lag ~12-18 meses)
     - Threshold: < -0.25%

  2. Sahm Rule
     - Regla de Claudia Sahm: si la tasa de desempleo sube ≥ 0.50 pp
       sobre su mínimo de los últimos 12 meses → recesión en curso
     - Umbral original: 0.50

  3. Conference Board Leading Indicators (proxy via FRED components)
     - Media móvil de cambio mensual en leading indicators
     - 3 caídas consecutivas → alerta

  4. Ensemble (weighted average)
     - Combina las tres probabilidades con pesos ajustables

Outputs:
  - recession_probability [0,1]
  - regime: "expansion" | "slowdown" | "recession" | "recovery"
  - sahm_indicator: valor actual del Sahm index
  - yield_curve_inversion_days: días consecutivos invertida
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from libs.shared.events import RecessionAlertEvent, MacroRegimeEvent, KafkaTopics
from libs.shared.kafka_client import KafkaProducerClient
from libs.shared.redis_client import RedisCache

logger = logging.getLogger(__name__)

# Umbrales
YIELD_INVERSION_THRESHOLD = float("-0.25")    # %
SAHM_THRESHOLD            = 0.50              # pp
LEADING_DECLINE_PERIODS   = 3

# Pesos ensemble
WEIGHTS = {"yield_curve": 0.40, "sahm": 0.40, "leading": 0.20}


class RecessionDetector:
    """
    Calcula probabilidad de recesión a partir de series macro de FRED.

    Consume el snapshot del FredCollector (series en Redis).
    """

    def __init__(
        self,
        producer: KafkaProducerClient,
        cache: RedisCache,
    ):
        self._producer = producer
        self._cache    = cache
        self._history: dict[str, list] = {
            "t10y2y":  [],
            "unrate":  [],
            "leading": [],
        }
        self._inversion_days: int = 0
        self._last_regime:    str = "expansion"
        self._last_prob:      float = 0.0

    async def evaluate(self, macro_snapshot: dict) -> dict:
        """
        Evalúa el estado actual del ciclo económico.

        Parameters
        ----------
        macro_snapshot : dict {series_id: {value, z_score, ...}}
            Output de FredCollector.get_current_snapshot()

        Returns
        -------
        dict con recession_probability, regime, indicators
        """
        p_yield  = self._yield_curve_signal(macro_snapshot)
        p_sahm   = self._sahm_signal(macro_snapshot)
        p_lead   = self._leading_indicator_signal(macro_snapshot)

        # Ensemble ponderado
        p_recession = (
            WEIGHTS["yield_curve"] * p_yield +
            WEIGHTS["sahm"]        * p_sahm  +
            WEIGHTS["leading"]     * p_lead
        )
        p_recession = float(np.clip(p_recession, 0.0, 1.0))

        # Clasificar régimen
        regime = self._classify_regime(p_recession, macro_snapshot)

        result = {
            "recession_probability":   round(p_recession, 4),
            "regime":                  regime,
            "p_yield_curve":           round(p_yield, 4),
            "p_sahm":                  round(p_sahm, 4),
            "p_leading":               round(p_lead, 4),
            "yield_inversion_days":    self._inversion_days,
            "sahm_value":              self._get_sahm_value(macro_snapshot),
            "rate_environment":        self._classify_rate_env(macro_snapshot),
            "ts":                      datetime.now(tz=timezone.utc).isoformat(),
        }

        # Cache Redis
        await self._cache.set("macro:recession_signal", result, ttl=3600)

        # Emitir eventos Kafka si hay cambios significativos
        await self._emit_events(p_recession, regime, macro_snapshot)

        self._last_prob   = p_recession
        self._last_regime = regime
        return result

    # ── Yield Curve ────────────────────────────────────────────────────

    def _yield_curve_signal(self, snap: dict) -> float:
        """Probabilidad de recesión basada en inversión de curva."""
        t10y2y_data = snap.get("T10Y2Y", {})
        if not t10y2y_data:
            return 0.15  # prior neutro

        value = float(t10y2y_data.get("value", 0.0))
        self._history["t10y2y"].append(value)

        # Contar días de inversión
        if value < YIELD_INVERSION_THRESHOLD:
            self._inversion_days += 1
        else:
            self._inversion_days = max(0, self._inversion_days - 1)

        # Probabilidad escalonada
        if self._inversion_days >= 252:   # ~1 año
            return 0.90
        elif self._inversion_days >= 126:  # ~6 meses
            return 0.75
        elif self._inversion_days >= 63:   # ~3 meses
            return 0.55
        elif value < YIELD_INVERSION_THRESHOLD:
            return 0.40
        elif value < 0:
            return 0.25
        else:
            return 0.10

    # ── Sahm Rule ──────────────────────────────────────────────────────

    def _sahm_signal(self, snap: dict) -> float:
        """
        Indicador de Sahm: U(3m_avg) - U_min(12m).
        > 0.50 → recesión activa.
        """
        unrate_data = snap.get("UNRATE", {})
        if not unrate_data:
            return 0.15

        current = float(unrate_data.get("value", 4.0))
        self._history["unrate"].append(current)

        sahm = self._get_sahm_value(snap)
        if sahm is None:
            return 0.15

        # Probabilidad sigmoid
        p = 1 / (1 + np.exp(-5 * (sahm - 0.35)))
        return float(np.clip(p, 0.05, 0.95))

    def _get_sahm_value(self, snap: dict) -> Optional[float]:
        """Calcula el índice Sahm usando el historial disponible."""
        history = self._history["unrate"]
        if len(history) < 13:
            return None

        recent  = history[-3:]
        last_12 = history[-13:-1]

        u_avg_3m = sum(recent) / len(recent)
        u_min_12m = min(last_12)

        return round(u_avg_3m - u_min_12m, 4)

    # ── Leading Indicators ─────────────────────────────────────────────

    def _leading_indicator_signal(self, snap: dict) -> float:
        """
        Proxy de Conference Board Leading Indicators usando componentes FRED:
        - INDPRO (industrial production)
        - ICSA (initial claims, invertido)
        - Yield curve slope
        """
        changes = []

        for series_id in ["INDPRO", "RSXFS"]:
            data = snap.get(series_id, {})
            if data:
                mom = data.get("mom_pct", 0)
                if mom is not None:
                    changes.append(mom)

        if not changes:
            return 0.15

        avg_change = sum(changes) / len(changes)
        self._history["leading"].append(avg_change)

        # Contar caídas consecutivas
        recent = self._history["leading"][-LEADING_DECLINE_PERIODS:]
        n_declines = sum(1 for c in recent if c < 0)

        if n_declines >= LEADING_DECLINE_PERIODS:
            return 0.65
        elif n_declines == LEADING_DECLINE_PERIODS - 1:
            return 0.40
        else:
            return 0.15

    # ── Regime classification ──────────────────────────────────────────

    def _classify_regime(self, p_recession: float, snap: dict) -> str:
        """
        Clasifica el régimen económico actual.

        expansion  : p < 0.25, crecimiento
        slowdown   : 0.25 <= p < 0.50, desaceleración
        recession  : p >= 0.50
        recovery   : p bajando desde recesión
        """
        if p_recession >= 0.55:
            return "recession"
        elif p_recession >= 0.30:
            if self._last_regime == "recession":
                return "recovery"
            return "slowdown"
        else:
            if self._last_regime in ("recession", "slowdown"):
                return "recovery"
            return "expansion"

    def _classify_rate_env(self, snap: dict) -> str:
        """Clasifica el entorno de tipos de interés."""
        dff = snap.get("DFF", {})
        if not dff:
            return "hold"

        z = dff.get("z_score", 0.0)
        mom = dff.get("mom_pct", 0.0) or 0.0

        if mom > 0.05:
            return "hiking"
        elif mom < -0.05:
            return "cutting"
        return "hold"

    # ── Event emission ─────────────────────────────────────────────────

    async def _emit_events(
        self, p: float, regime: str, snap: dict
    ):
        """Emite eventos Kafka si hay alertas o cambio de régimen."""
        # Alerta de recesión si supera umbrales
        if p >= 0.50 and self._last_prob < 0.50:
            event = RecessionAlertEvent(
                source="recession-detector",
                probability=p,
                model="ensemble_v1",
                threshold_breached=0.50,
                severity="alert" if p >= 0.70 else "warning",
            )
            await self._producer.send(KafkaTopics.RECESSION_ALERT, event)
            logger.warning("RECESSION ALERT: p=%.2f regime=%s", p, regime)

        # Cambio de régimen
        if regime != self._last_regime:
            macro_features = {
                k: float(v.get("value", 0))
                for k, v in snap.items()
                if isinstance(v, dict) and "value" in v
            }
            event = MacroRegimeEvent(
                source="recession-detector",
                regime=regime,
                confidence=1.0 - abs(p - 0.50) * 2,
                dominant_signal=(
                    "yield_curve" if p >= 0.50 else "expansion_indicators"
                ),
                rate_environment=self._classify_rate_env(snap),
                features=macro_features,
            )
            await self._producer.send(KafkaTopics.MACRO_REGIME, event)
            logger.info("Macro regime change: %s → %s", self._last_regime, regime)
