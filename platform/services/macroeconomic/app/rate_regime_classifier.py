"""
RateRegimeClassifier — Clasifica el entorno de tasas de la Fed.

Estados posibles:
  hiking    — Fed subiendo tasas activamente (hawkish)
  pausing   — Fed en pausa tras ciclo de alzas
  cutting   — Fed recortando tasas (dovish)
  neutral   — Sin sesgo claro, tasas estables por largo tiempo
  emergency — Recorte de emergencia (crisis)

Señales usadas:
  - FEDFUNDS: tasa de fondos federales actual
  - DFF: daily effective rate
  - T10Y2Y: pendiente de la curva (proxy forward-looking)
  - UMCSENT: sentimiento del consumidor (señal suavizada)
  - Velocidad de cambio de FEDFUNDS (3m momentum)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RateRegimeResult:
    environment:     str            # hiking | pausing | cutting | neutral | emergency
    current_rate:    float
    rate_momentum:   float          # bps/month  (+ve = hiking, -ve = cutting)
    is_inverted:     bool
    t10y2y:          Optional[float]
    description:     str


class RateRegimeClassifier:
    """
    Classifies Fed rate regime from FRED series.

    Usage
    -----
    clf = RateRegimeClassifier()
    result = clf.classify(indicators)
    """

    def classify(self, indicators: dict[str, float]) -> RateRegimeResult:
        """
        Parameters
        ----------
        indicators : dict
            Expects keys: FEDFUNDS, DFF, T10Y2Y, and optionally
            FEDFUNDS_3M_AGO, FEDFUNDS_6M_AGO (computed by FredCollector).
        """
        rate_now   = indicators.get("FEDFUNDS") or indicators.get("DFF") or 0.0
        rate_3m    = indicators.get("FEDFUNDS_3M_AGO", rate_now)
        rate_6m    = indicators.get("FEDFUNDS_6M_AGO", rate_now)
        t10y2y     = indicators.get("T10Y2Y")
        is_inverted = (t10y2y is not None and t10y2y < -0.10)

        # Momentum in bps/month
        momentum_3m = (rate_now - rate_3m) * 100 / 3     # bps per month
        momentum_6m = (rate_now - rate_6m) * 100 / 6

        # Classify
        if momentum_3m > 8:                          # >25 bps/quarter
            env = "hiking"
            desc = f"Active hiking cycle: +{momentum_3m:.0f} bps/mo"
        elif momentum_3m < -10:                      # emergency or fast cut
            if rate_now < rate_6m * 0.75:
                env = "emergency"
                desc = f"Emergency easing: rate dropped {(rate_now/rate_6m - 1)*100:.0f}% in 6m"
            else:
                env = "cutting"
                desc = f"Cutting cycle: {momentum_3m:.0f} bps/mo"
        elif momentum_3m < -3:                       # gradual cuts
            env = "cutting"
            desc = f"Gradual cutting: {momentum_3m:.0f} bps/mo"
        elif abs(momentum_3m) <= 3 and abs(momentum_6m) <= 5:
            # Stable — distinguish pause (recent activity) from neutral (long stable)
            if abs(rate_now - rate_6m) > 0.25:       # recently moved, now paused
                env = "pausing"
                desc = f"Post-hike pause at {rate_now:.2f}%"
            else:
                env = "neutral"
                desc = f"Rate stable at {rate_now:.2f}% for extended period"
        else:
            env = "neutral"
            desc = f"Mixed signals: 3m_mom={momentum_3m:.1f}, 6m_mom={momentum_6m:.1f}"

        # Override: if very inverted curve AND hiking → stress signal
        if is_inverted and env == "hiking":
            desc += " | CURVE DEEPLY INVERTED — recession risk high"

        result = RateRegimeResult(
            environment=env,
            current_rate=round(rate_now, 4),
            rate_momentum=round(momentum_3m, 2),
            is_inverted=is_inverted,
            t10y2y=round(t10y2y, 4) if t10y2y is not None else None,
            description=desc,
        )

        logger.info("rate_regime.classified",
                    env=env, rate=rate_now, mom=momentum_3m,
                    inverted=is_inverted)
        return result
