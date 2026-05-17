"""
Multi-Asset Position Sizing
===========================
Position sizing en UNIDADES REALES (lots para FX, contratos para futuros), no
en "fracción de equity".

PRINCIPIO ATR-based:
   risk_usd = equity × risk_pct
   stop_distance_usd = stop_distance_in_price × usd_per_unit_per_price_point
                       × usd_conversion_factor
   n_units = risk_usd / stop_distance_usd
   n_units = round_to_min_increment(n_units)

Esto garantiza que la pérdida máxima por trade ≈ risk_pct del equity,
independientemente del instrumento.

Cap adicional: respeta `max_units_per_trade` (anti-blow-up).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from instruments.specs import InstrumentSpec


# =====================================================================
# ATR
# =====================================================================

def compute_atr(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's smoothing aproximada con SMA para simplicidad).

    True Range = max(high - low, |high - close_prev|, |low - close_prev|)
    """
    high = prices["high"]
    low = prices["low"]
    close_prev = prices["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing ≈ EMA con alpha=1/period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


# =====================================================================
# ATR-BASED SIZER
# =====================================================================

@dataclass
class ATRRiskSizer:
    """
    Sizer basado en ATR + risk-per-trade.

    Parameters
    ----------
    risk_pct : float
        Fracción del equity arriesgada por trade (e.g., 0.005 = 0.5%).
    atr_stop_mult : float
        Distancia del stop en múltiplos de ATR (e.g., 2.0 = stop a 2 ATR).
    instrument : InstrumentSpec
        Instrumento a operar.
    max_units_per_trade : float | None
        Cap absoluto por trade (e.g., 5 contratos ES máximo).
    daily_loss_pct_pause : float
        Si pérdida acumulada del día ≥ este %, retorna 0 (pausa).
    """
    risk_pct: float = 0.005           # 0.5% por trade
    atr_stop_mult: float = 2.0
    instrument: InstrumentSpec = None  # type: ignore
    max_units_per_trade: Optional[float] = None
    daily_loss_pct_pause: float = 0.03  # -3% diario → pausa

    def __post_init__(self):
        if self.instrument is None:
            raise ValueError("instrument es obligatorio en ATRRiskSizer")
        # Estado interno para daily loss
        self._day_start_equity: float = 0.0
        self._current_day = None

    def _check_daily_pause(self, current_equity: float, current_date) -> bool:
        """True si trading está pausado por daily loss limit."""
        if self._current_day != current_date:
            self._current_day = current_date
            self._day_start_equity = current_equity
            return False
        if self._day_start_equity <= 0:
            return False
        loss_pct = (self._day_start_equity - current_equity) / self._day_start_equity
        return loss_pct >= self.daily_loss_pct_pause

    def __call__(
        self,
        signal: float,
        current_equity: float,
        current_price: float,
        current_atr: float,
        current_date=None,
        usd_conversion_factor: float = 1.0,
    ) -> float:
        """
        Calcula n_units objetivo (signed: positivo=long, negativo=short, 0=flat).

        Parameters
        ----------
        signal : float
            +1, 0, -1 (o probabilístico continuo).
        current_equity : float
            Equity actual en USD.
        current_price : float
            Precio del instrumento.
        current_atr : float
            ATR actual (en unidades de precio).
        current_date : optional
            Para daily loss tracking.
        usd_conversion_factor : float
            Factor para pairs no-USD-quoted. 1.0 para EURUSD, GBPUSD, ES, NQ.
            Para USDJPY: 1.0 / current_price.

        Returns
        -------
        float : n_units (signed, redondeado a min_size_increment válido).
        """
        # Sin señal o sin ATR → flat
        if signal == 0 or pd.isna(current_atr) or current_atr <= 0:
            return 0.0

        # Daily loss check
        if current_date is not None and self._check_daily_pause(
            current_equity, current_date
        ):
            return 0.0

        # Stop distance en USD
        stop_distance_price = self.atr_stop_mult * current_atr
        usd_per_unit_at_stop = (
            stop_distance_price
            * self.instrument.usd_per_unit_per_price_point
            * usd_conversion_factor
        )

        if usd_per_unit_at_stop <= 0:
            return 0.0

        risk_usd = current_equity * self.risk_pct
        n_units_raw = risk_usd / usd_per_unit_at_stop

        # Aplicar cap
        if self.max_units_per_trade is not None:
            n_units_raw = min(n_units_raw, self.max_units_per_trade)

        # Redondear hacia abajo a incremento válido
        n_units = self.instrument.round_to_min_increment(n_units_raw)

        # Aplicar dirección de señal
        return np.sign(signal) * n_units


# =====================================================================
# SIMPLE FIXED SIZER (para testing / debugging)
# =====================================================================

@dataclass
class FixedUnitsSizer:
    """Sizer trivial: n_units constante. Útil para tests y debugging."""
    n_units: float = 1.0
    instrument: InstrumentSpec = None  # type: ignore

    def __call__(
        self,
        signal: float,
        current_equity: float,
        current_price: float,
        current_atr: float,
        current_date=None,
        usd_conversion_factor: float = 1.0,
    ) -> float:
        if signal == 0:
            return 0.0
        return np.sign(signal) * self.n_units
