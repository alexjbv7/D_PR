"""
Dynamic Risk/Reward basado en Confianza del Modelo
====================================================
Ajusta el ratio TP/SL por trade según la probabilidad calibrada del modelo.

CONCEPTO CENTRAL:
  No todos los trades tienen el mismo potencial. Cuando el modelo tiene
  alta confianza (P(win)=0.80), el precio tiene más probabilidad de moverse
  hacia el objetivo → podemos aspirar a un TP más lejano.

  Con baja confianza (P(win)=0.42), mejor un TP conservador — si lo ponemos
  muy lejos, el mercado probablemente dará la vuelta antes de alcanzarlo.

MECÁNICA:
  SL_distance  = atr_sl_mult × ATR          ← siempre fijo (stop estructural)
  rr_dynamic   = f(P(win))                  ← varía de rr_min a rr_max
  TP_distance  = rr_dynamic × SL_distance   ← se mueve con la confianza

  Para long:
    SL_price = entry_price - SL_distance
    TP_price = entry_price + TP_distance

  Para short:
    SL_price = entry_price + SL_distance
    TP_price = entry_price - TP_distance

FORMAS DE MAPEO P(win) → R:R:

  'linear':
    Interpolación lineal entre [p_low, p_high] → [rr_min, rr_max].
    Simple. Puede ser inestable cerca de los bordes.

  'sigmoid':
    Transición suave tipo S. Meseta en valores extremos de P(win).
    Recomendado en producción: evita cambios bruscos de TP/SL por pequeñas
    variaciones de probabilidad.

  'stepped':
    3 niveles discretos (bajo / medio / alto) según la confianza.
    Más fácil de comunicar y monitorear operativamente.

COHERENCIA CON KELLY:
  El R:R dinámico no es solo para el TP — también se alimenta a Kelly:
    kelly_fraction_binary(p_win, rr=rr_dynamic)

  Resultado: alta confianza → R:R mayor → Kelly más agresivo → más tamaño.
  Baja confianza → R:R menor → Kelly más conservador → menos tamaño (o 0).

  Esto crea un sistema autoregulado donde el sizing y los niveles de salida
  están ambos determinados por la misma probabilidad calibrada.

PARÁMETROS CLAVE (sensibles):
  rr_min : demasiado bajo (< 0.8) → la mayoría de trades con EV negativo.
  rr_max : demasiado alto (> 5.0) → TP raramente se alcanza → win rate colapsa.
  p_low  : punto donde empezamos a escalar R:R. Debe ser ≥ threshold del filtro.
  p_high : punto de R:R máximo. No poner muy alto (0.90+) — raramente se alcanza.

  Valores conservadores recomendados para empezar:
    rr_min=1.2, rr_max=2.5, p_low=0.45, p_high=0.75, shape='sigmoid'
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# FUNCIÓN PURA: P(win) → R:R
# =====================================================================

def compute_dynamic_rr(
    p_win: float,
    rr_min: float = 1.2,
    rr_max: float = 2.5,
    p_low: float = 0.45,
    p_high: float = 0.75,
    shape: Literal["linear", "sigmoid", "stepped"] = "sigmoid",
) -> float:
    """
    Mapea P(win) ∈ [0,1] a un ratio R:R dinámico ∈ [rr_min, rr_max].

    Parameters
    ----------
    p_win : float
        Probabilidad calibrada de que el trade sea ganador.
    rr_min : float
        R:R mínimo (se aplica cuando p_win ≤ p_low).
    rr_max : float
        R:R máximo (se aplica cuando p_win ≥ p_high).
    p_low : float
        Umbral inferior de confianza: por debajo → rr_min.
    p_high : float
        Umbral superior de confianza: por encima → rr_max.
    shape : 'linear' | 'sigmoid' | 'stepped'
        Forma del mapeo.

    Returns
    -------
    float : R:R en [rr_min, rr_max].
    """
    if rr_min <= 0:
        raise ValueError(f"rr_min debe ser > 0, recibido: {rr_min}")
    if rr_max < rr_min:
        raise ValueError(f"rr_max ({rr_max}) debe ser >= rr_min ({rr_min})")
    if not (0 <= p_low < p_high <= 1):
        raise ValueError(
            f"Se requiere 0 <= p_low ({p_low}) < p_high ({p_high}) <= 1"
        )

    p_win = float(np.clip(p_win, 0.0, 1.0))

    if shape == "linear":
        return _rr_linear(p_win, rr_min, rr_max, p_low, p_high)
    elif shape == "sigmoid":
        return _rr_sigmoid(p_win, rr_min, rr_max, p_low, p_high)
    elif shape == "stepped":
        return _rr_stepped(p_win, rr_min, rr_max, p_low, p_high)
    else:
        raise ValueError(f"shape debe ser 'linear', 'sigmoid' o 'stepped'. Recibido: '{shape}'")


def _rr_linear(p_win, rr_min, rr_max, p_low, p_high):
    """Interpolación lineal entre p_low y p_high."""
    if p_win <= p_low:
        return float(rr_min)
    if p_win >= p_high:
        return float(rr_max)
    t = (p_win - p_low) / (p_high - p_low)
    return float(rr_min + t * (rr_max - rr_min))


def _rr_sigmoid(p_win, rr_min, rr_max, p_low, p_high):
    """
    Mapeo suave tipo S (sigmoid escalada).
    La transición se centra en (p_low + p_high) / 2.
    La "steepness" se calibra para que en p_low el resultado sea ≈ rr_min
    y en p_high sea ≈ rr_max (dentro del 5%).
    """
    center = (p_low + p_high) / 2.0
    # Steepness: elegida para que sigmoid(p_high) ≈ 0.95 y sigmoid(p_low) ≈ 0.05
    width = p_high - p_low
    if width < 1e-6:
        return float(rr_min) if p_win < center else float(rr_max)
    # log(0.95/0.05) ≈ 2.944; steepness = logit(0.95) / (width/2)
    steepness = 2.944 / (width / 2.0)
    sig = 1.0 / (1.0 + np.exp(-steepness * (p_win - center)))
    return float(rr_min + sig * (rr_max - rr_min))


def _rr_stepped(p_win, rr_min, rr_max, p_low, p_high):
    """
    3 niveles discretos: bajo / medio / alto.
    Umbral medio = (p_low + p_high) / 2.
    """
    rr_mid = (rr_min + rr_max) / 2.0
    mid = (p_low + p_high) / 2.0
    if p_win < p_low:
        return float(rr_min)
    elif p_win < mid:
        return float(rr_mid)
    elif p_win < p_high:
        return float(rr_mid)
    else:
        return float(rr_max)


# =====================================================================
# GESTOR DE NIVELES TP/SL
# =====================================================================

@dataclass
class DynamicRRManager:
    """
    Calcula niveles de TP y SL por trade usando R:R dinámico.

    Uso típico en walk-forward / backtesting:
        mgr = DynamicRRManager(atr_sl_mult=2.0, rr_min=1.2, rr_max=2.5)
        sl_price, tp_price = mgr.compute_levels(
            entry_price=5200.0,
            signal=+1,
            atr=18.5,
            p_win=0.68,
        )

    También puede usarse standalone con Kelly:
        rr = mgr.compute_rr(p_win)
        kelly_frac = kelly_fraction_binary(p_win, rr_ratio=rr, kelly_fraction=0.25)

    Parameters
    ----------
    atr_sl_mult : float
        Stop en múltiplos de ATR. Invariante — no cambia con la confianza.
    rr_min : float
        R:R mínimo (baja confianza).
    rr_max : float
        R:R máximo (alta confianza).
    p_low : float
        Confianza donde comienza la escala de R:R. Debe estar ≥ entry threshold.
    p_high : float
        Confianza donde R:R alcanza su máximo.
    shape : str
        Forma del mapeo P(win) → R:R.
    min_atr_multiple : float
        TP mínimo en múltiplos de ATR (cap por debajo, aunque R:R sea mínimo).
    max_atr_multiple : float
        TP máximo en múltiplos de ATR (cap por arriba, aunque R:R sea máximo).
    """
    atr_sl_mult: float = 2.0
    rr_min: float = 1.2
    rr_max: float = 2.5
    p_low: float = 0.45
    p_high: float = 0.75
    shape: str = "sigmoid"
    min_atr_multiple: Optional[float] = None  # si None, usa rr_min * atr_sl_mult
    max_atr_multiple: Optional[float] = None  # si None, usa rr_max * atr_sl_mult

    def compute_rr(self, p_win: float) -> float:
        """R:R dinámico para una probabilidad dada."""
        return compute_dynamic_rr(
            p_win=p_win,
            rr_min=self.rr_min,
            rr_max=self.rr_max,
            p_low=self.p_low,
            p_high=self.p_high,
            shape=self.shape,
        )

    def compute_levels(
        self,
        entry_price: float,
        signal: int,
        atr: float,
        p_win: float,
    ) -> tuple[float, float]:
        """
        Calcula precios de SL y TP para un trade.

        Parameters
        ----------
        entry_price : float
            Precio de entrada al trade.
        signal : int
            +1 (long) o -1 (short).
        atr : float
            ATR actual en unidades de precio.
        p_win : float
            Probabilidad calibrada de win para este trade.

        Returns
        -------
        (sl_price, tp_price) : tuple[float, float]
            Precios de stop loss y take profit.
        """
        if signal not in (1, -1):
            raise ValueError(f"signal debe ser +1 o -1, recibido: {signal}")
        if atr <= 0:
            raise ValueError(f"ATR debe ser > 0, recibido: {atr}")

        sl_distance = self.atr_sl_mult * atr
        rr = self.compute_rr(p_win)
        tp_distance = rr * sl_distance

        # Aplicar caps si están definidos
        if self.min_atr_multiple is not None:
            tp_distance = max(tp_distance, self.min_atr_multiple * atr)
        if self.max_atr_multiple is not None:
            tp_distance = min(tp_distance, self.max_atr_multiple * atr)

        if signal == 1:   # long
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:             # short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        return float(sl_price), float(tp_price)

    def levels_report(
        self,
        entry_price: float,
        signal: int,
        atr: float,
        p_win: float,
    ) -> dict:
        """
        Reporte completo de niveles para un trade. Útil para logging/debugging.
        """
        sl_price, tp_price = self.compute_levels(entry_price, signal, atr, p_win)
        rr = self.compute_rr(p_win)
        sl_distance = self.atr_sl_mult * atr
        tp_distance = rr * sl_distance
        direction = "LONG" if signal == 1 else "SHORT"

        return {
            "direction": direction,
            "entry_price": round(entry_price, 5),
            "sl_price": round(sl_price, 5),
            "tp_price": round(tp_price, 5),
            "sl_distance": round(sl_distance, 5),
            "tp_distance": round(tp_distance, 5),
            "rr_dynamic": round(rr, 4),
            "atr": round(atr, 5),
            "p_win": round(p_win, 4),
            "atr_sl_mult": self.atr_sl_mult,
            "shape": self.shape,
        }

    def rr_curve(
        self,
        p_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Devuelve la curva R:R vs P(win) para visualización/diagnóstico.
        Útil para verificar que el mapeo tiene el comportamiento esperado.

        Returns
        -------
        pd.DataFrame con columnas: ['p_win', 'rr', 'tp_mult']
        donde tp_mult = rr × atr_sl_mult (TP en múltiplos de ATR).
        """
        if p_values is None:
            p_values = np.linspace(0.30, 0.95, 50)

        records = []
        for p in p_values:
            rr = self.compute_rr(float(p))
            records.append({
                "p_win": round(float(p), 4),
                "rr": round(rr, 4),
                "tp_atr_mult": round(rr * self.atr_sl_mult, 4),
            })
        return pd.DataFrame(records)


# =====================================================================
# INTEGRACIÓN COMPLETA: Dynamic R:R + Kelly
# =====================================================================

def compute_full_sizing(
    signal: int,
    p_win: float,
    current_equity: float,
    current_price: float,
    current_atr: float,
    instrument,
    rr_manager: DynamicRRManager,
    kelly_fraction: float = 0.25,
    max_risk_pct: float = 0.02,
    min_risk_pct: float = 0.001,
    usd_conversion_factor: float = 1.0,
) -> dict:
    """
    Calcula sizing completo: R:R dinámico + Kelly fraccional + ATR stop.

    Este es el punto de integración de los tres componentes de riesgo:
      1. DynamicRRManager → R:R dinámico basado en P(win)
      2. KellyFractional   → fracción de equity basada en P(win) y R:R
      3. ATRSizing         → convierte riesgo USD en unidades del instrumento

    Parameters
    ----------
    signal : int
        +1 (long), -1 (short), 0 (sin señal).
    p_win : float
        P(win) calibrada para este trade.
    current_equity : float
        Equity actual en USD.
    current_price : float
        Precio de entrada.
    current_atr : float
        ATR actual.
    instrument : InstrumentSpec
        Especificación del instrumento.
    rr_manager : DynamicRRManager
        Gestor de R:R dinámico configurado.
    kelly_fraction : float
        Fracción de Kelly (0.25 = quarter Kelly).
    max_risk_pct : float
        Cap duro de riesgo por trade.
    min_risk_pct : float
        Riesgo mínimo viable.
    usd_conversion_factor : float
        Para pares no-USD-quoted.

    Returns
    -------
    dict con:
        - 'n_units': unidades a operar (0 si Kelly dice no entrar)
        - 'sl_price': precio de stop loss
        - 'tp_price': precio de take profit
        - 'risk_pct': fracción de equity arriesgada
        - 'risk_usd': riesgo en USD
        - 'rr_dynamic': R:R aplicado
        - 'kelly_raw': f* antes de aplicar kelly_fraction
    """
    from risk.kelly import kelly_fraction_binary

    if signal == 0 or current_atr <= 0:
        return {
            "n_units": 0.0, "sl_price": None, "tp_price": None,
            "risk_pct": 0.0, "risk_usd": 0.0, "rr_dynamic": None,
            "kelly_raw": 0.0,
        }

    # 1. R:R dinámico
    rr = rr_manager.compute_rr(p_win)

    # 2. Kelly con R:R dinámico
    from risk.kelly import kelly_fraction_binary
    f_kelly = kelly_fraction_binary(
        p_win=p_win,
        rr_ratio=rr,
        kelly_fraction=kelly_fraction,
        min_edge=0.0,
    )
    f_kelly = min(f_kelly, max_risk_pct)
    if f_kelly < min_risk_pct:
        return {
            "n_units": 0.0, "sl_price": None, "tp_price": None,
            "risk_pct": 0.0, "risk_usd": 0.0, "rr_dynamic": round(rr, 4),
            "kelly_raw": round(p_win - (1 - p_win) / rr, 4),
        }

    # 3. ATR sizing
    sl_distance = rr_manager.atr_sl_mult * current_atr
    usd_per_unit_at_stop = (
        sl_distance
        * instrument.usd_per_unit_per_price_point
        * usd_conversion_factor
    )
    if usd_per_unit_at_stop <= 0:
        return {
            "n_units": 0.0, "sl_price": None, "tp_price": None,
            "risk_pct": 0.0, "risk_usd": 0.0, "rr_dynamic": round(rr, 4),
            "kelly_raw": 0.0,
        }

    risk_usd = current_equity * f_kelly
    n_units_raw = risk_usd / usd_per_unit_at_stop
    n_units = instrument.round_to_min_increment(n_units_raw)
    n_units_signed = float(np.sign(signal) * n_units)

    # 4. Niveles TP/SL
    sl_price, tp_price = rr_manager.compute_levels(
        entry_price=current_price,
        signal=signal,
        atr=current_atr,
        p_win=p_win,
    )

    f_star = p_win - (1 - p_win) / rr

    return {
        "n_units": n_units_signed,
        "sl_price": round(sl_price, 5),
        "tp_price": round(tp_price, 5),
        "risk_pct": round(f_kelly, 5),
        "risk_usd": round(risk_usd, 2),
        "rr_dynamic": round(rr, 4),
        "kelly_raw": round(f_star, 4),
    }
