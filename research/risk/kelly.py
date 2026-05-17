"""
Kelly Fraccional para Position Sizing
======================================
Reemplaza el risk_pct fijo del ATRRiskSizer con un risk_pct dinámico
calculado a partir de las probabilidades calibradas del modelo.

KELLY CRITERION (fórmula binaria):
  f* = (p·b - q) / b  =  p - q/b

  donde:
    p = P(win)     — probabilidad calibrada de que el trade sea ganador
    q = 1 - p      — probabilidad de perder
    b = win / loss — ratio ganancia esperada / pérdida esperada (R:R)

  Si f* ≤ 0: Kelly dice "no entrar" (la apuesta es EV negativo).
  Si f* > 0: arriesgar esa fracción del capital.

KELLY FRACCIONAL (en producción siempre):
  f_kelly = kelly_fraction × f*

  kelly_fraction ∈ [0.10, 0.25] en la práctica.
  Kelly completo (fraction=1.0) es óptimo en teoría, ruinoso en práctica
  por la varianza de estimación de p y b.

INTEGRACIÓN CON ATR:
  El ATR determina la distancia del stop (en precio → en USD).
  Kelly determina cuánto arriesgar (fracción de equity).
  Juntos:
    risk_usd  = equity × f_kelly
    n_units   = risk_usd / (atr_stop_mult × ATR × usd_per_unit)

  Esto es idéntico a ATRRiskSizer pero con risk_pct = f_kelly (dinámico)
  en vez de risk_pct fijo.

ESTIMACIÓN DE R:R (b):
  Opción 1: fijo desde config (e.g., atr_tp_mult / atr_sl_mult = 3/2 = 1.5)
  Opción 2: empírico desde histórico de trades (función estimate_rr_ratio)

  Recomendación: usar el empírico cuando hay suficientes trades (≥50),
  fijo como prior cuando el histórico es escaso.

RELACIÓN CON EL FILTRO DE ENTRADA:
  - EntryFilter dice SI/NO entrar basado en P > threshold
  - KellySizer dice CUÁNTO arriesgar dado que ya decidimos entrar

  Son complementarios, no sustitutos:
    1. EntryFilter filtra señales de baja confianza
    2. KellySizer escala el sizing de las señales que pasan el filtro
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# FÓRMULAS KELLY — funciones puras (sin estado)
# =====================================================================

def kelly_fraction_binary(
    p_win: float,
    rr_ratio: float,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.0,
) -> float:
    """
    Kelly fraccional para una apuesta binaria (ganas b o pierdes 1).

    f* = p - (1-p) / b = p - q / b

    Parameters
    ----------
    p_win : float in [0, 1]
        Probabilidad calibrada de que el trade sea ganador.
    rr_ratio : float > 0
        Ratio ganancia/pérdida esperada. Si stop=2×ATR y target=3×ATR → b=1.5.
    kelly_fraction : float in (0, 1]
        Fracción de Kelly a usar. 0.25 = "quarter Kelly" (estándar en producción).
    min_edge : float
        f* mínima para considerar la apuesta atractiva. Default 0 (EV > 0).
        Usar valores > 0 para exigir un margen mínimo.

    Returns
    -------
    float : fracción del capital a arriesgar [0, kelly_fraction].
            0 si f* ≤ min_edge (no apostar).
    """
    if rr_ratio <= 0:
        raise ValueError(f"rr_ratio debe ser > 0, recibido: {rr_ratio}")
    p_win = float(np.clip(p_win, 0.0, 1.0))
    f_star = p_win - (1.0 - p_win) / rr_ratio
    if f_star <= min_edge:
        return 0.0
    return float(np.clip(kelly_fraction * f_star, 0.0, kelly_fraction))


def kelly_breakeven_probability(rr_ratio: float) -> float:
    """
    Probabilidad mínima de win para que Kelly recomiende apostar (f* > 0).

    Despejando f* > 0:
      p - (1-p)/b > 0  →  p·(1 + 1/b) > 1/b  →  p > 1/(1+b)

    Ejemplo: b=1.5 → p_min = 1/2.5 = 0.40 (necesitas >40% para tener EV+)
    """
    return 1.0 / (1.0 + rr_ratio)


def expected_value(p_win: float, rr_ratio: float) -> float:
    """
    Valor esperado por unidad arriesgada: EV = p·b - q·1 = p·b - (1-p)
    """
    return p_win * rr_ratio - (1.0 - p_win)


def estimate_rr_ratio(
    trade_returns: np.ndarray,
    min_trades: int = 20,
) -> float:
    """
    Estima el R:R empírico a partir del histórico de trades.

    R:R = mean(winning_trades) / |mean(losing_trades)|

    Parameters
    ----------
    trade_returns : np.ndarray
        Retornos de cada trade cerrado (positivos = ganadores, negativos = perdedores).
        NO incluir trades abiertos.
    min_trades : int
        Mínimo de trades para que la estimación sea válida.
        Si hay menos, devuelve 1.0 (R:R neutral).

    Returns
    -------
    float : R:R estimado (siempre > 0).
    """
    if len(trade_returns) < min_trades:
        logger.warning(
            f"Pocos trades para estimar R:R ({len(trade_returns)} < {min_trades}). "
            f"Usando R:R = 1.0 como prior."
        )
        return 1.0

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        logger.warning("Sin wins o sin losses para estimar R:R. Usando 1.0.")
        return 1.0

    avg_win = float(wins.mean())
    avg_loss = float(np.abs(losses.mean()))

    if avg_loss < 1e-10:
        return 10.0  # cap razonable si no hay pérdidas

    rr = avg_win / avg_loss
    logger.info(f"R:R empírico estimado: {rr:.3f} "
                f"(avg_win={avg_win:.5f}, avg_loss={avg_loss:.5f}, "
                f"n={len(trade_returns)})")
    return float(rr)


# =====================================================================
# KELLY ATR SIZER — combina Kelly + ATR stop
# =====================================================================

@dataclass
class KellyAtrSizer:
    """
    Position sizer que combina Kelly fraccional con stop basado en ATR.

    Reemplaza ATRRiskSizer cuando se tienen probabilidades calibradas.

    Flujo por trade:
      1. Recibe P(win) de las probabilidades calibradas del modelo.
      2. Calcula f_kelly = kelly_fraction × max(0, p - q/b).
      3. risk_usd = equity × f_kelly.
      4. n_units = risk_usd / (atr_stop_mult × ATR × usd_per_unit).
      5. Redondea y aplica caps.

    Parameters
    ----------
    instrument : InstrumentSpec
        Especificación del instrumento (usd_per_unit_per_price_point).
    rr_ratio : float
        R:R estimado. Si atr_tp_mult y atr_sl_mult están seteados, se calcula
        automáticamente como atr_tp_mult / atr_sl_mult.
    atr_sl_mult : float
        Stop loss en múltiplos de ATR (e.g., 2.0 = stop a 2×ATR del precio).
    atr_tp_mult : float | None
        Take profit en múltiplos de ATR. Si se proporciona, calcula R:R = tp/sl.
        Si es None, usa rr_ratio directamente.
    kelly_fraction : float
        Fracción de Kelly. Recomendado: 0.25 (quarter Kelly).
    max_risk_pct : float
        Cap duro: nunca arriesgar más de este % del equity por trade.
        Protección ante probabilidades mal calibradas o R:R sobreestimado.
    min_risk_pct : float
        Risk mínimo (si Kelly dice > 0 pero muy pequeño, usar este mínimo).
        Evita posiciones simbólicas (<0.1% de equity no tiene impacto real).
    min_edge : float
        f* mínima requerida para entrar. 0 = cualquier EV positivo.
        0.02 = solo si Kelly recomienda al menos 2% de fracción bruta.
    daily_loss_pct_pause : float
        Si la pérdida del día ≥ este %, no abrir nuevas posiciones.
    """
    instrument: object = None           # InstrumentSpec
    rr_ratio: float = 1.5               # R:R por defecto (puede actualizarse)
    atr_sl_mult: float = 2.0
    atr_tp_mult: Optional[float] = 3.0  # si None, usa rr_ratio directo
    kelly_fraction: float = 0.25
    max_risk_pct: float = 0.02          # cap duro: nunca > 2% equity por trade
    min_risk_pct: float = 0.001         # mínimo: 0.1% (si Kelly es muy pequeño → skip)
    min_edge: float = 0.0
    daily_loss_pct_pause: float = 0.03

    def __post_init__(self):
        if self.instrument is None:
            raise ValueError("instrument es obligatorio en KellyAtrSizer")
        # Calcular R:R desde ATR multiples si están disponibles
        if self.atr_tp_mult is not None and self.atr_sl_mult > 0:
            self.rr_ratio = self.atr_tp_mult / self.atr_sl_mult

        self._day_start_equity: float = 0.0
        self._current_day = None
        self._trade_log: list = []   # para estimar R:R empírico con el tiempo

    @property
    def effective_rr_ratio(self) -> float:
        """R:R efectivo actualmente en uso."""
        return self.rr_ratio

    def update_rr_from_history(self, trade_returns: np.ndarray) -> float:
        """
        Actualiza el R:R con estimación empírica.
        Llama a esto periódicamente (e.g., cada fold del walk-forward).

        Returns
        -------
        float : nuevo R:R estimado.
        """
        new_rr = estimate_rr_ratio(trade_returns)
        old_rr = self.rr_ratio
        self.rr_ratio = new_rr
        logger.info(f"R:R actualizado: {old_rr:.3f} -> {new_rr:.3f}")
        return new_rr

    def _check_daily_pause(self, equity: float, date) -> bool:
        if self._current_day != date:
            self._current_day = date
            self._day_start_equity = equity
            return False
        if self._day_start_equity <= 0:
            return False
        loss_pct = (self._day_start_equity - equity) / self._day_start_equity
        return loss_pct >= self.daily_loss_pct_pause

    def compute_risk_pct(self, p_win: float) -> float:
        """
        Calcula el risk_pct dinámico basado en Kelly.

        Returns 0 si Kelly dice no apostar (f* ≤ min_edge).
        """
        f = kelly_fraction_binary(
            p_win=p_win,
            rr_ratio=self.rr_ratio,
            kelly_fraction=self.kelly_fraction,
            min_edge=self.min_edge,
        )
        if f <= 0:
            return 0.0
        # Aplicar caps
        f = min(f, self.max_risk_pct)
        if f < self.min_risk_pct:
            return 0.0   # posición demasiado pequeña → skip
        return f

    def __call__(
        self,
        signal: float,
        p_win: float,
        current_equity: float,
        current_price: float,
        current_atr: float,
        current_date=None,
        usd_conversion_factor: float = 1.0,
    ) -> float:
        """
        Calcula n_units usando Kelly dinámico + stop ATR.

        Parameters
        ----------
        signal : float
            +1 (long), -1 (short), 0 (no operar).
        p_win : float
            P(win) calibrada. Para long: P(y=+1|x). Para short: P(y=-1|x).
            Se obtiene de model.predict_proba() → columna de la clase señalizada.
        current_equity : float
            Equity actual en USD.
        current_price : float
            Precio del instrumento.
        current_atr : float
            ATR actual (en unidades de precio).
        current_date : optional
            Para daily loss tracking.
        usd_conversion_factor : float
            Para pares no-USD-quoted. 1.0 para EURUSD, ES, NQ.

        Returns
        -------
        float : n_units signed. 0 si Kelly dice no entrar.
        """
        if signal == 0:
            return 0.0
        if pd.isna(current_atr) or current_atr <= 0:
            return 0.0

        # Daily loss check
        if current_date is not None and self._check_daily_pause(
            current_equity, current_date
        ):
            return 0.0

        # Kelly → risk_pct dinámico
        risk_pct = self.compute_risk_pct(p_win)
        if risk_pct <= 0:
            return 0.0

        # ATR → stop distance en USD
        stop_distance_price = self.atr_sl_mult * current_atr
        usd_per_unit_at_stop = (
            stop_distance_price
            * self.instrument.usd_per_unit_per_price_point
            * usd_conversion_factor
        )
        if usd_per_unit_at_stop <= 0:
            return 0.0

        risk_usd = current_equity * risk_pct
        n_units_raw = risk_usd / usd_per_unit_at_stop
        n_units = self.instrument.round_to_min_increment(n_units_raw)

        return float(np.sign(signal) * n_units)

    def sizing_report(self, p_win: float, equity: float, atr: float,
                      usd_conv: float = 1.0) -> dict:
        """
        Reporte detallado del sizing para una probabilidad dada.
        Útil para debugging y monitoreo.
        """
        risk_pct = self.compute_risk_pct(p_win)
        p_break = kelly_breakeven_probability(self.rr_ratio)
        ev = expected_value(p_win, self.rr_ratio)
        f_star = p_win - (1 - p_win) / self.rr_ratio

        stop_distance_price = self.atr_sl_mult * atr
        usd_per_unit_at_stop = (
            stop_distance_price
            * self.instrument.usd_per_unit_per_price_point
            * usd_conv
        )
        risk_usd = equity * risk_pct
        n_units_raw = risk_usd / usd_per_unit_at_stop if usd_per_unit_at_stop > 0 else 0
        n_units = self.instrument.round_to_min_increment(n_units_raw)

        return {
            "p_win": round(p_win, 4),
            "rr_ratio": round(self.rr_ratio, 4),
            "p_breakeven": round(p_break, 4),
            "edge": round(p_win - p_break, 4),
            "expected_value": round(ev, 4),
            "f_star": round(f_star, 4),
            "kelly_fraction_param": self.kelly_fraction,
            "f_kelly": round(self.kelly_fraction * max(0, f_star), 4),
            "risk_pct_applied": round(risk_pct, 4),
            "risk_usd": round(risk_usd, 2),
            "atr": round(atr, 6),
            "stop_distance_usd_per_unit": round(usd_per_unit_at_stop, 4),
            "n_units_raw": round(n_units_raw, 4),
            "n_units_final": n_units,
        }


# =====================================================================
# UTILIDAD: extraer p_win de predict_proba
# =====================================================================

def extract_p_win(
    proba: np.ndarray,
    signal: int,
    class_labels: list,
) -> float:
    """
    Extrae P(win) del array de probabilidades calibradas.

    Para un trade long  (signal=+1): P(win) = P(y=+1|x)
    Para un trade short (signal=-1): P(win) = P(y=-1|x)

    Parameters
    ----------
    proba : np.ndarray, shape (n_classes,)
        Probabilidades para UNA barra (una fila de predict_proba).
    signal : int
        +1 o -1.
    class_labels : list
        Lista de etiquetas en el mismo orden que las columnas de proba.
        Ejemplo: [-1, 0, 1].

    Returns
    -------
    float : P(win) en [0, 1].
    """
    target_class = int(signal)
    if target_class not in class_labels:
        raise ValueError(
            f"signal={signal} no está en class_labels={class_labels}"
        )
    idx = class_labels.index(target_class)
    return float(proba[idx])
