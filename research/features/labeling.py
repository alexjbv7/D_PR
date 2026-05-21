"""
Labeling Module — Triple-Barrier & Related Methods
====================================================
Implementaciones canónicas de etiquetado para series temporales financieras,
siguiendo López de Prado (AFML, cap. 3).

PRINCIPIO ANTI-LEAKAGE:
  Las barreras se calculan en `t` usando SOLO información disponible en `t`
  (volatilidad/ATR rolling sobre datos pasados). El futuro `[t+1, t+horizon]`
  solo se consulta para determinar cuál barrera se toca primero — nunca para
  calcular las barreras en sí.

Funciones canónicas:
  triple_barrier_labels      — barreras relativas a vol histórica (rolling std)
  triple_barrier_labels_atr  — barreras relativas a ATR (preferida para intraday)
  compute_atr_ewm            — ATR via EWM (cero look-ahead, listo para live)
  forward_return_label       — target de regresión simple (baseline)

Referencias:
  López de Prado, M. (2018). *Advances in Financial Machine Learning*, cap. 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# ATR — Average True Range (EWM, zero look-ahead)
# ============================================================================

def compute_atr_ewm(
    prices: pd.DataFrame,
    period: int = 14,
    col_close: str = "close",
    col_high: str = "high",
    col_low: str = "low",
) -> pd.Series:
    """
    ATR via EWM (Exponential Weighted Moving Average).

    Ventaja frente a SMA: sin cliff-edge al salir una barra antigua.
    Anti-leakage: usa solo información hasta la barra t inclusive.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame con columnas close, high, low.
    period : int
        Ventana EWM (alpha = 1/period).

    Returns
    -------
    pd.Series : ATR alineado con el índice de prices.
    """
    c = prices[col_close]
    h = prices[col_high]
    lo = prices[col_low]
    prev_c = c.shift(1)
    true_range = pd.concat(
        [h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


# ============================================================================
# TRIPLE-BARRIER (ATR-based) — versión preferida para barras OHLCV
# ============================================================================

def triple_barrier_labels_atr(
    close: pd.Series,
    atr: pd.Series,
    horizon: int = 5,
    upper_mult: float = 1.5,
    lower_mult: float = 1.5,
) -> pd.Series:
    """
    Triple-barrier labeling con barreras escaladas por ATR.

    Para cada barra t, observa las próximas `horizon` barras:
      +1  si close[t+k] ≥ close[t] + upper_mult × ATR[t]  (primera barrera tocada)
      -1  si close[t+k] ≤ close[t] − lower_mult × ATR[t]  (primera barrera tocada)
       0  si ninguna barrera se alcanza en `horizon` barras (timeout)

    Ventaja vs vol-based: ATR captura rango intrabar (high-low), más robusto
    ante gaps; es la opción natural cuando se dispone de OHLCV completo.

    Anti-leakage: la barrera usa `ATR[t]` (pasado), no vol forward.

    Parameters
    ----------
    close : pd.Series
        Serie de precios de cierre.
    atr : pd.Series
        ATR alineado con close (calcular con compute_atr_ewm).
    horizon : int
        Número de barras hacia adelante para buscar la primera barrera tocada.
    upper_mult, lower_mult : float
        Multiplicadores ATR para las barreras superior e inferior.

    Returns
    -------
    pd.Series con valores en {-1, 0, +1} y NaN en las últimas `horizon` barras.
    """
    n = len(close)
    labels = np.full(n, np.nan)
    close_arr = close.values
    atr_arr = atr.values

    for i in range(n - horizon):
        if np.isnan(atr_arr[i]) or atr_arr[i] == 0:
            continue
        entry = close_arr[i]
        upper = entry + upper_mult * atr_arr[i]
        lower = entry - lower_mult * atr_arr[i]
        label = 0
        for j in range(i + 1, i + horizon + 1):
            if close_arr[j] >= upper:
                label = 1
                break
            elif close_arr[j] <= lower:
                label = -1
                break
        labels[i] = label

    return pd.Series(labels, index=close.index, name="label")


# ============================================================================
# TRIPLE-BARRIER (vol-based) — versión log-vol, sin OHLCV completo
# ============================================================================

def triple_barrier_labels(
    close: pd.Series,
    horizon: int = 20,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    vol_period: int = 20,
) -> pd.Series:
    """
    Triple-barrier labeling con barreras escaladas por volatilidad rolling.

    Variante para datos sin high/low disponibles (solo close).
    Las barreras se calculan como `entry × exp(±mult × rolling_std)`.

    Parámetros
    ----------
    close : pd.Series
    horizon : int
    upper_mult, lower_mult : float
        Múltiplos de la desviación estándar para las barreras.
    vol_period : int
        Ventana rolling para la estimación de volatilidad.

    Returns
    -------
    pd.Series con valores en {-1, 0, +1}.
    """
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(vol_period).std()
    labels = pd.Series(index=close.index, dtype=float, name="label")

    for i in range(len(close) - horizon):
        if pd.isna(vol.iloc[i]) or vol.iloc[i] == 0:
            continue
        entry_price = close.iloc[i]
        upper_barrier = entry_price * np.exp(upper_mult * vol.iloc[i])
        lower_barrier = entry_price * np.exp(-lower_mult * vol.iloc[i])
        future = close.iloc[i + 1: i + 1 + horizon]

        hit_upper = future.index[future >= upper_barrier][0] if (future >= upper_barrier).any() else None
        hit_lower = future.index[future <= lower_barrier][0] if (future <= lower_barrier).any() else None

        if hit_upper is not None and hit_lower is not None:
            labels.iloc[i] = 1 if hit_upper < hit_lower else -1
        elif hit_upper is not None:
            labels.iloc[i] = 1
        elif hit_lower is not None:
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0

    return labels


# ============================================================================
# FORWARD RETURN LABEL — baseline de regresión
# ============================================================================

def forward_return_label(close: pd.Series, horizon: int = 20) -> pd.Series:
    """
    Target de regresión: retorno log forward a `horizon` barras.

    Más simple que triple-barrier; útil como baseline o cuando se quiere
    optimizar directamente la magnitud del retorno.

    Returns
    -------
    pd.Series de floats; NaN en las últimas `horizon` barras.
    """
    return np.log(close.shift(-horizon) / close).rename("fwd_return")
