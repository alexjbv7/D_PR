"""
Volatility Target Module
========================
Construye targets de prediccion de volatilidad para clasificacion 3-clase.

Target: regimen de volatilidad realizada en horizonte futuro.
- 0 = volatilidad BAJA (terciles inferior)
- 1 = volatilidad MEDIA
- 2 = volatilidad ALTA (tercil superior)

Los thresholds se calculan con quantiles ROLLING (sin look-ahead).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Volatilidad realizada en ventana.

    RV_t = sqrt(sum_{i=t-window+1}^{t} r_i^2 / window)
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std()


def future_volatility(close: pd.Series, horizon: int = 20) -> pd.Series:
    """
    Volatilidad realizada en los proximos `horizon` dias (mira al FUTURO).

    Esto es el TARGET. NO es feature.
    """
    log_ret = np.log(close / close.shift(1))
    # Volatilidad de dias t+1 a t+horizon (mira al futuro)
    return log_ret.shift(-horizon).rolling(horizon).std()


def vol_regime_target(
    close: pd.Series,
    horizon: int = 20,
    quantile_window: int = 252,
    n_regimes: int = 3,
) -> pd.Series:
    """
    Construye target de regimen de volatilidad futura.

    Para cada t:
    1. Calcula RV futura (t+1 a t+horizon)
    2. Calcula thresholds usando quantiles rolling de RV pasada
    3. Clasifica RV futura en uno de n_regimes

    Parameters
    ----------
    close : Serie de precios
    horizon : ventana futura en dias
    quantile_window : cuantos dias pasados para calcular thresholds
    n_regimes : 3 = baja/media/alta

    Returns
    -------
    Serie con valores en {0, 1, 2}
    """
    # Volatilidad realizada en cada barra (mira al pasado)
    rv_past = realized_volatility(close, window=horizon)

    # Volatilidad futura (target)
    rv_future = future_volatility(close, horizon=horizon)

    # Thresholds rolling basados en RV pasada
    target = pd.Series(index=close.index, dtype=float)

    for i in range(quantile_window, len(close) - horizon):
        # Datos pasados: RV historica
        past_rv = rv_past.iloc[i - quantile_window:i].dropna()
        if len(past_rv) < quantile_window // 2:
            continue

        # Quantiles para 3 regimenes: 33% y 67%
        thresholds = past_rv.quantile([1/n_regimes, 2/n_regimes]).values

        # Clasificar la RV FUTURA
        future_value = rv_future.iloc[i]
        if pd.isna(future_value):
            continue

        target.iloc[i] = sum(future_value > t for t in thresholds)

    return target


def vol_regime_descriptive_stats(target: pd.Series, close: pd.Series, horizon: int = 20) -> pd.DataFrame:
    """
    Para cada clase del target, devuelve estadisticas descriptivas
    de la volatilidad futura realizada.
    """
    rv_future = future_volatility(close, horizon=horizon)
    df = pd.DataFrame({'target': target, 'rv_future': rv_future}).dropna()

    stats = df.groupby('target')['rv_future'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    return stats * 100  # convertir a porcentaje