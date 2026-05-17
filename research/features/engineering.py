"""
Feature Engineering Module
==========================
Construcción de features para series temporales financieras siguiendo principios de
López de Prado (AFML). Énfasis en estacionariedad, justificación económica, y
prevención de look-ahead bias.

REGLA DE ORO: Toda feature en el tiempo `t` debe usar SOLO información disponible en `t`.
Si dudas, usa .shift(1) sobre cualquier valor que dependa del 'close' actual cuando lo
combines con un target futuro.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable


# ============================================================================
# 1. RETORNOS Y TRANSFORMACIONES BÁSICAS
# ============================================================================

def log_returns(close: pd.Series, periods: Iterable[int] = (1, 5, 10, 20)) -> pd.DataFrame:
    """Retornos log para múltiples horizontes (estacionarios, simétricos)."""
    out = pd.DataFrame(index=close.index)
    for p in periods:
        out[f'log_ret_{p}'] = np.log(close / close.shift(p))
    return out


def fractional_differentiation(series: pd.Series, d: float = 0.4, thresh: float = 1e-4) -> pd.Series:
    """
    Fractional differentiation (López de Prado, 2018).

    Logra estacionariedad preservando MÁS memoria que un diff entero.
    `d` típico entre 0.3 y 0.5 para precios.

    Implementación de window expansion fija con threshold de pesos.
    """
    # Calcular pesos de la serie binomial fraccional
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])
    width = len(weights)

    # Aplicar convolución
    result = pd.Series(index=series.index, dtype=float)
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1: i + 1].values
        if np.isnan(window).any():
            continue
        result.iloc[i] = np.dot(weights, window)
    return result


# ============================================================================
# 2. INDICADORES DE MOMENTUM
# ============================================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI clásico (Wilder). Bounded [0, 100]."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # EMA estilo Wilder
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD: line, signal, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        'macd_line': line,
        'macd_signal': sig,
        'macd_hist': line - sig
    })


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change."""
    return (close / close.shift(period) - 1) * 100


# ============================================================================
# 3. INDICADORES DE MEAN-REVERSION
# ============================================================================

def bollinger_pct_b(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """%B de Bollinger: posición del precio dentro de las bandas. >1 = sobre banda superior."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (close - lower) / (upper - lower)


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-score rolling. Mide desviación de la media móvil en unidades de std."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - ma) / std


# ============================================================================
# 4. INDICADORES DE VOLATILIDAD
# ============================================================================

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """
    Estimador de Parkinson: usa rangos high-low.
    Más eficiente que std de cierres (asume movimiento browniano).
    """
    factor = 1.0 / (4.0 * np.log(2.0))
    log_hl_sq = (np.log(high / low)) ** 2
    return np.sqrt(factor * log_hl_sq.rolling(period).mean())


def garman_klass_volatility(o, h, l, c, period: int = 20) -> pd.Series:
    """
    Estimador Garman-Klass: incorpora OHLC.
    Aún más eficiente que Parkinson cuando hay drift cero.
    """
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(rs.rolling(period).mean())


# ============================================================================
# 5. VOLUME / FLOW
# ============================================================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap_deviation(high, low, close, volume, period: int = 20) -> pd.Series:
    """Desviación porcentual del precio respecto al VWAP rolling."""
    typical = (high + low + close) / 3
    vwap = (typical * volume).rolling(period).sum() / volume.rolling(period).sum()
    return (close - vwap) / vwap


def mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    """Money Flow Index — versión de RSI ponderada por volumen."""
    typical = (high + low + close) / 3
    raw_mf = typical * volume
    direction = np.sign(typical.diff()).fillna(0)
    pos_mf = (raw_mf * (direction > 0)).rolling(period).sum()
    neg_mf = (raw_mf * (direction < 0)).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


# ============================================================================
# 6. FEATURES CALENDARIO Y RÉGIMEN
# ============================================================================

def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features cíclicas de calendario (sin/cos para preservar continuidad)."""
    df = pd.DataFrame(index=index)
    # Hora del día (cíclica)
    df['hour_sin'] = np.sin(2 * np.pi * index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * index.hour / 24)
    # Día de semana (cíclica)
    df['dow_sin'] = np.sin(2 * np.pi * index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * index.dayofweek / 7)
    # Indicador weekend
    df['is_weekend'] = (index.dayofweek >= 5).astype(int)
    return df


# ============================================================================
# 7. CONSTRUCTOR PRINCIPAL DE FEATURES
# ============================================================================

class FeatureBuilder:
    """
    Construye matriz de features completa a partir de OHLCV.

    IMPORTANTE: Todas las features están alineadas en `t`. El target debe ser
    construido por separado y representar información FUTURA (t+horizon).
    """

    def __init__(
        self,
        return_periods=(1, 5, 10, 20),
        rsi_periods=(7, 14, 21),
        vol_periods=(10, 20, 50),
        bb_periods=(20,),
    ):
        self.return_periods = return_periods
        self.rsi_periods = rsi_periods
        self.vol_periods = vol_periods
        self.bb_periods = bb_periods

    def build(self, df: pd.DataFrame, df_eth: pd.DataFrame = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame con columnas [open, high, low, close, volume] e indice datetime
        df_eth : DataFrame opcional con OHLCV de ETH para feature de correlacion cross-asset
        """
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Faltan columnas. Requeridas: {required}")

        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
        feats = pd.DataFrame(index=df.index)

        # Retornos log multiples horizontes
        feats = feats.join(log_returns(c, self.return_periods))

        # Momentum
        for p in self.rsi_periods:
            feats[f'rsi_{p}'] = rsi(c, p)
        feats = feats.join(macd(c))
        feats['roc_10'] = roc(c, 10)

        # Mean-reversion
        for p in self.bb_periods:
            feats[f'bb_pctb_{p}'] = bollinger_pct_b(c, p)
        feats['zscore_20'] = zscore(c, 20)

        # Volatilidad (en log para estacionariedad)
        for p in self.vol_periods:
            feats[f'atr_{p}'] = atr(h, l, c, p) / c
            feats[f'parkinson_{p}'] = parkinson_volatility(h, l, p)
            feats[f'gk_{p}'] = garman_klass_volatility(o, h, l, c, p)

        # Volumen / flujo
        feats['obv_zscore'] = zscore(obv(c, v), 50)
        feats['vwap_dev_20'] = vwap_deviation(h, l, c, v, 20)
        feats['mfi_14'] = mfi(h, l, c, v, 14)

        # Calendario
        feats = feats.join(calendar_features(df.index))

        # Volume-price relationship
        feats['volume_zscore'] = zscore(v, 20)
        feats['volume_ratio'] = v / v.rolling(20).mean()

        # Microestructura proxy (sin order book)
        feats['hl_range'] = (h - l) / c
        feats['oc_range'] = (c - o) / o
        feats['shadow_upper'] = (h - np.maximum(o, c)) / c
        feats['shadow_lower'] = (np.minimum(o, c) - l) / c

        # ============================================================
        # FEATURES AVANZADAS (sesion 2)
        # ============================================================

        # 1. Fractional differentiation del log-precio (preserva memoria + estacionariedad)
        log_price = np.log(c)
        feats['frac_diff_log_price'] = fractional_differentiation(log_price, d=0.4, thresh=1e-4)

        # 2. Regimen de volatilidad (0=baja, 1=media, 2=alta) usando quantiles rolling
        log_ret_1 = np.log(c / c.shift(1))
        feats['vol_regime'] = volatility_regime(log_ret_1, window=90, n_regimes=3)

        # 3. Correlacion rolling BTC-ETH (si se proporciona df_eth)
        if df_eth is not None:
            eth_log_ret = np.log(df_eth['close'] / df_eth['close'].shift(1))
            btc_log_ret = log_ret_1
            corr_30 = rolling_correlation(btc_log_ret, eth_log_ret, window=30)
            feats['corr_btc_eth_30'] = corr_30
        else:
            feats['corr_btc_eth_30'] = np.nan  # placeholder si no hay ETH

        return feats


# ============================================================================
# 8. CONSTRUCCIÓN DEL TARGET (con triple-barrier — López de Prado)
# ============================================================================
# Implementación canónica en features/labeling.py — re-exportada aquí para
# compatibilidad con código que importe desde features.engineering.

from features.labeling import (  # noqa: E402
    triple_barrier_labels,
    triple_barrier_labels_atr,
    forward_return_label,
    compute_atr_ewm,
)

# ============================================================================
# 9. FEATURES AVANZADAS (Sesion 2)
# ============================================================================

def volatility_regime(
    returns: pd.Series,
    window: int = 90,
    n_regimes: int = 3,
) -> pd.Series:
    """
    Clasifica cada barra en un regimen de volatilidad usando quantiles rolling.

    NO usa info futura: el quantile en t se calcula con datos de [t-window, t-1].

    Parameters
    ----------
    returns : retornos log
    window : ventana rolling (en barras) para calcular quantiles
    n_regimes : numero de regimenes (tipicamente 3: baja/media/alta)

    Returns
    -------
    Serie con valores 0..n_regimes-1
    """
    # Volatilidad rolling (std de retornos en ventana corta)
    short_vol = returns.rolling(20).std()

    # Quantiles rolling: para cada t, los thresholds vienen de los ultimos 'window' valores
    regime = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(short_vol)):
        past_vol = short_vol.iloc[i - window:i].dropna()
        if len(past_vol) < window // 2:
            continue
        # Calcular thresholds basados en datos pasados
        quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]  # ej: [0.33, 0.67] para 3 regimenes
        thresholds = past_vol.quantile(quantiles).values
        # Clasificar el valor actual
        current = short_vol.iloc[i]
        if pd.isna(current):
            continue
        regime.iloc[i] = sum(current > t for t in thresholds)

    return regime


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 30,
) -> pd.Series:
    """
    Correlacion rolling entre dos series.

    Antes de calcular, alinea las series por indice (importante si vienen
    de exchanges diferentes).
    """
    aligned = pd.concat([series_a, series_b], axis=1, join='inner')
    aligned.columns = ['a', 'b']
    return aligned['a'].rolling(window).corr(aligned['b'])