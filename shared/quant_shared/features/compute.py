"""
Cómputo canónico de los 19 features — puro numpy/pandas, sin Kafka ni Redis.

Esta es la implementación de referencia. Tanto:
  - research/features/feature_engineering.py  (backtesting)
  - platform/services/ml-feature-store/app/feature_streaming.py  (tiempo real)
deben producir valores numéricamente equivalentes para los mismos inputs.

Funciones estáticas — no hay estado, no hay I/O.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .definitions import FEATURES, FEATURE_NAMES, FEATURE_COUNT, FeatureDef


@dataclass
class FeatureVector:
    """Vector de 19 features listo para inference."""
    symbol: str
    ts: str
    # — Momentum y tendencia
    rsi_14:          float = 50.0
    macd_hist:       float = 0.0
    mom_1h:          float = 0.0
    mom_4h:          float = 0.0
    mom_24h:         float = 0.0
    # — Volatilidad
    atr_14:          float = 0.01
    bb_width:        float = 0.02
    vol_ratio_1h:    float = 1.0
    # — Microestructura
    ob_imbalance:    float = 0.0
    spread_bps:      float = 1.0
    vwap_deviation:  float = 0.0
    funding_rate:    float = 0.0
    oi_change_1h:    float = 0.0
    # — Contexto (externo)
    regime_id:       float = 0.0
    macro_leverage:  float = 1.0
    # — Tendencia técnica
    sma_cross:       float = 0.0
    adx_14:          float = 20.0
    # — On-chain (externo)
    reserve_z:       float = 0.0
    whale_sentiment: float = 0.0

    def to_array(self) -> np.ndarray:
        """Devuelve array numpy en el orden canónico de FEATURE_NAMES."""
        return np.array([
            self.rsi_14, self.macd_hist, self.mom_1h, self.mom_4h, self.mom_24h,
            self.atr_14, self.bb_width, self.vol_ratio_1h,
            self.ob_imbalance, self.spread_bps, self.vwap_deviation,
            self.funding_rate, self.oi_change_1h,
            self.regime_id, self.macro_leverage,
            self.sma_cross, self.adx_14,
            self.reserve_z, self.whale_sentiment,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "ts": self.ts,
            **{name: float(val) for name, val in
               zip(FEATURE_NAMES, self.to_array())}
        }


# ─── Indicadores técnicos (puro numpy) ──────────────────────────────────────

def rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI de Wilder. Devuelve 50.0 si hay datos insuficientes."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def macd_hist(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """MACD histogram. Requiere al menos slow+signal barras."""
    if len(prices) < slow + signal:
        return 0.0

    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        alpha = 2.0 / (n + 1)
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    ema_fast   = _ema(prices, fast)
    ema_slow   = _ema(prices, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line[slow - 1:], signal)
    return float(macd_line[-1] - signal_line[-1])


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average True Range normalizado por precio (ATR/close)."""
    if len(close) < period + 1:
        return 0.01
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        )
    )
    avg_tr = tr[-period:].mean()
    return float(avg_tr / (close[-1] + 1e-9))


def bollinger_width(prices: np.ndarray, period: int = 20) -> float:
    """Bollinger Width = 2*std / sma."""
    if len(prices) < period:
        return 0.02
    window = prices[-period:]
    mu, sd = window.mean(), window.std()
    if abs(mu) < 1e-9:
        return 0.02
    return float(2.0 * sd / mu)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """ADX de Wilder (0-100). Devuelve 20.0 si datos insuficientes."""
    n = len(close)
    if n < period * 2:
        return 20.0

    up_moves   = high[1:] - high[:-1]
    down_moves = low[:-1] - low[1:]
    plus_dm  = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
    minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)

    tr_arr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    )

    def _smooth(arr: np.ndarray) -> float:
        s = arr[:period].sum()
        for v in arr[period:]:
            s = s - s / period + v
        return s

    atr_s    = _smooth(tr_arr)
    plus_di  = 100 * _smooth(plus_dm)  / (atr_s + 1e-9)
    minus_di = 100 * _smooth(minus_dm) / (atr_s + 1e-9)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return float(np.clip(dx, 0, 100))


def momentum(prices: np.ndarray, lookback: int) -> float:
    """(prices[-1] / prices[-1-lookback]) - 1. Devuelve 0.0 si faltan datos."""
    if len(prices) <= lookback:
        return 0.0
    ref = prices[-(lookback + 1)]
    if abs(ref) < 1e-9:
        return 0.0
    return float((prices[-1] - ref) / ref)


def sma_cross(prices: np.ndarray, fast: int = 20, slow: int = 50) -> float:
    """(SMA_fast - SMA_slow) / SMA_slow. Devuelve 0.0 si faltan datos."""
    if len(prices) < slow:
        return 0.0
    sma_f = prices[-fast:].mean()
    sma_s = prices[-slow:].mean()
    if abs(sma_s) < 1e-9:
        return 0.0
    return float((sma_f - sma_s) / sma_s)


# ─── Función principal de cómputo ───────────────────────────────────────────

def compute_features(
    symbol: str,
    ts: str,
    closes:  np.ndarray,    # array de precios de cierre (orden cronológico)
    highs:   np.ndarray,
    lows:    np.ndarray,
    volumes: np.ndarray,
    # Microestructura (tick actual)
    ob_imbalance:   float = 0.0,
    spread_bps:     float = 1.0,
    vwap:           float = 0.0,
    funding_rate:   float = 0.0,
    oi_current:     float = 0.0,
    oi_1h_ago:      float = 0.0,
    # Contexto externo (desde Redis en producción, desde backtesting en research)
    regime_id:      float = 0.0,
    macro_leverage: float = 1.0,
    reserve_z:      float = 0.0,
    whale_sentiment: float = 0.0,
) -> FeatureVector:
    """
    Computa el vector canónico de 19 features.

    Entrada:
        closes, highs, lows, volumes — arrays numpy en orden cronológico
        (el último elemento es el tick más reciente)
    Salida:
        FeatureVector con todos los campos y to_array() listo para modelo
    """
    price = float(closes[-1]) if len(closes) > 0 else 1.0

    # Vol ratio: volumen último periodo vs media de 20 periodos
    vol_ratio = 1.0
    if len(volumes) >= 20:
        avg_vol = volumes[-20:].mean()
        vol_ratio = float(volumes[-1] / (avg_vol + 1e-9))

    # VWAP deviation
    vwap_dev = 0.0
    if abs(vwap) > 1e-9:
        vwap_dev = float((price - vwap) / vwap)

    # OI change 1h
    oi_chg = 0.0
    if abs(oi_1h_ago) > 1e-9:
        oi_chg = float((oi_current - oi_1h_ago) / oi_1h_ago)

    return FeatureVector(
        symbol=symbol,
        ts=ts,
        rsi_14         = rsi(closes, 14),
        macd_hist      = macd_hist(closes),
        mom_1h         = momentum(closes, 1),
        mom_4h         = momentum(closes, 4),
        mom_24h        = momentum(closes, 24),
        atr_14         = atr(highs, lows, closes, 14),
        bb_width       = bollinger_width(closes, 20),
        vol_ratio_1h   = vol_ratio,
        ob_imbalance   = ob_imbalance,
        spread_bps     = spread_bps,
        vwap_deviation = vwap_dev,
        funding_rate   = funding_rate,
        oi_change_1h   = oi_chg,
        regime_id      = regime_id,
        macro_leverage = macro_leverage,
        sma_cross      = sma_cross(closes, 20, 50),
        adx_14         = adx(highs, lows, closes, 14),
        reserve_z      = reserve_z,
        whale_sentiment = whale_sentiment,
    )
