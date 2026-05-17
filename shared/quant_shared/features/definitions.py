"""
Definición canónica de los 19 features de trading.

Esta es la fuente única de verdad. Tanto research/ (backtesting)
como platform/services/ml-feature-store (tiempo real) deben usar
estos nombres y ventanas exactas.

NUNCA cambiar el orden de FEATURE_NAMES sin versionar el cambio —
los modelos entrenados dependen del índice posicional.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDef:
    name: str
    description: str
    window: int        # periodos necesarios para computar (mínimo de datos)
    min_val: float     # valor mínimo esperado (para validación)
    max_val: float     # valor máximo esperado (para validación)
    default: float     # valor por defecto cuando hay datos insuficientes


# ─── Definición de los 19 features ──────────────────────────────────────────
# El orden es FIJO y se corresponde con el índice en el vector numpy.
# platform/services/ml-feature-store/app/feature_streaming.py debe
# producir features en este mismo orden.

FEATURES: list[FeatureDef] = [
    # Índice 0 — Momentum y tendencia
    FeatureDef("rsi_14",          "RSI de 14 periodos",                         14,   0.0, 100.0,  50.0),
    FeatureDef("macd_hist",       "MACD histogram (12/26/9)",                   35,  -1e4,   1e4,   0.0),
    FeatureDef("mom_1h",          "Momentum 1h: (close/close_1h_ago) - 1",       1,  -0.5,   0.5,   0.0),
    FeatureDef("mom_4h",          "Momentum 4h: (close/close_4h_ago) - 1",       4,  -0.5,   0.5,   0.0),
    FeatureDef("mom_24h",         "Momentum 24h: (close/close_24h_ago) - 1",    24,  -0.5,   0.5,   0.0),

    # Índice 5 — Volatilidad
    FeatureDef("atr_14",          "ATR de 14 periodos (normalizado por precio)", 14,   0.0,  0.20,   0.01),
    FeatureDef("bb_width",        "Bollinger Width: 2*std/sma (20 periodos)",    20,   0.0,  0.50,   0.02),
    FeatureDef("vol_ratio_1h",    "Volumen 1h vs media 20h",                     20,   0.0,  10.0,   1.0),

    # Índice 8 — Microestructura
    FeatureDef("ob_imbalance",    "Order book imbalance: (bid-ask)/(bid+ask)",    1,  -1.0,   1.0,   0.0),
    FeatureDef("spread_bps",      "Spread bid-ask en basis points",               1,   0.0, 100.0,   1.0),
    FeatureDef("vwap_deviation",  "(price - vwap) / vwap",                        1,  -0.1,   0.1,   0.0),
    FeatureDef("funding_rate",    "Funding rate (perpetuos), 8h",                 1,  -0.01, 0.01,   0.0),
    FeatureDef("oi_change_1h",    "Cambio OI 1h: (oi - oi_1h_ago) / oi_1h_ago",  1,  -0.5,  0.5,   0.0),

    # Índice 13 — Contexto macro/régimen (desde Redis)
    FeatureDef("regime_id",       "Régimen de mercado: 0=ranging, 1=trending_up, 2=trending_down, 3=volatile",
                                                                                   1,   0.0,   3.0,   0.0),
    FeatureDef("macro_leverage",  "Multiplicador de leverage macro (0.5-1.5)",     1,   0.5,   1.5,   1.0),

    # Índice 15 — Tendencia y momentum técnico
    FeatureDef("sma_cross",       "SMA 20 vs SMA 50: (sma20-sma50)/sma50",       50,  -0.1,   0.1,   0.0),
    FeatureDef("adx_14",          "ADX de 14 periodos (fuerza de tendencia)",     14,   0.0, 100.0,  20.0),

    # Índice 17 — On-chain (desde Redis)
    FeatureDef("reserve_z",       "Z-score de reservas en exchanges",              1,  -5.0,   5.0,   0.0),
    FeatureDef("whale_sentiment", "Sentimiento whale: -1 (bearish) a +1 (bullish)", 1, -1.0,  1.0,   0.0),
]

# Acceso rápido
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURE_COUNT: int = len(FEATURES)                         # debe ser 19
FEATURE_INDEX: dict[str, int] = {f.name: i for i, f in enumerate(FEATURES)}

assert FEATURE_COUNT == 19, f"Se esperaban 19 features, hay {FEATURE_COUNT}"
