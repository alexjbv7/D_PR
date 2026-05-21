"""
Instruments module: especificaciones de instrumentos financieros.

Una `InstrumentSpec` describe la microstructure de un instrumento (Forex pair o
futuro) lo suficiente para que el backtester pueda calcular P&L correctamente
en USD, aplicar spreads, comisiones y swap realistas.

Esto reemplaza el modelo "fracción de equity × precio" del motor crypto, que
está mal para FX (lots) y futuros (contratos enteros con tick_value).
"""
from .specs import (
    InstrumentSpec,
    ForexSpec,
    FutureSpec,
    AssetClass,
)
from .catalog import (
    get_instrument,
    list_instruments,
    EURUSD,
    GBPUSD,
    USDJPY,
    AUDUSD,
    ES,
    NQ,
    YM,
    RTY,
)

__all__ = [
    "InstrumentSpec",
    "ForexSpec",
    "FutureSpec",
    "AssetClass",
    "get_instrument",
    "list_instruments",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "ES",
    "NQ",
    "YM",
    "RTY",
]
