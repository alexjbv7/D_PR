# data/regimes.py
"""
Regimes Module
==============
Documenta periodos historicos con caracteristicas atipicas que pueden
contaminar el entrenamiento del modelo.

Filosofia: NO eliminamos datos, los ETIQUETAMOS. El usuario decide
si los usa para entrenar, validar, o solo para calcular features.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass
class Regime:
    name: str
    start: str
    end: str
    description: str
    affects_volume: bool = False
    affects_price: bool = False


# ============================================================
# REGIMENES DOCUMENTADOS
# ============================================================
KNOWN_REGIMES = [
    Regime(
        name='binance_zero_fee_btc',
        start='2022-07-08',
        end='2023-03-22',
        description=(
            'Binance ofrecio 0% maker/taker fees en BTC/USDT. '
            'Inflo artificialmente el volumen (hasta 85% del volumen '
            'semanal era zero-fee). Wash trading masivo. '
            'Datos de volumen no representativos.'
        ),
        affects_volume=True,
        affects_price=False,
    ),
    Regime(
        name='ftx_collapse',
        start='2022-11-06',
        end='2022-11-14',
        description=(
            'Colapso de FTX. Volatilidad extrema, contagio sistemico. '
            'Comportamiento atipico que no se repite normalmente.'
        ),
        affects_volume=True,
        affects_price=True,
    ),
    Regime(
        name='covid_crash',
        start='2020-03-09',
        end='2020-03-20',
        description=(
            'Crash COVID. Caida ~50% en BTC en 2 dias. '
            'Black swan que distorsiona modelos de volatilidad.'
        ),
        affects_volume=True,
        affects_price=True,
    ),
]


def add_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anade columnas booleanas al dataframe marcando cada regimen conocido.
    Tambien anade una columna agregada 'in_anomalous_regime'.
    """
    df = df.copy()
    in_any_anomaly = pd.Series(False, index=df.index)

    for regime in KNOWN_REGIMES:
        col_name = f'regime_{regime.name}'
        start = pd.Timestamp(regime.start, tz='UTC')
        end = pd.Timestamp(regime.end, tz='UTC')
        mask = (df.index >= start) & (df.index <= end)
        df[col_name] = mask
        in_any_anomaly |= mask

    df['in_anomalous_regime'] = in_any_anomaly
    return df


def filter_clean_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve solo las filas que NO estan en ningun regimen anomalo.
    Util para training/validation, NO para calcular features.
    """
    if 'in_anomalous_regime' not in df.columns:
        df = add_regime_flags(df)
    return df[~df['in_anomalous_regime']].copy()


def regime_summary(df: pd.DataFrame) -> str:
    """Imprime resumen de cuantas barras caen en cada regimen."""
    if 'in_anomalous_regime' not in df.columns:
        df = add_regime_flags(df)

    lines = ["Resumen de regimenes anomalos:"]
    lines.append("=" * 60)
    total = len(df)

    for regime in KNOWN_REGIMES:
        col = f'regime_{regime.name}'
        n = df[col].sum()
        pct = n / total * 100 if total > 0 else 0
        lines.append(f"  {regime.name:30s} {n:5d} barras  ({pct:5.1f}%)")

    n_clean = (~df['in_anomalous_regime']).sum()
    lines.append("-" * 60)
    lines.append(f"  Barras LIMPIAS (utiles): {n_clean} de {total} ({n_clean/total*100:.1f}%)")
    return "\n".join(lines)