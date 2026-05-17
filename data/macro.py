"""
Macro Data Module
=================
Descarga datos macro/tradfi para correlaciones cross-asset con BTC.

Estrategia: descarga ticker por ticker (mas lenta pero robusta) en lugar de
multi-ticker (que falla silenciosamente con algunos indices).
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance no instalado. Ejecuta: pip install yfinance")

logger = logging.getLogger(__name__)


TICKERS = {
    'sp500':  '^GSPC',
    'nasdaq': '^IXIC',
    'dxy':    'DX-Y.NYB',
    'vix':    '^VIX',
    'gold':   'GC=F',
}


def _download_single(ticker: str, start: str, end: str) -> pd.Series:
    """Descarga un solo ticker y devuelve la serie de cierres."""
    raw = yf.download(
        ticker,
        start=start, end=end,
        interval='1d',
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(f"yfinance devolvio dataframe vacio para {ticker}")

    # Manejar MultiIndex (yfinance >= 0.2.x devuelve MultiIndex incluso para 1 ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        # Buscar la columna 'Close' al primer nivel
        if 'Close' in raw.columns.get_level_values(0):
            close = raw['Close'].iloc[:, 0]  # primer (unico) ticker
        else:
            raise ValueError(f"No se encontro 'Close' en {ticker}, cols: {raw.columns}")
    else:
        if 'Close' in raw.columns:
            close = raw['Close']
        else:
            raise ValueError(f"No se encontro 'Close' en {ticker}, cols: {list(raw.columns)}")

    close.name = ticker
    return close


def fetch_macro(
    start: str = '2020-01-01',
    end: str = '2024-12-31',
    cache_dir: str = './cache',
    force_redownload: bool = False,
) -> pd.DataFrame:
    """
    Descarga close diario de los activos macro, ticker por ticker.

    Returns
    -------
    DataFrame con DatetimeIndex (UTC) y columnas: sp500, nasdaq, dxy, vix, gold
    """
    cache_path = Path(cache_dir) / f'macro_{start}_{end}.parquet'

    if cache_path.exists() and not force_redownload:
        cached = pd.read_parquet(cache_path)
        # Validar que ninguna columna sea 100% NaN
        all_nan_cols = cached.columns[cached.isna().all()].tolist()
        if not all_nan_cols:
            logger.info(f"Macro cargado desde cache: {cache_path}")
            return cached
        else:
            logger.warning(f"Cache invalido (columnas vacias: {all_nan_cols}). Re-descargando...")

    logger.info(f"Descargando datos macro de Yahoo Finance ticker por ticker...")

    series_dict = {}
    for name, ticker in TICKERS.items():
        try:
            print(f"  Descargando {name} ({ticker})...", end=' ', flush=True)
            s = _download_single(ticker, start, end)
            series_dict[name] = s
            print(f"OK ({len(s)} filas)")
        except Exception as e:
            print(f"FALLO: {e}")
            logger.error(f"Fallo {name}: {e}")

    if not series_dict:
        raise RuntimeError("No se pudo descargar NINGUN activo macro")

    # Combinar en un DataFrame con outer join (para preservar todas las fechas)
    closes = pd.concat(series_dict.values(), axis=1)
    closes.columns = list(series_dict.keys())

    # Forzar indice a UTC
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize('UTC')
    else:
        closes.index = closes.index.tz_convert('UTC')

    # Forward-fill: cuando un mercado esta cerrado, mantener ultimo close
    closes = closes.ffill()

    Path(cache_dir).mkdir(exist_ok=True)
    closes.to_parquet(cache_path)
    logger.info(f"Datos macro guardados: {cache_path}")

    return closes