"""
Multi-Asset Data Module
=======================
Descarga OHLCV de cualquier activo soportado por Yahoo Finance.

Usado para experimentos cross-asset (SPY, GLD, EURUSD, etc.)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf


# Catalogo de activos soportados con sus tickers Yahoo
ASSET_CATALOG = {
    'spy':    {'ticker': '^GSPC', 'name': 'S&P 500 Index'},
    'spy_etf': {'ticker': 'SPY', 'name': 'S&P 500 ETF'},
    'qqq':    {'ticker': 'QQQ', 'name': 'NASDAQ 100 ETF'},
    'gld':    {'ticker': 'GLD', 'name': 'Gold ETF'},
    'gold':   {'ticker': 'GC=F', 'name': 'Gold Futures'},
    'eurusd': {'ticker': 'EURUSD=X', 'name': 'EUR/USD'},
    'usdjpy': {'ticker': 'JPY=X', 'name': 'USD/JPY'},
    'gbpusd': {'ticker': 'GBPUSD=X', 'name': 'GBP/USD'},
    'tlt':    {'ticker': 'TLT', 'name': '20+ Yr Treasury Bond ETF'},
    'vix':    {'ticker': '^VIX', 'name': 'VIX Volatility Index'},
}


def fetch_asset(
    asset_id: str,
    start: str = '2010-01-01',
    end: str = '2025-01-31',
    cache_dir: str = './cache',
) -> pd.DataFrame:
    """
    Descarga OHLCV de un activo via Yahoo Finance.

    Returns
    -------
    DataFrame con columnas [open, high, low, close, volume] e indice DatetimeIndex UTC.
    """
    if asset_id not in ASSET_CATALOG:
        raise ValueError(f"Activo '{asset_id}' no soportado. Usa: {list(ASSET_CATALOG.keys())}")

    info = ASSET_CATALOG[asset_id]
    ticker = info['ticker']

    cache_path = Path(cache_dir) / f'{asset_id}_{start}_{end}.parquet'
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and not cached['close'].isna().all():
            print(f"  {asset_id} cargado de cache: {len(cached)} filas")
            return cached

    print(f"  Descargando {asset_id} ({ticker}) desde Yahoo...", end=' ', flush=True)

    raw = yf.download(
        ticker, start=start, end=end,
        interval='1d', progress=False, auto_adjust=True,
    )

    if raw.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")

    # Manejar MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        df = pd.DataFrame({
            'open': raw[('Open', ticker)],
            'high': raw[('High', ticker)],
            'low': raw[('Low', ticker)],
            'close': raw[('Close', ticker)],
            'volume': raw[('Volume', ticker)] if ('Volume', ticker) in raw.columns else 0.0,
        })
    else:
        df = pd.DataFrame({
            'open': raw['Open'],
            'high': raw['High'],
            'low': raw['Low'],
            'close': raw['Close'],
            'volume': raw['Volume'] if 'Volume' in raw.columns else 0.0,
        })

    # Eliminar filas con NaN en columnas criticas
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Indice UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    Path(cache_dir).mkdir(exist_ok=True)
    df.to_parquet(cache_path)
    print(f"OK ({len(df)} filas)")

    return df