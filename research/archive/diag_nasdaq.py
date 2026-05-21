# diag_nasdaq.py
import yfinance as yf
import pandas as pd

print("=" * 50)
print("Test 1: Descargar NASDAQ individualmente")
print("=" * 50)

try:
    nasdaq = yf.download('^IXIC', start='2020-01-01', end='2024-12-31',
                         interval='1d', progress=False, auto_adjust=True)
    print(f"Filas descargadas: {len(nasdaq)}")
    print(f"Columnas: {list(nasdaq.columns)}")
    print(f"Primeras filas:")
    print(nasdaq.head())
    print(f"\nUltimas filas:")
    print(nasdaq.tail())
    print(f"\nNaN totales: {nasdaq.isna().sum().sum()}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 50)
print("Test 2: Inspeccionar cache de macro")
print("=" * 50)

import os
cache_files = [f for f in os.listdir('./cache') if 'macro' in f.lower()]
print(f"Archivos cache: {cache_files}")

if cache_files:
    cached = pd.read_parquet(f'./cache/{cache_files[0]}')
    print(f"\nColumnas en cache: {list(cached.columns)}")
    print(f"NaN por columna:")
    print(cached.isna().sum())
    print(f"\nEstadisticas de NASDAQ en cache:")
    if 'nasdaq' in cached.columns:
        print(cached['nasdaq'].describe())
        print(f"\nPrimeros valores no-NaN:")
        print(cached['nasdaq'].dropna().head())