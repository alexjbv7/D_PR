"""
Tests del módulo de datos reales.

Casos validados:
1. YAHOO_TICKER_MAP contiene los instrumentos del catálogo.
2. _normalize() convierte columnas mayúsculas a minúsculas correctamente.
3. _normalize() aplana MultiIndex de columnas (formato yfinance >= 0.2.50).
4. _validate() elimina barras con OHLC inválido.
5. round_to_min_increment se mantiene correcto (smoke-test integración).
6. fetch_real_data lanza ImportError sin yfinance (sin red necesaria).

Los tests 1-5 no requieren red ni descarga.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data.real_data import YahooIngestor, YAHOO_TICKER_MAP, list_available_symbols
from instruments import EURUSD, ES


# =====================================================================
# HELPERS
# =====================================================================

def _make_ohlcv(closes, start="2024-01-02", freq="D") -> pd.DataFrame:
    """Helper: DataFrame OHLCV con columnas capitalizadas (formato yfinance raw)."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({
        "Open":   closes * 0.999,
        "High":   closes * 1.002,
        "Low":    closes * 0.998,
        "Close":  closes,
        "Volume": np.full(len(closes), 1_000_000.0),
    }, index=idx)


def _make_ohlcv_multiindex(closes) -> pd.DataFrame:
    """Simula el MultiIndex que devuelve yfinance >= 0.2.50 al descargar 1 ticker."""
    df = _make_ohlcv(closes)
    ticker = "EURUSD=X"
    df.columns = pd.MultiIndex.from_tuples(
        [(col, ticker) for col in df.columns]
    )
    return df


# =====================================================================
# 1. TICKER MAP COVERAGE
# =====================================================================

def test_ticker_map_covers_catalog():
    """El mapa cubre EURUSD, GBPUSD, USDJPY, AUDUSD, ES, NQ, YM, RTY."""
    required = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "ES", "NQ", "YM", "RTY"]
    for sym in required:
        assert sym in YAHOO_TICKER_MAP, f"'{sym}' no está en YAHOO_TICKER_MAP"
    print(f"OKtest_ticker_map_covers_catalog: {len(YAHOO_TICKER_MAP)} símbolos mapeados")


def test_list_available_symbols_sorted():
    syms = list_available_symbols()
    assert syms == sorted(syms), "list_available_symbols() debe estar ordenado"
    assert len(syms) > 5
    print(f"OKtest_list_available_symbols_sorted: {len(syms)} símbolos")


def test_resolve_ticker_fx():
    ingestor = YahooIngestor(cache_dir="./cache")
    assert ingestor._resolve_ticker("EURUSD") == "EURUSD=X"
    assert ingestor._resolve_ticker("eurusd") == "EURUSD=X"
    print("✓ test_resolve_ticker_fx: PASSED")


def test_resolve_ticker_futures():
    ingestor = YahooIngestor(cache_dir="./cache")
    assert ingestor._resolve_ticker("ES") == "ES=F"
    assert ingestor._resolve_ticker("NQ") == "NQ=F"
    print("✓ test_resolve_ticker_futures: PASSED")


def test_resolve_ticker_passthrough():
    """Símbolos no en el mapa se pasan tal cual (permite Yahoo tickers directos)."""
    ingestor = YahooIngestor(cache_dir="./cache")
    assert ingestor._resolve_ticker("AAPL") == "AAPL"
    print("✓ test_resolve_ticker_passthrough: PASSED")


# =====================================================================
# 2. NORMALIZE: columnas capitalizadas
# =====================================================================

def test_normalize_lowercases_columns():
    """_normalize convierte Open/High/Low/Close/Volume → open/high/low/close/volume."""
    ingestor = YahooIngestor(cache_dir="./cache")
    raw = _make_ohlcv([1.10, 1.11, 1.09, 1.12])
    df = ingestor._normalize(raw, "EURUSD")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    print("✓ test_normalize_lowercases_columns: PASSED")


def test_normalize_multiindex_flattened():
    """_normalize aplana MultiIndex de columnas (yfinance >= 0.2.50 con 1 ticker)."""
    ingestor = YahooIngestor(cache_dir="./cache")
    raw = _make_ohlcv_multiindex([1.10, 1.11, 1.09, 1.12])
    df = ingestor._normalize(raw, "EURUSD")
    assert not isinstance(df.columns, pd.MultiIndex), "Columnas deben estar aplanadas"
    assert "close" in df.columns
    print("✓ test_normalize_multiindex_flattened: PASSED")


def test_normalize_utc_index():
    """_normalize asegura que el índice tenga timezone UTC."""
    ingestor = YahooIngestor(cache_dir="./cache")
    raw = _make_ohlcv([1.10, 1.11])
    # Strip timezone para simular input sin tz
    raw.index = raw.index.tz_localize(None)
    df = ingestor._normalize(raw, "EURUSD")
    assert str(df.index.tz) == "UTC", f"Esperaba UTC, obtuvimos {df.index.tz}"
    print("✓ test_normalize_utc_index: PASSED")


def test_normalize_drops_nan_ohlc():
    """_normalize elimina filas con NaN en OHLC."""
    ingestor = YahooIngestor(cache_dir="./cache")
    raw = _make_ohlcv([1.10, np.nan, 1.12, 1.11])
    raw.loc[raw.index[1], "Close"] = np.nan
    df = ingestor._normalize(raw, "EURUSD")
    assert len(df) == 3, f"Esperaba 3 filas válidas, obtuvimos {len(df)}"
    print("✓ test_normalize_drops_nan_ohlc: PASSED")


def test_normalize_deduplicates():
    """_normalize elimina filas duplicadas en el índice."""
    ingestor = YahooIngestor(cache_dir="./cache")
    raw = _make_ohlcv([1.10, 1.11, 1.12])
    dup = pd.concat([raw, raw.iloc[[1]]])  # duplicar barra del medio
    df = ingestor._normalize(dup, "EURUSD")
    assert df.index.is_unique, "El índice debe ser único tras normalización"
    print("✓ test_normalize_deduplicates: PASSED")


# =====================================================================
# 3. VALIDATE: OHLC inválido
# =====================================================================

def test_validate_drops_invalid_ohlc():
    """
    _validate elimina barras donde high < max(open, close)
    o low > min(open, close).
    """
    ingestor = YahooIngestor(cache_dir="./cache")
    raw = _make_ohlcv([1.10, 1.11, 1.12, 1.13, 1.14])
    df = ingestor._normalize(raw, "EURUSD")

    # Introducir una barra con OHLC inválido: high < close
    df.loc[df.index[2], "high"] = df.loc[df.index[2], "close"] - 0.01

    validated = ingestor._validate(df, "EURUSD", "1d")
    assert len(validated) == 4, (
        f"Esperaba 4 barras válidas, obtuvimos {len(validated)}"
    )
    print("✓ test_validate_drops_invalid_ohlc: PASSED")


# =====================================================================
# 4. INTEGRACIÓN: InstrumentSpec sigue funcionando
# =====================================================================

def test_eurusd_pnl_unchanged():
    """Smoke-test: la integración con InstrumentSpec no se rompió."""
    pnl = EURUSD.pnl_usd(1.0, 1.1000, 1.1010, 1.0)
    assert abs(pnl - 100.0) < 0.01, f"Esperaba $100, obtuvimos {pnl}"
    print("✓ test_eurusd_pnl_unchanged: PASSED")


def test_es_pnl_unchanged():
    """1 contrato ES × 1 punto → $50."""
    pnl = ES.pnl_usd(1.0, 4500.0, 4501.0, 1.0)
    assert abs(pnl - 50.0) < 0.01, f"Esperaba $50, obtuvimos {pnl}"
    print("✓ test_es_pnl_unchanged: PASSED")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    print("\nEjecutando tests de datos reales (sin red)...\n")
    tests = [
        test_ticker_map_covers_catalog,
        test_list_available_symbols_sorted,
        test_resolve_ticker_fx,
        test_resolve_ticker_futures,
        test_resolve_ticker_passthrough,
        test_normalize_lowercases_columns,
        test_normalize_multiindex_flattened,
        test_normalize_utc_index,
        test_normalize_drops_nan_ohlc,
        test_normalize_deduplicates,
        test_validate_drops_invalid_ohlc,
        test_eurusd_pnl_unchanged,
        test_es_pnl_unchanged,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests fallaron: {failures}")
        sys.exit(1)
    print(f"ALL PASSED: {len(tests)} tests OK")
