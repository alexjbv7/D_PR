"""
Real Market Data Ingestion (Yahoo Finance)
==========================================
Fetches OHLCV data for FX pairs and futures from Yahoo Finance.
Maps catalog symbols (EURUSD, ES, NQ, …) to Yahoo tickers automatically.
Caches to parquet — same cache dir as OHLCVIngestor (crypto).

LIMITATIONS (conocidas y documentadas):
- Barras horarias: Yahoo limita a ~730 días de historia.
- Barras diarias: varios años disponibles.
- Futuros: front-month continuo SIN ajuste de roll. Artefactos de roll
  son visibles como saltos bruscos al vencer el contrato. Para backtests
  > 1 año en futuros, considera Norgate Data o CSI.
- FX: precios mid indicativos (no son fills ejecutables).
- Yahoo puede devolver gaps en días festivos o interrupciones de feed.

Para producción: reemplaza con Refinitiv, dukascopy-historical, o la API
de tu broker.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================================
# TICKER MAP
# =====================================================================

YAHOO_TICKER_MAP: dict[str, str] = {
    # FX majors (mid-price indicativo)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "EURGBP": "EURGBP=X",
    # E-mini Index Futures — front-month continuo (CME)
    "ES": "ES=F",
    "NQ": "NQ=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    # Micro E-minis
    "MES": "MES=F",
    "MNQ": "MNQ=F",
    # Commodities
    "GC": "GC=F",   # Gold
    "SI": "SI=F",   # Silver
    "CL": "CL=F",   # WTI Crude
    "NG": "NG=F",   # Natural Gas
    # Rates
    "ZN": "ZN=F",   # 10-Year T-Note
    "ZB": "ZB=F",   # 30-Year T-Bond
}

VALID_INTERVALS = frozenset({
    "1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m",
    "1d", "5d", "1wk", "1mo", "3mo",
})

# Intervalos con restricción de historia en Yahoo (~730 días)
INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m"})


# =====================================================================
# INGESTOR
# =====================================================================

class YahooIngestor:
    """
    Fetches y cachea OHLCV desde Yahoo Finance.

    Uso típico:
        ingestor = YahooIngestor(cache_dir="./cache")
        eurusd = ingestor.fetch("EURUSD", interval="1d", start="2018-01-01")
        es     = ingestor.fetch("ES",     interval="1d", start="2018-01-01")
    """

    def __init__(self, cache_dir: str | Path = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        interval: str = "1d",
        start: str = "2018-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Descarga (o carga del caché) OHLCV para el símbolo dado.

        Parameters
        ----------
        symbol : str
            Símbolo del catálogo (e.g., 'EURUSD', 'ES') o ticker Yahoo directo.
        interval : str
            '1d' (recomendado para backtests), '1h' (limitado a ~730 días).
        start, end : str
            Fechas en formato 'YYYY-MM-DD'. end=None → hoy.
        use_cache : bool
            Si True, carga del parquet si existe.
        force_reload : bool
            Si True, ignora caché y re-descarga.

        Returns
        -------
        pd.DataFrame con DatetimeIndex UTC y columnas [open, high, low, close, volume].
        Todas las columnas en unidades nativas del instrumento.
        """
        try:
            import yfinance as yf  # noqa: F401
        except ImportError:
            raise ImportError(
                "yfinance no está instalado. Ejecuta: pip install yfinance"
            )

        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"interval '{interval}' no válido. Opciones: {sorted(VALID_INTERVALS)}"
            )
        if interval in INTRADAY_INTERVALS:
            logger.warning(
                f"Intervalo intraday '{interval}': Yahoo Finance limita la "
                f"historia a ~730 días (últimos 2 años)."
            )

        end_str = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yahoo_ticker = self._resolve_ticker(symbol)
        cache_path = self._cache_path(symbol, interval, start, end_str)

        if use_cache and not force_reload and cache_path.exists():
            logger.info(f"Cache hit: {cache_path}")
            df = pd.read_parquet(cache_path)
            logger.info(
                f"{symbol}: {len(df)} barras cargadas del caché "
                f"({df.index[0].date()} → {df.index[-1].date()})"
            )
            return df

        df = self._download(yahoo_ticker, interval, start, end_str, symbol)
        df = self._normalize(df, symbol)
        df = self._validate(df, symbol, interval)

        df.to_parquet(cache_path)
        logger.info(
            f"{symbol} guardado en caché: {cache_path} ({len(df)} barras)"
        )
        return df

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _resolve_ticker(self, symbol: str) -> str:
        key = symbol.upper().replace("/", "")
        return YAHOO_TICKER_MAP.get(key, symbol)  # fallback: asume Yahoo directo

    def _cache_path(self, symbol: str, interval: str, start: str, end: str) -> Path:
        safe = symbol.upper().replace("=", "").replace("/", "")
        return self.cache_dir / f"yahoo_{safe}_{interval}_{start}_{end}.parquet"

    def _download(
        self,
        ticker: str,
        interval: str,
        start: str,
        end: str,
        symbol: str,
    ) -> pd.DataFrame:
        import yfinance as yf

        logger.info(
            f"Descargando {ticker} de Yahoo Finance "
            f"(interval={interval}, {start} → {end})..."
        )
        # yfinance >= 0.2.x puede devolver MultiIndex de columnas al descargar
        # un solo ticker. Manejamos ambos formatos en _normalize().
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_column=False,  # yfinance >= 0.2.50
            )
        except TypeError:
            # Versiones anteriores no tienen multi_level_column
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

        if df is None or len(df) == 0:
            raise ValueError(
                f"Yahoo Finance no devolvió datos para '{ticker}' ({symbol}). "
                f"Verifica que el ticker sea válido y que el rango de fechas tenga datos."
            )
        return df

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Estandariza columnas y asegura DatetimeIndex UTC."""
        # Aplanar MultiIndex de columnas (yfinance >= 0.2.x con un ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]

        # Columnas requeridas
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{symbol}: faltan columnas {missing} en los datos de Yahoo. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )

        if "volume" not in df.columns:
            df["volume"] = 0.0

        df = df[["open", "high", "low", "close", "volume"]].copy()

        # Eliminar filas con NaN en OHLC
        n_before = len(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(f"{symbol}: eliminadas {n_dropped} barras con NaN en OHLC")

        # Asegurar DatetimeIndex UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        return df

    def _validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        """Sanity checks: OHLC válido, outliers log-retorno."""
        # OHLC consistency
        invalid = (
            (df["high"] < df[["open", "close"]].max(axis=1)) |
            (df["low"] > df[["open", "close"]].min(axis=1))
        )
        n_invalid = int(invalid.sum())
        if n_invalid > 0:
            logger.warning(
                f"{symbol}: {n_invalid} barras con OHLC inválido — eliminando"
            )
            df = df[~invalid]

        # Outliers en log-retornos (> 10 sigma → probable error de feed)
        if len(df) > 10:
            log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
            sigma = log_ret.std()
            if sigma > 0:
                z = (log_ret - log_ret.mean()) / sigma
                n_outliers = int((z.abs() > 10).sum())
                if n_outliers > 0:
                    logger.warning(
                        f"{symbol}: {n_outliers} outliers extremos (>10σ) en "
                        f"log-retornos. Pueden ser artefactos de roll en futuros."
                    )

        logger.info(
            f"{symbol} ({interval}): {len(df)} barras, "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df


# =====================================================================
# CONVENIENCE
# =====================================================================

def fetch_real_data(
    symbol: str,
    interval: str = "1d",
    start: str = "2018-01-01",
    end: Optional[str] = None,
    cache_dir: str | Path = "./cache",
    use_cache: bool = True,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Wrapper de una línea para obtener OHLCV real limpio.

    Ejemplo
    -------
    >>> from data.real_data import fetch_real_data
    >>> eurusd = fetch_real_data("EURUSD", interval="1d", start="2018-01-01")
    >>> es     = fetch_real_data("ES",     interval="1d", start="2018-01-01")
    """
    return YahooIngestor(cache_dir=cache_dir).fetch(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        use_cache=use_cache,
        force_reload=force_reload,
    )


def list_available_symbols() -> list[str]:
    """Retorna todos los símbolos del catálogo mapeados a Yahoo."""
    return sorted(YAHOO_TICKER_MAP.keys())
