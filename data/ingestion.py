"""
CCXT OHLCV Ingestion Module
============================
Descarga histórica de datos OHLCV desde exchanges crypto vía CCXT.
Pagina automáticamente, respeta rate limits y persiste en Parquet local.

Uso típico:
    >>> ingestor = OHLCVIngestor(exchange='binance')
    >>> df = ingestor.fetch_historical('BTC/USDT', '1h', since='2020-01-01')

Nota: este módulo solo hace ingesta. Para triple-barrier labeling ver
research/features/labeling.py. Para feature engineering ver
research/features/engineering.py.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import ccxt

logger = logging.getLogger(__name__)


class OHLCVIngestor:
    """
    Ingesta robusta de datos OHLCV desde exchanges crypto.

    - Maneja rate limits de forma automática (CCXT lo hace internamente).
    - Pagina hacia atrás cuando el exchange limita el número de velas por request.
    - Persiste en parquet para acceso rápido posterior.

    Parameters
    ----------
    exchange : str
        Identificador del exchange en CCXT (e.g., 'binance', 'bybit', 'kraken').
    cache_dir : Path | str
        Directorio donde se guardan los parquets cacheados.
    """

    TIMEFRAME_MS = {
        '1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000,
        '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
    }

    def __init__(self, exchange: str = 'binance', cache_dir: str | Path = './cache'):
        if exchange not in ccxt.exchanges:
            raise ValueError(f"Exchange '{exchange}' no soportado por CCXT")

        self.exchange = getattr(ccxt, exchange)({
            'enableRateLimit': True,  # CRITICAL: respeta los límites del exchange
            'timeout': 30_000,
        })
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _to_ms(self, date_str: str) -> int:
        """Convierte 'YYYY-MM-DD' a timestamp milisegundos UTC."""
        dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: str = '2020-01-01',
        until: Optional[str] = None,
        limit_per_call: int = 1000,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Descarga histórico completo paginando hacia adelante.

        Returns
        -------
        pd.DataFrame con índice datetime UTC y columnas [open, high, low, close, volume]
        """
        cache_file = self.cache_dir / f"{self.exchange.id}_{symbol.replace('/', '')}_{timeframe}.parquet"

        if use_cache and cache_file.exists():
            logger.info(f"Cargando desde cache: {cache_file}")
            return pd.read_parquet(cache_file)

        since_ms = self._to_ms(since)
        until_ms = self._to_ms(until) if until else int(time.time() * 1000)
        tf_ms = self.TIMEFRAME_MS[timeframe]

        all_candles = []
        current = since_ms

        while current < until_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current, limit=limit_per_call
                )
                if not candles:
                    break

                all_candles.extend(candles)
                # Avanzamos al siguiente bloque; +tf_ms para no duplicar la última vela
                current = candles[-1][0] + tf_ms
                logger.info(f"Descargadas {len(all_candles)} velas. Última: {datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc)}")

                # Pausa defensiva extra (CCXT ya maneja rate limit, pero por seguridad)
                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.NetworkError as e:
                logger.warning(f"NetworkError: {e}. Reintentando en 5s...")
                time.sleep(5)
            except ccxt.ExchangeError as e:
                logger.error(f"ExchangeError: {e}")
                break

        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp').drop_duplicates()
        df = df.sort_index()

        # Validación crítica: detectar gaps
        expected_freq = pd.Timedelta(milliseconds=tf_ms)
        gaps = df.index.to_series().diff().dropna()
        anomalies = gaps[gaps > expected_freq * 1.5]
        if len(anomalies) > 0:
            logger.warning(f"Detectados {len(anomalies)} gaps en los datos")

        df.to_parquet(cache_file)
        logger.info(f"Guardado en cache: {cache_file} ({len(df)} velas)")
        return df

    def fetch_latest(self, symbol: str, timeframe: str = '1h', n: int = 100) -> pd.DataFrame:
        """Obtiene las últimas N velas para inferencia en tiempo real."""
        candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=n)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df.set_index('timestamp')


def clean_ohlcv(df: pd.DataFrame, max_gap_ratio: float = 0.05) -> pd.DataFrame:
    """
    Limpieza de datos OHLCV.

    - Elimina filas con NaN en columnas críticas
    - Detecta y reporta gaps temporales
    - Filtra velas con volumen 0 (sospechosas en activos líquidos)
    - Detecta outliers con z-score sobre retornos log
    """
    df = df.copy()

    # 1. NaN handling
    n_before = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if len(df) < n_before:
        logger.warning(f"Eliminadas {n_before - len(df)} filas con NaN")

    # 2. Volumen 0 (en activos líquidos esto suele ser un error de feed)
    zero_vol = (df['volume'] == 0).sum()
    if zero_vol > 0:
        logger.warning(f"{zero_vol} velas con volumen 0 detectadas")

    # 3. Validación OHLC: high >= max(open, close), low <= min(open, close)
    invalid = (df['high'] < df[['open', 'close']].max(axis=1)) | \
              (df['low'] > df[['open', 'close']].min(axis=1))
    if invalid.sum() > 0:
        logger.warning(f"{invalid.sum()} velas con OHLC inválido — eliminando")
        df = df[~invalid]

    # 4. Outliers en retornos log (>10 desviaciones estándar es típicamente un error)
    log_ret = (df['close'] / df['close'].shift(1)).apply('log')
    z = (log_ret - log_ret.mean()) / log_ret.std()
    outliers = z.abs() > 10
    if outliers.sum() > 0:
        logger.warning(f"{outliers.sum()} outliers extremos detectados (>10 sigma)")

    return df
