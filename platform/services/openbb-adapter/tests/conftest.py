"""
Fixtures compartidas para los tests del openbb-adapter.

Estrategia de mocking:
  - OpenBB SDK mockeado completamente — no llamadas reales a proveedores
  - Redis mockeado con AsyncMock — sin Redis real
  - Kafka producer mockeado — sin Kafka real
  - El OpenBBClient se puede testear en modo "obb=None" (disabled)
    para testear el comportamiento de fallback y cache

Importar el módulo `app` antes de configurar mocks de libs externas
no es necesario aquí porque el cliente hace lazy import de `openbb`.
"""
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.cache import ResponseCache
from app.client import OpenBBClient
from app.config import Settings


# ── Configuración base ────────────────────────────────────────────────────────

@pytest.fixture
def settings() -> Settings:
    """Settings con valores de test (sin API keys reales)."""
    return Settings(
        fred_api_key="test-fred-key",
        kafka_servers="",
        redis_url="",
        crypto_symbols="BTC,ETH",
        macro_poll_interval=9999,   # deshabilitar polling en tests
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Redis mock que simula get/set/setex/delete/exists."""
    redis = AsyncMock()
    redis.get    = AsyncMock(return_value=None)  # cache MISS por defecto
    redis.setex  = AsyncMock(return_value=True)
    redis.set    = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    return redis


@pytest.fixture
def cache(mock_redis: AsyncMock) -> ResponseCache:
    return ResponseCache(mock_redis)


@pytest.fixture
def cache_no_redis() -> ResponseCache:
    """Cache sin Redis — simula entorno degradado."""
    return ResponseCache(redis=None)


@pytest.fixture
def mock_obb() -> MagicMock:
    """Mock del SDK de OpenBB — simula respuestas de providers."""
    obb = MagicMock()

    # ── Crypto OHLCV ──────────────────────────────────────────────
    _ohlcv_result = MagicMock()
    import pandas as pd
    _ohlcv_result.to_df.return_value = pd.DataFrame({
        "date":   ["2026-01-01", "2026-01-02", "2026-01-03"],
        "open":   [50_000.0, 51_000.0, 52_000.0],
        "high":   [51_000.0, 52_000.0, 53_000.0],
        "low":    [49_000.0, 50_000.0, 51_000.0],
        "close":  [50_500.0, 51_500.0, 52_500.0],
        "volume": [1000.0,   1100.0,   1200.0],
    })
    obb.crypto.price.historical.return_value = _ohlcv_result

    # ── FRED series ────────────────────────────────────────────────
    _fred_result = MagicMock()
    _fred_result.to_df.return_value = pd.DataFrame({
        "date":  ["2026-01-01", "2026-02-01", "2026-03-01"],
        "value": [4.0, 4.1, 4.2],
    })
    obb.economy.fred_series.return_value = _fred_result

    # ── Yield curve ───────────────────────────────────────────────
    _yc_result = MagicMock()
    _yc_result.to_df.return_value = pd.DataFrame({
        "maturity": ["1M", "3M", "1Y", "2Y", "10Y"],
        "rate":     [5.0,  5.1,  4.9,  4.8,  4.5],
    })
    obb.fixedincome.government.yield_curve.return_value = _yc_result

    # ── Options chain ─────────────────────────────────────────────
    _options_result = MagicMock()
    _options_result.to_df.return_value = pd.DataFrame({
        "option_type":        ["call", "put", "call", "put"],
        "strike":             [50000, 50000, 55000, 55000],
        "implied_volatility": [0.65,  0.70,  0.60,  0.75],
        "delta":              [0.55,  -0.45, 0.40,  -0.60],
        "open_interest":      [100,   80,    60,    90],
    })
    obb.derivatives.options.chains.return_value = _options_result

    # ── SEC filings ───────────────────────────────────────────────
    _sec_result = MagicMock()
    _sec_result.to_df.return_value = pd.DataFrame({
        "filed_at":  ["2026-01-15", "2026-02-10"],
        "form_type": ["10-K", "10-Q"],
        "symbol":    ["IBIT", "IBIT"],
    })
    obb.equity.fundamental.filings.return_value = _sec_result

    return obb


@pytest.fixture
def client(settings: Settings, cache: ResponseCache, mock_obb: MagicMock) -> OpenBBClient:
    """OpenBBClient con OpenBB SDK mockeado."""
    c = OpenBBClient(settings, cache)
    c._obb = mock_obb   # inyectar mock directamente (lazy import bypass)
    return c


@pytest.fixture
def client_no_obb(settings: Settings, cache: ResponseCache) -> OpenBBClient:
    """OpenBBClient sin SDK — simula OpenBB no instalado."""
    c = OpenBBClient(settings, cache)
    c._obb = None
    return c
