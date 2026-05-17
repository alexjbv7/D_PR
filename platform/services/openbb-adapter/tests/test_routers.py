"""
Tests para los routers FastAPI del openbb-adapter.

Usa TestClient de FastAPI con el cliente OpenBB mockeado
via dependency_overrides — sin llamadas reales a proveedores.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mockear dependencias externas ANTES de importar app.main
import sys
for _mod in ["openbb", "aiokafka", "aioredis", "apscheduler",
             "apscheduler.schedulers.asyncio",
             "apscheduler.triggers.interval"]:
    sys.modules.setdefault(_mod, MagicMock())

from app.main import app, get_obb_client
from app.client import OpenBBClient
from app.cache import ResponseCache
from app.config import Settings


# ── Client mock para dependency override ─────────────────────────────────────

def _build_mock_client() -> OpenBBClient:
    settings = Settings(kafka_servers="", redis_url="")
    cache    = ResponseCache(redis=None)
    client   = OpenBBClient(settings, cache)
    client._obb = None  # no SDK, todos los métodos retornan []

    # Sobreescribir métodos con AsyncMock que retornan datos concretos
    client.get_crypto_ohlcv  = AsyncMock(return_value=[
        {"date": "2026-01-01", "open": 50000.0, "close": 50500.0, "volume": 1000.0}
    ])
    client.get_fred_series = AsyncMock(return_value=[
        {"date": "2026-01-01", "value": 4.0},
        {"date": "2026-02-01", "value": 4.1},
    ])
    client.get_yield_curve = AsyncMock(return_value=[
        {"maturity": "2Y", "rate": 4.8},
        {"maturity": "10Y", "rate": 4.5},
    ])
    client.get_options_chain = AsyncMock(return_value=[
        {"option_type": "call", "strike": 50000, "implied_volatility": 0.65},
        {"option_type": "put",  "strike": 50000, "implied_volatility": 0.70},
        {"option_type": "call", "strike": 55000, "implied_volatility": 0.60},
        {"option_type": "put",  "strike": 55000, "implied_volatility": 0.75},
    ])
    client.get_sec_filings    = AsyncMock(return_value=[
        {"filed_at": "2026-01-15", "form_type": "10-K", "symbol": "IBIT"}
    ])
    client.get_cftc_cot       = AsyncMock(return_value=[
        {"date": "2026-01-01", "net_speculative": 1500.0}
    ])
    client.get_news           = AsyncMock(return_value=[])
    client.get_crypto_funding_rate = AsyncMock(return_value=[])
    client.get_sec_rss_litigation  = AsyncMock(return_value=[])
    client.get_institutional_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture(scope="module")
def test_client():
    mock_client = _build_mock_client()
    app.dependency_overrides[get_obb_client] = lambda: mock_client
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ts" in data


# ── /macro ────────────────────────────────────────────────────────────────────

class TestMacroRoutes:
    def test_fred_series_returns_list(self, test_client):
        resp = test_client.get("/macro/fred?series_id=UNRATE")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["date"] == "2026-01-01"

    def test_fred_series_not_found_on_empty(self, test_client):
        mock = _build_mock_client()
        mock.get_fred_series = AsyncMock(return_value=[])
        app.dependency_overrides[get_obb_client] = lambda: mock
        resp = test_client.get("/macro/fred?series_id=BADID")
        assert resp.status_code == 404
        app.dependency_overrides[get_obb_client] = lambda: _build_mock_client()

    def test_yield_curve_returns_list(self, test_client):
        resp = test_client.get("/macro/yield_curve")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_series_list_endpoint(self, test_client):
        resp = test_client.get("/macro/series")
        assert resp.status_code == 200
        body = resp.json()
        assert "series" in body
        assert body["count"] == len(body["series"])
        assert "UNRATE" in body["series"]


# ── /crypto ───────────────────────────────────────────────────────────────────

class TestCryptoRoutes:
    def test_ohlcv_btc(self, test_client):
        resp = test_client.get("/crypto/ohlcv/BTC")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert "close" in data[0]

    def test_ohlcv_not_found_on_empty(self, test_client):
        mock = _build_mock_client()
        mock.get_crypto_ohlcv = AsyncMock(return_value=[])
        app.dependency_overrides[get_obb_client] = lambda: mock
        resp = test_client.get("/crypto/ohlcv/UNKNOWN")
        assert resp.status_code == 503
        app.dependency_overrides[get_obb_client] = lambda: _build_mock_client()

    def test_news_returns_list(self, test_client):
        resp = test_client.get("/crypto/news?symbols=BTC,ETH")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ── /derivatives ──────────────────────────────────────────────────────────────

class TestDerivativesRoutes:
    def test_options_chain_btc(self, test_client):
        resp = test_client.get("/derivatives/options/BTC")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 4

    def test_put_call_ratio(self, test_client):
        resp = test_client.get("/derivatives/options/BTC/pcr")
        assert resp.status_code == 200
        body = resp.json()
        assert "put_call_ratio" in body
        assert body["puts"] == 2
        assert body["calls"] == 2
        assert body["put_call_ratio"] == pytest.approx(1.0)

    def test_options_chain_not_found_on_empty(self, test_client):
        mock = _build_mock_client()
        mock.get_options_chain = AsyncMock(return_value=[])
        app.dependency_overrides[get_obb_client] = lambda: mock
        resp = test_client.get("/derivatives/options/UNKNOWN")
        assert resp.status_code == 503
        app.dependency_overrides[get_obb_client] = lambda: _build_mock_client()


# ── /regulators ───────────────────────────────────────────────────────────────

class TestRegulatorsRoutes:
    def test_sec_filings_ibit(self, test_client):
        resp = test_client.get("/regulators/sec/filings/IBIT")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["form_type"] == "10-K"

    def test_sec_filings_not_found_on_empty(self, test_client):
        mock = _build_mock_client()
        mock.get_sec_filings = AsyncMock(return_value=[])
        app.dependency_overrides[get_obb_client] = lambda: mock
        resp = test_client.get("/regulators/sec/filings/NOPE")
        assert resp.status_code == 404
        app.dependency_overrides[get_obb_client] = lambda: _build_mock_client()

    def test_cftc_cot_returns_list(self, test_client):
        resp = test_client.get("/regulators/cftc/cot")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["net_speculative"] == pytest.approx(1500.0)

    def test_sec_rss_returns_list(self, test_client):
        resp = test_client.get("/regulators/sec/rss_litigation")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
