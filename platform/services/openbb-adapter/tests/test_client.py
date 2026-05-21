"""
Tests para OpenBBClient.

Cubre:
  - Retorna datos correctos cuando OpenBB responde
  - Retorna lista vacía cuando OpenBB falla (resilencia)
  - Retorna lista vacía cuando obb=None (SDK no instalado)
  - Cache hit previene llamada al SDK
  - Fallback entre providers
  - Put/call ratio calculado correctamente
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.client import OpenBBClient


# ── get_crypto_ohlcv ─────────────────────────────────────────────────────────

class TestGetCryptoOHLCV:
    async def test_returns_list_of_dicts(self, client):
        data = await client.get_crypto_ohlcv("BTC", interval="1d")
        assert isinstance(data, list)
        assert len(data) == 3
        assert "close" in data[0] or "date" in data[0]

    async def test_no_obb_returns_empty(self, client_no_obb):
        data = await client_no_obb.get_crypto_ohlcv("BTC")
        assert data == []

    async def test_cache_hit_skips_sdk(self, client, mock_redis):
        payload = [{"date": "2026-01-01", "close": 50000.0}]
        mock_redis.get.return_value = json.dumps(payload)
        data = await client.get_crypto_ohlcv("BTC")
        assert data == payload
        # SDK no fue llamado
        client._obb.crypto.price.historical.assert_not_called()

    async def test_provider_error_returns_empty(self, client):
        client._obb.crypto.price.historical.side_effect = Exception("provider down")
        data = await client.get_crypto_ohlcv("BTC")
        assert data == []

    async def test_1m_interval_sets_ttl_60(self, client, mock_redis):
        mock_redis.get.return_value = None  # cache MISS
        await client.get_crypto_ohlcv("BTC", interval="1m")
        call_args = mock_redis.setex.call_args[0]
        assert call_args[1] == 60  # TTL 60s para 1m

    async def test_1d_interval_sets_ttl_3600(self, client, mock_redis):
        mock_redis.get.return_value = None
        await client.get_crypto_ohlcv("BTC", interval="1d")
        call_args = mock_redis.setex.call_args[0]
        assert call_args[1] == 3600

    async def test_symbol_usd_suffix(self, client):
        await client.get_crypto_ohlcv("ETH", interval="1d")
        call_kwargs = client._obb.crypto.price.historical.call_args[1]
        assert call_kwargs["symbol"] == "ETHUSD"


# ── get_fred_series ───────────────────────────────────────────────────────────

class TestGetFredSeries:
    async def test_returns_sorted_records(self, client):
        data = await client.get_fred_series("UNRATE")
        assert isinstance(data, list)
        assert len(data) == 3
        # Ordenados cronológicamente
        dates = [d["date"] for d in data]
        assert dates == sorted(dates)

    async def test_each_record_has_date_and_value(self, client):
        data = await client.get_fred_series("DGS10")
        for record in data:
            assert "date"  in record
            assert "value" in record
            assert isinstance(record["value"], float)

    async def test_no_obb_returns_empty(self, client_no_obb):
        assert await client_no_obb.get_fred_series("UNRATE") == []

    async def test_cache_hit_skips_sdk(self, client, mock_redis):
        payload = [{"date": "2026-01-01", "value": 4.0}]
        mock_redis.get.return_value = json.dumps(payload)
        data = await client.get_fred_series("UNRATE")
        assert data == payload
        client._obb.economy.fred_series.assert_not_called()

    async def test_provider_error_returns_empty(self, client):
        client._obb.economy.fred_series.side_effect = RuntimeError("FRED error")
        assert await client.get_fred_series("BADID") == []


# ── get_yield_curve ───────────────────────────────────────────────────────────

class TestGetYieldCurve:
    async def test_returns_list(self, client):
        data = await client.get_yield_curve()
        assert isinstance(data, list)
        assert len(data) == 5

    async def test_no_obb_returns_empty(self, client_no_obb):
        assert await client_no_obb.get_yield_curve() == []

    async def test_cache_hit_skips_sdk(self, client, mock_redis):
        payload = [{"maturity": "10Y", "rate": 4.5}]
        mock_redis.get.return_value = json.dumps(payload)
        data = await client.get_yield_curve()
        assert data == payload
        client._obb.fixedincome.government.yield_curve.assert_not_called()


# ── get_options_chain ─────────────────────────────────────────────────────────

class TestGetOptionsChain:
    async def test_returns_list_of_contracts(self, client):
        data = await client.get_options_chain("BTC")
        assert isinstance(data, list)
        assert len(data) == 4

    async def test_no_obb_returns_empty(self, client_no_obb):
        assert await client_no_obb.get_options_chain("BTC") == []

    async def test_provider_error_returns_empty(self, client):
        client._obb.derivatives.options.chains.side_effect = Exception("deribit down")
        assert await client.get_options_chain("BTC") == []


# ── get_put_call_ratio ────────────────────────────────────────────────────────

class TestPutCallRatio:
    async def test_correct_ratio(self, client):
        """Mock tiene 2 puts y 2 calls → PCR = 1.0"""
        pcr = await client.get_put_call_ratio("BTC")
        assert pcr == pytest.approx(1.0)

    async def test_no_chain_returns_none(self, client):
        client._obb.derivatives.options.chains.side_effect = Exception("down")
        pcr = await client.get_put_call_ratio("BTC")
        assert pcr is None

    async def test_no_calls_returns_none(self, client, mock_obb):
        """Si no hay calls, división por cero → None."""
        import pandas as pd
        only_puts = mock_obb.derivatives.options.chains.return_value
        only_puts.to_df.return_value = pd.DataFrame({
            "option_type": ["put", "put"],
            "strike":      [50000, 55000],
        })
        pcr = await client.get_put_call_ratio("BTC")
        assert pcr is None


# ── get_sec_filings ───────────────────────────────────────────────────────────

class TestGetSECFilings:
    async def test_returns_filings(self, client):
        data = await client.get_sec_filings("IBIT")
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_no_obb_returns_empty(self, client_no_obb):
        assert await client_no_obb.get_sec_filings("IBIT") == []


# ── get_news ──────────────────────────────────────────────────────────────────

class TestGetNews:
    async def test_no_obb_returns_empty(self, client_no_obb):
        assert await client_no_obb.get_news(["BTC", "ETH"]) == []

    async def test_provider_error_returns_empty(self, client):
        client._obb.news.world.news.side_effect = Exception("fmp down")
        result = await client.get_news(["BTC"])
        assert result == []
