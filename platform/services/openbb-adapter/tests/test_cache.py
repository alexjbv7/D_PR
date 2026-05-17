"""
Tests para ResponseCache.

Cubre:
  - Cache hit / miss
  - TTL correcto por tipo de clave
  - Serialización JSON de distintos tipos
  - Comportamiento sin Redis (modo degradado)
  - Error handling silencioso
"""
import json
from unittest.mock import AsyncMock

import pytest

from app.cache import ResponseCache, _TTL_MAP, _ttl_for_key


# ── TTL selection ─────────────────────────────────────────────────────────────

class TestTTLSelection:
    def test_crypto_1m_ttl(self):
        assert _ttl_for_key("crypto:1m:BTC:2026-01-01") == 60

    def test_crypto_1h_ttl(self):
        assert _ttl_for_key("crypto:1h:ETH:2026-01-01") == 3_600

    def test_macro_daily_ttl(self):
        assert _ttl_for_key("macro:daily:UNRATE:2020-01-01") == 86_400

    def test_macro_monthly_ttl(self):
        assert _ttl_for_key("macro:monthly:GDP") == 604_800

    def test_options_ttl(self):
        assert _ttl_for_key("options:BTC:all") == 300

    def test_cot_ttl(self):
        assert _ttl_for_key("cot:legacy_fut") == 604_800

    def test_news_ttl(self):
        assert _ttl_for_key("news:BTC,ETH:20") == 300

    def test_unknown_key_defaults(self):
        assert _ttl_for_key("unknown:something:else") == 300


# ── Cache sin Redis ───────────────────────────────────────────────────────────

class TestCacheNoRedis:
    def test_available_false(self, cache_no_redis):
        assert cache_no_redis.available is False

    async def test_get_returns_none(self, cache_no_redis):
        result = await cache_no_redis.get("any:key")
        assert result is None

    async def test_set_returns_false(self, cache_no_redis):
        ok = await cache_no_redis.set("any:key", [{"a": 1}])
        assert ok is False

    async def test_exists_returns_false(self, cache_no_redis):
        assert await cache_no_redis.exists("any:key") is False


# ── Cache con Redis mock ──────────────────────────────────────────────────────

class TestCacheWithRedis:
    def test_available_true(self, cache):
        assert cache.available is True

    async def test_get_miss_returns_none(self, cache, mock_redis):
        mock_redis.get.return_value = None
        result = await cache.get("crypto:1d:BTC:2020-01-01")
        assert result is None

    async def test_get_hit_returns_deserialized(self, cache, mock_redis):
        payload = [{"date": "2026-01-01", "value": 50_000.0}]
        mock_redis.get.return_value = json.dumps(payload)
        result = await cache.get("crypto:1d:BTC:2020-01-01")
        assert result == payload

    async def test_set_calls_setex_with_correct_ttl(self, cache, mock_redis):
        data = [{"date": "2026-01-01", "value": 4.2}]
        await cache.set("macro:daily:UNRATE:2010-01-01", data)
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[1] == 86_400   # TTL para macro daily

    async def test_set_custom_ttl_overrides(self, cache, mock_redis):
        await cache.set("any:key", {"x": 1}, ttl=999)
        args = mock_redis.setex.call_args[0]
        assert args[1] == 999

    async def test_set_serializes_with_default_str(self, cache, mock_redis):
        """Objetos no-JSON-serializables se convierten a str (no lanza excepción)."""
        from datetime import datetime
        data = {"ts": datetime(2026, 1, 1)}
        ok = await cache.set("any:key", data)
        assert ok is True

    async def test_get_bad_json_returns_none(self, cache, mock_redis):
        mock_redis.get.return_value = "not-json-{"
        result = await cache.get("crypto:1d:BTC")
        assert result is None

    async def test_delete_calls_redis(self, cache, mock_redis):
        await cache.delete("some:key")
        mock_redis.delete.assert_called_once_with("some:key")

    async def test_exists_returns_true(self, cache, mock_redis):
        mock_redis.exists.return_value = 1
        assert await cache.exists("some:key") is True

    async def test_redis_error_is_silent_on_get(self, cache, mock_redis):
        mock_redis.get.side_effect = Exception("Redis down")
        result = await cache.get("any:key")
        assert result is None

    async def test_redis_error_is_silent_on_set(self, cache, mock_redis):
        mock_redis.setex.side_effect = Exception("Redis down")
        ok = await cache.set("any:key", [{"a": 1}])
        assert ok is False
