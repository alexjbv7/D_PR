"""
Tests for DiscordSignalNotifier (app/signal_notifier.py).

Coverage
--------
- format_signal_embed: BUY green, SELL red, correct fields
- DiscordSignalNotifier.from_env: reads DISCORD_SIGNAL_WEBHOOK
- notify: skips when disabled, skips low confidence, posts valid signal
- _post: handles 429 with Retry-After, handles 5xx retry, handles success
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.signal_notifier import (
    DiscordSignalNotifier,
    _COLOR_GREEN,
    _COLOR_RED,
    format_signal_embed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _signal(direction: int = 1, confidence: float = 0.45, **kwargs) -> dict:
    base = {
        "event_id":      "test-001",
        "event_type":    "TradingSignalEvent",
        "ts":            "2026-06-07T22:00:00+00:00",
        "symbol":        "BTC/USD",
        "strategy":      "regime_adaptive",
        "direction":     direction,
        "p_win":         0.62,
        "confidence":    confidence,
        "regime":        "trending_up",
        "position_size": 0.02,
        "venue":         "alpaca",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# format_signal_embed
# ---------------------------------------------------------------------------


class TestFormatSignalEmbed:
    def test_buy_is_green(self):
        body = format_signal_embed(_signal(direction=1))
        assert body["embeds"][0]["color"] == _COLOR_GREEN

    def test_sell_is_red(self):
        body = format_signal_embed(_signal(direction=-1))
        assert body["embeds"][0]["color"] == _COLOR_RED

    def test_content_not_empty(self):
        body = format_signal_embed(_signal())
        assert body["content"]
        assert "BTC/USD" in body["content"]

    def test_content_within_2000_chars(self):
        body = format_signal_embed(_signal())
        assert len(body["content"]) <= 2000

    def test_embed_has_required_fields(self):
        body = format_signal_embed(_signal())
        embed = body["embeds"][0]
        assert "title" in embed
        assert "color" in embed
        assert "fields" in embed
        assert "footer" in embed

    def test_field_names_present(self):
        body = format_signal_embed(_signal())
        field_names = {f["name"] for f in body["embeds"][0]["fields"]}
        assert "p_win" in field_names
        assert "Confidence" in field_names
        assert "Size (kelly)" in field_names

    def test_symbol_in_title(self):
        body = format_signal_embed(_signal(symbol="ETH/USD"))
        assert "ETH/USD" in body["embeds"][0]["title"]

    def test_buy_label_in_content(self):
        body = format_signal_embed(_signal(direction=1))
        assert "BUY" in body["content"]

    def test_sell_label_in_content(self):
        body = format_signal_embed(_signal(direction=-1))
        assert "SELL" in body["content"]

    def test_p_win_formatted_as_percent(self):
        body = format_signal_embed(_signal(p_win=0.62))
        fields = {f["name"]: f["value"] for f in body["embeds"][0]["fields"]}
        assert "62.00%" in fields["p_win"]

    def test_position_size_formatted_as_percent(self):
        body = format_signal_embed(_signal(position_size=0.02))
        fields = {f["name"]: f["value"] for f in body["embeds"][0]["fields"]}
        assert "2.00%" in fields["Size (kelly)"]

    def test_missing_optional_fields_dont_crash(self):
        minimal = {"direction": 1, "symbol": "AAPL"}
        body = format_signal_embed(minimal)
        assert body["embeds"][0]["color"] == _COLOR_GREEN


# ---------------------------------------------------------------------------
# DiscordSignalNotifier construction
# ---------------------------------------------------------------------------


class TestDiscordSignalNotifierConstruction:
    def test_disabled_when_no_webhook(self):
        n = DiscordSignalNotifier(webhook_url="")
        assert not n._enabled

    def test_enabled_when_webhook_set(self):
        n = DiscordSignalNotifier(webhook_url="https://discord.com/api/webhooks/fake")
        assert n._enabled

    def test_from_env_reads_webhook(self):
        with patch.dict(os.environ, {"DISCORD_SIGNAL_WEBHOOK": "https://example.com/hook"}):
            n = DiscordSignalNotifier.from_env()
        assert n.webhook_url == "https://example.com/hook"
        assert n._enabled

    def test_from_env_disabled_when_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "DISCORD_SIGNAL_WEBHOOK"}
        with patch.dict(os.environ, env, clear=True):
            n = DiscordSignalNotifier.from_env()
        assert not n._enabled

    def test_from_env_min_confidence(self):
        with patch.dict(os.environ, {
            "DISCORD_SIGNAL_WEBHOOK": "https://x.com",
            "DISCORD_SIGNAL_MIN_CONFIDENCE": "0.4",
        }):
            n = DiscordSignalNotifier.from_env()
        assert n.min_confidence == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# notify() — disabled path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNotifyDisabled:
    async def test_notify_noop_when_disabled(self):
        n = DiscordSignalNotifier(webhook_url="")
        # Should not raise and should not call _post
        with patch.object(n, "_post", new_callable=AsyncMock) as mock_post:
            await n.notify(_signal())
        mock_post.assert_not_called()

    async def test_notify_skips_low_confidence(self):
        n = DiscordSignalNotifier(
            webhook_url="https://discord.com/api/webhooks/fake",
            min_confidence=0.5,
        )
        with patch.object(n, "_post", new_callable=AsyncMock) as mock_post:
            await n.notify(_signal(confidence=0.3))
            # Allow event loop to flush tasks
            await asyncio.sleep(0)
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# _post() — HTTP behaviour
# ---------------------------------------------------------------------------


def _make_response(status: int, headers: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


@pytest.mark.asyncio
class TestPostBehaviour:
    async def test_post_success(self):
        n = DiscordSignalNotifier(webhook_url="https://discord.com/api/webhooks/fake")
        resp = _make_response(204)

        with patch("app.signal_notifier.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await n._post(_signal())
        mock_client.post.assert_called_once()

    async def test_post_retries_on_5xx(self):
        n = DiscordSignalNotifier(
            webhook_url="https://discord.com/api/webhooks/fake",
            max_retries=2,
        )
        resp_500 = _make_response(500)
        resp_204 = _make_response(204)

        calls = [resp_500, resp_204]

        with patch("app.signal_notifier.httpx.AsyncClient") as mock_client_cls, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=calls)
            mock_client_cls.return_value = mock_client

            await n._post(_signal())
        assert mock_client.post.call_count == 2

    async def test_post_handles_429_with_retry_after(self):
        n = DiscordSignalNotifier(
            webhook_url="https://discord.com/api/webhooks/fake",
            max_retries=2,
        )
        resp_429 = _make_response(429, headers={"Retry-After": "1"})
        resp_204 = _make_response(204)

        with patch("app.signal_notifier.httpx.AsyncClient") as mock_client_cls, \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[resp_429, resp_204])
            mock_client_cls.return_value = mock_client

            await n._post(_signal())
        mock_sleep.assert_called()

    async def test_post_gives_up_after_max_retries(self):
        n = DiscordSignalNotifier(
            webhook_url="https://discord.com/api/webhooks/fake",
            max_retries=2,
        )
        resp_500 = _make_response(500)

        with patch("app.signal_notifier.httpx.AsyncClient") as mock_client_cls, \
             patch("asyncio.sleep", new_callable=AsyncMock):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp_500)
            mock_client_cls.return_value = mock_client

            # Should not raise even after exhausting retries
            await n._post(_signal())
        assert mock_client.post.call_count == 2
