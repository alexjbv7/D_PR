"""
Discord signal notifier for the strategy-orchestrator (FIX: 0 trades visibility).

Sends a formatted Discord embed to a webhook URL every time a TradingSignalEvent
is emitted to Kafka.  Designed as fire-and-forget: failures are logged but never
propagate to the main signal pipeline.

Environment variables
---------------------
DISCORD_SIGNAL_WEBHOOK : str
    Full Discord webhook URL.  If absent or empty, notifications are silently
    skipped (no error raised — keeps the orchestrator functional without Discord).

DISCORD_SIGNAL_MIN_CONFIDENCE : float (default 0.0)
    Only notify when signal["confidence"] >= this value.
    Useful to suppress low-conviction noise.

Embed format
------------
Color:
  GREEN  (#2ECC71) — BUY  (direction > 0)
  RED    (#E74C3C) — SELL (direction < 0)
  GRAY   (#95A5A6) — HOLD (direction == 0, not emitted in practice)

Fields: symbol | direction | p_win | confidence | regime | position_size | venue

References
----------
Discord webhook API: https://discord.com/developers/docs/resources/webhook
Rate limits: 5 req / 2s per webhook (429 with Retry-After header).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Discord color palette
_COLOR_GREEN  = 0x2ECC71   # BUY
_COLOR_RED    = 0xE74C3C   # SELL
_COLOR_GRAY   = 0x95A5A6   # neutral / unknown

_DISCORD_CONTENT_MAX = 2000
_DISCORD_DESC_MAX    = 4096

# Direction labels
_DIR_LABEL = {1: "🟢 BUY", -1: "🔴 SELL", 0: "⬜ HOLD"}
_DIR_COLOR = {1: _COLOR_GREEN, -1: _COLOR_RED, 0: _COLOR_GRAY}


def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def format_signal_embed(signal: dict) -> dict:
    """
    Build a Discord webhook POST body from a TradingSignalEvent dict.

    Parameters
    ----------
    signal : dict
        TradingSignalEvent payload (from _generate_signal).

    Returns
    -------
    dict
        JSON-serialisable body ready for POST to a Discord webhook.
    """
    direction = int(signal.get("direction", 0))
    symbol    = str(signal.get("symbol", "?"))
    p_win     = float(signal.get("p_win", 0.0))
    confidence= float(signal.get("confidence", 0.0))
    regime    = str(signal.get("regime", "?"))
    pos_size  = float(signal.get("position_size", 0.0))
    venue     = str(signal.get("venue", "?"))
    strategy  = str(signal.get("strategy", "?"))
    ts        = signal.get("ts", datetime.now(tz=timezone.utc).isoformat())

    label = _DIR_LABEL.get(direction, "?")
    color = _DIR_COLOR.get(direction, _COLOR_GRAY)

    content = _truncate(
        f"**{label}** `{symbol}` — p_win `{p_win:.2%}` conf `{confidence:.2f}`",
        _DISCORD_CONTENT_MAX,
    )

    embed = {
        "title": f"{label}  {symbol}",
        "color": color,
        "description": (
            f"**Strategy:** `{strategy}`\n"
            f"**Venue:** `{venue}`\n"
            f"**Regime:** `{regime}`\n"
        ),
        "fields": [
            {"name": "p_win",        "value": f"`{p_win:.2%}`",      "inline": True},
            {"name": "Confidence",   "value": f"`{confidence:.4f}`", "inline": True},
            {"name": "Size (kelly)", "value": f"`{pos_size:.2%}`",   "inline": True},
            {"name": "Venue",        "value": f"`{venue}`",          "inline": True},
            {"name": "Regime",       "value": f"`{regime}`",         "inline": True},
        ],
        "footer": {"text": f"quant_bot · {ts[:19]}Z"},
        "timestamp": ts if ts.endswith("Z") or "+" in ts else ts + "Z",
    }

    return {"content": content, "embeds": [embed]}


class DiscordSignalNotifier:
    """
    Async Discord webhook notifier for trading signals.

    Parameters
    ----------
    webhook_url : str | None
        Discord webhook URL.  If None or empty, notify() is a no-op.
    min_confidence : float
        Minimum signal["confidence"] to post (filters low-conviction noise).
    timeout : float
        HTTP request timeout in seconds.
    max_retries : int
        Number of retries on 429 / 5xx before giving up.

    Examples
    --------
    >>> notifier = DiscordSignalNotifier(webhook_url="https://discord.com/api/webhooks/...")
    >>> await notifier.notify(signal)
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        min_confidence: float = 0.0,
        timeout: float = 8.0,
        max_retries: int = 3,
    ) -> None:
        self.webhook_url   = webhook_url or ""
        self.min_confidence = min_confidence
        self.timeout       = timeout
        self.max_retries   = max_retries
        self._enabled      = bool(self.webhook_url)

        if not self._enabled:
            logger.info(
                "discord_notifier.disabled — set DISCORD_SIGNAL_WEBHOOK to enable"
            )

    @classmethod
    def from_env(cls) -> "DiscordSignalNotifier":
        """Construct from environment variables."""
        return cls(
            webhook_url    = os.getenv("DISCORD_SIGNAL_WEBHOOK", ""),
            min_confidence = float(os.getenv("DISCORD_SIGNAL_MIN_CONFIDENCE", "0.0")),
        )

    async def notify(self, signal: dict) -> None:
        """
        Post a signal embed to Discord.  Fire-and-forget: never raises.

        Parameters
        ----------
        signal : dict
            TradingSignalEvent payload from _generate_signal().
        """
        if not self._enabled:
            return

        confidence = float(signal.get("confidence", 0.0))
        if confidence < self.min_confidence:
            logger.debug(
                "discord_notifier.skip_low_confidence conf=%.4f min=%.4f",
                confidence, self.min_confidence,
            )
            return

        asyncio.create_task(self._post(signal))

    async def _post(self, signal: dict) -> None:
        """Internal: POST to webhook with retry on 429/5xx."""
        body = format_signal_embed(signal)
        symbol    = signal.get("symbol", "?")
        direction = signal.get("direction", 0)

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.webhook_url, json=body)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "2"))
                    wait = min(retry_after, 30.0)
                    logger.warning(
                        "discord_notifier.rate_limited attempt=%d retry_in=%.1fs",
                        attempt, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    logger.warning(
                        "discord_notifier.server_error attempt=%d status=%d",
                        attempt, resp.status_code,
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                logger.debug(
                    "discord_notifier.sent symbol=%s direction=%s",
                    symbol, direction,
                )
                return

            except httpx.TransportError as exc:
                logger.warning(
                    "discord_notifier.transport_error attempt=%d err=%s",
                    attempt, exc,
                )
                await asyncio.sleep(2 ** attempt)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "discord_notifier.unexpected_error attempt=%d err=%s",
                    attempt, exc,
                )
                return  # don't retry on unknown errors

        logger.error(
            "discord_notifier.failed_after_retries symbol=%s retries=%d",
            symbol, self.max_retries,
        )
