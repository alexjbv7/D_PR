"""Discord webhook formatter + sender for briefings.

Discord webhook constraints:
- ``content`` max 2000 chars
- Embed ``description`` max 4096 chars, total embed max 6000 chars
- Max 10 embeds per message
- Rate limit: 5 req / 2s per webhook (HTTP 429 with ``Retry-After`` header)

Design:
- Always include a short TL;DR in ``content`` (visible in mobile notif).
- Put the full markdown briefing inside an embed ``description`` (truncated if
  needed). The full ``.md`` lives on disk; Discord is a glanceable summary.
- ``post_to_discord()`` is idempotent on transient errors (429, 5xx) via tenacity.
- Color coding: green (positive PnL), red (negative or alerts), gray (flat/info).
- ``post_to_discord`` accepts an optional ``client`` for DI in tests.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_DISCORD_DESC_MAX = 4096
_DISCORD_CONTENT_MAX = 2000
_DISCORD_EMBED_TOTAL_MAX = 6000
_TRUNC_NOTE = "\n\n_…truncado. Ver .md completo en tools/briefing/output/._"

# Discord color codes (decimal RGB)
COLOR_GREEN = 0x2ECC71  # positive PnL / OK
COLOR_RED = 0xE74C3C    # negative PnL / alerts
COLOR_GRAY = 0x95A5A6   # flat / informational
COLOR_BLUE = 0x3498DB   # drill / scheduled action
COLOR_YELLOW = 0xF1C40F  # warning


@dataclass(frozen=True)
class DiscordPayload:
    """Discord webhook POST body."""

    content: str
    embed_title: str
    embed_description: str
    embed_color: int
    embed_fields: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        desc = self.embed_description
        if len(desc) > _DISCORD_DESC_MAX:
            desc = desc[: _DISCORD_DESC_MAX - len(_TRUNC_NOTE)] + _TRUNC_NOTE
        content = self.content[:_DISCORD_CONTENT_MAX]
        embed: dict[str, Any] = {
            "title": self.embed_title[:256],
            "description": desc,
            "color": self.embed_color,
        }
        if self.embed_fields:
            embed["fields"] = self.embed_fields[:25]  # Discord max 25 fields
        return {"content": content, "embeds": [embed]}


def build_daily_payload(
    *,
    date_str: str,
    pnl_realized: str,
    pnl_pct: float,
    trades_total: int,
    alerts_count: int,
    open_positions_count: int,
    drift_events: int,
    md_full: str,
) -> DiscordPayload:
    """Construct payload for a daily briefing."""
    if alerts_count > 0 or pnl_pct < 0:
        color = COLOR_RED
    elif pnl_pct > 0:
        color = COLOR_GREEN
    else:
        color = COLOR_GRAY
    tldr = (
        f"**Daily briefing — {date_str}**  "
        f"PnL `{pnl_realized}` ({pnl_pct:+.2f}%) · "
        f"Trades `{trades_total}` · Alerts `{alerts_count}` · "
        f"Drift `{drift_events}`"
    )
    fields = [
        {"name": "PnL realizado", "value": f"`{pnl_realized}` ({pnl_pct:+.2f}%)", "inline": True},
        {"name": "Trades", "value": f"`{trades_total}`", "inline": True},
        {"name": "Posiciones abiertas", "value": f"`{open_positions_count}`", "inline": True},
        {"name": "Alertas", "value": f"`{alerts_count}`", "inline": True},
        {"name": "Drift events", "value": f"`{drift_events}`", "inline": True},
    ]
    return DiscordPayload(
        content=tldr,
        embed_title=f"Daily Briefing — {date_str}",
        embed_description=md_full,
        embed_color=color,
        embed_fields=fields,
    )


def build_weekly_payload(
    *,
    week_iso: str,
    weekly_pnl: str,
    weekly_pnl_pct: float,
    trades_total: int,
    sharpe_7d: float,
    sparkline: str,
    md_full: str,
    max_dd_pct: float | None = None,
) -> DiscordPayload:
    """Construct payload for a weekly review.

    The natural weekly KPI in :class:`WeeklyMetrics` is rolling Sharpe (no
    ``win_rate`` field exists at weekly granularity). Optional ``max_dd_pct``
    surfaces drawdown — a more useful weekly risk signal than win rate.
    """
    color = COLOR_GREEN if weekly_pnl_pct > 0 else COLOR_RED if weekly_pnl_pct < 0 else COLOR_GRAY
    tldr = (
        f"**Weekly review — {week_iso}**  "
        f"PnL `{weekly_pnl}` ({weekly_pnl_pct:+.2f}%) · "
        f"Sharpe7d `{sharpe_7d:+.2f}` · {sparkline}"
    )
    fields = [
        {"name": "PnL semana", "value": f"`{weekly_pnl}` ({weekly_pnl_pct:+.2f}%)", "inline": True},
        {"name": "Trades", "value": f"`{trades_total}`", "inline": True},
        {"name": "Sharpe 7d", "value": f"`{sharpe_7d:+.2f}`", "inline": True},
    ]
    if max_dd_pct is not None:
        fields.append(
            {"name": "Max DD", "value": f"`{max_dd_pct:+.2f}%`", "inline": True},
        )
    fields.append(
        {"name": "PnL diario", "value": f"`{sparkline}`", "inline": False},
    )
    return DiscordPayload(
        content=tldr,
        embed_title=f"Weekly Review — {week_iso}",
        embed_description=md_full,
        embed_color=color,
        embed_fields=fields,
    )


def build_simple_payload(
    *,
    title: str,
    summary: str,
    body: str = "",
    color: int = COLOR_BLUE,
) -> DiscordPayload:
    """Generic payload (drills, ad-hoc alerts)."""
    return DiscordPayload(
        content=summary,
        embed_title=title,
        embed_description=body or summary,
        embed_color=color,
        embed_fields=[],
    )


async def post_to_discord(
    webhook_url: str,
    payload: DiscordPayload,
    *,
    timeout: float = 10.0,
    max_attempts: int = 5,
    client: httpx.AsyncClient | None = None,
) -> None:
    """POST to a Discord webhook with retry on 429 / 5xx.

    Discord returns ``Retry-After`` in seconds (float) on 429. We honor it
    via tenacity's exponential backoff capped at 30s.

    Parameters
    ----------
    client
        Optional pre-built ``AsyncClient`` — useful for tests with
        ``httpx.MockTransport``. When ``None`` (production), a default client
        is created and closed within this call.
    """
    body = payload.to_json()

    async def _send(c: httpx.AsyncClient) -> httpx.Response:
        resp = await c.post(webhook_url, json=body, timeout=timeout)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "2"))
            await asyncio.sleep(min(retry_after, 30.0))
            resp.raise_for_status()
        resp.raise_for_status()
        return resp

    async def _run(c: httpx.AsyncClient) -> None:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
            reraise=True,
        ):
            with attempt:
                await _send(c)

    if client is None:
        async with httpx.AsyncClient() as owned:
            await _run(owned)
    else:
        await _run(client)
