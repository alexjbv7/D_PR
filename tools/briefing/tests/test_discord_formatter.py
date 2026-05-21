"""Tests for the Discord webhook formatter + sender.

Covers:
- payload construction (daily, weekly, simple)
- color selection by PnL sign / alerts presence
- to_json() respects Discord size limits (content/title/description/fields)
- truncation note appended when description exceeds 4096 chars
- post_to_discord retries on 429 and 5xx, succeeds on 204
- weekly payload uses Sharpe semantics (not win_rate) and surfaces max_dd_pct
"""
from __future__ import annotations

import httpx
import pytest

from briefing.discord_formatter import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    DiscordPayload,
    build_daily_payload,
    build_simple_payload,
    build_weekly_payload,
    post_to_discord,
)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def test_daily_payload_green_on_positive_pnl_no_alerts() -> None:
    payload = build_daily_payload(
        date_str="2026-05-19",
        pnl_realized="500.00",
        pnl_pct=0.5,
        trades_total=12,
        alerts_count=0,
        open_positions_count=3,
        drift_events=0,
        md_full="# Daily briefing — 2026-05-19\n\nAll systems nominal.",
    )
    assert payload.embed_color == COLOR_GREEN
    assert "2026-05-19" in payload.content
    assert "+0.50%" in payload.content
    assert len(payload.embed_fields) == 5


def test_daily_payload_red_on_negative_pnl() -> None:
    payload = build_daily_payload(
        date_str="2026-05-19",
        pnl_realized="-300.00",
        pnl_pct=-0.3,
        trades_total=8,
        alerts_count=0,
        open_positions_count=1,
        drift_events=0,
        md_full="negative day",
    )
    assert payload.embed_color == COLOR_RED


def test_daily_payload_red_when_alerts_present_even_with_positive_pnl() -> None:
    payload = build_daily_payload(
        date_str="2026-05-19",
        pnl_realized="100.00",
        pnl_pct=0.1,
        trades_total=5,
        alerts_count=2,
        open_positions_count=0,
        drift_events=0,
        md_full="alerts fired",
    )
    assert payload.embed_color == COLOR_RED


def test_daily_payload_gray_when_flat() -> None:
    payload = build_daily_payload(
        date_str="2026-05-19",
        pnl_realized="0.00",
        pnl_pct=0.0,
        trades_total=0,
        alerts_count=0,
        open_positions_count=0,
        drift_events=0,
        md_full="flat day",
    )
    assert payload.embed_color == COLOR_GRAY


def test_weekly_payload_uses_sharpe_not_winrate() -> None:
    """Regression test: weekly payload must surface Sharpe, not 'win_rate'.

    WeeklyMetrics has sharpe_rolling_7d but no win_rate field. The previous
    buggy wiring passed Sharpe (e.g. 0.5) into a slot formatted as ``{:.0%}``
    which would render as "50%". This test pins the corrected contract.
    """
    payload = build_weekly_payload(
        week_iso="2026-W20",
        weekly_pnl="1234.00",
        weekly_pnl_pct=1.23,
        trades_total=40,
        sharpe_7d=0.85,
        sparkline="▁▃▅▆▇▆▅",
        md_full="weekly md",
    )
    assert "Sharpe7d" in payload.content
    assert "+0.85" in payload.content
    assert "85%" not in payload.content
    field_names = {f["name"] for f in payload.embed_fields}
    assert "Sharpe 7d" in field_names
    assert "Win rate" not in field_names


def test_weekly_payload_includes_max_dd_when_provided() -> None:
    payload = build_weekly_payload(
        week_iso="2026-W20",
        weekly_pnl="-500",
        weekly_pnl_pct=-0.5,
        trades_total=30,
        sharpe_7d=-0.2,
        sparkline="▇▆▅▄▃▂▁",
        md_full="weekly md",
        max_dd_pct=-3.4,
    )
    field_names = {f["name"] for f in payload.embed_fields}
    assert "Max DD" in field_names
    assert payload.embed_color == COLOR_RED


def test_weekly_payload_omits_max_dd_when_none() -> None:
    payload = build_weekly_payload(
        week_iso="2026-W20",
        weekly_pnl="0",
        weekly_pnl_pct=0.0,
        trades_total=0,
        sharpe_7d=0.0,
        sparkline="▁▁▁▁▁▁▁",
        md_full="weekly md",
        max_dd_pct=None,
    )
    field_names = {f["name"] for f in payload.embed_fields}
    assert "Max DD" not in field_names


def test_simple_payload_defaults_to_blue() -> None:
    payload = build_simple_payload(
        title="Drill: outage simulation",
        summary="Day-5 drill triggered.",
        body="Detailed body…",
    )
    assert payload.embed_color == COLOR_BLUE
    assert payload.embed_description == "Detailed body…"


def test_simple_payload_body_defaults_to_summary() -> None:
    payload = build_simple_payload(title="t", summary="s")
    assert payload.embed_description == "s"


# ---------------------------------------------------------------------------
# to_json() — Discord size constraints
# ---------------------------------------------------------------------------


def test_to_json_truncates_description_over_4096() -> None:
    long_desc = "x" * 5000
    payload = DiscordPayload(
        content="ok",
        embed_title="t",
        embed_description=long_desc,
        embed_color=COLOR_GRAY,
        embed_fields=[],
    )
    body = payload.to_json()
    desc = body["embeds"][0]["description"]
    assert len(desc) <= 4096
    assert "truncado" in desc


def test_to_json_truncates_content_over_2000() -> None:
    long_content = "y" * 2500
    payload = DiscordPayload(
        content=long_content,
        embed_title="t",
        embed_description="d",
        embed_color=COLOR_GRAY,
        embed_fields=[],
    )
    body = payload.to_json()
    assert len(body["content"]) <= 2000


def test_to_json_truncates_title_over_256() -> None:
    long_title = "z" * 300
    payload = DiscordPayload(
        content="ok",
        embed_title=long_title,
        embed_description="d",
        embed_color=COLOR_GRAY,
        embed_fields=[],
    )
    body = payload.to_json()
    assert len(body["embeds"][0]["title"]) <= 256


def test_to_json_caps_fields_at_25() -> None:
    too_many = [{"name": f"n{i}", "value": f"v{i}", "inline": True} for i in range(30)]
    payload = DiscordPayload(
        content="ok",
        embed_title="t",
        embed_description="d",
        embed_color=COLOR_GRAY,
        embed_fields=too_many,
    )
    body = payload.to_json()
    assert len(body["embeds"][0]["fields"]) == 25


def test_to_json_omits_fields_key_when_empty() -> None:
    payload = DiscordPayload(
        content="ok",
        embed_title="t",
        embed_description="d",
        embed_color=COLOR_GRAY,
        embed_fields=[],
    )
    body = payload.to_json()
    assert "fields" not in body["embeds"][0]


# ---------------------------------------------------------------------------
# post_to_discord — retry / rate-limit handling
# ---------------------------------------------------------------------------


def _make_payload() -> DiscordPayload:
    return DiscordPayload(
        content="hello",
        embed_title="t",
        embed_description="d",
        embed_color=COLOR_BLUE,
        embed_fields=[],
    )


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _fast_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tenacity retries effectively zero-wait so tests stay snappy."""
    import briefing.discord_formatter as mod

    async def _zero(_s: float) -> None:
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", _zero)


@pytest.mark.asyncio
async def test_post_to_discord_success_204() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(204)

    async with _mock_client(handler) as client:
        await post_to_discord(
            "https://discord.test/webhooks/abc",
            _make_payload(),
            client=client,
        )
    assert captured == ["https://discord.test/webhooks/abc"]


@pytest.mark.asyncio
async def test_post_to_discord_retries_then_succeeds_on_5xx() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(204)

    async with _mock_client(handler) as client:
        await post_to_discord(
            "https://discord.test/webhooks/abc",
            _make_payload(),
            max_attempts=5,
            client=client,
        )
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_post_to_discord_honors_retry_after_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On 429, code path reads Retry-After header and sleeps before retrying."""
    calls = {"n": 0}
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"})
        return httpx.Response(204)

    import briefing.discord_formatter as mod

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    async with _mock_client(handler) as client:
        await post_to_discord(
            "https://discord.test/webhooks/abc",
            _make_payload(),
            max_attempts=3,
            client=client,
        )
    assert calls["n"] == 2
    assert slept and slept[0] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_post_to_discord_gives_up_after_max_attempts() -> None:
    """Persistent 5xx must propagate after max_attempts."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    async with _mock_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await post_to_discord(
                "https://discord.test/webhooks/abc",
                _make_payload(),
                max_attempts=2,
                client=client,
            )
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_post_to_discord_sends_correct_json_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content.decode())
        return httpx.Response(204)

    payload = build_daily_payload(
        date_str="2026-05-19",
        pnl_realized="100",
        pnl_pct=0.1,
        trades_total=5,
        alerts_count=0,
        open_positions_count=2,
        drift_events=0,
        md_full="# md",
    )
    async with _mock_client(handler) as client:
        await post_to_discord(
            "https://discord.test/webhooks/abc",
            payload,
            client=client,
        )
    body = captured["body"]
    assert "content" in body
    assert "embeds" in body
    assert len(body["embeds"]) == 1
    assert body["embeds"][0]["color"] == COLOR_GREEN
    assert body["embeds"][0]["title"].startswith("Daily Briefing")
