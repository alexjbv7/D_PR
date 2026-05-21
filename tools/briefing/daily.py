"""
Daily briefing CLI::

    python -m tools.briefing.daily --date 2026-05-19
    python -m tools.briefing.daily --date 2026-05-19 --discord-webhook $DISCORD_WEBHOOK_URL
"""
from __future__ import annotations

import asyncio
from datetime import date as date_cls
from pathlib import Path

import click
import httpx

from .discord_formatter import build_daily_payload, post_to_discord
from .metrics_collector import MetricsCollector
from .renderer import render_template
from .slack_formatter import md_to_slack_mrkdwn
from .wiring import build_collector

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


@click.command()
@click.option("--date", "date_str", required=True, help="YYYY-MM-DD (America/New_York)")
@click.option("--output-dir", default=str(_DEFAULT_OUTPUT), show_default=True)
@click.option("--slack-webhook", envvar="SLACK_WEBHOOK", default=None)
@click.option("--discord-webhook", envvar="DISCORD_WEBHOOK_URL", default=None)
def main(date_str: str, output_dir: str, slack_webhook: str | None, discord_webhook: str | None) -> None:
    """Generate daily briefing markdown; optionally post to Slack and/or Discord."""
    asyncio.run(_main(date_str, output_dir, slack_webhook, discord_webhook))


async def _main(
    date_str: str,
    output_dir: str,
    slack_webhook: str | None,
    discord_webhook: str | None,
) -> None:
    target_date = date_cls.fromisoformat(date_str)
    collector = await _build_collector()
    metrics = await collector.collect_daily(target_date)
    md = render_template("daily.md.j2", metrics=metrics)

    out_path = Path(output_dir) / f"{date_str}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    click.echo(f"Wrote {out_path}")

    if slack_webhook:
        slack_text = md_to_slack_mrkdwn(md)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                slack_webhook,
                json={"text": slack_text},
                timeout=10.0,
            )
            response.raise_for_status()
        click.echo("Sent to Slack")

    if discord_webhook:
        payload = build_daily_payload(
            date_str=date_str,
            pnl_realized=str(metrics.get("pnl_realized", "N/A")),
            pnl_pct=float(metrics.get("pnl_pct", 0.0)),
            trades_total=int(metrics.get("trades_total", 0)),
            alerts_count=int(metrics.get("alerts_count", 0)),
            open_positions_count=int(metrics.get("open_positions_count", 0)),
            drift_events=int(metrics.get("drift_events", 0)),
            md_full=md,
        )
        await post_to_discord(discord_webhook, payload)
        click.echo("Sent to Discord")


async def _build_collector() -> MetricsCollector:
    return build_collector()


if __name__ == "__main__":
    main()