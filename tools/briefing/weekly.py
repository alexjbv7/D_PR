"""
Weekly briefing CLI::

    python -m tools.briefing.weekly --week 2026-W20
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import click
import httpx

from .metrics_collector import MetricsCollector
from .renderer import render_template
from .slack_formatter import md_to_slack_mrkdwn
from .wiring import build_collector

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
_SPARK_CHARS = "▁▂▃▄▅▆▇"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_CHARS[0] * len(values)
    span = hi - lo
    out: list[str] = []
    for v in values:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


@click.command()
@click.option("--week", "week_iso", required=True, help="ISO week e.g. 2026-W20")
@click.option("--output-dir", default=str(_DEFAULT_OUTPUT), show_default=True)
@click.option("--slack-webhook", envvar="SLACK_WEBHOOK", default=None)
def main(week_iso: str, output_dir: str, slack_webhook: str | None) -> None:
    """Generate weekly aggregated briefing markdown."""
    asyncio.run(_main(week_iso, output_dir, slack_webhook))


async def _main(
    week_iso: str,
    output_dir: str,
    slack_webhook: str | None,
) -> None:
    collector = await _build_collector()
    metrics = await collector.collect_weekly(week_iso)
    spark = _sparkline(metrics.daily_pnl_series)
    md = render_template("weekly.md.j2", metrics=metrics, pnl_sparkline=spark)
    safe_name = week_iso.replace("-", "")
    out_path = Path(output_dir) / f"weekly_{safe_name}.md"
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


async def _build_collector() -> MetricsCollector:
    return build_collector()


if __name__ == "__main__":
    main()
