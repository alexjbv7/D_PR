"""Tests for weekly briefing generation."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from tools.briefing.metrics_collector import MetricsCollector, StubMetricsBackend, _parse_iso_week
from tools.briefing.renderer import render_template
from tools.briefing.weekly import _sparkline


@pytest.mark.asyncio
async def test_weekly_reporter_generates_report() -> None:
    collector = MetricsCollector(StubMetricsBackend())
    metrics = await collector.collect_weekly("2026-W20")
    md = render_template(
        "weekly.md.j2",
        metrics=metrics,
        pnl_sparkline=_sparkline(metrics.daily_pnl_series),
    )
    for section in (
        "Weekly Briefing — 2026-W20",
        "## Cumulative P&L",
        "## Risk metrics",
        "## Top 5 winners",
        "## Allocator posterior snapshot",
        "## Auto-detected TODOs",
    ):
        assert section in md


def test_parse_iso_week() -> None:
    start, end = _parse_iso_week("2026-W20")
    assert start.isocalendar()[1] == 20
    assert (end - start).days == 6


def test_sparkline_non_empty() -> None:
    assert len(_sparkline([0.0, 1.0, 0.5, 2.0])) >= 4
