"""Tests for daily briefing generation."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from tools.briefing.metrics_collector import DailyMetrics, OpenPositionRow
from tools.briefing.renderer import render_template
from tools.briefing.slack_formatter import md_to_slack_mrkdwn


def _sample_metrics(**overrides: object) -> DailyMetrics:
    base = DailyMetrics(
        date_et=date(2026, 5, 19),
        equity_start=Decimal("100000"),
        equity_end=Decimal("100500"),
        pnl_realized=Decimal("500"),
        pnl_unrealized=Decimal("0"),
        pnl_realized_pct=0.5,
        trades_total=12,
        trades_by_horizon={"intraday": 8, "swing": 3, "daily": 1},
        win_rate=0.58,
        avg_kelly_used=0.18,
        alpaca_latency_p99_ms=145.2,
        alerts_fired=[],
        reconciler_discrepancies=0,
        drift_events=0,
        pdt_warnings=0,
        open_positions=[],
        next_macro_events_et=["FOMC 14:00 ET"],
        next_earnings_count=3,
        generated_at_utc=datetime(2026, 5, 20, 3, 0, tzinfo=timezone.utc),
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_daily_briefing_generates_all_sections() -> None:
    md = render_template("daily.md.j2", metrics=_sample_metrics())
    for section in (
        "Daily Briefing — 2026-05-19",
        "## P&L",
        "## Operación",
        "## Salud sistema",
        "## Alertas disparadas",
        "## Posiciones abiertas",
        "## Próximos eventos",
    ):
        assert section in md


def test_daily_briefing_handles_empty_alerts() -> None:
    md = render_template("daily.md.j2", metrics=_sample_metrics(alerts_fired=[]))
    assert "Ninguna." in md


def test_daily_briefing_handles_flat_positions() -> None:
    md = render_template("daily.md.j2", metrics=_sample_metrics(open_positions=[]))
    assert "Flat." in md
    assert "| Symbol |" not in md


def test_daily_briefing_handles_missing_current_price() -> None:
    pos = OpenPositionRow(
        symbol="AAPL",
        side="long",
        qty=Decimal("10"),
        avg_entry=Decimal("150"),
        current_price=None,
        unrealized_pnl=None,
    )
    md = render_template("daily.md.j2", metrics=_sample_metrics(open_positions=[pos]))
    assert "—" in md
    assert "None" not in md


def test_slack_formatter_wraps_tables() -> None:
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    out = md_to_slack_mrkdwn(md)
    assert "```" in out
