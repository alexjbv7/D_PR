"""CLI integration tests (stub backend when Postgres unavailable)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from click.testing import CliRunner

from tools.briefing.daily import main as daily_main
from tools.briefing.weekly import main as weekly_main


def test_daily_cli_writes_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        daily_main,
        ["--date", "2026-05-19", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "2026-05-19.md"
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "Daily Briefing — 2026-05-19" in text
    assert "## P&L" in text


def test_weekly_cli_writes_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        weekly_main,
        ["--week", "2026-W20", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "weekly_2026W20.md"
    assert out.is_file()
    assert "Weekly Briefing — 2026-W20" in out.read_text(encoding="utf-8")
