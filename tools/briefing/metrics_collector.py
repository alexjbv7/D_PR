"""
MetricsCollector — Prometheus + Postgres + Alpaca for briefing reports.

Production wiring uses real clients; tests inject a stub backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


@dataclass
class OpenPositionRow:
    """Flattened position row for Jinja templates."""

    symbol: str
    side: Any
    qty: Decimal
    avg_entry: Decimal
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None


@dataclass
class DailyMetrics:
    date_et: date
    equity_start: Decimal
    equity_end: Decimal
    pnl_realized: Decimal
    pnl_unrealized: Decimal
    pnl_realized_pct: float
    trades_total: int
    trades_by_horizon: dict[str, int]
    win_rate: float
    avg_kelly_used: float
    alpaca_latency_p99_ms: float
    alerts_fired: list[dict[str, str]]
    reconciler_discrepancies: int
    drift_events: int
    pdt_warnings: int
    open_positions: list[OpenPositionRow]
    next_macro_events_et: list[str]
    next_earnings_count: int
    generated_at_utc: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )


@dataclass
class WeeklyMetrics:
    week_iso: str
    start_date_et: date
    end_date_et: date
    equity_start: Decimal
    equity_end: Decimal
    pnl_total: Decimal
    pnl_by_horizon: dict[str, Decimal]
    sharpe_rolling_7d: float
    sharpe_rolling_30d: float | None
    max_drawdown_pct: float
    trades_total: int
    top_5_winners: list[dict[str, Any]]
    top_5_losers: list[dict[str, Any]]
    drift_events_by_feature: dict[str, int]
    alerts_by_severity: dict[str, int]
    allocator_snapshot: dict[str, dict[str, float]]
    auto_detected_todos: list[str]
    daily_pnl_series: list[float] = field(default_factory=list)
    allocator_decisions_low_sample: list[dict[str, Any]] = field(default_factory=list)
    generated_at_utc: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
    )


@runtime_checkable
class MetricsBackend(Protocol):
    async def equity_at(self, ts: datetime) -> Decimal: ...
    async def pnl_realized(self, start: datetime, end: datetime) -> Decimal: ...
    async def pnl_unrealized(self, ts: datetime) -> Decimal: ...
    async def trades_count(self, start: datetime, end: datetime) -> int: ...
    async def trades_by_horizon(self, start: datetime, end: datetime) -> dict[str, int]: ...
    async def win_rate(self, start: datetime, end: datetime) -> float: ...
    async def avg_kelly(self, start: datetime, end: datetime) -> float: ...
    async def prom_p99(self, metric: str, start: datetime, end: datetime) -> float: ...
    async def alerts_fired(self, start: datetime, end: datetime) -> list[dict[str, str]]: ...
    async def recon_count(self, start: datetime, end: datetime) -> int: ...
    async def drift_count(self, start: datetime, end: datetime) -> int: ...
    async def pdt_count(self, start: datetime, end: datetime) -> int: ...
    async def open_positions(self) -> list[OpenPositionRow]: ...
    async def next_macro_events(self) -> list[str]: ...
    async def next_earnings_count(self) -> int: ...
    async def drift_by_feature(self, start: datetime, end: datetime) -> dict[str, int]: ...
    async def alerts_by_severity(self, start: datetime, end: datetime) -> dict[str, int]: ...
    async def allocator_snapshot(self) -> dict[str, dict[str, float]]: ...
    async def top_trades(
        self, start: datetime, end: datetime, *, winners: bool, limit: int,
    ) -> list[dict[str, Any]]: ...
    async def sharpe_rolling(self, end: datetime, days: int) -> float | None: ...
    async def max_drawdown_pct(self, start: datetime, end: datetime) -> float: ...
    async def daily_pnl_series(self, start: date, end: date) -> list[float]: ...
    async def allocator_low_sample_decisions(
        self, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]: ...
    async def pnl_by_horizon(self, start: datetime, end: datetime) -> dict[str, Decimal]: ...


class StubMetricsBackend:
    """Zero/empty backend for dry-runs and unit tests."""

    async def equity_at(self, ts: datetime) -> Decimal:
        return Decimal("100000")

    async def pnl_realized(self, start: datetime, end: datetime) -> Decimal:
        return Decimal("0")

    async def pnl_unrealized(self, ts: datetime) -> Decimal:
        return Decimal("0")

    async def trades_count(self, start: datetime, end: datetime) -> int:
        return 0

    async def trades_by_horizon(self, start: datetime, end: datetime) -> dict[str, int]:
        return {"intraday": 0, "swing": 0, "daily": 0}

    async def win_rate(self, start: datetime, end: datetime) -> float:
        return 0.0

    async def avg_kelly(self, start: datetime, end: datetime) -> float:
        return 0.0

    async def prom_p99(self, metric: str, start: datetime, end: datetime) -> float:
        return 0.0

    async def alerts_fired(self, start: datetime, end: datetime) -> list[dict[str, str]]:
        return []

    async def recon_count(self, start: datetime, end: datetime) -> int:
        return 0

    async def drift_count(self, start: datetime, end: datetime) -> int:
        return 0

    async def pdt_count(self, start: datetime, end: datetime) -> int:
        return 0

    async def open_positions(self) -> list[OpenPositionRow]:
        return []

    async def next_macro_events(self) -> list[str]:
        return []

    async def next_earnings_count(self) -> int:
        return 0

    async def drift_by_feature(self, start: datetime, end: datetime) -> dict[str, int]:
        return {}

    async def alerts_by_severity(self, start: datetime, end: datetime) -> dict[str, int]:
        return {"P0": 0, "P1": 0, "P2": 0}

    async def allocator_snapshot(self) -> dict[str, dict[str, float]]:
        return {
            h: {"alpha": 20.0, "beta": 20.0, "mean": 0.5}
            for h in ("intraday", "swing", "daily")
        }

    async def top_trades(
        self, start: datetime, end: datetime, *, winners: bool, limit: int,
    ) -> list[dict[str, Any]]:
        return []

    async def sharpe_rolling(self, end: datetime, days: int) -> float | None:
        if days > 14:
            return None
        return 0.0

    async def max_drawdown_pct(self, start: datetime, end: datetime) -> float:
        return 0.0

    async def daily_pnl_series(self, start: date, end: date) -> list[float]:
        days = (end - start).days + 1
        return [0.0] * max(days, 1)

    async def allocator_low_sample_decisions(
        self, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]:
        return []

    async def pnl_by_horizon(self, start: datetime, end: datetime) -> dict[str, Decimal]:
        return {"intraday": Decimal("0"), "swing": Decimal("0"), "daily": Decimal("0")}


def _day_bounds_et(date_et: date) -> tuple[datetime, datetime]:
    start_et = datetime.combine(date_et, datetime.min.time(), tzinfo=_ET)
    start_utc = start_et.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=1)
    return start_utc, end_utc


def _parse_iso_week(week_iso: str) -> tuple[date, date]:
    """Parse ``2026-W20`` → (Monday, Sunday) in ET."""
    year_str, week_str = week_iso.upper().split("-W", maxsplit=1)
    year, week = int(year_str), int(week_str)
    monday = date.fromisocalendar(year, week, 1)
    sunday = date.fromisocalendar(year, week, 7)
    return monday, sunday


class MetricsCollector:
    def __init__(self, backend: MetricsBackend) -> None:
        self._backend = backend

    async def collect_daily(self, date_et: date) -> DailyMetrics:
        start_utc, end_utc = _day_bounds_et(date_et)
        equity_start = await self._backend.equity_at(start_utc)
        equity_end = await self._backend.equity_at(end_utc)
        pnl_realized = await self._backend.pnl_realized(start_utc, end_utc)
        pnl_unrealized = await self._backend.pnl_unrealized(end_utc)
        pct = 0.0
        if equity_start > 0:
            pct = float(pnl_realized / equity_start * Decimal("100"))
        return DailyMetrics(
            date_et=date_et,
            equity_start=equity_start,
            equity_end=equity_end,
            pnl_realized=pnl_realized,
            pnl_unrealized=pnl_unrealized,
            pnl_realized_pct=pct,
            trades_total=await self._backend.trades_count(start_utc, end_utc),
            trades_by_horizon=await self._backend.trades_by_horizon(start_utc, end_utc),
            win_rate=await self._backend.win_rate(start_utc, end_utc),
            avg_kelly_used=await self._backend.avg_kelly(start_utc, end_utc),
            alpaca_latency_p99_ms=await self._backend.prom_p99(
                "alpaca_submit_latency_seconds", start_utc, end_utc,
            ),
            alerts_fired=await self._backend.alerts_fired(start_utc, end_utc),
            reconciler_discrepancies=await self._backend.recon_count(start_utc, end_utc),
            drift_events=await self._backend.drift_count(start_utc, end_utc),
            pdt_warnings=await self._backend.pdt_count(start_utc, end_utc),
            open_positions=await self._backend.open_positions(),
            next_macro_events_et=await self._backend.next_macro_events(),
            next_earnings_count=await self._backend.next_earnings_count(),
        )

    async def collect_weekly(self, week_iso: str) -> WeeklyMetrics:
        start_et, end_et = _parse_iso_week(week_iso)
        start_utc, _ = _day_bounds_et(start_et)
        _, end_utc = _day_bounds_et(end_et)
        equity_start = await self._backend.equity_at(start_utc)
        equity_end = await self._backend.equity_at(end_utc)
        pnl_total = await self._backend.pnl_realized(start_utc, end_utc)
        pnl_by = await self._backend.pnl_by_horizon(start_utc, end_utc)
        todos: list[str] = []
        pdt = await self._backend.pdt_count(start_utc, end_utc)
        if pdt > 10:
            todos.append(
                f"PDT warnings={pdt} this week — review position sizing on repeat symbols",
            )
        return WeeklyMetrics(
            week_iso=week_iso,
            start_date_et=start_et,
            end_date_et=end_et,
            equity_start=equity_start,
            equity_end=equity_end,
            pnl_total=pnl_total,
            pnl_by_horizon=pnl_by,
            sharpe_rolling_7d=(await self._backend.sharpe_rolling(end_utc, 7)) or 0.0,
            sharpe_rolling_30d=await self._backend.sharpe_rolling(end_utc, 30),
            max_drawdown_pct=await self._backend.max_drawdown_pct(start_utc, end_utc),
            trades_total=await self._backend.trades_count(start_utc, end_utc),
            top_5_winners=await self._backend.top_trades(
                start_utc, end_utc, winners=True, limit=5,
            ),
            top_5_losers=await self._backend.top_trades(
                start_utc, end_utc, winners=False, limit=5,
            ),
            drift_events_by_feature=await self._backend.drift_by_feature(start_utc, end_utc),
            alerts_by_severity=await self._backend.alerts_by_severity(start_utc, end_utc),
            allocator_snapshot=await self._backend.allocator_snapshot(),
            auto_detected_todos=todos,
            daily_pnl_series=await self._backend.daily_pnl_series(start_et, end_et),
            allocator_decisions_low_sample=await self._backend.allocator_low_sample_decisions(
                start_utc, end_utc,
            ),
        )
