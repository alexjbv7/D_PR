"""
PostgresMetricsBackend — read-only queries for daily/weekly briefings.

Uses psycopg2 in a thread pool so the async collector API stays uniform.
"""
from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")
from zoneinfo import ZoneInfo

import yaml

from .metrics_collector import OpenPositionRow
from .prom_client import query_p99_seconds

_ET = ZoneInfo("America/New_York")
_HORIZON_RE = re.compile(r"(intraday|swing|daily)", re.IGNORECASE)


def _horizon_from_strategy(strategy: str | None) -> str:
    if not strategy:
        return "unknown"
    m = _HORIZON_RE.search(strategy)
    return m.group(1).lower() if m else "unknown"


class PostgresMetricsBackend:
    """Metrics backed by TimescaleDB / Postgres (read-only)."""

    def __init__(
        self,
        dsn: str,
        *,
        prom_url: str | None = None,
        macro_events_path: Path | None = None,
    ) -> None:
        self._dsn = dsn
        self._prom_url = prom_url
        self._macro_path = macro_events_path or (
            Path(__file__).resolve().parents[2] / "data" / "macro" / "events_2026.yaml"
        )

    def _connect(self) -> Any:
        import psycopg2  # type: ignore[import-untyped]

        return psycopg2.connect(self._dsn)

    async def _run(self, fn: Callable[[], _T]) -> _T:
        return await asyncio.to_thread(fn)

    async def equity_at(self, ts: datetime) -> Decimal:
        def _q() -> Decimal:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(SUM(
                            avg_entry * qty + COALESCE(unrealized_pnl, 0)
                        ), 0)
                        FROM orders.positions
                        WHERE ts_updated <= %s
                        """,
                        (ts,),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        return Decimal(str(row[0]))
            return Decimal("0")

        return await self._run(_q)

    async def pnl_realized(self, start: datetime, end: datetime) -> Decimal:
        def _q() -> Decimal:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(SUM(realized_pnl), 0)
                        FROM risk.allocator_updates
                        WHERE ts_utc >= %s AND ts_utc < %s
                        """,
                        (start, end),
                    )
                    row = cur.fetchone()
                    return Decimal(str(row[0])) if row and row[0] is not None else Decimal("0")

        return await self._run(_q)

    async def pnl_unrealized(self, ts: datetime) -> Decimal:
        def _q() -> Decimal:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(SUM(unrealized_pnl), 0)
                        FROM orders.positions
                        WHERE ts_updated <= %s
                        """,
                        (ts,),
                    )
                    row = cur.fetchone()
                    return Decimal(str(row[0])) if row and row[0] is not None else Decimal("0")

        return await self._run(_q)

    async def trades_count(self, start: datetime, end: datetime) -> int:
        def _q() -> int:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM orders.results
                        WHERE status = 'FILLED'
                          AND ts_updated >= %s AND ts_updated < %s
                        """,
                        (start, end),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0

        return await self._run(_q)

    async def trades_by_horizon(self, start: datetime, end: datetime) -> dict[str, int]:
        def _q() -> dict[str, int]:
            counts: dict[str, int] = {"intraday": 0, "swing": 0, "daily": 0}
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT i.strategy, COUNT(*)
                        FROM orders.results r
                        JOIN orders.intents i ON i.intent_id = r.intent_id
                        WHERE r.status = 'FILLED'
                          AND r.ts_updated >= %s AND r.ts_updated < %s
                        GROUP BY i.strategy
                        """,
                        (start, end),
                    )
                    for strategy, cnt in cur.fetchall():
                        h = _horizon_from_strategy(str(strategy) if strategy else None)
                        if h in counts:
                            counts[h] += int(cnt)
            return counts

        return await self._run(_q)

    async def win_rate(self, start: datetime, end: datetime) -> float:
        def _q() -> float:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
                            COUNT(*)
                        FROM risk.allocator_updates
                        WHERE ts_utc >= %s AND ts_utc < %s
                        """,
                        (start, end),
                    )
                    row = cur.fetchone()
                    if not row or not row[1]:
                        return 0.0
                    return float(row[0]) / float(row[1])

        return await self._run(_q)

    async def avg_kelly(self, start: datetime, end: datetime) -> float:
        def _q() -> float:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COALESCE(AVG(kelly_fraction), 0)
                        FROM orders.intents
                        WHERE risk_decision = 'approved'
                          AND ts >= %s AND ts < %s
                          AND kelly_fraction IS NOT NULL
                        """,
                        (start, end),
                    )
                    row = cur.fetchone()
                    return float(row[0]) if row and row[0] is not None else 0.0

        return await self._run(_q)

    async def prom_p99(self, metric: str, start: datetime, end: datetime) -> float:
        if not self._prom_url:
            return 0.0
        secs = await asyncio.to_thread(
            query_p99_seconds, self._prom_url, metric, start, end,
        )
        return secs * 1000.0

    async def alerts_fired(self, start: datetime, end: datetime) -> list[dict[str, str]]:
        def _q() -> list[dict[str, str]]:
            alerts: list[dict[str, str]] = []
            with self._connect() as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute(
                            """
                            SELECT trigger_reason, horizon, ts
                            FROM drift.retrain_history
                            WHERE ts >= %s AND ts < %s
                              AND NOT suppressed
                            ORDER BY ts DESC
                            LIMIT 20
                            """,
                            (start, end),
                        )
                        for reason, horizon, ts in cur.fetchall():
                            alerts.append({
                                "id": f"retrain-{horizon}",
                                "severity": "P1",
                                "message": f"{reason} @ {ts}",
                            })
                    except Exception:
                        pass
            return alerts

        return await self._run(_q)

    async def recon_count(self, start: datetime, end: datetime) -> int:
        return 0

    async def drift_count(self, start: datetime, end: datetime) -> int:
        def _q() -> int:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM drift.psi_history
                        WHERE ts >= %s AND ts < %s
                          AND severity IN ('moderate', 'severe')
                        """,
                        (start, end),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0

        return await self._run(_q)

    async def pdt_count(self, start: datetime, end: datetime) -> int:
        def _q() -> int:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute(
                            """
                            SELECT COUNT(*)
                            FROM risk.position_actions
                            WHERE ts_utc >= %s AND ts_utc < %s
                            """,
                            (start, end),
                        )
                        row = cur.fetchone()
                        return int(row[0]) if row else 0
                    except Exception:
                        return 0

        return await self._run(_q)

    async def open_positions(self) -> list[OpenPositionRow]:
        def _q() -> list[OpenPositionRow]:
            rows: list[OpenPositionRow] = []
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT symbol, side, qty, avg_entry,
                               current_price, unrealized_pnl
                        FROM orders.positions
                        WHERE qty > 0
                        ORDER BY symbol
                        """,
                    )
                    for sym, side, qty, avg, cur_px, upnl in cur.fetchall():
                        rows.append(
                            OpenPositionRow(
                                symbol=str(sym),
                                side=str(side),
                                qty=Decimal(str(qty)),
                                avg_entry=Decimal(str(avg)),
                                current_price=(
                                    Decimal(str(cur_px)) if cur_px is not None else None
                                ),
                                unrealized_pnl=(
                                    Decimal(str(upnl)) if upnl is not None else None
                                ),
                            ),
                        )
            return rows

        return await self._run(_q)

    async def next_macro_events(self) -> list[str]:
        def _load() -> list[str]:
            if not self._macro_path.is_file():
                return []
            raw = yaml.safe_load(self._macro_path.read_text(encoding="utf-8")) or {}
            today = datetime.now(tz=_ET).date()
            tomorrow = today + timedelta(days=1)
            out: list[str] = []
            for ev in raw.get("events", []):
                if not isinstance(ev, dict):
                    continue
                d_str = ev.get("date")
                label = ev.get("label", "MACRO")
                if not d_str:
                    continue
                d = date.fromisoformat(str(d_str))
                if today <= d <= tomorrow:
                    out.append(f"{label} {d_str} ET")
            return out

        return await self._run(_load)

    async def next_earnings_count(self) -> int:
        return 0

    async def drift_by_feature(
        self, start: datetime, end: datetime,
    ) -> dict[str, int]:
        def _q() -> dict[str, int]:
            out: dict[str, int] = {}
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT feature_name, COUNT(*)
                        FROM drift.psi_history
                        WHERE ts >= %s AND ts < %s
                          AND severity IN ('moderate', 'severe')
                        GROUP BY feature_name
                        ORDER BY COUNT(*) DESC
                        LIMIT 20
                        """,
                        (start, end),
                    )
                    for feat, cnt in cur.fetchall():
                        out[str(feat)] = int(cnt)
            return out

        return await self._run(_q)

    async def alerts_by_severity(
        self, start: datetime, end: datetime,
    ) -> dict[str, int]:
        fired = await self.alerts_fired(start, end)
        return {
            "P0": sum(1 for a in fired if a.get("severity") == "P0"),
            "P1": sum(1 for a in fired if a.get("severity") == "P1"),
            "P2": sum(1 for a in fired if a.get("severity") == "P2"),
        }

    async def allocator_snapshot(self) -> dict[str, dict[str, float]]:
        def _q() -> dict[str, dict[str, float]]:
            snap: dict[str, dict[str, float]] = {}
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT horizon, alpha, beta
                        FROM risk.allocator_state
                        ORDER BY horizon
                        """,
                    )
                    for horizon, alpha, beta in cur.fetchall():
                        a = float(alpha)
                        b = float(beta)
                        snap[str(horizon)] = {
                            "alpha": a,
                            "beta": b,
                            "mean": a / (a + b) if (a + b) > 0 else 0.5,
                        }
            return snap

        return await self._run(_q)

    async def top_trades(
        self,
        start: datetime,
        end: datetime,
        *,
        winners: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        def _q() -> list[dict[str, Any]]:
            order = "DESC" if winners else "ASC"
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT trade_id, realized_pnl, horizon
                        FROM risk.allocator_updates
                        WHERE ts_utc >= %s AND ts_utc < %s
                        ORDER BY realized_pnl {order}
                        LIMIT %s
                        """,
                        (start, end, limit),
                    )
                    return [
                        {
                            "symbol": str(trade_id),
                            "pnl": Decimal(str(pnl)),
                            "reason": str(horizon),
                        }
                        for trade_id, pnl, horizon in cur.fetchall()
                    ]

        return await self._run(_q)

    async def sharpe_rolling(self, end: datetime, days: int) -> float | None:
        start = end - timedelta(days=days)
        series = await self.daily_pnl_series(start.date(), end.date())
        if len(series) < 2:
            return None if days > 14 else 0.0
        import statistics

        mean = statistics.mean(series)
        stdev = statistics.pstdev(series)
        if stdev == 0:
            return 0.0
        return float(mean / stdev * (252 ** 0.5))

    async def max_drawdown_pct(self, start: datetime, end: datetime) -> float:
        series = await self.daily_pnl_series(start.date(), end.date())
        if not series:
            return 0.0
        peak = 0.0
        equity = 0.0
        max_dd = 0.0
        for pnl in series:
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                dd = (peak - equity) / peak * 100.0
                max_dd = max(max_dd, dd)
        return max_dd

    async def daily_pnl_series(self, start: date, end: date) -> list[float]:
        def _q() -> list[float]:
            out: list[float] = []
            d = start
            while d <= end:
                day_start = datetime.combine(d, datetime.min.time(), tzinfo=_ET).astimezone(
                    timezone.utc,
                )
                day_end = day_start + timedelta(days=1)
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT COALESCE(SUM(realized_pnl), 0)
                            FROM risk.allocator_updates
                            WHERE ts_utc >= %s AND ts_utc < %s
                            """,
                            (day_start, day_end),
                        )
                        row = cur.fetchone()
                        out.append(float(row[0]) if row and row[0] is not None else 0.0)
                d += timedelta(days=1)
            return out

        return await self._run(_q)

    async def pnl_by_horizon(self, start: datetime, end: datetime) -> dict[str, Decimal]:
        def _q() -> dict[str, Decimal]:
            out = {h: Decimal("0") for h in ("intraday", "swing", "daily")}
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT horizon, COALESCE(SUM(realized_pnl), 0)
                        FROM risk.allocator_updates
                        WHERE ts_utc >= %s AND ts_utc < %s
                        GROUP BY horizon
                        """,
                        (start, end),
                    )
                    for horizon, total in cur.fetchall():
                        h = str(horizon).lower()
                        if h in out:
                            out[h] = Decimal(str(total))
            return out

        return await self._run(_q)

    async def allocator_low_sample_decisions(
        self, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]:
        return []
