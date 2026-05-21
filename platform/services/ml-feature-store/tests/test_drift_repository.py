"""
Tests — drift/drift_repository.py
=================================

Verifies closed-day window semantics for drift queries (cron @ 03:00 UTC).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

_SVC = os.path.dirname(os.path.dirname(__file__))
if _SVC not in sys.path:
    sys.path.insert(0, _SVC)

# Optional dep — same bootstrap as test_drift_cron when structlog is not installed.
if "structlog" not in sys.modules:
    _sl = MagicMock()
    _sl.get_logger.return_value = MagicMock()
    sys.modules["structlog"] = _sl

from app.drift.drift_repository import DriftRepository


def _closed_window_bounds(
    window_days: int,
    now: datetime,
) -> tuple[datetime, datetime]:
    """Mirror SQL: ts >= now - (window_days+1)d AND ts < now - 1d."""
    end = now - timedelta(days=1)
    start = now - timedelta(days=window_days + 1)
    return start, end


def _make_pool_with_rows(rows: list[dict[str, Any]]) -> MagicMock:
    """Pool whose fetch applies the same closed-window filter as the SQL."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    async def _fetch(sql: str, horizon: str, model_version: str, window_days: int) -> list:
        now = datetime.now(tz=timezone.utc)
        start, end = _closed_window_bounds(window_days, now)
        out = []
        for r in rows:
            if r["horizon"] != horizon or r["model_version"] != model_version:
                continue
            if r.get("true_label") is None:
                continue
            ts = r["ts"]
            if start <= ts < end:
                out.append({"probas": r["probas"], "true_label": r["true_label"]})
        return out

    conn.fetch = AsyncMock(side_effect=_fetch)
    return pool


@pytest.mark.asyncio
async def test_fetch_recent_predictions_excludes_current_day() -> None:
    """100 rows today + 100 yesterday → window_days=2 returns only yesterday."""
    now = datetime.now(tz=timezone.utc)
    today = now - timedelta(hours=3)
    yesterday = now - timedelta(days=1, hours=12)

    rows: list[dict[str, Any]] = []
    for i in range(100):
        rows.append({
            "ts": today,
            "horizon": "swing",
            "model_version": "xgb_swing_v3",
            "probas": [0.6, 0.2, 0.2],
            "true_label": 0,
        })
    for i in range(100):
        rows.append({
            "ts": yesterday,
            "horizon": "swing",
            "model_version": "xgb_swing_v3",
            "probas": [0.5, 0.3, 0.2],
            "true_label": 1,
        })

    repo = DriftRepository(_make_pool_with_rows(rows))
    result = await repo.fetch_recent_predictions(
        horizon="swing",
        model_version="xgb_swing_v3",
        window_days=2,
    )

    assert len(result) == 100
    assert all(r["true_label"] == 1 for r in result)


@pytest.mark.asyncio
async def test_fetch_recent_predictions_sql_uses_closed_window() -> None:
    """Captured SQL must exclude the current UTC day."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    repo = DriftRepository(pool)
    await repo.fetch_recent_predictions("swing", "xgb_swing_v3", window_days=7)

    sql = conn.fetch.call_args[0][0]
    assert "($3 + 1)" in sql or "(($3 + 1)" in sql
    assert "ts <  NOW() - INTERVAL '1 day'" in sql


@pytest.mark.asyncio
async def test_fetch_recent_feature_values_sql_excludes_current_day() -> None:
    """Feature window query uses the same closed-day upper bound."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    repo = DriftRepository(pool)
    await repo.fetch_recent_feature_values("rsi_14", window_days=7)

    sql = conn.fetch.call_args[0][0]
    assert "(($2 + 1)" in sql
    assert "ts <  NOW() - INTERVAL '1 day'" in sql
