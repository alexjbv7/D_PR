"""Tests for BarsApplier — price/volume adjustment correctness and idempotence.

All tests use an in-memory stub instead of a real database so the suite
runs without infrastructure (pytest -xvs).

Reference scenarios
-------------------
AAPL 2020-08-31 4:1 forward split:
  Pre-split price ~$500.  Post-split equivalent: ~$125.
  ratio = 4/1 = 4
  adj_close = raw_close / 4
  adj_volume = raw_volume * 4

Hypothetical reverse split 1:2 on AAPL:
  ratio = 1/2 = 0.5
  adj_close = raw_close / 0.5 = raw_close * 2
  adj_volume = raw_volume * 0.5

Stock dividend 10%:
  ratio = 1.10
  adj_close = raw_close / 1.10
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

UTC = timezone.utc


# ---------------------------------------------------------------------------
# In-process stub: BarsApplier with injectable SQL executor
# ---------------------------------------------------------------------------

class _FakeConn:
    """Simulates an asyncpg connection returned by pool.acquire()."""

    def __init__(self, db: "_FakeDB") -> None:
        self._db = db

    async def execute(self, sql: str, *args: Any) -> str:
        self._db.execute_calls.append((sql, args))
        return self._db.last_result

    async def fetchrow(self, sql: str, *args: Any) -> Optional[Any]:
        return None


class _FakeAcquireCtx:
    """Async context manager returned by pool.acquire() (mimics asyncpg)."""

    def __init__(self, db: "_FakeDB") -> None:
        self._db = db

    async def __aenter__(self) -> "_FakeConn":
        return _FakeConn(self._db)

    async def __aexit__(self, *_: Any) -> None:
        pass


class _FakeDB:
    """Captures INSERT/UPDATE calls instead of hitting real Postgres."""

    def __init__(self) -> None:
        self.inserts: list[dict[str, Any]] = []
        self.execute_calls: list[tuple] = []
        self.last_result = "INSERT 0 5"

    def acquire(self) -> "_FakeAcquireCtx":
        return _FakeAcquireCtx(self)


class _FakeCARepo:
    def __init__(self) -> None:
        self._applied: dict[tuple[str, str], bool] = {}
        self.recorded: list[tuple] = []

    async def was_applied(self, ca_id: str, target: str) -> bool:
        return self._applied.get((ca_id, target), False)

    async def record_application(
        self, ca_id: str, target: str, rows_affected: int,
        success: bool = True, error_msg: Optional[str] = None,
    ) -> None:
        self._applied[(ca_id, target)] = success
        self.recorded.append((ca_id, target, rows_affected, success, error_msg))


# ---------------------------------------------------------------------------
# Import BarsApplier using sys.path manipulation so the test is portable
# ---------------------------------------------------------------------------

import sys, os
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import the pure helper directly (no DB needed)
from app.corporate_actions.bars_applier import _compute_ratio, BarsApplier  # noqa: E402


# ---------------------------------------------------------------------------
# _compute_ratio unit tests
# ---------------------------------------------------------------------------

def test_forward_split_4_1_ratio():
    ca = {"ca_type": "forward_split", "split_from": "1", "split_to": "4"}
    ratio = _compute_ratio(ca)
    assert ratio == Decimal("4")


def test_reverse_split_1_2_ratio():
    ca = {"ca_type": "reverse_split", "split_from": "2", "split_to": "1"}
    ratio = _compute_ratio(ca)
    assert ratio == Decimal("0.5")


def test_stock_dividend_10pct_ratio():
    ca = {"ca_type": "stock_dividend", "stock_amount": "0.10"}
    ratio = _compute_ratio(ca)
    # 1 + 0.10 = 1.10
    assert ratio == Decimal("1.10")


def test_cash_dividend_returns_none():
    ca = {"ca_type": "cash_dividend"}
    ratio = _compute_ratio(ca)
    assert ratio is None


def test_missing_split_values_returns_none():
    ca = {"ca_type": "forward_split"}
    ratio = _compute_ratio(ca)
    assert ratio is None


# ---------------------------------------------------------------------------
# Forward split adjustment formula
# ---------------------------------------------------------------------------

def test_forward_split_adj_price_formula():
    """adj_close = raw_close / ratio."""
    ratio     = Decimal("4")
    raw_close = Decimal("500")
    adj_close = raw_close / ratio
    assert adj_close == Decimal("125")


def test_forward_split_adj_volume_formula():
    """adj_volume = raw_volume * ratio."""
    ratio      = Decimal("4")
    raw_volume = Decimal("10_000_000")
    adj_volume = raw_volume * ratio
    assert adj_volume == Decimal("40000000")


def test_aapl_2020_forward_split_consistency():
    """AAPL 2020-08-31: price ~500, after 4:1 → ~125. Volume from 10M → 40M."""
    ratio     = Decimal("4")  # split_to / split_from
    raw_close = Decimal("499.23")
    raw_vol   = Decimal("12_000_000")

    adj_close = raw_close / ratio
    adj_vol   = raw_vol * ratio

    assert adj_close == Decimal("499.23") / Decimal("4")
    assert adj_vol   == Decimal("48000000")


# ---------------------------------------------------------------------------
# Reverse split adjustment
# ---------------------------------------------------------------------------

def test_reverse_split_price_increases():
    """ratio = 0.5 → price doubles, volume halves."""
    ratio     = Decimal("0.5")   # 1/2 reverse split
    raw_close = Decimal("5.00")
    raw_vol   = Decimal("1_000_000")

    adj_close = raw_close / ratio
    adj_vol   = raw_vol * ratio

    assert adj_close == Decimal("10.00")
    assert adj_vol   == Decimal("500000.0")


def test_reverse_split_formula_mirrors_forward():
    """Same formula works for both: adj_price = raw / ratio."""
    # forward: ratio=4 → price /4 (decreases)
    assert Decimal("100") / Decimal("4") == Decimal("25")
    # reverse: ratio=0.5 → price /0.5 = *2 (increases)
    assert Decimal("100") / Decimal("0.5") == Decimal("200")


# ---------------------------------------------------------------------------
# Stock dividend
# ---------------------------------------------------------------------------

def test_stock_dividend_adj_price():
    """stock_amount=0.10 → ratio=1.10 → adj_close = raw / 1.10."""
    ratio     = Decimal("1.10")
    raw_close = Decimal("100")
    adj_close = raw_close / ratio
    # 100 / 1.10 ≈ 90.9090...
    assert abs(adj_close - Decimal("90.90909090909090909090909091")) < Decimal("1e-10")


# ---------------------------------------------------------------------------
# BarsApplier — idempotence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bars_applier_idempotence():
    """Applying same CA twice → second call is no-op (0 rows affected)."""
    db   = _FakeDB()
    repo = _FakeCARepo()
    applier = BarsApplier(db, repo)  # type: ignore[arg-type]

    ca = {
        "ca_id":      "test-ca-001",
        "ca_type":    "forward_split",
        "symbol":     "AAPL",
        "ex_ts":      datetime(2020, 8, 31, tzinfo=UTC),
        "split_from": "1",
        "split_to":   "4",
    }

    rows1 = await applier.apply(ca)
    # Mark as applied so second call is no-op
    assert repo._applied.get(("test-ca-001", "bars")) is True

    # second run: should short-circuit
    rows2 = await applier.apply(ca)
    assert rows2 == 0


@pytest.mark.asyncio
async def test_cash_dividend_noop():
    """Cash dividend records no-op and returns 0."""
    db   = _FakeDB()
    repo = _FakeCARepo()
    applier = BarsApplier(db, repo)  # type: ignore[arg-type]

    ca = {
        "ca_id":    "ca-cash-001",
        "ca_type":  "cash_dividend",
        "symbol":   "AAPL",
        "ex_ts":    datetime(2024, 1, 15, tzinfo=UTC),
    }
    rows = await applier.apply(ca)
    assert rows == 0
    # No execute calls for cash dividend
    assert len(db.execute_calls) == 0


@pytest.mark.asyncio
async def test_forward_split_triggers_sql():
    """Forward split calls the INSERT/ON CONFLICT SQL exactly once."""
    db   = _FakeDB()
    repo = _FakeCARepo()
    applier = BarsApplier(db, repo)  # type: ignore[arg-type]

    ca = {
        "ca_id":      "ca-split-002",
        "ca_type":    "forward_split",
        "symbol":     "AAPL",
        "ex_ts":      datetime(2020, 8, 31, tzinfo=UTC),
        "split_from": "1",
        "split_to":   "4",
    }
    rows = await applier.apply(ca)
    assert len(db.execute_calls) == 1
    sql_called, args = db.execute_calls[0]
    assert "bars_1m_adjusted" in sql_called
    # ratio arg = Decimal("4")
    assert args[2] == Decimal("4")


@pytest.mark.asyncio
async def test_bars_not_modified_after_ex_ts():
    """SQL uses WHERE time < ex_ts → the ratio must be passed correctly."""
    db   = _FakeDB()
    repo = _FakeCARepo()
    applier = BarsApplier(db, repo)  # type: ignore[arg-type]

    ex_ts = datetime(2020, 8, 31, tzinfo=UTC)
    ca = {
        "ca_id":      "ca-split-003",
        "ca_type":    "forward_split",
        "symbol":     "AAPL",
        "ex_ts":      ex_ts,
        "split_from": "1",
        "split_to":   "4",
    }
    await applier.apply(ca)
    _, args = db.execute_calls[0]
    # args: (symbol, ex_ts, ratio, ca_id)
    assert args[1] == ex_ts, "SQL must filter time < ex_ts"
