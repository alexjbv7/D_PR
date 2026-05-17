"""
Tests for the Reconciler.

We mock ``Router.get_positions_all`` with an AsyncMock and feed positions
into a real MemoryRepository.  This exercises the diff logic, callbacks,
and the kill-switch streak counter without spinning up the background task.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_shared.schemas.orders import OrderSide, Position

from app.brokers.base import BrokerError
from app.reconciler import Discrepancy, ReconcileReport, Reconciler
from app.repository import MemoryRepository


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _make_router(positions=None) -> MagicMock:
    r = MagicMock()
    r.get_positions_all = AsyncMock(return_value=list(positions or []))
    return r


def _pos(symbol, qty="0.01", side=OrderSide.BUY,
         avg_entry="65000", venue="binance") -> Position:
    return Position(
        symbol=symbol, side=side,
        qty=Decimal(qty), avg_entry=Decimal(avg_entry),
        venue=venue,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_rejects_bad_interval():
    with pytest.raises(ValueError, match="interval_sec"):
        Reconciler(_make_router(), MemoryRepository(), interval_sec=0)


def test_rejects_bad_threshold():
    with pytest.raises(ValueError, match="failure_threshold"):
        Reconciler(_make_router(), MemoryRepository(), failure_threshold=0)


# ---------------------------------------------------------------------------
# Diff logic — clean / phantom / missing / qty / side
# ---------------------------------------------------------------------------

async def test_clean_when_no_positions():
    rec = Reconciler(_make_router([]), MemoryRepository())
    report = await rec.reconcile_once()
    assert report.ok
    assert report.broker_count   == 0
    assert report.internal_count == 0


async def test_clean_when_matched():
    repo = MemoryRepository()
    pos  = _pos("BTCUSDT")
    await repo.upsert_position(pos)
    rec = Reconciler(_make_router([pos]), repo)

    report = await rec.reconcile_once()
    assert report.ok
    assert report.broker_count   == 1
    assert report.internal_count == 1


async def test_phantom_detected():
    """Broker has a position the engine doesn't know about."""
    repo = MemoryRepository()
    rec  = Reconciler(_make_router([_pos("BTCUSDT")]), repo)

    report = await rec.reconcile_once()
    assert not report.ok
    assert len(report.discrepancies) == 1
    assert report.discrepancies[0].kind   == "PHANTOM"
    assert report.discrepancies[0].symbol == "BTCUSDT"


async def test_missing_detected():
    """Engine tracks a position the broker doesn't have."""
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT"))
    rec = Reconciler(_make_router([]), repo)

    report = await rec.reconcile_once()
    assert any(d.kind == "MISSING" for d in report.discrepancies)


async def test_qty_mismatch_detected():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT", qty="0.02"))
    rec = Reconciler(_make_router([_pos("BTCUSDT", qty="0.01")]), repo)

    report = await rec.reconcile_once()
    assert any(d.kind == "QTY_MISMATCH" for d in report.discrepancies)


async def test_side_mismatch_detected():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT", side=OrderSide.BUY))
    rec = Reconciler(_make_router([_pos("BTCUSDT", side=OrderSide.SELL)]), repo)

    report = await rec.reconcile_once()
    assert any(d.kind == "SIDE_MISMATCH" for d in report.discrepancies)


async def test_qty_tolerance_absorbs_tiny_diffs():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT", qty="0.010000000001"))
    rec = Reconciler(
        _make_router([_pos("BTCUSDT", qty="0.01")]),
        repo,
        qty_tolerance=Decimal("1e-6"),
    )
    report = await rec.reconcile_once()
    assert report.ok


# ---------------------------------------------------------------------------
# Multi-venue diff
# ---------------------------------------------------------------------------

async def test_same_symbol_two_venues_diffed_independently():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT", venue="binance"))
    await repo.upsert_position(_pos("BTCUSDT", venue="bybit", qty="0.02"))
    broker = [
        _pos("BTCUSDT", venue="binance"),
        _pos("BTCUSDT", venue="bybit",  qty="0.01"),     # qty mismatch
    ]
    rec    = Reconciler(_make_router(broker), repo)
    report = await rec.reconcile_once()
    assert len(report.discrepancies) == 1
    assert report.discrepancies[0].venue  == "bybit"
    assert report.discrepancies[0].kind   == "QTY_MISMATCH"


# ---------------------------------------------------------------------------
# Broker errors are tolerated
# ---------------------------------------------------------------------------

async def test_broker_error_logged_and_treated_as_empty():
    router = _make_router()
    router.get_positions_all = AsyncMock(side_effect=BrokerError("network"))
    rec    = Reconciler(router, MemoryRepository())
    report = await rec.reconcile_once()
    assert report.broker_count == 0       # gracefully degraded


# ---------------------------------------------------------------------------
# Streak counter + kill switch callback
# ---------------------------------------------------------------------------

async def test_streak_resets_on_clean_cycle():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT"))
    rec = Reconciler(_make_router([]), repo, failure_threshold=10)

    # 1st cycle: missing
    report = await rec.reconcile_once()
    await rec._handle_report(report)
    assert rec._consecutive_failures == 1

    # Fix the discrepancy, run a clean cycle
    rec.router.get_positions_all = AsyncMock(return_value=[_pos("BTCUSDT")])
    report = await rec.reconcile_once()
    await rec._handle_report(report)
    assert rec._consecutive_failures == 0


async def test_kill_switch_trips_after_threshold():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT"))
    kill = AsyncMock()
    rec = Reconciler(
        _make_router([]),                    # phantom MISSING every cycle
        repo,
        failure_threshold=3,
        kill_switch_callback=kill,
    )
    for _ in range(3):
        report = await rec.reconcile_once()
        await rec._handle_report(report)

    kill.assert_awaited_once()
    reason_arg = kill.await_args.args[0]
    assert "consecutive" in reason_arg.lower()


async def test_kill_switch_fires_only_once():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT"))
    kill = AsyncMock()
    rec = Reconciler(
        _make_router([]), repo,
        failure_threshold=2, kill_switch_callback=kill,
    )
    for _ in range(5):
        report = await rec.reconcile_once()
        await rec._handle_report(report)

    assert kill.await_count == 1


async def test_kill_switch_callback_error_does_not_crash():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT"))
    kill = AsyncMock(side_effect=RuntimeError("redis down"))
    rec  = Reconciler(
        _make_router([]), repo,
        failure_threshold=1, kill_switch_callback=kill,
    )
    report = await rec.reconcile_once()
    await rec._handle_report(report)        # must not raise
    kill.assert_awaited_once()


# ---------------------------------------------------------------------------
# Discrepancy callback
# ---------------------------------------------------------------------------

async def test_on_discrepancy_called_with_report():
    repo = MemoryRepository()
    await repo.upsert_position(_pos("BTCUSDT"))
    cb = AsyncMock()
    rec = Reconciler(_make_router([]), repo, on_discrepancy=cb)

    report = await rec.reconcile_once()
    await rec._handle_report(report)

    cb.assert_awaited_once()
    arg = cb.await_args.args[0]
    assert isinstance(arg, ReconcileReport)
    assert not arg.ok


async def test_on_discrepancy_not_called_when_clean():
    cb = AsyncMock()
    rec = Reconciler(_make_router([]), MemoryRepository(), on_discrepancy=cb)
    report = await rec.reconcile_once()
    await rec._handle_report(report)
    cb.assert_not_awaited()


# ---------------------------------------------------------------------------
# Background task lifecycle
# ---------------------------------------------------------------------------

async def test_start_and_stop_lifecycle():
    rec = Reconciler(_make_router([]), MemoryRepository(), interval_sec=1)
    rec.start()
    assert rec._task is not None
    await asyncio.sleep(0)         # yield once
    await rec.stop()
    assert rec._task is None


async def test_start_is_idempotent():
    rec = Reconciler(_make_router([]), MemoryRepository(), interval_sec=1)
    rec.start()
    first_task = rec._task
    rec.start()
    assert rec._task is first_task
    await rec.stop()


# ---------------------------------------------------------------------------
# Discrepancy dataclass
# ---------------------------------------------------------------------------

def test_discrepancy_is_frozen():
    d = Discrepancy(kind="PHANTOM", venue="binance", symbol="BTCUSDT", detail="")
    with pytest.raises(Exception):
        d.kind = "OTHER"     # type: ignore[misc]
