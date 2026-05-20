"""
Tests — allocator/update_consumer.py
====================================

3 cases verifying idempotency and event filtering:
  1. test_idempotent_same_trade_id          (DoD-6 — second apply is a no-op)
  2. test_only_fills_update                 (REJECTED / CANCELLED ignored)
  3. test_only_closes_update                (no realized_pnl ⇒ skip)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.allocator.posterior       import BetaPosterior
from app.allocator.update_consumer import AllocatorUpdateConsumer


def _ts() -> datetime:
    return datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _warmstart() -> BetaPosterior:
    return BetaPosterior(
        alpha=Decimal("20"), beta=Decimal("20"), last_update_ts=_ts(),
    )


def _make_repo(
    *,
    initial: BetaPosterior | None = None,
    already_applied: bool = False,
) -> AsyncMock:
    repo = AsyncMock()
    state = {"swing": initial or _warmstart()}

    async def _load(horizon: str) -> BetaPosterior:
        return state[horizon]

    async def _save(horizon: str, p: BetaPosterior) -> None:
        state[horizon] = p

    repo.load.side_effect = _load
    repo.save.side_effect = _save
    repo.was_update_applied = AsyncMock(return_value=already_applied)
    repo.record_update     = AsyncMock(return_value=None)

    # lock_horizon must return a real async context manager.
    real_lock = asyncio.Lock()
    repo.lock_horizon = lambda h: real_lock  # type: ignore[assignment]
    return repo


# ----------------------------------------------------------------------
# DoD-6 — idempotency
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotent_same_trade_id() -> None:
    """Applying the same trade_id twice → second call is a no-op."""
    repo = _make_repo(already_applied=False)
    consumer = AllocatorUpdateConsumer(repo=repo, kafka_producer=None)

    payload = {
        "result_id":    "trade-001",
        "status":       "FILLED",
        "realized_pnl": "100",
        "strategy":     "xgb_swing_v3",
        "ts_updated":   _ts().isoformat(),
    }

    p1 = await consumer.handle_event(payload)
    assert p1 is not None
    # Now mark as already applied for the second invocation.
    repo.was_update_applied = AsyncMock(return_value=True)
    p2 = await consumer.handle_event(payload)
    assert p2 is None  # skipped
    # record_update should have been called exactly once.
    assert repo.record_update.await_count == 1


# ----------------------------------------------------------------------
# Status filtering
# ----------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["REJECTED", "CANCELLED", "PENDING", "PARTIAL"])
async def test_only_fills_update(status: str) -> None:
    repo = _make_repo()
    consumer = AllocatorUpdateConsumer(repo=repo, kafka_producer=None)
    payload = {
        "result_id":    "t-x",
        "status":       status,
        "realized_pnl": "10",
        "strategy":     "xgb_swing_v3",
    }
    result = await consumer.handle_event(payload)
    assert result is None
    repo.record_update.assert_not_awaited()


# ----------------------------------------------------------------------
# Realized-pnl required
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_only_closes_update() -> None:
    """A FILLED entry (no realized_pnl) must not update the posterior."""
    repo = _make_repo()
    consumer = AllocatorUpdateConsumer(repo=repo, kafka_producer=None)
    payload = {
        "result_id": "t-entry",
        "status":    "FILLED",
        "strategy":  "xgb_swing_v3",
        # NO realized_pnl
    }
    result = await consumer.handle_event(payload)
    assert result is None
    repo.record_update.assert_not_awaited()


# ----------------------------------------------------------------------
# Bonus — happy path proves win increments α; loss increments β
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_win_increments_alpha() -> None:
    repo = _make_repo()
    consumer = AllocatorUpdateConsumer(repo=repo, kafka_producer=None)
    payload = {
        "result_id":    "t-win",
        "status":       "FILLED",
        "realized_pnl": "42",
        "strategy":     "xgb_swing_v3",
        "ts_updated":   _ts().isoformat(),
    }
    new_p = await consumer.handle_event(payload)
    assert new_p is not None
    assert new_p.alpha == Decimal("21")  # 20 + 1
    assert new_p.beta  == Decimal("20")


@pytest.mark.asyncio
async def test_happy_path_loss_increments_beta() -> None:
    repo = _make_repo()
    consumer = AllocatorUpdateConsumer(repo=repo, kafka_producer=None)
    payload = {
        "result_id":    "t-loss",
        "status":       "FILLED",
        "realized_pnl": "-10",
        "strategy":     "xgb_swing_v3",
        "ts_updated":   _ts().isoformat(),
    }
    new_p = await consumer.handle_event(payload)
    assert new_p is not None
    assert new_p.alpha == Decimal("20")
    assert new_p.beta  == Decimal("21")
