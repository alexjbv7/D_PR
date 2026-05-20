"""
Tests — locks/horizon_lock.py
=============================

4 cases:
  1. test_lock_prevents_double_open          (DoD-4)
  2. test_lock_different_directions_no_block (no false positives)
  3. test_lock_ttl_expires                   (60s default → use short ttl in tests)
  4. test_lock_race_condition_simulation     (asyncio.gather of 5 acquires)
"""
from __future__ import annotations

import asyncio

import pytest

from app.locks.horizon_lock import (
    LockAcquisitionError,
    SameSymbolDirectionLock,
)


@pytest.mark.asyncio
async def test_lock_prevents_double_open() -> None:
    """Two concurrent acquires for the same (symbol, direction) serialise.

    The second waits until the first releases.
    """
    lock = SameSymbolDirectionLock(ttl_seconds=5)
    order: list[str] = []

    async def caller(tag: str, hold: float) -> None:
        async with await lock.acquire("AAPL", 1):
            order.append(f"{tag}-enter")
            await asyncio.sleep(hold)
            order.append(f"{tag}-exit")

    # Start A; give it a head-start, then start B.
    a = asyncio.create_task(caller("A", 0.1))
    await asyncio.sleep(0.01)
    b = asyncio.create_task(caller("B", 0.0))

    await asyncio.gather(a, b)

    # The trace must be strictly serial.
    assert order == ["A-enter", "A-exit", "B-enter", "B-exit"], order


@pytest.mark.asyncio
async def test_lock_different_directions_no_block() -> None:
    """(AAPL, +1) and (AAPL, -1) are independent."""
    lock = SameSymbolDirectionLock(ttl_seconds=5)
    order: list[str] = []

    async def long_holder() -> None:
        async with await lock.acquire("AAPL", 1):
            order.append("long-enter")
            await asyncio.sleep(0.1)
            order.append("long-exit")

    async def short_holder() -> None:
        async with await lock.acquire("AAPL", -1):
            order.append("short-enter")
            await asyncio.sleep(0.05)
            order.append("short-exit")

    await asyncio.gather(long_holder(), short_holder())

    # Both should enter before either exits — interleaved is fine, no waits.
    assert order.index("long-enter")  < order.index("short-exit")
    assert order.index("short-enter") < order.index("long-exit")


@pytest.mark.asyncio
async def test_lock_ttl_expires() -> None:
    """Hold a lock longer than TTL → next acquire force-releases the stale one."""
    lock = SameSymbolDirectionLock(ttl_seconds=1)

    # Manually acquire without releasing — simulates a stuck caller.
    guard1 = await lock.acquire("AAPL", 1)
    assert lock.is_locked("AAPL", 1)

    # Sleep past TTL so the next acquire triggers stale recovery.
    await asyncio.sleep(1.3)

    # A new acquire should now succeed (the stale holder gets force-released).
    guard2 = await lock.acquire("AAPL", 1, timeout=1.0)
    assert lock.is_locked("AAPL", 1)

    # Cleanup
    await guard2.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_lock_race_condition_simulation() -> None:
    """5 concurrent acquires → exactly 1 holds at any moment, all 5 finish."""
    lock = SameSymbolDirectionLock(ttl_seconds=10)
    in_critical: list[int] = []
    max_concurrent = 0
    current = 0
    lock_local = asyncio.Lock()  # for counter mutation only

    async def runner(idx: int) -> None:
        nonlocal current, max_concurrent
        async with await lock.acquire("AAPL", 1):
            async with lock_local:
                current += 1
                in_critical.append(idx)
                max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.02)
            async with lock_local:
                current -= 1

    await asyncio.gather(*[runner(i) for i in range(5)])

    assert max_concurrent == 1, f"expected serial access, observed {max_concurrent} concurrent"
    assert sorted(in_critical) == [0, 1, 2, 3, 4]
