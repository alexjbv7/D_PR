"""
Tests — AlpacaRateLimiter (token bucket).
==========================================

Verifies:
  * Separate buckets for "trading" and "data".
  * Burst up to capacity without blocking.
  * Exhaustion → awaits until refill.
  * Token count never goes negative.
  * Unknown bucket name raises KeyError.
  * High-volume calls throttle correctly (latency ≥ expected refill).
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.brokers._alpaca.rate_limiter import AlpacaRateLimiter, _TokenBucket


# ---------------------------------------------------------------------------
# _TokenBucket unit tests
# ---------------------------------------------------------------------------

async def test_bucket_starts_full():
    b = _TokenBucket(capacity=10, refill_period=60.0)
    assert b.tokens >= 9.9  # float rounding tolerance


async def test_acquire_consumes_one_token():
    b = _TokenBucket(capacity=10, refill_period=60.0)
    await b.acquire()
    assert b.tokens < 10


async def test_burst_up_to_capacity_without_blocking():
    """10 rapid acquires on a capacity-10 bucket should complete instantly."""
    b = _TokenBucket(capacity=10, refill_period=60.0)
    t0 = time.monotonic()
    for _ in range(10):
        await b.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"Burst should be near-instant, took {elapsed:.2f}s"


async def test_exhaustion_blocks_until_refill():
    """After draining the bucket, the next acquire must wait for refill."""
    # Capacity 2, refill 2 tokens/sec → 1 token every 0.5s
    b = _TokenBucket(capacity=2, refill_period=1.0)
    await b.acquire()
    await b.acquire()
    # bucket empty — next call should block ~0.5s
    t0 = time.monotonic()
    await b.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.3, f"Should have waited for refill, only {elapsed:.2f}s"


async def test_token_count_never_negative():
    """Even under contention, available tokens never drop below 0."""
    b = _TokenBucket(capacity=5, refill_period=1.0)
    for _ in range(20):
        await b.acquire()
        assert b.tokens >= -0.01  # tiny float tolerance


async def test_timeout_raises():
    """acquire(timeout=…) raises TimeoutError when bucket cannot refill in time."""
    b = _TokenBucket(capacity=1, refill_period=60.0)
    await b.acquire()  # drain

    with pytest.raises(asyncio.TimeoutError):
        await b.acquire(timeout=0.1)


# ---------------------------------------------------------------------------
# AlpacaRateLimiter tests
# ---------------------------------------------------------------------------

async def test_separate_buckets():
    """'trading' and 'data' are independent — draining one doesn't block the other."""
    limiter = AlpacaRateLimiter(trading_rpm=2, data_rpm=2)

    # Drain trading
    await limiter.acquire("trading")
    await limiter.acquire("trading")

    # Data should still have tokens
    t0 = time.monotonic()
    await limiter.acquire("data")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, "data bucket should have tokens"


async def test_unknown_bucket_raises():
    limiter = AlpacaRateLimiter()
    with pytest.raises(KeyError, match="unknown_bucket"):
        await limiter.acquire("unknown_bucket")


async def test_available_reports_token_count():
    limiter = AlpacaRateLimiter(trading_rpm=10, data_rpm=10)
    initial = limiter.available("trading")
    assert initial >= 9.0

    await limiter.acquire("trading")
    after = limiter.available("trading")
    assert after < initial


async def test_throttling_high_volume():
    """
    With capacity=5/sec (300 rpm), 10 calls should take at least ~1 second
    for the last 5 to refill.
    """
    limiter = AlpacaRateLimiter(trading_rpm=5, data_rpm=5)
    # rpm=5 with refill_period=60s means 5 tokens/60s = 1 token/12s
    # That's too slow. Use a custom bucket for this test.

    from app.brokers._alpaca.rate_limiter import _TokenBucket

    # 5 tokens, refill in 1 second → 5 tokens/sec
    fast_bucket = _TokenBucket(capacity=5, refill_period=1.0)

    t0 = time.monotonic()
    for _ in range(10):
        await fast_bucket.acquire()
    elapsed = time.monotonic() - t0

    # 5 instant + 5 needing refill (1 token every 0.2s) → ~1.0s minimum
    assert elapsed >= 0.8, f"Expected throttling, got {elapsed:.2f}s"
    # But shouldn't be absurdly long
    assert elapsed < 5.0, f"Took too long: {elapsed:.2f}s"
