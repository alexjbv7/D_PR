"""
Tests for AlpacaCircuitBreaker (P1-001).

Covers the full CLOSED → OPEN → HALF_OPEN → CLOSED cycle plus edge cases.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.brokers._alpaca.circuit_breaker import (
    AlpacaCircuitBreaker,
    CBState,
    CircuitBreakerOpenError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _pass(cb: AlpacaCircuitBreaker) -> str:
    """Simulate a successful call."""
    async with cb:
        return "ok"


async def _fail(cb: AlpacaCircuitBreaker) -> None:
    """Simulate a failing call."""
    async with cb:
        raise RuntimeError("simulated failure")


async def _trip(cb: AlpacaCircuitBreaker, n: int = 5) -> None:
    """Force the breaker OPEN by simulating n failures."""
    for _ in range(n):
        with pytest.raises((RuntimeError, CircuitBreakerOpenError)):
            await _fail(cb)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_closed():
    cb = AlpacaCircuitBreaker()
    assert cb.state == CBState.CLOSED
    assert not cb.is_open


def test_config_validation():
    with pytest.raises(ValueError):
        AlpacaCircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        AlpacaCircuitBreaker(window_sec=0)
    with pytest.raises(ValueError):
        AlpacaCircuitBreaker(recovery_timeout_sec=-1)


# ---------------------------------------------------------------------------
# CLOSED → success
# ---------------------------------------------------------------------------

async def test_success_in_closed_state():
    cb = AlpacaCircuitBreaker()
    result = await _pass(cb)
    assert result == "ok"
    assert cb.state == CBState.CLOSED


# ---------------------------------------------------------------------------
# CLOSED → OPEN
# ---------------------------------------------------------------------------

async def test_opens_after_threshold_failures():
    cb = AlpacaCircuitBreaker(failure_threshold=3, window_sec=60)
    await _trip(cb, n=3)
    assert cb.state == CBState.OPEN
    assert cb.is_open


async def test_open_rejects_calls():
    cb = AlpacaCircuitBreaker(failure_threshold=2, window_sec=60)
    await _trip(cb, n=2)

    with pytest.raises(CircuitBreakerOpenError):
        await _pass(cb)


async def test_failures_outside_window_do_not_count():
    cb = AlpacaCircuitBreaker(failure_threshold=3, window_sec=0.05)
    # 2 failures, then wait for window to expire
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await _fail(cb)
    await asyncio.sleep(0.1)   # window expired
    # 2 more failures — still under threshold because old ones expired
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await _fail(cb)
    assert cb.state == CBState.CLOSED


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN
# ---------------------------------------------------------------------------

async def test_transitions_to_half_open_after_recovery_timeout():
    cb = AlpacaCircuitBreaker(failure_threshold=2, window_sec=60, recovery_timeout_sec=0.05)
    await _trip(cb, n=2)
    assert cb.state == CBState.OPEN

    await asyncio.sleep(0.1)   # recovery timeout elapsed

    # Next __aenter__ should transition to HALF_OPEN, not reject
    result = await _pass(cb)
    assert result == "ok"
    assert cb.state == CBState.CLOSED   # probe succeeded → CLOSED


# ---------------------------------------------------------------------------
# HALF_OPEN → OPEN (probe failure)
# ---------------------------------------------------------------------------

async def test_probe_failure_returns_to_open():
    cb = AlpacaCircuitBreaker(failure_threshold=2, window_sec=60, recovery_timeout_sec=0.05)
    await _trip(cb, n=2)
    await asyncio.sleep(0.1)   # → HALF_OPEN on next call

    with pytest.raises((RuntimeError, CircuitBreakerOpenError)):
        await _fail(cb)

    assert cb.state == CBState.OPEN


# ---------------------------------------------------------------------------
# HALF_OPEN → CLOSED (probe success)
# ---------------------------------------------------------------------------

async def test_probe_success_closes_breaker():
    cb = AlpacaCircuitBreaker(failure_threshold=2, window_sec=60, recovery_timeout_sec=0.05)
    await _trip(cb, n=2)
    await asyncio.sleep(0.1)

    await _pass(cb)
    assert cb.state == CBState.CLOSED
    assert not cb.is_open


# ---------------------------------------------------------------------------
# Success resets failure counter
# ---------------------------------------------------------------------------

async def test_success_clears_failure_counter():
    cb = AlpacaCircuitBreaker(failure_threshold=3, window_sec=60)
    # 2 failures (one below threshold)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await _fail(cb)
    # success should reset the counter
    await _pass(cb)
    # 2 more failures should not open (counter was cleared)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await _fail(cb)
    assert cb.state == CBState.CLOSED
