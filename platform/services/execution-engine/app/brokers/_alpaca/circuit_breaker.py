"""
AlpacaCircuitBreaker — three-state circuit breaker for Alpaca API calls.
=========================================================================

States
------
CLOSED (normal)
    Calls pass through.  On each failure the failure counter increments.
    When ``failure_threshold`` failures occur within ``window_sec`` seconds
    the breaker transitions to OPEN.

OPEN (protecting)
    All calls are rejected immediately with :class:`CircuitBreakerOpenError`.
    After ``recovery_timeout_sec`` the breaker transitions to HALF_OPEN.

HALF_OPEN (probing)
    The next call is allowed through as a probe.
    * Success  → CLOSED (counter reset).
    * Failure  → OPEN again (timer restarted).

Design decisions
----------------
* Asyncio-safe: uses :class:`asyncio.Lock` for state mutation.
* Time-windowed failures: a sliding deque of timestamps; only failures within
  ``window_sec`` count toward the threshold.
* Prometheus metrics are optional; imported lazily.

References
----------
* Nygard, *Release It!* (2018) §5 — Circuit Breaker pattern.
* CLAUDE.md §2.7 (Circuit breakers): "if executor receives > 5 errors 5xx in
  60s → read-only mode."
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus (optional)
# ---------------------------------------------------------------------------

_HAS_PROMETHEUS = False
_CB_STATE: Any = None
_CB_CALLS: Any = None

try:
    from prometheus_client import Counter, Gauge

    _CB_STATE = Gauge(
        "alpaca_circuit_breaker_state",
        "Circuit breaker state: 0=CLOSED, 1=OPEN, 2=HALF_OPEN",
    )
    _CB_CALLS = Counter(
        "alpaca_circuit_breaker_calls_total",
        "Circuit breaker call outcomes",
        ["outcome"],   # success | rejected | failure
    )
    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    pass

_STATE_METRIC = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}


def _prom_state(name: str) -> None:
    if _HAS_PROMETHEUS and _CB_STATE is not None:
        _CB_STATE.set(_STATE_METRIC[name])


def _prom_call(outcome: str) -> None:
    if _HAS_PROMETHEUS and _CB_CALLS is not None:
        _CB_CALLS.labels(outcome=outcome).inc()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the breaker is OPEN."""


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class CBState(Enum):
    CLOSED   = "CLOSED"
    OPEN     = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class AlpacaCircuitBreaker:
    """
    Three-state circuit breaker for Alpaca API calls.

    Parameters
    ----------
    failure_threshold : int
        Number of failures within *window_sec* that triggers OPEN state.
        Default: 5 (matches CLAUDE.md §2.7).
    window_sec : float
        Sliding window for failure counting.  Default: 60 s.
    recovery_timeout_sec : float
        Time to stay in OPEN before transitioning to HALF_OPEN.  Default: 30 s.

    Examples
    --------
    >>> cb = AlpacaCircuitBreaker()
    >>> async def submit():
    ...     async with cb:
    ...         return await _do_alpaca_call()
    """

    def __init__(
        self,
        failure_threshold:    int   = 5,
        window_sec:           float = 60.0,
        recovery_timeout_sec: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if window_sec <= 0 or recovery_timeout_sec <= 0:
            raise ValueError("window_sec and recovery_timeout_sec must be > 0")

        self.failure_threshold    = failure_threshold
        self.window_sec           = window_sec
        self.recovery_timeout_sec = recovery_timeout_sec

        self._state: CBState = CBState.CLOSED
        self._failure_times: deque[float] = deque()
        self._opened_at: Optional[float]  = None
        self._lock = asyncio.Lock()

        _prom_state("CLOSED")

    # -----------------------------------------------------------------------
    # Public read-only state
    # -----------------------------------------------------------------------

    @property
    def state(self) -> CBState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CBState.OPEN

    # -----------------------------------------------------------------------
    # Context manager interface
    # -----------------------------------------------------------------------

    async def __aenter__(self) -> "AlpacaCircuitBreaker":
        async with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CBState.OPEN:
                _prom_call("rejected")
                raise CircuitBreakerOpenError(
                    f"Circuit breaker OPEN — call rejected "
                    f"(recovery in {self._time_until_recovery():.0f}s)"
                )
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val:  Any,
        exc_tb:   Any,
    ) -> bool:
        async with self._lock:
            if exc_type is None:
                self._on_success()
                _prom_call("success")
            else:
                self._on_failure()
                _prom_call("failure")
        return False  # never suppress exceptions

    # -----------------------------------------------------------------------
    # Internal state machine
    # -----------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """OPEN → HALF_OPEN when recovery timeout has elapsed."""
        if (
            self._state == CBState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.recovery_timeout_sec
        ):
            self._state = CBState.HALF_OPEN
            logger.warning(
                "circuit_breaker.half_open recovery_timeout=%.0fs",
                self.recovery_timeout_sec,
            )
            _prom_state("HALF_OPEN")

    def _on_success(self) -> None:
        if self._state in (CBState.HALF_OPEN, CBState.CLOSED):
            self._state = CBState.CLOSED
            self._failure_times.clear()
            self._opened_at = None
            _prom_state("CLOSED")
            if self._state == CBState.HALF_OPEN:   # was half-open → log
                logger.info("circuit_breaker.closed probe_succeeded")

    def _on_failure(self) -> None:
        now = time.monotonic()
        self._failure_times.append(now)
        # Prune timestamps outside the window
        cutoff = now - self.window_sec
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()

        if self._state == CBState.HALF_OPEN:
            # Probe failed → back to OPEN
            self._state     = CBState.OPEN
            self._opened_at = now
            logger.error("circuit_breaker.open probe_failed")
            _prom_state("OPEN")

        elif self._state == CBState.CLOSED:
            recent = sum(
                1 for t in self._failure_times if t >= cutoff
            )
            if recent >= self.failure_threshold:
                self._state     = CBState.OPEN
                self._opened_at = now
                logger.error(
                    "circuit_breaker.open failures=%d window=%.0fs",
                    recent, self.window_sec,
                )
                _prom_state("OPEN")

    def _time_until_recovery(self) -> float:
        if self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self.recovery_timeout_sec - elapsed)
