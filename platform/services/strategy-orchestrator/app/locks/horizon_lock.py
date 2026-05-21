"""
SameSymbolDirectionLock — process-local mutex keyed by ``(symbol, direction)``.

Goal
----
Prevent two horizons from opening positions on the *same symbol in the same
direction* at the same time.  If both intraday and swing emit a long signal
on AAPL within the allocator's 5-second buffer, only one executes.

Why this granularity?
---------------------
* Not (symbol, horizon)  — would NOT block cross-horizon double-opens.
* Not symbol alone       — would prevent legitimate "close long, open short".
* (symbol, direction)    — exactly what we want.

TTL
---
A bug downstream (broker timeout, executor crash) can leave a lock orphaned
forever, which would silently disable trading for that pair.  We attach a
TTL: when the lock has been held longer than ``ttl_seconds``, the next
caller forces release with a WARNING log.

Acquisition semantics
---------------------
``acquire`` returns an async context manager (``LockGuard``).  Use it as:

    try:
        async with await lock.acquire("AAPL", 1) as guard:
            ...
    except LockAcquisitionError:
        ...

The wait timeout is ``ttl_seconds`` by default; pass ``timeout`` to override.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class LockAcquisitionError(RuntimeError):
    """Raised when a lock cannot be acquired within the timeout."""


@dataclass
class _LockEntry:
    lock:          asyncio.Lock
    acquired_at:   float                # monotonic seconds; 0 when free


class LockGuard:
    """Async context manager returned by ``SameSymbolDirectionLock.acquire``.

    Releases the lock and clears its TTL on exit.  Safe to use across awaits
    (the underlying asyncio.Lock is preserved).
    """

    def __init__(
        self,
        parent: "SameSymbolDirectionLock",
        key:    tuple[str, int],
    ) -> None:
        self._parent = parent
        self._key    = key

    async def __aenter__(self) -> "LockGuard":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self._parent._release(self._key)


class SameSymbolDirectionLock:
    """Per-(symbol, direction) async lock with TTL-based stale recovery.

    Parameters
    ----------
    ttl_seconds : int
        Max time a lock may be held before the next acquire force-releases it.
        Default 60.  Tests use a smaller value.
    """

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._ttl: int = ttl_seconds
        self._entries: dict[tuple[str, int], _LockEntry] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def acquire(
        self,
        symbol:    str,
        direction: int,
        timeout:   Optional[float] = None,
    ) -> LockGuard:
        """Acquire the lock for ``(symbol, direction)``.

        Raises
        ------
        LockAcquisitionError
            If the lock could not be acquired within ``timeout``
            (default = ttl_seconds).
        """
        key = (symbol, direction)
        self._reap_stale()

        entry = self._entries.get(key)
        if entry is None:
            entry = _LockEntry(lock=asyncio.Lock(), acquired_at=0.0)
            self._entries[key] = entry

        wait = timeout if timeout is not None else float(self._ttl)
        try:
            await asyncio.wait_for(entry.lock.acquire(), timeout=wait)
        except asyncio.TimeoutError as exc:
            raise LockAcquisitionError(
                f"could not acquire lock for {key} within {wait:.1f}s"
            ) from exc

        entry.acquired_at = time.monotonic()
        return LockGuard(self, key)

    def is_locked(self, symbol: str, direction: int) -> bool:
        """Inspection helper (useful in tests)."""
        entry = self._entries.get((symbol, direction))
        return entry is not None and entry.lock.locked()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _release(self, key: tuple[str, int]) -> None:
        entry = self._entries.get(key)
        if entry is None or not entry.lock.locked():
            return
        entry.lock.release()
        entry.acquired_at = 0.0

    def _reap_stale(self) -> None:
        """Force-release any lock held longer than ttl_seconds."""
        now = time.monotonic()
        for key, entry in list(self._entries.items()):
            if entry.lock.locked() and entry.acquired_at > 0:
                if now - entry.acquired_at > self._ttl:
                    logger.warning(
                        "horizon_lock.stale_release",
                        symbol=key[0], direction=key[1],
                        held_seconds=round(now - entry.acquired_at, 2),
                    )
                    entry.lock.release()
                    entry.acquired_at = 0.0


__all__ = ["SameSymbolDirectionLock", "LockGuard", "LockAcquisitionError"]
