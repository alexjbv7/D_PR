"""
AlpacaRateLimiter — asyncio-native token bucket for Alpaca API rate limits.
============================================================================

Alpaca enforces **two separate** rate limits:
  - Trading API:  200 requests / minute
  - Data API:     200 requests / minute

Each bucket refills linearly (1 token every ``60 / capacity`` seconds).
When a bucket is empty, :meth:`acquire` awaits until a token is available —
**no blocking ``time.sleep``**, purely ``asyncio.Event`` based.

Usage
-----
::

    limiter = AlpacaRateLimiter()
    await limiter.acquire("trading")   # blocks if exhausted
    # ... do the request ...

Thread safety: single-loop only (standard for asyncio services).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class _TokenBucket:
    """
    Simple asyncio token-bucket rate limiter.

    Parameters
    ----------
    capacity : int
        Maximum tokens (= burst size = requests per window).
    refill_period : float
        Seconds over which *all* ``capacity`` tokens refill (e.g. 60.0 for
        200 tokens/minute).
    """

    def __init__(self, capacity: int, refill_period: float):
        self._capacity      = capacity
        self._refill_rate   = capacity / refill_period   # tokens per second
        self._tokens        = float(capacity)
        self._last_refill   = time.monotonic()
        self._lock          = asyncio.Lock()
        self._has_tokens    = asyncio.Event()
        self._has_tokens.set()

    @property
    def tokens(self) -> float:
        """Current available tokens (informational, not thread-safe)."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        now     = time.monotonic()
        elapsed = now - self._last_refill
        added   = elapsed * self._refill_rate
        if added > 0:
            self._tokens      = min(self._capacity, self._tokens + added)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._has_tokens.set()

    async def acquire(self, timeout: Optional[float] = None) -> None:
        """
        Wait until a token is available, then consume one.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait.  ``None`` waits indefinitely.

        Raises
        ------
        asyncio.TimeoutError
            If ``timeout`` elapses before a token becomes available.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    if self._tokens < 1.0:
                        self._has_tokens.clear()
                    return

            # Wait for refill
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError("rate_limiter.timeout")

            # Sleep until roughly 1 token is available
            sleep_s = 1.0 / self._refill_rate
            if remaining is not None:
                sleep_s = min(sleep_s, remaining)
            await asyncio.sleep(sleep_s)


class AlpacaRateLimiter:
    """
    Dual-bucket rate limiter matching Alpaca's published limits.

    Parameters
    ----------
    trading_rpm : int
        Trading-endpoint limit (default 200 requests/minute).
    data_rpm : int
        Data-endpoint limit (default 200 requests/minute).
    """

    def __init__(
        self,
        trading_rpm: int = 200,
        data_rpm:    int = 200,
    ):
        self._buckets: dict[str, _TokenBucket] = {
            "trading": _TokenBucket(capacity=trading_rpm, refill_period=60.0),
            "data":    _TokenBucket(capacity=data_rpm,    refill_period=60.0),
        }

    async def acquire(
        self,
        bucket: str = "trading",
        timeout: Optional[float] = None,
    ) -> None:
        """
        Consume one token from the named bucket.

        Parameters
        ----------
        bucket : str
            ``"trading"`` or ``"data"``.
        timeout : float, optional
            Maximum wait in seconds.

        Raises
        ------
        KeyError
            If ``bucket`` is not recognised.
        asyncio.TimeoutError
            If the bucket cannot refill within ``timeout``.
        """
        b = self._buckets.get(bucket)
        if b is None:
            raise KeyError(f"Unknown bucket: {bucket!r}. Valid: {list(self._buckets)}")
        await b.acquire(timeout=timeout)

    def available(self, bucket: str = "trading") -> float:
        """Return current token count for ``bucket`` (informational)."""
        return self._buckets[bucket].tokens
