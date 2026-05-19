"""
retry_with_jitter — tenacity decorator for Alpaca API retries.
================================================================

Retries **only** on transient errors:
  - HTTP 5xx  (server error)
  - HTTP 429  (rate limited)

Client errors (4xx excl. 429) propagate immediately.

Usage
-----
::

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def submit(self, intent):
        ...

Design: wraps ``tenacity.retry`` with custom ``retry_if_exception``
that inspects the error message or HTTP status code.
"""
from __future__ import annotations

import functools
import logging
import re
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# Patterns we treat as transient (matched against exception string or attrs)
_DEFAULT_RETRYABLE_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)

# Regex to extract HTTP status from alpaca-py error messages
_STATUS_RE = re.compile(r"\b(4\d{2}|5\d{2})\b")


def _is_retryable(exc: BaseException, retryable_statuses: tuple[int, ...]) -> bool:
    """
    Determine whether ``exc`` represents a transient server error.

    Checks (in order):
      1. ``exc.status_code`` attribute (alpaca-py / httpx style).
      2. ``exc.status`` attribute.
      3. Integer status codes extracted from the string representation.

    Returns True only if the status is in ``retryable_statuses``.
    """
    # 1. Attribute-based detection
    for attr in ("status_code", "status"):
        code = getattr(exc, attr, None)
        if code is not None:
            try:
                return int(code) in retryable_statuses
            except (ValueError, TypeError):
                pass

    # 2. Regex on the error message
    msg = str(exc)
    match = _STATUS_RE.search(msg)
    if match:
        return int(match.group(1)) in retryable_statuses

    # 3. ConnectionError / TimeoutError → treat as transient
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    return False


_P = ParamSpec("_P")
_T = TypeVar("_T")


def retry_with_jitter(
    max_attempts: int = 3,
    base_delay:   float = 0.5,
    max_delay:    float = 10.0,
    jitter:       float = 1.0,
    retryable_statuses: tuple[int, ...] = _DEFAULT_RETRYABLE_STATUSES,
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Awaitable[_T]]]:
    """
    Decorator: retry an async method on transient Alpaca errors.

    Parameters
    ----------
    max_attempts : int
        Total attempts (including the first).  Default 3.
    base_delay : float
        Initial backoff in seconds.  Default 0.5.
    max_delay : float
        Maximum backoff in seconds.  Default 10.
    jitter : float
        Random jitter (seconds) added to each wait.  Default 1.
    retryable_statuses : tuple[int, ...]
        HTTP status codes that trigger a retry.

    Returns
    -------
    Callable
        Decorator applicable to async methods.
    """

    def decorator(fn: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        @retry(
            retry=retry_if_exception(
                lambda exc: _is_retryable(exc, retryable_statuses)
            ),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(
                initial=base_delay,
                max=max_delay,
                jitter=jitter,
            ),
            reraise=True,
        )
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            return await fn(*args, **kwargs)

        return wrapper

    return decorator
