"""
Tests — retry_with_jitter decorator.
======================================

Verifies:
  * 5xx → retries up to max_attempts, last attempt succeeds → OK.
  * 4xx (non-429) → no retry, raises immediately.
  * 429 → retried (treated as transient).
  * ConnectionError / TimeoutError → retried.
  * Successful call on first attempt → no extra attempts.
"""
from __future__ import annotations

import pytest

from app.brokers._alpaca.retry import retry_with_jitter, _is_retryable


# ---------------------------------------------------------------------------
# _is_retryable unit tests
# ---------------------------------------------------------------------------

class _StatusCodeExc(Exception):
    def __init__(self, status_code: int, msg: str = ""):
        self.status_code = status_code
        super().__init__(msg)


def test_retryable_5xx():
    assert _is_retryable(_StatusCodeExc(500), (429, 500, 502, 503, 504))
    assert _is_retryable(_StatusCodeExc(503), (429, 500, 502, 503, 504))


def test_retryable_429():
    assert _is_retryable(_StatusCodeExc(429), (429, 500, 502, 503, 504))


def test_not_retryable_4xx():
    assert not _is_retryable(_StatusCodeExc(400), (429, 500, 502, 503, 504))
    assert not _is_retryable(_StatusCodeExc(403), (429, 500, 502, 503, 504))
    assert not _is_retryable(_StatusCodeExc(404), (429, 500, 502, 503, 504))
    assert not _is_retryable(_StatusCodeExc(422), (429, 500, 502, 503, 504))


def test_retryable_by_string_match():
    exc = RuntimeError("HTTP 503 Service Unavailable")
    assert _is_retryable(exc, (429, 500, 502, 503, 504))


def test_not_retryable_string_400():
    exc = RuntimeError("HTTP 400 Bad Request")
    assert not _is_retryable(exc, (429, 500, 502, 503, 504))


def test_retryable_connection_error():
    assert _is_retryable(ConnectionError("reset"), (429, 500))


def test_retryable_timeout_error():
    assert _is_retryable(TimeoutError("timed out"), (429, 500))


def test_not_retryable_generic():
    assert not _is_retryable(ValueError("bad value"), (429, 500))


# ---------------------------------------------------------------------------
# Decorator integration tests
# ---------------------------------------------------------------------------

async def test_5xx_5xx_200_retries_and_succeeds():
    """Simulate 5xx, 5xx, then success → 3 attempts total, returns OK."""
    attempts = [0]

    @retry_with_jitter(max_attempts=3, base_delay=0.01, jitter=0.0)
    async def flaky_call():
        attempts[0] += 1
        if attempts[0] < 3:
            raise _StatusCodeExc(500, "Internal Server Error")
        return "ok"

    result = await flaky_call()
    assert result == "ok"
    assert attempts[0] == 3


async def test_4xx_no_retry():
    """A 4xx error should NOT be retried — raises on first attempt."""
    attempts = [0]

    @retry_with_jitter(max_attempts=3, base_delay=0.01, jitter=0.0)
    async def bad_request():
        attempts[0] += 1
        raise _StatusCodeExc(400, "Bad Request")

    with pytest.raises(_StatusCodeExc):
        await bad_request()
    assert attempts[0] == 1


async def test_429_is_retried():
    """429 is transient and should be retried."""
    attempts = [0]

    @retry_with_jitter(max_attempts=3, base_delay=0.01, jitter=0.0)
    async def rate_limited():
        attempts[0] += 1
        if attempts[0] < 2:
            raise _StatusCodeExc(429, "Too Many Requests")
        return "ok"

    result = await rate_limited()
    assert result == "ok"
    assert attempts[0] == 2


async def test_all_5xx_exhausts_retries():
    """If all 3 attempts fail with 5xx, the last exception propagates."""
    attempts = [0]

    @retry_with_jitter(max_attempts=3, base_delay=0.01, jitter=0.0)
    async def always_fails():
        attempts[0] += 1
        raise _StatusCodeExc(502, "Bad Gateway")

    with pytest.raises(_StatusCodeExc, match="Bad Gateway"):
        await always_fails()
    assert attempts[0] == 3


async def test_success_on_first_attempt():
    """No retry needed when the first call succeeds."""
    attempts = [0]

    @retry_with_jitter(max_attempts=3, base_delay=0.01, jitter=0.0)
    async def works_fine():
        attempts[0] += 1
        return 42

    result = await works_fine()
    assert result == 42
    assert attempts[0] == 1


async def test_connection_error_is_retried():
    """ConnectionError should be treated as transient."""
    attempts = [0]

    @retry_with_jitter(max_attempts=3, base_delay=0.01, jitter=0.0)
    async def connection_drop():
        attempts[0] += 1
        if attempts[0] < 2:
            raise ConnectionError("connection reset")
        return "recovered"

    result = await connection_drop()
    assert result == "recovered"
    assert attempts[0] == 2
