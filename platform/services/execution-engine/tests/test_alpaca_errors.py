"""
Tests — alpaca_errors.py (typed error hierarchy + classify_alpaca_error).
==========================================================================

All tests are pure unit tests (no network, no SDK).

Covers
------
* All typed subclasses are subclasses of BrokerError AND AlpacaError.
* `classify_alpaca_error` routes correctly by:
  - Explicit HTTP code (401, 403, 404, 429, 5xx).
  - HTTP code embedded in message string.
  - Substring message matching (PDT, buying power, market closed, …).
  - Fallback to generic AlpacaError.
* `__cause__` is set on every returned error.
* All classes exported in `__all__`.
"""
from __future__ import annotations

import pytest

from app.brokers.base import BrokerError
from app.brokers.alpaca_errors import (
    AlpacaAuthError,
    AlpacaDuplicateOrderError,
    AlpacaError,
    AlpacaFractionalNotSupportedError,
    AlpacaInsufficientFundsError,
    AlpacaInvalidQuantityError,
    AlpacaMarketClosedError,
    AlpacaPDTViolationError,
    AlpacaRateLimitError,
    AlpacaServerError,
    AlpacaSymbolNotFoundError,
    classify_alpaca_error,
    __all__ as _ALL,
)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

TYPED_CLASSES = [
    AlpacaAuthError,
    AlpacaDuplicateOrderError,
    AlpacaFractionalNotSupportedError,
    AlpacaInsufficientFundsError,
    AlpacaInvalidQuantityError,
    AlpacaMarketClosedError,
    AlpacaPDTViolationError,
    AlpacaRateLimitError,
    AlpacaServerError,
    AlpacaSymbolNotFoundError,
]


@pytest.mark.parametrize("cls", TYPED_CLASSES)
def test_subclass_of_alpaca_error(cls):
    assert issubclass(cls, AlpacaError)


@pytest.mark.parametrize("cls", TYPED_CLASSES)
def test_subclass_of_broker_error(cls):
    assert issubclass(cls, BrokerError)


def test_alpaca_error_is_broker_error():
    assert issubclass(AlpacaError, BrokerError)


@pytest.mark.parametrize("cls", TYPED_CLASSES)
def test_instantiatable(cls):
    err = cls("test message")
    assert str(err) == "test message"


# ---------------------------------------------------------------------------
# classify_alpaca_error — HTTP code routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected_cls", [
    (401, AlpacaAuthError),
    (403, AlpacaAuthError),
    (404, AlpacaSymbolNotFoundError),
    (429, AlpacaRateLimitError),
    (500, AlpacaServerError),
    (502, AlpacaServerError),
    (503, AlpacaServerError),
    (504, AlpacaServerError),
])
def test_classify_by_http_code(code, expected_cls):
    exc    = Exception(f"HTTP {code} error")
    result = classify_alpaca_error(exc, http_code=code)
    assert isinstance(result, expected_cls)


def test_classify_5xx_family():
    """Any 5xx code should map to AlpacaServerError."""
    exc    = Exception("server fault")
    result = classify_alpaca_error(exc, http_code=599)
    assert isinstance(result, AlpacaServerError)


# ---------------------------------------------------------------------------
# classify_alpaca_error — HTTP code in message
# ---------------------------------------------------------------------------

def test_classify_429_in_message():
    exc    = Exception("429 Too Many Requests")
    result = classify_alpaca_error(exc)
    assert isinstance(result, AlpacaRateLimitError)


def test_classify_403_in_message():
    exc    = Exception("403 Forbidden")
    result = classify_alpaca_error(exc)
    assert isinstance(result, AlpacaAuthError)


def test_classify_404_in_message():
    exc    = Exception("404 Not Found: asset not found")
    result = classify_alpaca_error(exc)
    assert isinstance(result, AlpacaSymbolNotFoundError)


def test_classify_500_in_message():
    exc    = Exception("500 Internal Server Error")
    result = classify_alpaca_error(exc)
    assert isinstance(result, AlpacaServerError)


# ---------------------------------------------------------------------------
# classify_alpaca_error — message substring matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg,expected_cls", [
    # Insufficient funds
    ("insufficient buying power for this order", AlpacaInsufficientFundsError),
    ("Insufficient funds to complete the order", AlpacaInsufficientFundsError),
    # PDT
    ("pattern day trade rule would be violated", AlpacaPDTViolationError),
    ("day trade restrictions apply",             AlpacaPDTViolationError),
    ("This order would create a day trade",      AlpacaPDTViolationError),
    # Market closed
    ("market is closed for this symbol",         AlpacaMarketClosedError),
    ("outside market hours",                     AlpacaMarketClosedError),
    ("asset is not tradable",                    AlpacaMarketClosedError),
    ("symbol is halted",                         AlpacaMarketClosedError),
    # Duplicate
    ("duplicate client_order_id detected",       AlpacaDuplicateOrderError),
    ("already exists in system",                 AlpacaDuplicateOrderError),
    # Fractional
    ("fractional shares are not supported here", AlpacaFractionalNotSupportedError),
    ("fractions not allowed for this order type",AlpacaFractionalNotSupportedError),
    # Invalid quantity
    ("invalid quantity: must be whole number",   AlpacaInvalidQuantityError),
    ("qty must be at least 1",                   AlpacaInvalidQuantityError),
    # Symbol
    ("symbol not found in our system",           AlpacaSymbolNotFoundError),
    ("asset not found",                          AlpacaSymbolNotFoundError),
    # Auth
    ("unauthorized: invalid api key",            AlpacaAuthError),
    ("authentication failed",                    AlpacaAuthError),
    # Rate limit
    ("too many requests, please slow down",      AlpacaRateLimitError),
    ("rate limit exceeded",                      AlpacaRateLimitError),
])
def test_classify_by_message(msg, expected_cls):
    exc    = Exception(msg)
    result = classify_alpaca_error(exc)
    assert isinstance(result, expected_cls), (
        f"Expected {expected_cls.__name__} for msg='{msg}', "
        f"got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# classify_alpaca_error — fallback
# ---------------------------------------------------------------------------

def test_classify_unknown_returns_alpaca_error():
    exc    = Exception("some completely unrecognized error string")
    result = classify_alpaca_error(exc)
    assert isinstance(result, AlpacaError)
    assert type(result) is AlpacaError   # generic, not a subclass


def test_classify_no_args():
    exc    = Exception()
    result = classify_alpaca_error(exc)
    assert isinstance(result, AlpacaError)


# ---------------------------------------------------------------------------
# __cause__ is set
# ---------------------------------------------------------------------------

def test_cause_is_set_on_result():
    original = ValueError("original SDK error")
    result   = classify_alpaca_error(original, http_code=429)
    assert result.__cause__ is original


def test_cause_is_set_for_message_match():
    original = RuntimeError("insufficient buying power right now")
    result   = classify_alpaca_error(original)
    assert result.__cause__ is original


def test_cause_is_set_for_fallback():
    original = Exception("mystery")
    result   = classify_alpaca_error(original)
    assert result.__cause__ is original


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------

def test_all_includes_classify():
    assert "classify_alpaca_error" in _ALL


def test_all_includes_typed_classes():
    expected = {
        "AlpacaError",
        "AlpacaAuthError",
        "AlpacaRateLimitError",
        "AlpacaInsufficientFundsError",
        "AlpacaPDTViolationError",
        "AlpacaMarketClosedError",
        "AlpacaDuplicateOrderError",
        "AlpacaInvalidQuantityError",
        "AlpacaFractionalNotSupportedError",
        "AlpacaSymbolNotFoundError",
        "AlpacaServerError",
    }
    missing = expected - set(_ALL)
    assert not missing, f"Missing from __all__: {missing}"


# ---------------------------------------------------------------------------
# Catch semantics (isinstance in except)
# ---------------------------------------------------------------------------

def test_catch_as_broker_error():
    """Any AlpacaError can be caught as BrokerError."""
    exc    = Exception("429")
    result = classify_alpaca_error(exc)
    try:
        raise result
    except BrokerError as caught:
        assert isinstance(caught, AlpacaRateLimitError)


def test_catch_as_alpaca_error():
    exc    = Exception("rate limit exceeded")
    result = classify_alpaca_error(exc)
    try:
        raise result
    except AlpacaError as caught:
        assert caught is result
