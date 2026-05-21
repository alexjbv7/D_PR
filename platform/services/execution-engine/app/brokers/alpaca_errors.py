"""
Typed Alpaca error hierarchy.
=====================================================================

Extends the base :class:`~.base.BrokerError` with Alpaca-specific subclasses
mapped from HTTP status codes and error message patterns returned by the
`alpaca-py` SDK and the underlying Alpaca REST API.

Usage
-----
Swap generic ``BrokerError`` raises in ``alpaca.py`` for typed subclasses so
callers can react precisely::

    from app.brokers.alpaca_errors import classify_alpaca_error, AlpacaRateLimitError

    try:
        raw = await to_thread(client.submit_order, ...)
    except Exception as exc:
        typed = classify_alpaca_error(exc)
        if isinstance(typed, AlpacaRateLimitError):
            await asyncio.sleep(backoff)
        raise typed from exc

References
----------
* Alpaca error codes: https://docs.alpaca.markets/reference/error-codes
* Architecture doc §1.4 (Errores tipados)
"""
from __future__ import annotations

import re
from typing import Optional

from .base import BrokerError


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class AlpacaError(BrokerError):
    """Base class for all Alpaca-specific broker errors."""


class AlpacaAuthError(AlpacaError):
    """Authentication or authorization failure (HTTP 401 / 403, bad API key)."""


class AlpacaRateLimitError(AlpacaError):
    """Alpaca responded with HTTP 429 Too Many Requests."""


class AlpacaInsufficientFundsError(AlpacaError):
    """
    Account does not have enough buying power / cash for the requested order.

    Alpaca message examples
    -----------------------
    * "insufficient buying power"
    * "insufficient funds for this order"
    """


class AlpacaPDTViolationError(AlpacaError):
    """
    The order would violate the Pattern Day Trader (PDT) rule.

    Accounts with < $25,000 equity may execute at most 3 day trades per
    rolling 5-business-day window.
    """


class AlpacaMarketClosedError(AlpacaError):
    """
    The requested symbol is not currently tradable (market closed, halted,
    or ``extended_hours`` flag missing).
    """


class AlpacaDuplicateOrderError(AlpacaError):
    """
    A duplicate ``client_order_id`` was submitted (HTTP 422).

    The execution engine uses ``intent_id`` as the idempotency key, so this
    error indicates a retry that has already been accepted by Alpaca.
    """


class AlpacaInvalidQuantityError(AlpacaError):
    """
    The order quantity is invalid (zero, negative, or non-integer for
    assets that require whole shares).
    """


class AlpacaFractionalNotSupportedError(AlpacaError):
    """
    Fractional shares are not supported for the requested order type or
    symbol (e.g. stop orders with fractional qty).
    """


class AlpacaSymbolNotFoundError(AlpacaError):
    """Symbol not found, not listed, or not tradable on Alpaca (HTTP 404)."""


class AlpacaServerError(AlpacaError):
    """Alpaca server-side error (HTTP 5xx)."""


class BracketRejectedError(AlpacaError):
    """
    Alpaca rejected bracket order (e.g. stop_price too close to entry,
    incompatible asset, fractional + bracket combo).
    """


class TrailingStopRejectedError(AlpacaError):
    """Alpaca rejected trailing stop (e.g. trail amount out of range)."""


class BracketIncompatibleOrderTypeError(AlpacaError):
    """
    Caller passed LIMIT_MAKER with sl_price+tp_price.
    Alpaca bracket requires LIMIT (not post-only) or MARKET.
    """


# Alias used in architecture docs / Semana 2 naming
FractionalShareNotAllowedError = AlpacaFractionalNotSupportedError
InsufficientBuyingPowerError = AlpacaInsufficientFundsError
MarketClosedError = AlpacaMarketClosedError
AssetNotTradableError = AlpacaSymbolNotFoundError


# ---------------------------------------------------------------------------
# Internal classification rules
# ---------------------------------------------------------------------------

_CODE_TO_CLASS: dict[int, type[AlpacaError]] = {
    401: AlpacaAuthError,
    403: AlpacaAuthError,
    404: AlpacaSymbolNotFoundError,
    429: AlpacaRateLimitError,
    500: AlpacaServerError,
    502: AlpacaServerError,
    503: AlpacaServerError,
    504: AlpacaServerError,
}

# (substring_in_lowercase_message, error_class)
# Checked in order; first match wins.
_MSG_RULES: list[tuple[str, type[AlpacaError]]] = [
    ("insufficient buying power",    AlpacaInsufficientFundsError),
    ("insufficient funds",           AlpacaInsufficientFundsError),
    ("pattern day trad",             AlpacaPDTViolationError),
    ("day trade",                    AlpacaPDTViolationError),
    (" pdt ",                        AlpacaPDTViolationError),
    ("market is closed",             AlpacaMarketClosedError),
    ("outside market hours",         AlpacaMarketClosedError),
    ("not tradable",                 AlpacaMarketClosedError),
    ("asset is not tradable",        AlpacaMarketClosedError),
    ("halted",                       AlpacaMarketClosedError),
    ("duplicate client_order_id",    AlpacaDuplicateOrderError),
    ("already exists",               AlpacaDuplicateOrderError),
    ("fractional not allowed",       AlpacaFractionalNotSupportedError),
    ("fractional shares are not",    AlpacaFractionalNotSupportedError),
    ("fractions",                    AlpacaFractionalNotSupportedError),
    ("invalid quantity",             AlpacaInvalidQuantityError),
    ("qty must be",                  AlpacaInvalidQuantityError),
    ("quantity must be",             AlpacaInvalidQuantityError),
    ("min qty",                      AlpacaInvalidQuantityError),
    ("symbol not found",             AlpacaSymbolNotFoundError),
    ("asset not found",              AlpacaSymbolNotFoundError),
    ("ticker not found",             AlpacaSymbolNotFoundError),
    ("unauthorized",                 AlpacaAuthError),
    ("forbidden",                    AlpacaAuthError),
    ("invalid api key",              AlpacaAuthError),
    ("api key",                      AlpacaAuthError),
    ("authentication failed",        AlpacaAuthError),
    ("too many requests",            AlpacaRateLimitError),
    ("rate limit",                   AlpacaRateLimitError),
    ("take_profit",                  BracketRejectedError),
    ("stop_loss",                    BracketRejectedError),
    ("bracket",                      BracketRejectedError),
    ("order_class",                  BracketRejectedError),
    ("trail_percent",                TrailingStopRejectedError),
    ("trail_price",                  TrailingStopRejectedError),
    ("trailing stop",                TrailingStopRejectedError),
    ("post-only",                    BracketIncompatibleOrderTypeError),
    ("post only",                    BracketIncompatibleOrderTypeError),
]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def classify_alpaca_error(
    exc: BaseException,
    http_code: int = 0,
) -> AlpacaError:
    """
    Translate a raw Alpaca exception into the most specific typed subclass.

    Parameters
    ----------
    exc : BaseException
        Original exception from ``alpaca-py`` or the HTTP layer.
    http_code : int, optional
        HTTP status code (e.g. 422, 429, 500) if it can be extracted
        externally.  Used as primary signal; message matching is fallback.

    Returns
    -------
    AlpacaError
        Typed error with ``__cause__`` set to *exc*.

    Examples
    --------
    >>> err = Exception("403 Forbidden: invalid API key")
    >>> typed = classify_alpaca_error(err, http_code=403)
    >>> isinstance(typed, AlpacaAuthError)
    True

    >>> err2 = Exception("insufficient buying power for this order")
    >>> isinstance(classify_alpaca_error(err2), AlpacaInsufficientFundsError)
    True
    """
    # 1. Explicit HTTP code (caller-supplied) → direct lookup
    if http_code and http_code in _CODE_TO_CLASS:
        cls = _CODE_TO_CLASS[http_code]
        typed = cls(str(exc))
        typed.__cause__ = exc
        return typed

    # Catch 5xx family
    if http_code >= 500:
        typed = AlpacaServerError(str(exc))
        typed.__cause__ = exc
        return typed

    # 2. HTTP code embedded in the exception message
    msg_lower = str(exc).lower()
    http_match = re.search(r"\b(4\d{2}|5\d{2})\b", msg_lower)
    if http_match:
        code = int(http_match.group(1))
        if code in _CODE_TO_CLASS:
            cls = _CODE_TO_CLASS[code]
            typed = cls(str(exc))
            typed.__cause__ = exc
            return typed
        if code >= 500:
            typed = AlpacaServerError(str(exc))
            typed.__cause__ = exc
            return typed

    # 3. Substring rules (case-insensitive)
    for pattern, cls in _MSG_RULES:
        if pattern in msg_lower:
            typed = cls(str(exc))
            typed.__cause__ = exc
            return typed

    # 4. Generic fallback
    typed = AlpacaError(str(exc))
    typed.__cause__ = exc
    return typed


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
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
    "BracketRejectedError",
    "TrailingStopRejectedError",
    "BracketIncompatibleOrderTypeError",
    "FractionalShareNotAllowedError",
    "InsufficientBuyingPowerError",
    "MarketClosedError",
    "AssetNotTradableError",
    "classify_alpaca_error",
]
