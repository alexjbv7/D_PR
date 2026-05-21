"""Session phase enum and helpers for equities vs crypto."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from quant_shared.symbols import is_equity

UTC = timezone.utc


class SessionPhase(str, Enum):
    """Trading session phase for feature-engine and risk."""

    PRE_MARKET = "pre_market"
    RTH = "rth"
    POST_MARKET = "post_market"
    CLOSED_EQUITY = "closed_equity"
    CRYPTO_24_7 = "crypto_24_7"


def classify_equity_phase(
    ts_utc: datetime,
    market_open_utc: datetime | None,
    market_close_utc: datetime | None,
) -> SessionPhase:
    """
  Classify phase for an equity on a calendar day given RTH bounds (UTC).

  Parameters
  ----------
  ts_utc : datetime
      Instant in UTC.
  market_open_utc, market_close_utc : datetime | None
      RTH bounds from ``pandas_market_calendars``; ``None`` if no session.
  """
    if market_open_utc is None or market_close_utc is None:
        return SessionPhase.CLOSED_EQUITY
    if ts_utc < market_open_utc:
        return SessionPhase.PRE_MARKET
    if ts_utc < market_close_utc:
        return SessionPhase.RTH
    return SessionPhase.POST_MARKET


def get_session_phase(symbol: str, ts_utc: datetime) -> SessionPhase:
    """
    Return session phase for ``symbol`` at ``ts_utc`` (delegates to singleton calendar).

    Crypto symbols always return :attr:`SessionPhase.CRYPTO_24_7`.
    """
    if not is_equity(symbol):
        return SessionPhase.CRYPTO_24_7
    from quant_shared.calendar.market_calendar import market_calendar

    return market_calendar.session_phase(symbol, ts_utc)
