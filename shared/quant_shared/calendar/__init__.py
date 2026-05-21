"""Cross-service US equity market calendar (NYSE/NASDAQ RTH)."""
from __future__ import annotations

from .errors import MarketClosedError
from .feature_encoding import session_phase_value
from .market_calendar import MarketCalendar, market_calendar
from .session_phase import SessionPhase, get_session_phase

__all__ = [
    "MarketCalendar",
    "MarketClosedError",
    "SessionPhase",
    "get_session_phase",
    "market_calendar",
    "session_phase_value",
]

# Re-exported via market_calendar instance methods:
# is_trading_day, last_n_trading_dates
