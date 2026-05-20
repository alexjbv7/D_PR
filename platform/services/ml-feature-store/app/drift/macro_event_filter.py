"""
MacroEventFilter — suppresses RetrainTriggerEvent within ±window_days of
scheduled FOMC, NFP, and CPI events.

Rationale (ADR-031)
-------------------
Model drift detected around scheduled macro events is often transient:
the feature distribution shifts for 1-3 bars then reverts.  Retraining
immediately after an FOMC print risks overfitting to a single-day
volatility spike.  We suppress the retrain trigger and re-evaluate on
the next cron run.

Note: suppression does NOT suppress the DriftDetectedEvent / ECEDriftEvent
alerts — those are always emitted.  Only the RetrainTriggerEvent is
suppressed; it is still published but with ``suppressed=True`` for audit.

Calendar source
---------------
Dates are hardcoded for 2025-2026.  A live calendar feed can be injected
via the ``extra_events`` constructor argument.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

__all__ = ["MacroEventFilter"]

# ---------------------------------------------------------------------------
# 2025-2026 macro event calendar
# ---------------------------------------------------------------------------

_MACRO_EVENTS_RAW: list[tuple[int, int, int, str]] = [
    # ---- FOMC ----
    # 2025
    (2025, 1, 29, "FOMC"), (2025, 3, 19, "FOMC"), (2025, 5, 7,  "FOMC"),
    (2025, 6, 18, "FOMC"), (2025, 7, 30, "FOMC"), (2025, 9, 17, "FOMC"),
    (2025, 10, 29, "FOMC"), (2025, 12, 10, "FOMC"),
    # 2026
    (2026, 1, 28, "FOMC"), (2026, 3, 18, "FOMC"), (2026, 5,  6, "FOMC"),
    (2026, 6, 17, "FOMC"), (2026, 7, 29, "FOMC"), (2026, 9, 16, "FOMC"),
    (2026, 10, 28, "FOMC"), (2026, 12, 9, "FOMC"),

    # ---- NFP (first Friday of each month) ----
    # 2025
    (2025, 1,  3, "NFP"), (2025, 2,  7, "NFP"), (2025, 3,  7, "NFP"),
    (2025, 4,  4, "NFP"), (2025, 5,  2, "NFP"), (2025, 6,  6, "NFP"),
    (2025, 7,  4, "NFP"), (2025, 8,  1, "NFP"), (2025, 9,  5, "NFP"),
    (2025, 10, 3, "NFP"), (2025, 11, 7, "NFP"), (2025, 12, 5, "NFP"),
    # 2026
    (2026, 1,  9, "NFP"), (2026, 2,  6, "NFP"), (2026, 3,  6, "NFP"),
    (2026, 4,  3, "NFP"), (2026, 5,  8, "NFP"), (2026, 6,  5, "NFP"),
    (2026, 7,  2, "NFP"), (2026, 8,  7, "NFP"), (2026, 9,  4, "NFP"),
    (2026, 10, 2, "NFP"), (2026, 11, 6, "NFP"), (2026, 12, 4, "NFP"),

    # ---- CPI (typically 10th–15th each month) ----
    # 2025
    (2025, 1, 15, "CPI"), (2025, 2, 12, "CPI"), (2025, 3, 12, "CPI"),
    (2025, 4, 10, "CPI"), (2025, 5, 13, "CPI"), (2025, 6, 11, "CPI"),
    (2025, 7, 11, "CPI"), (2025, 8, 12, "CPI"), (2025, 9, 10, "CPI"),
    (2025, 10, 15, "CPI"), (2025, 11, 13, "CPI"), (2025, 12, 10, "CPI"),
    # 2026
    (2026, 1, 14, "CPI"), (2026, 2, 11, "CPI"), (2026, 3, 11, "CPI"),
    (2026, 4, 10, "CPI"), (2026, 5, 13, "CPI"), (2026, 6, 10, "CPI"),
    (2026, 7, 14, "CPI"), (2026, 8, 12, "CPI"), (2026, 9, 10, "CPI"),
    (2026, 10, 14, "CPI"), (2026, 11, 12, "CPI"), (2026, 12, 10, "CPI"),
]

_DEFAULT_EVENTS: list[tuple[date, str]] = [
    (date(y, m, d), label) for y, m, d, label in _MACRO_EVENTS_RAW
]


class MacroEventFilter:
    """Suppress retrain triggers around macro event dates.

    Parameters
    ----------
    extra_events : sequence of (date, label), optional
        Additional event dates to inject (e.g. from a live calendar API).
    window_days : int
        Default suppression window on each side of an event (inclusive).
        If ts.date() ∈ [event_date − window, event_date + window] → suppressed.
    """

    def __init__(
        self,
        extra_events: Sequence[tuple[date, str]] | None = None,
        window_days: int = 2,
    ) -> None:
        self._events: list[tuple[date, str]] = list(_DEFAULT_EVENTS)
        if extra_events:
            self._events.extend(extra_events)
        self._default_window = window_days

    def is_suppressed(
        self,
        ts: datetime | date,
        window_days: int | None = None,
    ) -> bool:
        """Return True if *ts* falls within ±window_days of any macro event.

        Parameters
        ----------
        ts : datetime (UTC preferred) or date
        window_days : override per-call; falls back to constructor default
        """
        check_date = ts.date() if isinstance(ts, datetime) else ts
        window     = window_days if window_days is not None else self._default_window
        return any(
            abs((check_date - ev_date).days) <= window
            for ev_date, _ in self._events
        )

    def nearest_event(
        self, ts: datetime | date
    ) -> tuple[date, str, int] | None:
        """Return ``(event_date, label, delta_days)`` for the closest event.

        Returns None if no events are registered.
        """
        check_date = ts.date() if isinstance(ts, datetime) else ts
        if not self._events:
            return None
        nearest = min(
            self._events,
            key=lambda ev: abs((check_date - ev[0]).days),
        )
        return nearest[0], nearest[1], abs((check_date - nearest[0]).days)
