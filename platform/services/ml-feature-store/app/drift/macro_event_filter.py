"""
MacroEventFilter — suppresses RetrainTriggerEvent within ±window_days of
scheduled FOMC, NFP, CPI, and GDP events.

Rationale (ADR-031)
-------------------
Model drift detected around scheduled macro events is often transient.
Retraining immediately after a macro print risks overfitting to a
single-day volatility spike.

Note: suppression does NOT suppress DriftDetectedEvent / ECEDriftEvent
alerts — only the actionable retrain trigger (``suppressed=True`` on Kafka).

Calendar source (priority)
--------------------------
1. ``data/macro/events_2026.yaml`` at repo root (canonical for 2026)
2. Built-in 2025–2026 fallback list (if YAML missing)
3. ``extra_events`` constructor injection (emergency / ad-hoc)
"""
from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
from typing import Sequence

__all__ = ["MacroEventFilter", "load_macro_events_from_yaml"]

def _default_macro_events_path() -> Path:
    """Resolve the optional macro calendar across repo and container layouts."""
    env_path = os.getenv("MACRO_EVENTS_PATH")
    if env_path:
        return Path(env_path)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "macro" / "events_2026.yaml"
        if candidate.is_file():
            return candidate
    return Path("data") / "macro" / "events_2026.yaml"


_DEFAULT_YAML = _default_macro_events_path()

# Fallback when YAML is absent (2025–2026 subset)
_MACRO_EVENTS_RAW: list[tuple[int, int, int, str]] = [
    (2025, 1, 29, "FOMC"), (2025, 3, 19, "FOMC"), (2025, 5, 7, "FOMC"),
    (2025, 6, 18, "FOMC"), (2025, 7, 30, "FOMC"), (2025, 9, 17, "FOMC"),
    (2025, 10, 29, "FOMC"), (2025, 12, 10, "FOMC"),
    (2026, 1, 28, "FOMC"), (2026, 3, 18, "FOMC"), (2026, 5, 6, "FOMC"),
    (2026, 6, 17, "FOMC"), (2026, 7, 29, "FOMC"), (2026, 9, 16, "FOMC"),
    (2026, 10, 28, "FOMC"), (2026, 12, 9, "FOMC"),
    (2026, 1, 9, "NFP"), (2026, 2, 6, "NFP"), (2026, 3, 6, "NFP"),
    (2026, 4, 3, "NFP"), (2026, 5, 8, "NFP"), (2026, 6, 5, "NFP"),
    (2026, 7, 2, "NFP"), (2026, 8, 7, "NFP"), (2026, 9, 4, "NFP"),
    (2026, 10, 2, "NFP"), (2026, 11, 6, "NFP"), (2026, 12, 4, "NFP"),
    (2026, 1, 14, "CPI"), (2026, 2, 11, "CPI"), (2026, 3, 11, "CPI"),
    (2026, 4, 10, "CPI"), (2026, 5, 13, "CPI"), (2026, 6, 10, "CPI"),
    (2026, 7, 14, "CPI"), (2026, 8, 12, "CPI"), (2026, 9, 10, "CPI"),
    (2026, 10, 14, "CPI"), (2026, 11, 12, "CPI"), (2026, 12, 10, "CPI"),
]

_FALLBACK_EVENTS: list[tuple[date, str]] = [
    (date(y, m, d), label) for y, m, d, label in _MACRO_EVENTS_RAW
]


def load_macro_events_from_yaml(
    path: Path | None = None,
) -> list[tuple[date, str]]:
    """Load ``(date, label)`` tuples from ``data/macro/events_2026.yaml``."""
    yaml_path = path or _DEFAULT_YAML
    if not yaml_path.is_file():
        return []

    try:
        import yaml
    except ImportError:
        return _parse_yaml_minimal(yaml_path)

    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    events: list[tuple[date, str]] = []
    for row in data.get("events", []):
        if not row:
            continue
        d = row.get("date")
        label = str(row.get("label", "MACRO"))
        if isinstance(d, date):
            ev_date = d
        else:
            ev_date = date.fromisoformat(str(d)[:10])
        events.append((ev_date, label))
    return events


def _parse_yaml_minimal(path: Path) -> list[tuple[date, str]]:
    """Parse inline ``- { date: \"...\", label: X }`` entries without PyYAML."""
    import re

    events: list[tuple[date, str]] = []
    text = path.read_text(encoding="utf-8")
    for m in re.finditer(
        r'date:\s*"?(\d{4}-\d{2}-\d{2})"?\s*,\s*label:\s*(\w+)',
        text,
    ):
        events.append((date.fromisoformat(m.group(1)), m.group(2)))
    return events


class MacroEventFilter:
    """Suppress retrain triggers around macro event dates.

    Parameters
    ----------
    extra_events : sequence of (date, label), optional
    window_days : int
        Suppression window on each side of an event (inclusive).
    yaml_path : Path, optional
        Override path to the macro calendar YAML.
    """

    def __init__(
        self,
        extra_events: Sequence[tuple[date, str]] | None = None,
        window_days: int = 2,
        yaml_path: Path | None = None,
    ) -> None:
        loaded = load_macro_events_from_yaml(yaml_path)
        self._events: list[tuple[date, str]] = (
            loaded if loaded else list(_FALLBACK_EVENTS)
        )
        if extra_events:
            self._events.extend(extra_events)
        self._default_window = window_days

    def is_suppressed(
        self,
        ts: datetime | date,
        window_days: int | None = None,
    ) -> bool:
        """True if *ts* falls within ±window_days of any macro event."""
        check_date = ts.date() if isinstance(ts, datetime) else ts
        window = window_days if window_days is not None else self._default_window
        return any(
            abs((check_date - ev_date).days) <= window
            for ev_date, _ in self._events
        )

    def nearest_event(
        self, ts: datetime | date
    ) -> tuple[date, str, int] | None:
        """Return ``(event_date, label, delta_days)`` for the closest event."""
        check_date = ts.date() if isinstance(ts, datetime) else ts
        if not self._events:
            return None
        nearest = min(
            self._events,
            key=lambda ev: abs((check_date - ev[0]).days),
        )
        return nearest[0], nearest[1], abs((check_date - nearest[0]).days)
