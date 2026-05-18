"""Tests for quant_shared.calendar — NYSE/NASDAQ RTH, DST, holidays, half-days."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_shared.calendar import (
    MarketCalendar,
    SessionPhase,
    get_session_phase,
    market_calendar,
    session_phase_value,
)

UTC = timezone.utc


def _utc(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Group A — Crypto (always open)
# ---------------------------------------------------------------------------

def test_crypto_is_open_any_ts():
    assert market_calendar.is_open("BTCUSDT", _utc(2026, 1, 1, 3)) is True


def test_crypto_is_open_saturday():
    assert market_calendar.is_open("ETHUSD", _utc(2026, 5, 16, 3)) is True


def test_crypto_session_phase():
    assert get_session_phase("SOLUSDT", _utc(2026, 5, 19, 12)) == SessionPhase.CRYPTO_24_7


def test_session_phase_value_rth():
    assert session_phase_value("AAPL", _utc(2026, 5, 19, 15)) == 3.0


# ---------------------------------------------------------------------------
# Group B — RTH timing (2026-05-19 EDT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hour,minute,expected",
    [
        (13, 29, False),
        (13, 30, True),
        (19, 59, True),
        (20, 0, False),
        (20, 1, False),
        (12, 0, False),
    ],
)
def test_aapl_rth_boundaries_may_19_2026(hour, minute, expected):
    ts = _utc(2026, 5, 19, hour, minute)
    assert market_calendar.is_open("AAPL", ts) is expected


# ---------------------------------------------------------------------------
# Group C — DST transitions
# ---------------------------------------------------------------------------

def test_dst_starts_march_9_2026_open():
    assert market_calendar.is_open("AAPL", _utc(2026, 3, 9, 13, 30)) is True


def test_pre_dst_march_6_2026_open():
    assert market_calendar.is_open("AAPL", _utc(2026, 3, 6, 14, 30)) is True


def test_dst_ends_november_2_2026_open():
    assert market_calendar.is_open("AAPL", _utc(2026, 11, 2, 14, 30)) is True


def test_pre_dst_end_october_30_2026_open():
    assert market_calendar.is_open("AAPL", _utc(2026, 10, 30, 13, 30)) is True


# ---------------------------------------------------------------------------
# Group D — Holidays NYSE
# ---------------------------------------------------------------------------

def test_new_years_day_2026_closed():
    assert market_calendar.is_open("AAPL", _utc(2026, 1, 1, 15)) is False


def test_mlk_day_2026_closed():
    assert market_calendar.is_open("AAPL", _utc(2026, 1, 19, 15)) is False


def test_good_friday_2026_closed():
    assert market_calendar.is_open("AAPL", _utc(2026, 4, 3, 15)) is False


def test_independence_day_observed_2026_closed():
    # July 4 2026 is Saturday → market closed that day
    assert market_calendar.is_open("AAPL", _utc(2026, 7, 4, 15)) is False


def test_independence_day_sunday_observed_monday():
    """Jul 4 2021 (Sunday) → NYSE closed Monday Jul 5 2021."""
    assert market_calendar.is_open("AAPL", _utc(2021, 7, 5, 15)) is False


def test_independence_day_saturday_observed_friday():
    """Jul 4 2020 (Saturday) → NYSE closed Friday Jul 3 2020."""
    assert market_calendar.is_open("AAPL", _utc(2020, 7, 3, 15)) is False


def test_thanksgiving_2026_closed():
    assert market_calendar.is_open("AAPL", _utc(2026, 11, 26, 15)) is False


def test_christmas_2026_closed():
    assert market_calendar.is_open("AAPL", _utc(2026, 12, 25, 15)) is False


# ---------------------------------------------------------------------------
# Group E — Half-days
# ---------------------------------------------------------------------------

def test_day_after_thanksgiving_2026_half_day():
    assert market_calendar.is_half_day("AAPL", _utc(2026, 11, 27, 12)) is True
    assert market_calendar.is_open("AAPL", _utc(2026, 11, 27, 17, 59)) is True
    assert market_calendar.is_open("AAPL", _utc(2026, 11, 27, 18, 0)) is False


def test_christmas_eve_2026_half_day():
    assert market_calendar.is_half_day("AAPL", _utc(2026, 12, 24, 12)) is True


def test_july_3_2026_independence_observed_closed():
    # July 4 2026 is Saturday → NYSE observes Friday July 3 (full close per calendar)
    assert market_calendar.is_open("AAPL", _utc(2026, 7, 3, 15)) is False
    assert market_calendar.is_half_day("AAPL", _utc(2026, 7, 3, 12)) is False


def test_half_day_post_market_after_early_close():
    ts = _utc(2026, 11, 27, 18, 30)
    assert market_calendar.session_phase("AAPL", ts) == SessionPhase.POST_MARKET


# ---------------------------------------------------------------------------
# Group F — next_open
# ---------------------------------------------------------------------------

def test_next_open_friday_evening_to_monday():
    ts = _utc(2026, 5, 15, 22)  # Friday
    nxt = market_calendar.next_open("AAPL", ts)
    assert nxt.weekday() == 0  # Monday
    assert nxt.hour == 13 and nxt.minute == 30


def test_next_open_tuesday_pre_market_same_day():
    ts = _utc(2026, 5, 19, 12)
    nxt = market_calendar.next_open("AAPL", ts)
    assert nxt == _utc(2026, 5, 19, 13, 30)


def test_next_open_after_thanksgiving():
    ts = _utc(2026, 11, 26, 15)
    nxt = market_calendar.next_open("AAPL", ts)
    assert nxt.date() == datetime(2026, 11, 27, tzinfo=UTC).date()


def test_next_open_crypto_returns_same_ts():
    ts = _utc(2026, 5, 16, 3)
    assert market_calendar.next_open("BTCUSDT", ts) == ts


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_naive_datetime_raises():
    naive = datetime(2026, 5, 19, 13, 30)
    with pytest.raises(ValueError, match="timezone-aware"):
        market_calendar.is_open("AAPL", naive)


def test_nasdaq_ticker_uses_nasdaq_calendar():
    assert market_calendar.is_open("NVDA", _utc(2026, 5, 19, 15)) is True


# ---------------------------------------------------------------------------
# Group G — Performance (pytest-benchmark)
# ---------------------------------------------------------------------------

@pytest.fixture
def market_calendar_singleton() -> MarketCalendar:
    return MarketCalendar()


def test_cache_hit_ratio_above_99pct(market_calendar_singleton: MarketCalendar) -> None:
    """Tras 100 calls a is_open() sobre fechas dentro de caché, hit ratio > 0.99."""
    cal = market_calendar_singleton
    cal._cache_hits = 0
    cal._cache_misses = 0

    base_ts = datetime(2026, 5, 19, 14, 0, tzinfo=UTC)
    for i in range(100):
        cal.is_open("AAPL", base_ts + timedelta(hours=i))

    total = cal._cache_hits + cal._cache_misses
    hit_ratio = cal._cache_hits / total if total else 0.0
    assert hit_ratio > 0.99, f"hit_ratio={hit_ratio}, expected >0.99"


def test_is_open_benchmark_warm_cache(benchmark):
    ts = _utc(2026, 5, 19, 15)
    for _ in range(50):
        market_calendar.is_open("AAPL", ts)
    benchmark(market_calendar.is_open, "AAPL", ts)


def test_cold_calendar_first_is_open_under_50ms():
    cal = MarketCalendar()
    ts = _utc(2026, 6, 1, 15)
    import time

    t0 = time.perf_counter()
    cal.is_open("AAPL", ts)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50


def test_warm_cache_many_calls_fast():
    ts = _utc(2026, 5, 19, 15)
    import time

    for _ in range(100):
        market_calendar.is_open("AAPL", ts)
    t0 = time.perf_counter()
    for _ in range(1000):
        market_calendar.is_open("AAPL", ts)
    per_call_us = (time.perf_counter() - t0) / 1000 * 1e6
    assert per_call_us < 1000  # < 1 ms per call on warm cache
