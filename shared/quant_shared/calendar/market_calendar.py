"""NYSE/NASDAQ session calendar with in-memory schedule cache."""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone
from typing import Final, cast
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from quant_shared.symbols import is_equity

from .session_phase import SessionPhase, classify_equity_phase

UTC: Final = timezone.utc
ET: Final = ZoneInfo("America/New_York")
_CACHE_TTL = timedelta(hours=24)
_CACHE_HORIZON_DAYS = 400
_CACHE_LOOKBACK_DAYS = 400
_NORMAL_SESSION_SECONDS = 6.5 * 3600

# Schedule cache window: ±400 calendar days from today (see docs/adr/016-shared-calendar-module.md).
# Supports walk-forward / historical is_open() without per-call pandas_market_calendars loads.

# TODO(@alex 2026-08-01): cargar mapping completo desde Alpaca /v2/assets
NASDAQ_ONLY: Final[frozenset[str]] = frozenset(
    {
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AVGO",
        "COST",
        "NFLX",
        "ADBE",
        "PEP",
        "AMD",
        "CSCO",
        "INTC",
        "CMCSA",
        "QCOM",
        "TXN",
        "AMGN",
        "INTU",
    }
)


def _ts_to_utc(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        ts = ts.tz_localize(ET)
    return cast(datetime, ts.tz_convert(UTC).to_pydatetime())


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (use UTC)")
    return ts.astimezone(UTC)


def _et_date(ts_utc: datetime) -> date:
    return ts_utc.astimezone(ET).date()


class MarketCalendar:
    """
    Wrapper over ``pandas_market_calendars`` with a 24 h in-memory schedule cache.

    Attributes
    ----------
    _xnys, _xnas : pandas_market_calendars.MarketCalendar
        NYSE and NASDAQ calendars.
    _cache : dict[str, pd.DataFrame]
        Exchange code → schedule DataFrame.
    _cache_expires_at : datetime
        UTC instant when cache must refresh.
    """

    def __init__(self) -> None:
        self._xnys = mcal.get_calendar("XNYS")
        self._xnas = mcal.get_calendar("NASDAQ")
        self._cache: dict[str, pd.DataFrame] = {}
        self._day_index: dict[str, dict[date, int]] = {}
        self._bounds_cache: dict[tuple[str, date], tuple[datetime, datetime] | None] = {}
        self._cache_expires_at = datetime.min.replace(tzinfo=UTC)
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._refresh_cache()

    def _exchange_for_symbol(self, symbol: str) -> str:
        if symbol.upper() in NASDAQ_ONLY:
            return "XNAS"
        return "XNYS"

    def _calendar_for_exchange(self, exchange: str) -> mcal.MarketCalendar:
        return self._xnas if exchange == "XNAS" else self._xnys

    def _refresh_cache_if_needed(self, through: date | None = None) -> None:
        if datetime.now(tz=UTC) >= self._cache_expires_at:
            self._cache_misses += 1
            self._refresh_cache(through=through)
            return
        if through is not None:
            for ex in ("XNYS", "XNAS"):
                df = self._cache.get(ex)
                if df is not None and not df.empty:
                    last = df.index[-1]
                    last_d = last.date() if hasattr(last, "date") else pd.Timestamp(last).date()
                    if through > last_d:
                        self._cache_misses += 1
                        self._refresh_cache(through=through)
                        return
        self._cache_hits += 1

    def _refresh_cache(self, through: date | None = None) -> None:
        """Load schedule [today-lookback, today+horizon] for all exchanges. TTL 24 h."""
        with self._lock:
            today = datetime.now(tz=UTC).date()
            start = today - timedelta(days=_CACHE_LOOKBACK_DAYS)
            end = through or (today + timedelta(days=_CACHE_HORIZON_DAYS))
            if through is not None and through > end:
                end = through
            self._bounds_cache.clear()
            for exchange in ("XNYS", "XNAS"):
                cal = self._calendar_for_exchange(exchange)
                df = cal.schedule(
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                )
                self._cache[exchange] = df
                self._day_index[exchange] = self._build_day_index(df)
            self._cache_expires_at = datetime.now(tz=UTC) + _CACHE_TTL

    @staticmethod
    def _build_day_index(df: pd.DataFrame) -> dict[date, int]:
        out: dict[date, int] = {}
        for i in range(len(df)):
            ts = pd.Timestamp(df.index[i])
            if ts.tzinfo is None:
                ts = ts.tz_localize(ET)
            else:
                ts = ts.tz_convert(ET)
            out[ts.date()] = i
        return out

    def _schedule_for(self, symbol: str) -> pd.DataFrame:
        exchange = self._exchange_for_symbol(symbol)
        self._refresh_cache_if_needed()
        return self._cache[exchange]

    def _row_for_date(self, exchange: str, session_date: date) -> pd.Series | None:
        df = self._cache.get(exchange)
        if df is None or df.empty:
            return None
        idx = self._day_index.get(exchange, {}).get(session_date)
        if idx is None:
            return None
        return df.iloc[idx]

    def _bounds_utc(
        self, symbol: str, session_date: date
    ) -> tuple[datetime | None, datetime | None]:
        self._refresh_cache_if_needed()
        exchange = self._exchange_for_symbol(symbol)
        key = (exchange, session_date)
        if key in self._bounds_cache:
            cached = self._bounds_cache[key]
            if cached is None:
                return None, None
            return cached
        row = self._row_for_date(exchange, session_date)
        if row is None:
            self._bounds_cache[key] = None
            return None, None
        bounds = (
            _ts_to_utc(pd.Timestamp(row["market_open"])),
            _ts_to_utc(pd.Timestamp(row["market_close"])),
        )
        self._bounds_cache[key] = bounds
        return bounds

    def is_open(self, symbol: str, ts_utc: datetime) -> bool:
        """
        True if ``symbol`` is tradable at ``ts_utc`` (RTH for US equities).

        Crypto symbols always return ``True``.
        """
        ts_utc = _ensure_utc(ts_utc)
        if not is_equity(symbol):
            return True
        open_utc, close_utc = self._bounds_utc(symbol, _et_date(ts_utc))
        if open_utc is None or close_utc is None:
            return False
        return open_utc <= ts_utc < close_utc

    def next_open(self, symbol: str, ts_utc: datetime) -> datetime:
        """UTC datetime of the next RTH open. Crypto returns ``ts_utc`` unchanged."""
        ts_utc = _ensure_utc(ts_utc)
        if not is_equity(symbol):
            return ts_utc

        search_end = _et_date(ts_utc) + timedelta(days=30)
        self._refresh_cache_if_needed(through=search_end)

        cursor = _et_date(ts_utc)
        while cursor <= search_end:
            open_utc, close_utc = self._bounds_utc(symbol, cursor)
            if open_utc is None:
                cursor += timedelta(days=1)
                continue
            if ts_utc < open_utc:
                return open_utc
            if close_utc is not None and ts_utc >= close_utc:
                cursor += timedelta(days=1)
                continue
            # In session: next RTH open is the following trading day.
            probe = cursor + timedelta(days=1)
            while probe <= search_end:
                next_open_utc, _ = self._bounds_utc(symbol, probe)
                if next_open_utc is not None:
                    return next_open_utc
                probe += timedelta(days=1)
            break
            cursor += timedelta(days=1)

        open_utc, _ = self._bounds_utc(symbol, search_end)
        if open_utc is not None:
            return open_utc
        return ts_utc + timedelta(days=1)

    def session_phase(self, symbol: str, ts_utc: datetime) -> SessionPhase:
        """Session phase at ``ts_utc`` (RTH bounds from exchange calendar)."""
        ts_utc = _ensure_utc(ts_utc)
        if not is_equity(symbol):
            return SessionPhase.CRYPTO_24_7
        open_utc, close_utc = self._bounds_utc(symbol, _et_date(ts_utc))
        return classify_equity_phase(ts_utc, open_utc, close_utc)

    def is_half_day(self, symbol: str, ts_utc: datetime) -> bool:
        """True on early-close sessions (e.g. day after Thanksgiving)."""
        ts_utc = _ensure_utc(ts_utc)
        if not is_equity(symbol):
            return False
        row = self._row_for_date(self._exchange_for_symbol(symbol), _et_date(ts_utc))
        if row is None:
            return False
        open_ts = pd.Timestamp(row["market_open"])
        close_ts = pd.Timestamp(row["market_close"])
        duration = float((close_ts - open_ts).total_seconds())
        return duration < _NORMAL_SESSION_SECONDS - 60.0


market_calendar = MarketCalendar()
