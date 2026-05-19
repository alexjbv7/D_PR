"""
Earnings calendar features for the daily horizon.

Sources (in priority order):
  1. Local cache  data/earnings/{symbol}_earnings.parquet
  2. yfinance (if installed)
  3. Synthetic stub — all zeros with WARN log

Blackout window: 5 trading days pre-earnings + 1 trading day post.
This mirrors typical quant fund blackout policies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parents[3] / "data" / "earnings"
_BLACKOUT_PRE_DAYS = 5
_BLACKOUT_POST_DAYS = 1


def _load_from_yfinance(symbol: str) -> pd.DatetimeIndex | None:
    try:
        import yfinance as yf  # type: ignore[import]

        ticker = yf.Ticker(symbol)
        cal: Any = ticker.calendar
        if cal is not None and hasattr(cal, "get"):
            earnings_dates_raw = cal.get("Earnings Date", [])
            if earnings_dates_raw:
                return pd.DatetimeIndex(pd.to_datetime(earnings_dates_raw))
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance earnings fetch failed for %s: %s", symbol, exc)
    return None


def _load_from_cache(symbol: str) -> pd.DatetimeIndex | None:
    path = _CACHE_DIR / f"{symbol}_earnings.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return pd.DatetimeIndex(pd.to_datetime(df["earnings_date"]))
    return None


def get_earnings_dates(symbol: str) -> pd.DatetimeIndex:
    """
    Return known future and past earnings dates for symbol.

    Falls back to empty index with a warning when no source is available.
    """
    dates = _load_from_cache(symbol)
    if dates is not None and len(dates) > 0:
        return dates

    dates = _load_from_yfinance(symbol)
    if dates is not None and len(dates) > 0:
        return dates

    logger.warning(
        "No earnings dates available for %s. "
        "Earnings features will be zero. "
        "TODO(@alex): populate data/earnings/ from Alpaca or Polygon earnings API.",
        symbol,
    )
    return pd.DatetimeIndex([])


def compute_earnings_features(
    symbol: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute days-to-earnings, days-since-earnings, and blackout flag.

    Parameters
    ----------
    symbol : str
        Ticker symbol (used to load earnings calendar).
    dates : pd.DatetimeIndex
        Bar timestamps for which to compute features.

    Returns
    -------
    pd.DataFrame with columns:
        days_to_earnings      — calendar days to next earnings (NaN if none known)
        days_since_earnings   — calendar days since last earnings (NaN if none)
        is_earnings_blackout  — 1 if within 5d pre + 1d post window, else 0
    """
    earnings_idx = get_earnings_dates(symbol)

    days_to: list[float] = []
    days_since: list[float] = []
    blackout: list[int] = []

    for dt in dates:
        dt_ts = pd.Timestamp(dt)

        future = earnings_idx[earnings_idx > dt_ts]
        past   = earnings_idx[earnings_idx <= dt_ts]

        to_next   = float((future.min() - dt_ts).days) if len(future) > 0 else np.nan
        since_last = float((dt_ts - past.max()).days) if len(past) > 0 else np.nan

        in_blackout = 0
        if not np.isnan(to_next) and to_next <= _BLACKOUT_PRE_DAYS:
            in_blackout = 1
        if not np.isnan(since_last) and since_last <= _BLACKOUT_POST_DAYS:
            in_blackout = 1

        days_to.append(to_next)
        days_since.append(since_last)
        blackout.append(in_blackout)

    return pd.DataFrame(
        {
            "days_to_earnings":    days_to,
            "days_since_earnings": days_since,
            "is_earnings_blackout": blackout,
        },
        index=dates,
    )
