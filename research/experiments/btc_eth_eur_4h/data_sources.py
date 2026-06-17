"""Unified 4H OHLCV loader for the BTC/ETH/EUR experiment.

Routing
-------
* Crypto (``BTC/USD``, ``ETH/USD``): delegated to the canonical
  ``data.drl_dataset.fetch_ohlcv_frame`` (Alpaca). Imported lazily so this
  module (and its tests) stay importable without the Alpaca SDK / credentials.
* Spot FX (``EUR/USD``): Alpaca has no FX feed, so bars come from yfinance.
  yfinance exposes 1H intraday (capped at ~730 days of history); 4H bars are
  built by causal resampling (open=first, high=max, low=min, close=last,
  volume=sum). This is the known limitation accepted for FX: the 4H FX sample
  is ~2 years vs the multi-year crypto sample.

Both paths return the SAME schema the gate consumes:
``DataFrame[open, high, low, close, volume]`` with a UTC ``DatetimeIndex``.

The network/SDK call is isolated in ``_yf_download`` (yfinance) and in the
lazily-imported Alpaca wrapper so tests can monkeypatch them — no network or
credentials required for unit tests (mirrors ``data.drl_dataset._fetch_ohlcv``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import is_fx

logger = logging.getLogger(__name__)

#: yfinance hard cap for 1H intraday history.
_YF_1H_MAX_DAYS: int = 729

#: experiment timeframe -> pandas resample rule.
_PANDAS_RULE: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1D"}

_OHLCV_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
_OHLC_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _fx_to_yahoo(symbol: str) -> str:
    """Map an FX pair to its Yahoo Finance ticker (``EUR/USD`` -> ``EURUSD=X``)."""
    base, quote = (p.strip().upper() for p in symbol.split("/"))
    return f"{base}{quote}=X"


def _yf_download(
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame:
    """Fetch bars from yfinance, normalized to lowercase OHLCV columns.

    Isolated for monkeypatching in tests (no network there). Flattens the
    MultiIndex columns yfinance returns for a single ticker and lowercases
    them; does not alter the index timezone (handled by the caller).

    Parameters
    ----------
    ticker : str
        Yahoo ticker (e.g. ``"EURUSD=X"``).
    start, end : str
        ISO dates.
    interval : str
        yfinance interval (``"1h"``).

    Returns
    -------
    pd.DataFrame
        Columns ``[open, high, low, close, volume]`` (subset present), raw index.
    """
    import yfinance as yf  # lazy: keep module importable without yfinance

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no bars for {ticker} [{start}..{end}] @ {interval}"
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    present = [c for c in _OHLCV_COLS if c in raw.columns]
    return raw[present]


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a UTC tz-aware DatetimeIndex, sorted ascending."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    return out.sort_index()


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Causally resample an intraday OHLCV frame to ``timeframe``.

    Bars are left-labeled and left-closed (timestamped at their open, matching
    Alpaca's convention); empty buckets (weekend/holiday gaps in FX) are
    dropped so no synthetic flat bars enter the feature pipeline.
    """
    rule = _PANDAS_RULE.get(timeframe)
    if rule is None:
        raise NotImplementedError(f"unsupported timeframe {timeframe!r}")
    agg = {k: v for k, v in _OHLC_AGG.items() if k in df.columns}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=[c for c in ("open", "high", "low", "close") if c in out])


def load_fx_yfinance(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    *,
    timeframe: str = "4h",
) -> pd.DataFrame:
    """Load spot-FX 4H bars from yfinance (resampled from 1H).

    Parameters
    ----------
    symbol : str
        FX pair (``"EUR/USD"``).
    start, end : str or datetime
        Requested window. ``start`` is clamped forward to stay within
        yfinance's ~730-day 1H cap (a warning is logged when clamped).
    timeframe : str
        Target timeframe (``"4h"``).

    Returns
    -------
    pd.DataFrame
        ``[open, high, low, close, volume]`` indexed by a UTC DatetimeIndex.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    floor = end_ts - timedelta(days=_YF_1H_MAX_DAYS)
    if start_ts < floor:
        logger.warning(
            "yfinance 1H cap: clamping %s start %s -> %s (max %d days)",
            symbol, start_ts.date(), floor.date(), _YF_1H_MAX_DAYS,
        )
        start_ts = floor

    raw = _yf_download(
        _fx_to_yahoo(symbol),
        start_ts.date().isoformat(),
        end_ts.date().isoformat(),
        interval="1h",
    )
    raw = _ensure_utc(raw)
    out = _resample_ohlcv(raw, timeframe)
    if out.empty:
        raise RuntimeError(f"no {timeframe} bars after resampling {symbol}")
    if "volume" in out.columns and float(out["volume"].fillna(0).abs().sum()) == 0.0:
        rng = (out["high"] - out["low"]).abs()
        out["volume"] = rng.mask(rng <= 0).ffill().bfill().fillna(1.0)
    logger.info(
        "loaded FX %s: %d %s bars [%s .. %s]",
        symbol, len(out), timeframe, out.index.min().date(), out.index.max().date(),
    )
    return out[[c for c in _OHLCV_COLS if c in out.columns]]


def load_crypto_alpaca(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    *,
    timeframe: str = "4h",
    feed: str = "iex",
) -> pd.DataFrame:
    """Load crypto 4H bars via the canonical Alpaca path (lazy import).

    Thin pass-through to ``data.drl_dataset.fetch_ohlcv_frame`` so the crypto
    leg reuses the audited ingestor/cache and returns the identical schema.
    """
    from data.drl_dataset import fetch_ohlcv_frame  # lazy: pulls features/sklearn

    df = fetch_ohlcv_frame(symbol, start, end, timeframe=timeframe, feed=feed)
    return df[[c for c in _OHLCV_COLS if c in df.columns]]


def load_raw_ohlcv(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    *,
    timeframe: str = "4h",
    feed: str = "iex",
) -> pd.DataFrame:
    """Load raw OHLCV for ``symbol``, routing crypto->Alpaca and FX->yfinance.

    Parameters
    ----------
    symbol : str
        Instrument in ``BASE/QUOTE`` form (``"BTC/USD"``, ``"EUR/USD"``).
    start, end : str or datetime
        Window (ISO or datetime).
    timeframe : str
        Bar timeframe (``"4h"``).
    feed : str
        Alpaca feed for the crypto leg.

    Returns
    -------
    pd.DataFrame
        ``[open, high, low, close, volume]``, UTC DatetimeIndex — the schema the
        ADR-040 gate (``models.drl.dsr_gate``) consumes.
    """
    if is_fx(symbol):
        return load_fx_yfinance(symbol, start, end, timeframe=timeframe)
    return load_crypto_alpaca(symbol, start, end, timeframe=timeframe, feed=feed)
