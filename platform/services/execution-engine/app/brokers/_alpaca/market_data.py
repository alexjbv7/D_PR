"""
AlpacaMarketData — real-time and historical market data via alpaca-py.
=====================================================================

Wraps ``alpaca.data`` clients (Stock + Crypto) behind a single async-friendly
class.  Every network call is offloaded to :func:`asyncio.to_thread` so the
event loop stays responsive.

Capabilities
------------
* Historical OHLCV bars (any timeframe: 1Min → 1Month).
* Latest bar / latest quote / latest trade per symbol.
* Full snapshot (latest quote + trade + minute bar + daily bar).
* ``get_last_price(symbol)`` — single Decimal, used by the execution engine's
  risk gate for pre-trade price validation.

Data feed
---------
* **Stocks**: ``iex`` (free, 15-min delayed quotes but real-time bars) or
  ``sip`` (paid, all US exchanges, real-time everything).
* **Crypto**: no feed option — always real-time from Alpaca's matching engine.

References
----------
* Alpaca Data API v2: https://docs.alpaca.markets/docs/about-market-data-api
* alpaca-py data guide: https://alpaca.markets/sdks/python/market_data.html
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from .rate_limiter import AlpacaRateLimiter
from .retry import retry_with_jitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    from alpaca.data.historical import (                                # type: ignore
        CryptoHistoricalDataClient,
        StockHistoricalDataClient,
    )
    from alpaca.data.requests import (                                  # type: ignore
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
        CryptoSnapshotRequest,
        StockBarsRequest,
        StockLatestQuoteRequest,
        StockSnapshotRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit          # type: ignore
    _HAS_DATA_SDK = True
except ImportError:  # pragma: no cover
    _HAS_DATA_SDK = False
    StockHistoricalDataClient  = None   # type: ignore[assignment,misc]
    CryptoHistoricalDataClient = None   # type: ignore[assignment,misc]
    StockBarsRequest           = None   # type: ignore[assignment,misc]
    CryptoBarsRequest          = None   # type: ignore[assignment,misc]
    TimeFrame                  = None   # type: ignore[assignment,misc]
    TimeFrameUnit              = None   # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Prometheus (optional)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram  # type: ignore[import-untyped]

    DATA_REQUESTS = Counter(
        "alpaca_data_requests_total",
        "Alpaca market-data requests by method",
        ["method"],  # bars | quote | snapshot | last_price
    )
    DATA_LATENCY = Histogram(
        "alpaca_data_latency_seconds",
        "Alpaca market-data call latency",
        buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
    )
    _HAS_PROM = True
except ImportError:  # pragma: no cover
    _HAS_PROM = False
    DATA_REQUESTS = None  # type: ignore[assignment]
    DATA_LATENCY  = None  # type: ignore[assignment]


def _record(method: str) -> None:
    if _HAS_PROM:
        DATA_REQUESTS.labels(method=method).inc()


# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------

_TF_MAP: dict[str, Any] = {}


def _init_tf_map() -> None:
    """Populate once after confirming the SDK is present."""
    if _TF_MAP or not _HAS_DATA_SDK:
        return
    _TF_MAP.update({
        "1min":   TimeFrame(1,  TimeFrameUnit.Minute),
        "5min":   TimeFrame(5,  TimeFrameUnit.Minute),
        "15min":  TimeFrame(15, TimeFrameUnit.Minute),
        "30min":  TimeFrame(30, TimeFrameUnit.Minute),
        "1h":     TimeFrame(1,  TimeFrameUnit.Hour),
        "4h":     TimeFrame(4,  TimeFrameUnit.Hour),
        "1d":     TimeFrame(1,  TimeFrameUnit.Day),
        "1w":     TimeFrame(1,  TimeFrameUnit.Week),
        "1m":     TimeFrame(1,  TimeFrameUnit.Month),
    })


def resolve_timeframe(tf: str) -> Any:
    """Map a human-friendly string to an ``alpaca.data.TimeFrame``."""
    _init_tf_map()
    key = tf.lower().strip()
    if key not in _TF_MAP:
        raise ValueError(
            f"Unknown timeframe '{tf}'. Choose from: {sorted(_TF_MAP.keys())}"
        )
    return _TF_MAP[key]


# ---------------------------------------------------------------------------
# Bar dataclass (plain dict — no Pydantic dependency inside broker layer)
# ---------------------------------------------------------------------------

def _bar_to_dict(bar: Any, symbol: str) -> dict:
    """Normalise an alpaca-py Bar object → plain dict."""
    return {
        "symbol":    symbol,
        "timestamp": bar.timestamp.isoformat() if hasattr(bar.timestamp, "isoformat") else str(bar.timestamp),
        "open":      float(bar.open),
        "high":      float(bar.high),
        "low":       float(bar.low),
        "close":     float(bar.close),
        "volume":    float(bar.volume),
        "vwap":      float(bar.vwap) if hasattr(bar, "vwap") and bar.vwap else None,
        "trade_count": int(bar.trade_count) if hasattr(bar, "trade_count") and bar.trade_count else None,
    }


# ---------------------------------------------------------------------------
# AlpacaMarketData
# ---------------------------------------------------------------------------

class AlpacaMarketData:
    """
    Async wrapper around Alpaca's Stock + Crypto data clients.

    Parameters
    ----------
    api_key : str
        Alpaca API key.
    api_secret : str, optional
        Alpaca API secret.  Some data endpoints work without it.
    feed : str
        Stock data feed: ``"iex"`` (free) or ``"sip"`` (paid, real-time).
    rate_limiter : AlpacaRateLimiter, optional
        Shared rate limiter (uses the ``"data"`` bucket).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        feed: str = "iex",
        rate_limiter: Optional[AlpacaRateLimiter] = None,
    ):
        if not _HAS_DATA_SDK:
            raise RuntimeError(
                "alpaca-py data SDK not available. Run: pip install alpaca-py"
            )
        self._api_key    = api_key
        self._api_secret = api_secret or ""
        self._feed       = feed
        self._rate_limiter = rate_limiter or AlpacaRateLimiter()

        self._stock_client:  Any = None
        self._crypto_client: Any = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def connect(self) -> None:
        """Create underlying SDK clients (sync — lightweight, no network)."""
        _init_tf_map()
        self._stock_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._api_secret if self._api_secret else None,
        )
        self._crypto_client = CryptoHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._api_secret if self._api_secret else None,
        )
        logger.info(
            "alpaca.market_data.connected feed=%s key=%s***",
            self._feed,
            self._api_key[:4] if self._api_key else "?",
        )

    def close(self) -> None:
        self._stock_client  = None
        self._crypto_client = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch historical OHLCV bars.

        Parameters
        ----------
        symbol : str
            Alpaca-format symbol (``"AAPL"`` or ``"BTC/USD"``).
        timeframe : str
            One of ``1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w, 1m``.
        start, end : datetime, optional
            UTC range.  If omitted, Alpaca returns the most recent bars.
        limit : int
            Max bars to return (default 200).

        Returns
        -------
        list[dict]
            Each dict has keys: symbol, timestamp, open, high, low, close,
            volume, vwap, trade_count.
        """
        await self._rate_limiter.acquire("data")
        _record("bars")
        tf = resolve_timeframe(timeframe)
        is_crypto = "/" in symbol

        if is_crypto:
            return await self._get_crypto_bars(symbol, tf, start, end, limit)
        return await self._get_stock_bars(symbol, tf, start, end, limit)

    async def get_latest_bar(self, symbol: str) -> Optional[dict]:
        """Return the most recent completed bar for *symbol*."""
        await self._rate_limiter.acquire("data")
        _record("bars")
        is_crypto = "/" in symbol
        try:
            if is_crypto:
                raw = await self._crypto_latest_bar(symbol)
            else:
                raw = await self._stock_latest_bar(symbol)
            return _bar_to_dict(raw, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("market_data.latest_bar_failed %s: %s", symbol, exc)
            return None

    async def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Return the latest bid/ask quote for *symbol*."""
        await self._rate_limiter.acquire("data")
        _record("quote")
        is_crypto = "/" in symbol
        try:
            if is_crypto:
                raw = await self._crypto_latest_quote(symbol)
            else:
                raw = await self._stock_latest_quote(symbol)
            return {
                "symbol":    symbol,
                "bid_price": float(raw.bid_price) if raw.bid_price else None,
                "ask_price": float(raw.ask_price) if raw.ask_price else None,
                "bid_size":  float(raw.bid_size)  if raw.bid_size  else None,
                "ask_size":  float(raw.ask_size)  if raw.ask_size  else None,
                "timestamp": raw.timestamp.isoformat() if hasattr(raw, "timestamp") and raw.timestamp else None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("market_data.latest_quote_failed %s: %s", symbol, exc)
            return None

    async def get_snapshot(self, symbol: str) -> Optional[dict]:
        """
        Return a full snapshot: latest quote + trade + minute bar + daily bar.
        """
        await self._rate_limiter.acquire("data")
        _record("snapshot")
        is_crypto = "/" in symbol
        try:
            if is_crypto:
                snap = await self._crypto_snapshot(symbol)
            else:
                snap = await self._stock_snapshot(symbol)
            return self._snapshot_to_dict(snap, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("market_data.snapshot_failed %s: %s", symbol, exc)
            return None

    async def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """
        Best-effort last traded price (used by the execution engine).

        Tries latest trade → latest quote midpoint → latest bar close.
        """
        _record("last_price")
        snap = await self.get_snapshot(symbol)
        if snap is None:
            return None

        # 1. Latest trade price
        trade = snap.get("latest_trade")
        if trade and trade.get("price"):
            return Decimal(str(trade["price"]))

        # 2. Midpoint of latest quote
        quote = snap.get("latest_quote")
        if quote and quote.get("bid_price") and quote.get("ask_price"):
            mid = (quote["bid_price"] + quote["ask_price"]) / 2
            return Decimal(str(mid))

        # 3. Latest minute bar close
        mbar = snap.get("minute_bar")
        if mbar and mbar.get("close"):
            return Decimal(str(mbar["close"]))

        return None

    # -----------------------------------------------------------------------
    # Private — stock
    # -----------------------------------------------------------------------

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _get_stock_bars(
        self, symbol: str, tf: Any, start: Any, end: Any, limit: int,
    ) -> list[dict]:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
            feed=self._feed,
        )
        barset = await asyncio.to_thread(self._stock_client.get_stock_bars, req)
        bars = barset.get(symbol, []) if hasattr(barset, "get") else barset.data.get(symbol, [])
        return [_bar_to_dict(b, symbol) for b in bars]

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _stock_latest_bar(self, symbol: str) -> Any:
        result = await asyncio.to_thread(
            self._stock_client.get_stock_latest_bar, symbol,
        )
        # result can be a dict {symbol: Bar} or a Bar directly
        if isinstance(result, dict):
            return result[symbol]
        return result

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _stock_latest_quote(self, symbol: str) -> Any:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
        result = await asyncio.to_thread(
            self._stock_client.get_stock_latest_quote, req,
        )
        if isinstance(result, dict):
            return result[symbol]
        return result

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _stock_snapshot(self, symbol: str) -> Any:
        req = StockSnapshotRequest(symbol_or_symbols=symbol, feed=self._feed)
        result = await asyncio.to_thread(
            self._stock_client.get_stock_snapshot, req,
        )
        if isinstance(result, dict):
            return result[symbol]
        return result

    # -----------------------------------------------------------------------
    # Private — crypto
    # -----------------------------------------------------------------------

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _get_crypto_bars(
        self, symbol: str, tf: Any, start: Any, end: Any, limit: int,
    ) -> list[dict]:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )
        barset = await asyncio.to_thread(self._crypto_client.get_crypto_bars, req)
        bars = barset.get(symbol, []) if hasattr(barset, "get") else barset.data.get(symbol, [])
        return [_bar_to_dict(b, symbol) for b in bars]

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _crypto_latest_bar(self, symbol: str) -> Any:
        result = await asyncio.to_thread(
            self._crypto_client.get_crypto_latest_bar, symbol,
        )
        if isinstance(result, dict):
            return result[symbol]
        return result

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _crypto_latest_quote(self, symbol: str) -> Any:
        req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        result = await asyncio.to_thread(
            self._crypto_client.get_crypto_latest_quote, req,
        )
        if isinstance(result, dict):
            return result[symbol]
        return result

    @retry_with_jitter(max_attempts=3, base_delay=0.5)
    async def _crypto_snapshot(self, symbol: str) -> Any:
        req = CryptoSnapshotRequest(symbol_or_symbols=symbol)
        result = await asyncio.to_thread(
            self._crypto_client.get_crypto_snapshot, req,
        )
        if isinstance(result, dict):
            return result[symbol]
        return result

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _snapshot_to_dict(snap: Any, symbol: str) -> dict:
        """Normalise an alpaca-py Snapshot → plain dict."""
        result: dict[str, Any] = {"symbol": symbol}

        # Latest trade
        if hasattr(snap, "latest_trade") and snap.latest_trade:
            t = snap.latest_trade
            result["latest_trade"] = {
                "price":     float(t.price),
                "size":      float(t.size) if hasattr(t, "size") else None,
                "timestamp": t.timestamp.isoformat() if hasattr(t, "timestamp") and t.timestamp else None,
            }

        # Latest quote
        if hasattr(snap, "latest_quote") and snap.latest_quote:
            q = snap.latest_quote
            result["latest_quote"] = {
                "bid_price": float(q.bid_price) if q.bid_price else None,
                "ask_price": float(q.ask_price) if q.ask_price else None,
                "bid_size":  float(q.bid_size)  if q.bid_size  else None,
                "ask_size":  float(q.ask_size)  if q.ask_size  else None,
                "timestamp": q.timestamp.isoformat() if hasattr(q, "timestamp") and q.timestamp else None,
            }

        # Minute bar
        if hasattr(snap, "minute_bar") and snap.minute_bar:
            result["minute_bar"] = _bar_to_dict(snap.minute_bar, symbol)

        # Daily bar
        if hasattr(snap, "daily_bar") and snap.daily_bar:
            result["daily_bar"] = _bar_to_dict(snap.daily_bar, symbol)

        # Previous daily bar (stocks only)
        if hasattr(snap, "previous_daily_bar") and snap.previous_daily_bar:
            result["previous_daily_bar"] = _bar_to_dict(snap.previous_daily_bar, symbol)

        return result
