"""
AlpacaBarsIngestor — historical OHLCV bar ingestion via alpaca-py.
==================================================================

Downloads OHLCV bars for any list of Alpaca-supported symbols (US equities,
ETFs, and crypto) and persists them to partitioned Parquet files for
offline research and walk-forward back-testing.

Storage layout
--------------
::

    {output_dir}/
        bars/
            {timeframe}/
                {safe_symbol}.parquet     # all historical bars for symbol
        manifest.json                     # per-symbol watermarks + stats

Where ``safe_symbol`` replaces ``/`` with ``_`` (e.g. ``BTC_USD``).

Design decisions
----------------
* **Standalone** — no imports from ``platform/``.  Uses ``alpaca-py`` directly.
* **Sync** — research scripts / notebooks run outside asyncio; sync SDK calls
  are natural here.  Use ``AlpacaBarsIngestor`` as a context manager or call
  ``connect()`` / ``close()`` explicitly.
* **Incremental** — if a Parquet file already exists, the ingestor uses its
  last timestamp as ``start`` so only new bars are fetched and merged.
* **Rate-limit aware** — a configurable inter-request sleep keeps traffic
  below Alpaca's 200 req/min free-tier limit.
* **Error resilient** — a failure for one symbol is logged and counted; the
  loop continues for the remaining symbols.

Semana 1 deliverable (roadmap §5, `docs/architecture/alpaca_integration.md`)
----------------------------------------------------------------------------
* Universe top-200 MVP + ingest 4h bars.
* Parquet + manifest as the verifiable artifact.
* ``pytest tests/test_alpaca_bars.py`` must pass 100 % without network.

References
----------
* alpaca-py data docs: https://alpaca.markets/sdks/python/market_data.html
* ADR-010: UTC + Decimal in all platform schemas (bars use float for perf)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional alpaca-py import
# ---------------------------------------------------------------------------

try:
    from alpaca.data.historical import (             # type: ignore
        CryptoHistoricalDataClient,
        StockHistoricalDataClient,
    )
    from alpaca.data.requests import (               # type: ignore
        CryptoBarsRequest,
        StockBarsRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
    _HAS_ALPACA = True
except ImportError:  # pragma: no cover
    _HAS_ALPACA = False
    StockHistoricalDataClient  = None   # type: ignore[assignment,misc]
    CryptoHistoricalDataClient = None   # type: ignore[assignment,misc]
    StockBarsRequest           = None   # type: ignore[assignment,misc]
    CryptoBarsRequest          = None   # type: ignore[assignment,misc]
    TimeFrame                  = None   # type: ignore[assignment,misc]
    TimeFrameUnit              = None   # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------

# Maps human string → (amount, unit_name)
# Unit name is resolved to the real TimeFrameUnit once SDK is present.
_TF_SPECS: dict[str, tuple[int, str]] = {
    "1min":  (1,  "Minute"),
    "5min":  (5,  "Minute"),
    "15min": (15, "Minute"),
    "30min": (30, "Minute"),
    "1h":    (1,  "Hour"),
    "4h":    (4,  "Hour"),
    "1d":    (1,  "Day"),
    "1w":    (1,  "Week"),
    "1m":    (1,  "Month"),
}


def _resolve_timeframe(tf: str) -> Any:
    """Return an alpaca-py ``TimeFrame`` for *tf* (e.g. ``"4h"``)."""
    if not _HAS_ALPACA:
        raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")
    key = tf.lower().strip()
    if key not in _TF_SPECS:
        raise ValueError(
            f"Unknown timeframe '{tf}'. Supported: {sorted(_TF_SPECS)}"
        )
    amount, unit_name = _TF_SPECS[key]
    unit = getattr(TimeFrameUnit, unit_name)
    return TimeFrame(amount, unit)


def _safe_symbol(symbol: str) -> str:
    """Sanitise *symbol* for use as a filename component (``/`` → ``_``)."""
    return symbol.replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class SymbolReport:
    """Per-symbol result produced by :meth:`AlpacaBarsIngestor.ingest`."""
    symbol:    str
    success:   bool
    bar_count: int         = 0
    last_ts:   Optional[str] = None   # ISO-8601 UTC
    error:     Optional[str] = None


@dataclass
class IngestReport:
    """Aggregate result of an :meth:`AlpacaBarsIngestor.ingest` run."""
    started_at:    str
    completed_at:  str
    timeframe:     str
    total_symbols: int
    succeeded:     int
    failed:        int
    reports:       list[SymbolReport] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_symbols == 0:
            return 0.0
        return self.succeeded / self.total_symbols


# ---------------------------------------------------------------------------
# Bar → dict helper
# ---------------------------------------------------------------------------

def _bar_to_dict(bar: Any, symbol: str) -> dict:
    """Normalise an alpaca-py Bar object → plain dict (UTC timestamps)."""
    ts = bar.timestamp
    if hasattr(ts, "isoformat"):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    return {
        "symbol":      symbol,
        "timestamp":   ts_str,
        "open":        float(bar.open),
        "high":        float(bar.high),
        "low":         float(bar.low),
        "close":       float(bar.close),
        "volume":      float(bar.volume),
        "vwap":        float(bar.vwap)       if getattr(bar, "vwap", None)       is not None else None,
        "trade_count": int(bar.trade_count)  if getattr(bar, "trade_count", None) is not None else None,
    }


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Convert a list of bar dicts to a time-indexed DataFrame (UTC)."""
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# AlpacaBarsIngestor
# ---------------------------------------------------------------------------

class AlpacaBarsIngestor:
    """
    Download and persist historical OHLCV bars from Alpaca.

    Parameters
    ----------
    api_key : str
        Alpaca API key.
    api_secret : str, optional
        Alpaca API secret.
    output_dir : str or Path
        Root directory for Parquet output.  Created if absent.
    feed : str
        Stock data feed — ``"iex"`` (free) or ``"sip"`` (paid).
    request_sleep : float
        Seconds to sleep between API calls (default 0.35 ≈ 170 req/min,
        safely under the 200 req/min free-tier limit).

    Examples
    --------
    ::

        with AlpacaBarsIngestor("MY_KEY", "MY_SECRET") as ingestor:
            report = ingestor.ingest(
                symbols=["AAPL", "MSFT", "BTC/USD"],
                timeframe="4h",
                start=datetime(2022, 1, 1, tzinfo=timezone.utc),
            )
            print(f"Ingested {report.succeeded}/{report.total_symbols} symbols")

        # Load back
        df = ingestor.load("AAPL", timeframe="4h")
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        output_dir: str | Path = "data/alpaca_bars",
        feed: str = "iex",
        request_sleep: float = 0.35,
    ):
        self._api_key       = api_key
        self._api_secret    = api_secret or ""
        self._output_dir    = Path(output_dir)
        self._feed          = feed
        self._request_sleep = request_sleep

        self._stock_client:  Any = None
        self._crypto_client: Any = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def connect(self) -> None:
        """Create Alpaca SDK clients (lightweight — no network call)."""
        if not _HAS_ALPACA:
            raise RuntimeError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            )
        secret = self._api_secret if self._api_secret else None
        self._stock_client = StockHistoricalDataClient(
            api_key=self._api_key, secret_key=secret,
        )
        self._crypto_client = CryptoHistoricalDataClient(
            api_key=self._api_key, secret_key=secret,
        )
        logger.info(
            "alpaca_bars.connected feed=%s key=%s***",
            self._feed,
            self._api_key[:4] if self._api_key else "?",
        )

    def close(self) -> None:
        """Release SDK clients."""
        self._stock_client  = None
        self._crypto_client = None

    def __enter__(self) -> "AlpacaBarsIngestor":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Parquet storage helpers
    # -----------------------------------------------------------------------

    def _bars_dir(self, timeframe: str) -> Path:
        return self._output_dir / "bars" / timeframe

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        return self._bars_dir(timeframe) / f"{_safe_symbol(symbol)}.parquet"

    def _manifest_path(self) -> Path:
        return self._output_dir / self.MANIFEST_FILE

    def _save_df(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        """Write *df* to Parquet, merging with any existing file."""
        path = self._parquet_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df])
            # De-duplicate on timestamp index, keep last (newest data wins)
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = df

        combined.to_parquet(path, engine="pyarrow", compression="snappy")
        return path

    def _load_df(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load an existing Parquet file, or None if it does not exist."""
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alpaca_bars.load_failed %s: %s", symbol, exc)
            return None

    # -----------------------------------------------------------------------
    # Manifest helpers
    # -----------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -----------------------------------------------------------------------
    # Fetch helpers
    # -----------------------------------------------------------------------

    def _fetch_stock_bars(
        self,
        symbol: str,
        tf: Any,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: int,
    ) -> list[dict]:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
            feed=self._feed,
        )
        barset = self._stock_client.get_stock_bars(req)
        raw_bars = (
            barset.get(symbol, []) if hasattr(barset, "get")
            else getattr(barset, "data", {}).get(symbol, [])
        )
        return [_bar_to_dict(b, symbol) for b in raw_bars]

    def _fetch_crypto_bars(
        self,
        symbol: str,
        tf: Any,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: int,
    ) -> list[dict]:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )
        barset = self._crypto_client.get_crypto_bars(req)
        raw_bars = (
            barset.get(symbol, []) if hasattr(barset, "get")
            else getattr(barset, "data", {}).get(symbol, [])
        )
        return [_bar_to_dict(b, symbol) for b in raw_bars]

    def _fetch_bars(
        self,
        symbol: str,
        tf: Any,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: int,
    ) -> list[dict]:
        """Route to stock or crypto client based on symbol format."""
        is_crypto = "/" in symbol
        if is_crypto:
            return self._fetch_crypto_bars(symbol, tf, start, end, limit)
        return self._fetch_stock_bars(symbol, tf, start, end, limit)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def ingest(
        self,
        symbols: list[str],
        timeframe: str = "4h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 10_000,
        incremental: bool = True,
    ) -> IngestReport:
        """
        Download bars for *symbols* and persist to Parquet.

        Parameters
        ----------
        symbols : list[str]
            Alpaca-format symbols: ``"AAPL"`` for equities, ``"BTC/USD"`` for
            crypto.
        timeframe : str
            Bar timeframe — see ``_TF_SPECS`` for supported values.
        start : datetime, optional
            UTC start of the date range.  If *incremental* is ``True`` and a
            Parquet file already exists, the watermark from the manifest takes
            precedence over this parameter (only newer bars are fetched).
        end : datetime, optional
            UTC end of the date range.  Defaults to "now".
        limit : int
            Maximum bars per request (default 10,000 ≈ 10 years of 4h equity
            bars, or ~5 years of 4h crypto bars).
        incremental : bool
            When ``True``, use the last-bar timestamp from the manifest as
            ``start`` for symbols that already have data (avoid re-downloading
            history).

        Returns
        -------
        IngestReport
            Summary with per-symbol :class:`SymbolReport` entries.

        Notes
        -----
        Failed symbols are logged and counted in ``IngestReport.failed``;
        they do **not** abort the loop.
        """
        if self._stock_client is None:
            raise RuntimeError(
                "AlpacaBarsIngestor.connect() must be called first "
                "(or use the context manager)."
            )

        tf = _resolve_timeframe(timeframe)
        manifest = self._load_manifest()

        started_at = datetime.now(tz=timezone.utc).isoformat()
        reports: list[SymbolReport] = []
        succeeded = 0
        failed    = 0

        for sym in symbols:
            # Determine start watermark for incremental mode
            effective_start = start
            if incremental and sym in manifest:
                last_ts_str = manifest[sym].get("last_ts")
                if last_ts_str:
                    try:
                        last_ts = datetime.fromisoformat(last_ts_str)
                        # Add a tiny buffer to avoid re-fetching the last bar
                        if effective_start is None or last_ts > effective_start:
                            effective_start = last_ts
                    except ValueError:
                        pass

            try:
                logger.info(
                    "alpaca_bars.fetch symbol=%s tf=%s start=%s limit=%d",
                    sym, timeframe, effective_start, limit,
                )
                bars = self._fetch_bars(sym, tf, effective_start, end, limit)

                if not bars:
                    logger.warning("alpaca_bars.empty_response symbol=%s", sym)
                    # Still a success — no new bars available
                    reports.append(SymbolReport(symbol=sym, success=True, bar_count=0))
                    succeeded += 1
                else:
                    df = _bars_to_df(bars)
                    self._save_df(df, sym, timeframe)

                    last_ts = df.index[-1].isoformat()
                    manifest[sym] = {
                        "timeframe":        timeframe,
                        "last_ts":          last_ts,
                        "bar_count":        len(df),
                        "updated_at":       datetime.now(tz=timezone.utc).isoformat(),
                    }
                    self._save_manifest(manifest)

                    reports.append(
                        SymbolReport(
                            symbol=sym, success=True,
                            bar_count=len(df), last_ts=last_ts,
                        )
                    )
                    succeeded += 1
                    logger.info(
                        "alpaca_bars.saved symbol=%s bars=%d last_ts=%s",
                        sym, len(df), last_ts,
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error("alpaca_bars.error symbol=%s: %s", sym, exc)
                reports.append(SymbolReport(symbol=sym, success=False, error=str(exc)))
                failed += 1

            # Rate-limit guard between requests
            if self._request_sleep > 0:
                time.sleep(self._request_sleep)

        completed_at = datetime.now(tz=timezone.utc).isoformat()
        return IngestReport(
            started_at    = started_at,
            completed_at  = completed_at,
            timeframe     = timeframe,
            total_symbols = len(symbols),
            succeeded     = succeeded,
            failed        = failed,
            reports       = reports,
        )

    def load(
        self,
        symbol: str,
        timeframe: str = "4h",
    ) -> Optional[pd.DataFrame]:
        """
        Load saved Parquet bars for *symbol*.

        Parameters
        ----------
        symbol : str
            Alpaca-format symbol.
        timeframe : str
            Bar timeframe matching the stored file.

        Returns
        -------
        pd.DataFrame or None
            Time-indexed DataFrame (UTC), or ``None`` if no data found.

        Notes
        -----
        This is a lightweight read path — it does **not** require
        :meth:`connect` to have been called.
        """
        return self._load_df(symbol, timeframe)

    def manifest(self) -> dict:
        """Return the current manifest dictionary (per-symbol metadata)."""
        return self._load_manifest()

    # -----------------------------------------------------------------------
    # Convenience class methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        output_dir: str | Path = "data/alpaca_bars",
        feed: str = "iex",
    ) -> "AlpacaBarsIngestor":
        """
        Build an ingestor from environment variables.

        Reads ``ALPACA_API_KEY`` and ``ALPACA_API_SECRET`` from the process
        environment (or a ``.env`` file if ``python-dotenv`` is installed).
        """
        import os
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_API_SECRET", "")
        if not api_key:
            raise RuntimeError(
                "ALPACA_API_KEY not set.  Export the variable or populate .env"
            )
        return cls(api_key=api_key, api_secret=api_secret,
                   output_dir=output_dir, feed=feed)
