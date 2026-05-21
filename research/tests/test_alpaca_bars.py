"""
Tests — AlpacaBarsIngestor.
===========================================================

All tests run **without network access** — alpaca-py SDK calls are
monkeypatched with lightweight stubs.

Covers
------
* `_bar_to_dict` normalisation.
* `_bars_to_df` — correct UTC index.
* `_resolve_timeframe` — known / unknown strings.
* `AlpacaBarsIngestor.ingest`:
  - Stock bars persisted to Parquet.
  - Crypto bars routed to crypto client.
  - Manifest updated after successful ingest.
  - Failed symbol skipped; loop continues.
  - Incremental ingest uses last watermark as `start`.
  - Empty response counts as success.
* `AlpacaBarsIngestor.load` — reads saved Parquet.
* `IngestReport.success_rate` calculation.
"""
from __future__ import annotations

import json
import time as _time_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import research.data.alpaca_bars as bars_mod
from research.data.alpaca_bars import (
    AlpacaBarsIngestor,
    IngestReport,
    SymbolReport,
    _bar_to_dict,
    _bars_to_df,
    _resolve_timeframe,
)


# ---------------------------------------------------------------------------
# Stub bar object
# ---------------------------------------------------------------------------

class _StubBar:
    def __init__(self, o=100.0, h=110.0, l=95.0, c=105.0, v=1_000.0,
                 vwap=103.0, tc=50,
                 ts: datetime | None = None):
        self.open        = o
        self.high        = h
        self.low         = l
        self.close       = c
        self.volume      = v
        self.vwap        = vwap
        self.trade_count = tc
        self.timestamp   = ts or datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture: patch alpaca-py presence + stub SDK
# ---------------------------------------------------------------------------

@pytest.fixture
def _patch_sdk(monkeypatch, tmp_path):
    """
    Ensure bars_mod thinks alpaca-py is present and inject fake clients.
    Returns (ingestor, stock_client, crypto_client).
    """
    monkeypatch.setattr(bars_mod, "_HAS_ALPACA", True)

    # Stub TimeFrame / TimeFrameUnit
    class _TFU:
        Minute = "Minute"
        Hour   = "Hour"
        Day    = "Day"
        Week   = "Week"
        Month  = "Month"

    class _TF:
        def __init__(self, amount, unit):
            self.amount = amount
            self.unit   = unit

    monkeypatch.setattr(bars_mod, "TimeFrame",     _TF,  raising=False)
    monkeypatch.setattr(bars_mod, "TimeFrameUnit", _TFU, raising=False)

    # Stub StockBarsRequest / CryptoBarsRequest
    for name in ("StockBarsRequest", "CryptoBarsRequest"):
        monkeypatch.setattr(
            bars_mod, name,
            type(name, (), {"__init__": lambda self, **kw: self.__dict__.update(kw)}),
            raising=False,
        )

    # Fake clients
    stock_client  = MagicMock()
    crypto_client = MagicMock()
    monkeypatch.setattr(bars_mod, "StockHistoricalDataClient",
                        lambda **_: stock_client, raising=False)
    monkeypatch.setattr(bars_mod, "CryptoHistoricalDataClient",
                        lambda **_: crypto_client, raising=False)

    ingestor = AlpacaBarsIngestor(
        api_key="test-key", api_secret="test-secret",
        output_dir=tmp_path,
        request_sleep=0,    # disable sleep in tests
    )
    ingestor.connect()
    # Inject fake clients directly (connect replaced them with MagicMock lambdas;
    # re-inject after connect)
    ingestor._stock_client  = stock_client
    ingestor._crypto_client = crypto_client

    return ingestor, stock_client, crypto_client


# ---------------------------------------------------------------------------
# _bar_to_dict
# ---------------------------------------------------------------------------

def test_bar_to_dict_keys():
    b = _StubBar()
    d = _bar_to_dict(b, "AAPL")
    for key in ("symbol", "timestamp", "open", "high", "low", "close",
                "volume", "vwap", "trade_count"):
        assert key in d, f"Missing key: {key}"


def test_bar_to_dict_values():
    b = _StubBar(o=100, h=110, l=95, c=105, v=999, vwap=103, tc=7)
    d = _bar_to_dict(b, "AAPL")
    assert d["symbol"]      == "AAPL"
    assert d["open"]        == 100.0
    assert d["close"]       == 105.0
    assert d["vwap"]        == 103.0
    assert d["trade_count"] == 7


def test_bar_to_dict_no_vwap():
    b = _StubBar()
    del b.vwap             # remove attribute
    d = _bar_to_dict(b, "X")
    assert d["vwap"] is None


def test_bar_to_dict_utc_timestamp():
    ts = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    b  = _StubBar(ts=ts)
    d  = _bar_to_dict(b, "BTC/USD")
    assert "2026-05-18" in d["timestamp"]


# ---------------------------------------------------------------------------
# _bars_to_df
# ---------------------------------------------------------------------------

def test_bars_to_df_empty():
    df = _bars_to_df([])
    assert df.empty


def test_bars_to_df_index_is_utc():
    b = _StubBar(ts=datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc))
    dicts = [_bar_to_dict(b, "AAPL")]
    df = _bars_to_df(dicts)
    assert not df.empty
    assert df.index.tz is not None
    assert df.index.tzinfo is not None


def test_bars_to_df_sorted():
    ts1 = datetime(2026, 5, 18, 8,  0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    bars = [
        _bar_to_dict(_StubBar(ts=ts3), "X"),
        _bar_to_dict(_StubBar(ts=ts1), "X"),
        _bar_to_dict(_StubBar(ts=ts2), "X"),
    ]
    df = _bars_to_df(bars)
    assert list(df.index) == sorted(df.index)


# ---------------------------------------------------------------------------
# _resolve_timeframe
# ---------------------------------------------------------------------------

def test_resolve_timeframe_1h(_patch_sdk):
    tf = _resolve_timeframe("1h")
    assert tf.amount == 1
    assert tf.unit   == "Hour"


def test_resolve_timeframe_4h(_patch_sdk):
    tf = _resolve_timeframe("4h")
    assert tf.amount == 4


def test_resolve_timeframe_case_insensitive(_patch_sdk):
    tf = _resolve_timeframe("4H")
    assert tf.amount == 4


def test_resolve_timeframe_unknown(_patch_sdk):
    with pytest.raises(ValueError, match="Unknown timeframe"):
        _resolve_timeframe("3h")


# ---------------------------------------------------------------------------
# ingest — stock bars
# ---------------------------------------------------------------------------

def test_ingest_stock_creates_parquet(_patch_sdk, tmp_path):
    ingestor, stock_client, _ = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {
        "AAPL": [_StubBar(ts=ts), _StubBar(c=110, ts=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc))]
    }

    report = ingestor.ingest(["AAPL"], timeframe="4h")

    assert report.succeeded == 1
    assert report.failed    == 0
    parquet_path = tmp_path / "bars" / "4h" / "AAPL.parquet"
    assert parquet_path.exists(), "Parquet file not created"


def test_ingest_stock_bar_count(_patch_sdk):
    ingestor, stock_client, _ = _patch_sdk
    ts1 = datetime(2026, 5, 18, 8,  0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {
        "MSFT": [_StubBar(ts=ts1), _StubBar(ts=ts2)]
    }
    report = ingestor.ingest(["MSFT"], timeframe="4h")
    assert report.reports[0].bar_count == 2


def test_ingest_stock_last_ts_recorded(_patch_sdk):
    ingestor, stock_client, _ = _patch_sdk
    ts = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {"AAPL": [_StubBar(ts=ts)]}

    report = ingestor.ingest(["AAPL"], timeframe="1d")
    assert report.reports[0].last_ts is not None
    assert "2026-05-18" in report.reports[0].last_ts


# ---------------------------------------------------------------------------
# ingest — crypto bars
# ---------------------------------------------------------------------------

def test_ingest_crypto_routes_to_crypto_client(_patch_sdk):
    ingestor, stock_client, crypto_client = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    crypto_client.get_crypto_bars.return_value = {
        "BTC/USD": [_StubBar(o=67000, c=68000, ts=ts)]
    }

    report = ingestor.ingest(["BTC/USD"], timeframe="4h")

    assert report.succeeded == 1
    crypto_client.get_crypto_bars.assert_called_once()
    stock_client.get_stock_bars.assert_not_called()


def test_ingest_crypto_parquet_safe_name(_patch_sdk, tmp_path):
    ingestor, _, crypto_client = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    crypto_client.get_crypto_bars.return_value = {"BTC/USD": [_StubBar(ts=ts)]}

    ingestor.ingest(["BTC/USD"], timeframe="4h")

    parquet_path = tmp_path / "bars" / "4h" / "BTC_USD.parquet"
    assert parquet_path.exists(), "Crypto parquet should use _ separator"


# ---------------------------------------------------------------------------
# ingest — manifest
# ---------------------------------------------------------------------------

def test_ingest_manifest_written(_patch_sdk, tmp_path):
    ingestor, stock_client, _ = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {"AAPL": [_StubBar(ts=ts)]}

    ingestor.ingest(["AAPL"], timeframe="4h")

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "AAPL" in manifest
    assert manifest["AAPL"]["timeframe"] == "4h"
    assert "last_ts" in manifest["AAPL"]


def test_ingest_multiple_symbols_manifest(_patch_sdk, tmp_path):
    ingestor, stock_client, _ = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.side_effect = [
        {"AAPL": [_StubBar(ts=ts)]},
        {"MSFT": [_StubBar(c=400, ts=ts)]},
    ]

    ingestor.ingest(["AAPL", "MSFT"], timeframe="4h")

    manifest = ingestor.manifest()
    assert "AAPL" in manifest
    assert "MSFT" in manifest


# ---------------------------------------------------------------------------
# ingest — error resilience
# ---------------------------------------------------------------------------

def test_failed_symbol_skipped(_patch_sdk):
    ingestor, stock_client, _ = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.side_effect = [
        RuntimeError("API error"),              # AAPL fails
        {"MSFT": [_StubBar(c=400, ts=ts)]},     # MSFT succeeds
    ]

    report = ingestor.ingest(["AAPL", "MSFT"], timeframe="4h")

    assert report.failed    == 1
    assert report.succeeded == 1

    aapl_rep = next(r for r in report.reports if r.symbol == "AAPL")
    msft_rep = next(r for r in report.reports if r.symbol == "MSFT")
    assert not aapl_rep.success
    assert aapl_rep.error is not None
    assert msft_rep.success


def test_empty_response_counts_as_success(_patch_sdk):
    ingestor, stock_client, _ = _patch_sdk
    stock_client.get_stock_bars.return_value = {"AAPL": []}

    report = ingestor.ingest(["AAPL"], timeframe="4h")

    assert report.succeeded == 1
    assert report.reports[0].bar_count == 0


# ---------------------------------------------------------------------------
# ingest — incremental mode
# ---------------------------------------------------------------------------

def test_incremental_uses_manifest_watermark(_patch_sdk, tmp_path):
    ingestor, stock_client, _ = _patch_sdk

    # First ingest: 2 bars
    ts1 = datetime(2026, 5, 15, 8, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {
        "AAPL": [_StubBar(ts=ts1), _StubBar(ts=ts2)]
    }
    ingestor.ingest(["AAPL"], timeframe="4h", incremental=False)

    # Second ingest: 1 new bar
    ts3 = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {"AAPL": [_StubBar(ts=ts3)]}
    ingestor.ingest(["AAPL"], timeframe="4h", incremental=True)

    # The second call should have received a `start` argument
    call_args = stock_client.get_stock_bars.call_args_list[-1]
    # The request object will have start != None
    req = call_args[0][0]
    assert req.start is not None, "incremental should pass watermark as start"


def test_incremental_merges_bars(_patch_sdk, tmp_path):
    """After two ingests, the Parquet file contains bars from both."""
    ingestor, stock_client, _ = _patch_sdk

    ts1 = datetime(2026, 5, 15, 8, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 16, 8, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 5, 17, 8, 0, tzinfo=timezone.utc)

    stock_client.get_stock_bars.side_effect = [
        {"AAPL": [_StubBar(ts=ts1), _StubBar(ts=ts2)]},
        {"AAPL": [_StubBar(ts=ts3)]},
    ]

    ingestor.ingest(["AAPL"], timeframe="4h", incremental=False)
    ingestor.ingest(["AAPL"], timeframe="4h", incremental=True)

    df = ingestor.load("AAPL", timeframe="4h")
    assert df is not None
    assert len(df) == 3


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def test_load_returns_none_if_no_file(_patch_sdk):
    ingestor, _, _ = _patch_sdk
    result = ingestor.load("UNKNOWN", timeframe="4h")
    assert result is None


def test_load_returns_dataframe(_patch_sdk, tmp_path):
    ingestor, stock_client, _ = _patch_sdk
    ts = datetime(2026, 5, 18, 8, 0, tzinfo=timezone.utc)
    stock_client.get_stock_bars.return_value = {"NVDA": [_StubBar(ts=ts)]}

    ingestor.ingest(["NVDA"], timeframe="1d")
    df = ingestor.load("NVDA", timeframe="1d")

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# IngestReport helpers
# ---------------------------------------------------------------------------

def test_success_rate_all_ok():
    r = IngestReport(
        started_at="a", completed_at="b", timeframe="4h",
        total_symbols=3, succeeded=3, failed=0,
    )
    assert r.success_rate == 1.0


def test_success_rate_partial():
    r = IngestReport(
        started_at="a", completed_at="b", timeframe="4h",
        total_symbols=4, succeeded=3, failed=1,
    )
    assert r.success_rate == 0.75


def test_success_rate_zero_symbols():
    r = IngestReport(
        started_at="a", completed_at="b", timeframe="4h",
        total_symbols=0, succeeded=0, failed=0,
    )
    assert r.success_rate == 0.0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_connects_and_closes(_patch_sdk, monkeypatch, tmp_path):
    """Using ``with`` statement should call connect/close."""
    monkeypatch.setattr(bars_mod, "_HAS_ALPACA", True)

    connected = []
    closed    = []

    ingestor = AlpacaBarsIngestor(
        api_key="k", output_dir=tmp_path, request_sleep=0
    )
    original_connect = ingestor.connect
    original_close   = ingestor.close

    def fake_connect():
        connected.append(True)
        original_connect()

    def fake_close():
        closed.append(True)
        original_close()

    ingestor.connect = fake_connect  # type: ignore[method-assign]
    ingestor.close   = fake_close    # type: ignore[method-assign]

    with ingestor:
        pass

    assert connected
    assert closed
