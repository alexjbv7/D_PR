"""
Tests — AlpacaMarketData (OHLCV bars, quotes, snapshots, last_price).
======================================================================

Uses lightweight stubs for the alpaca-py data SDK so tests run without
network access or real credentials.

Verifies:
  * get_bars returns normalised dicts with required keys.
  * get_bars routes crypto (``/`` in symbol) vs stock correctly.
  * get_latest_bar, get_latest_quote, get_snapshot return structured dicts.
  * get_last_price extracts a Decimal from snapshot.
  * resolve_timeframe maps human-friendly strings to SDK objects.
  * Unknown timeframe raises ValueError.
  * Rate limiter "data" bucket is used for every call.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# We patch the SDK presence flag so the module loads even when alpaca-py
# is not installed in the test environment.
import app.brokers._alpaca.market_data as md_mod


# ---------------------------------------------------------------------------
# Stub objects that mimic alpaca-py SDK responses
# ---------------------------------------------------------------------------

class _StubBar:
    def __init__(self, o=100, h=110, l=95, c=105, v=1000, vwap=103, tc=50):
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.vwap = vwap
        self.trade_count = tc
        self.timestamp = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)


class _StubQuote:
    def __init__(self, bid=104.5, ask=105.5, bs=100, asize=200):
        self.bid_price = bid
        self.ask_price = ask
        self.bid_size = bs
        self.ask_size = asize
        self.timestamp = datetime(2026, 5, 18, 12, 0, 1, tzinfo=timezone.utc)


class _StubTrade:
    def __init__(self, price=105.0, size=10):
        self.price = price
        self.size = size
        self.timestamp = datetime(2026, 5, 18, 12, 0, 2, tzinfo=timezone.utc)


class _StubSnapshot:
    def __init__(self):
        self.latest_trade = _StubTrade()
        self.latest_quote = _StubQuote()
        self.minute_bar = _StubBar()
        self.daily_bar = _StubBar(o=100, h=115, l=90, c=108, v=50000)
        self.previous_daily_bar = _StubBar(o=95, h=105, l=90, c=100, v=45000)


# ---------------------------------------------------------------------------
# Fake SDK clients
# ---------------------------------------------------------------------------

class _FakeStockClient:
    def get_stock_bars(self, req: Any):
        sym = req.symbol_or_symbols
        return {sym: [_StubBar(), _StubBar(o=105, h=112, l=100, c=110, v=1200)]}

    def get_stock_latest_bar(self, symbol: str):
        return {symbol: _StubBar()}

    def get_stock_latest_quote(self, req: Any):
        sym = req.symbol_or_symbols
        return {sym: _StubQuote()}

    def get_stock_snapshot(self, req: Any):
        sym = req.symbol_or_symbols
        return {sym: _StubSnapshot()}


class _FakeCryptoClient:
    def get_crypto_bars(self, req: Any):
        sym = req.symbol_or_symbols
        return {sym: [_StubBar(o=67000, h=68000, l=66500, c=67800, v=300)]}

    def get_crypto_latest_bar(self, symbol: str):
        return {symbol: _StubBar(o=67000, h=67500, l=66800, c=67200, v=50)}

    def get_crypto_latest_quote(self, req: Any):
        sym = req.symbol_or_symbols
        return {sym: _StubQuote(bid=67100, ask=67200, bs=1, asize=2)}

    def get_crypto_snapshot(self, req: Any):
        sym = req.symbol_or_symbols
        snap = _StubSnapshot()
        snap.latest_trade = _StubTrade(price=67150, size=0.5)
        snap.latest_quote = _StubQuote(bid=67100, ask=67200)
        snap.previous_daily_bar = None  # crypto has no previous daily
        return {sym: snap}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _patch_sdk(monkeypatch):
    """Ensure the module thinks alpaca-py data SDK is present."""
    monkeypatch.setattr(md_mod, "_HAS_DATA_SDK", True)

    # Stub the request classes (the real ones just store kwargs)
    for name in (
        "StockBarsRequest", "CryptoBarsRequest",
        "StockLatestQuoteRequest", "CryptoLatestQuoteRequest",
        "StockSnapshotRequest", "CryptoSnapshotRequest",
    ):
        monkeypatch.setattr(md_mod, name, type(name, (), {"__init__": lambda self, **kw: self.__dict__.update(kw)}), raising=False)

    # Stub TimeFrame + TimeFrameUnit
    class _TFU:
        Minute = "Minute"
        Hour = "Hour"
        Day = "Day"
        Week = "Week"
        Month = "Month"

    class _TF:
        def __init__(self, amount, unit):
            self.amount = amount
            self.unit = unit

    monkeypatch.setattr(md_mod, "TimeFrame", _TF, raising=False)
    monkeypatch.setattr(md_mod, "TimeFrameUnit", _TFU, raising=False)

    # Reset the TF map so it gets rebuilt with our stubs
    md_mod._TF_MAP.clear()


@pytest.fixture
def market_data(_patch_sdk):
    """Create an AlpacaMarketData instance with fake clients."""
    md = md_mod.AlpacaMarketData.__new__(md_mod.AlpacaMarketData)
    md._api_key = "test-key"
    md._api_secret = "test-secret"
    md._feed = "iex"
    md._rate_limiter = md_mod.AlpacaRateLimiter(trading_rpm=200, data_rpm=200)
    md._stock_client = _FakeStockClient()
    md._crypto_client = _FakeCryptoClient()
    return md


# ---------------------------------------------------------------------------
# resolve_timeframe
# ---------------------------------------------------------------------------

def test_resolve_timeframe_known(_patch_sdk):
    tf = md_mod.resolve_timeframe("1h")
    assert tf.amount == 1
    assert tf.unit == "Hour"


def test_resolve_timeframe_case_insensitive(_patch_sdk):
    tf = md_mod.resolve_timeframe("4H")
    assert tf.amount == 4


def test_resolve_timeframe_unknown(_patch_sdk):
    with pytest.raises(ValueError, match="Unknown timeframe"):
        md_mod.resolve_timeframe("3h")


# ---------------------------------------------------------------------------
# get_bars — stock
# ---------------------------------------------------------------------------

async def test_get_bars_stock(market_data):
    bars = await market_data.get_bars("AAPL", timeframe="1h", limit=100)
    assert len(bars) == 2
    bar = bars[0]
    for key in ("symbol", "timestamp", "open", "high", "low", "close", "volume"):
        assert key in bar, f"missing key: {key}"
    assert bar["symbol"] == "AAPL"
    assert bar["open"] == 100.0
    assert bar["close"] == 105.0


async def test_get_bars_stock_has_vwap(market_data):
    bars = await market_data.get_bars("MSFT", timeframe="1d")
    assert bars[0]["vwap"] == 103.0


# ---------------------------------------------------------------------------
# get_bars — crypto
# ---------------------------------------------------------------------------

async def test_get_bars_crypto(market_data):
    bars = await market_data.get_bars("BTC/USD", timeframe="1h")
    assert len(bars) == 1
    assert bars[0]["symbol"] == "BTC/USD"
    assert bars[0]["open"] == 67000.0


async def test_crypto_detected_by_slash(market_data):
    """Slash in symbol routes to crypto client."""
    bars = await market_data.get_bars("ETH/USD", timeframe="1d")
    # The fake crypto client returns 1 bar; stock would return 2.
    assert len(bars) == 1


# ---------------------------------------------------------------------------
# get_latest_bar
# ---------------------------------------------------------------------------

async def test_latest_bar_stock(market_data):
    bar = await market_data.get_latest_bar("AAPL")
    assert bar is not None
    assert bar["symbol"] == "AAPL"
    assert bar["close"] == 105.0


async def test_latest_bar_crypto(market_data):
    bar = await market_data.get_latest_bar("BTC/USD")
    assert bar is not None
    assert bar["close"] == 67200.0


# ---------------------------------------------------------------------------
# get_latest_quote
# ---------------------------------------------------------------------------

async def test_latest_quote_stock(market_data):
    q = await market_data.get_latest_quote("AAPL")
    assert q is not None
    assert q["bid_price"] == 104.5
    assert q["ask_price"] == 105.5


async def test_latest_quote_crypto(market_data):
    q = await market_data.get_latest_quote("BTC/USD")
    assert q is not None
    assert q["bid_price"] == 67100.0


# ---------------------------------------------------------------------------
# get_snapshot
# ---------------------------------------------------------------------------

async def test_snapshot_stock(market_data):
    snap = await market_data.get_snapshot("AAPL")
    assert snap is not None
    assert snap["symbol"] == "AAPL"
    assert "latest_trade" in snap
    assert "latest_quote" in snap
    assert "minute_bar" in snap
    assert "daily_bar" in snap
    assert snap["latest_trade"]["price"] == 105.0


async def test_snapshot_crypto(market_data):
    snap = await market_data.get_snapshot("BTC/USD")
    assert snap is not None
    assert snap["latest_trade"]["price"] == 67150.0
    # Crypto stub has no previous_daily_bar
    assert "previous_daily_bar" not in snap


# ---------------------------------------------------------------------------
# get_last_price
# ---------------------------------------------------------------------------

async def test_last_price_from_trade(market_data):
    price = await market_data.get_last_price("AAPL")
    assert price is not None
    assert isinstance(price, Decimal)
    assert price == Decimal("105.0")


async def test_last_price_crypto(market_data):
    price = await market_data.get_last_price("BTC/USD")
    assert price is not None
    assert price == Decimal("67150")


async def test_last_price_falls_back_to_quote(market_data):
    """When latest_trade is missing, fall back to quote midpoint."""
    # Make the snapshot return no trade
    class _NoTradeSnapshot:
        latest_trade = None
        latest_quote = _StubQuote(bid=100, ask=102)
        minute_bar = _StubBar()
        daily_bar = _StubBar()

    class _ClientNoTrade:
        def get_stock_snapshot(self, req):
            return {req.symbol_or_symbols: _NoTradeSnapshot()}

    market_data._stock_client = _ClientNoTrade()
    price = await market_data.get_last_price("AAPL")
    assert price is not None
    # Midpoint of 100 and 102 = 101
    assert price == Decimal("101.0")


async def test_last_price_returns_none_on_failure(market_data):
    """If snapshot completely fails, return None."""
    class _BrokenClient:
        def get_stock_snapshot(self, req):
            raise RuntimeError("network error")

    market_data._stock_client = _BrokenClient()
    price = await market_data.get_last_price("AAPL")
    assert price is None


# ---------------------------------------------------------------------------
# Rate limiter integration
# ---------------------------------------------------------------------------

async def test_data_bucket_is_consumed(market_data):
    """Every market data call should consume from the 'data' bucket."""
    initial = market_data._rate_limiter.available("data")
    await market_data.get_bars("AAPL", timeframe="1h")
    after = market_data._rate_limiter.available("data")
    assert after < initial


# ---------------------------------------------------------------------------
# AlpacaAdapter.get_last_price delegation
# ---------------------------------------------------------------------------

async def test_adapter_get_last_price_delegates(market_data, _patch_sdk, monkeypatch):
    """AlpacaAdapter.get_last_price delegates to AlpacaMarketData."""
    from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig
    import app.brokers.alpaca as alpaca_mod

    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", True)

    class _Side:
        BUY = SimpleNamespace(value="buy")
        SELL = SimpleNamespace(value="sell")

    class _TIF:
        GTC = SimpleNamespace(value="gtc")
        IOC = SimpleNamespace(value="ioc")
        FOK = SimpleNamespace(value="fok")
        DAY = SimpleNamespace(value="day")

    monkeypatch.setattr(alpaca_mod, "_AC_Side", _Side, raising=False)
    monkeypatch.setattr(alpaca_mod, "_AC_TIF", _TIF, raising=False)
    alpaca_mod._SIDE_TO_ALPACA.clear()
    alpaca_mod._TIF_TO_ALPACA.clear()

    class _DummyClient:
        def get_all_positions(self):
            return []
        def get_account(self):
            return SimpleNamespace(
                id="acc-1", equity="10000", cash="10000",
                last_equity="10000", initial_margin="0", currency="USD",
            )

    cfg = AlpacaConfig(api_key="k", api_secret="s", paper=True)
    adapter = AlpacaAdapter(
        config=cfg,
        client_factory=lambda: _DummyClient(),
        market_data=market_data,
    )
    await adapter.connect()

    price = await adapter.get_last_price("AAPL")
    assert price is not None
    assert isinstance(price, Decimal)

    await adapter.close()


async def test_adapter_get_last_price_none_without_market_data(_patch_sdk, monkeypatch):
    """Without market_data, get_last_price returns None (base default)."""
    from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig
    import app.brokers.alpaca as alpaca_mod

    monkeypatch.setattr(alpaca_mod, "_HAS_ALPACA", True)

    class _DummyClient:
        def get_all_positions(self):
            return []
        def get_account(self):
            return SimpleNamespace(
                id="acc-1", equity="10000", cash="10000",
                last_equity="10000", initial_margin="0", currency="USD",
            )

    cfg = AlpacaConfig(api_key="k", api_secret="s", paper=True)
    adapter = AlpacaAdapter(config=cfg, client_factory=lambda: _DummyClient())
    await adapter.connect()

    price = await adapter.get_last_price("AAPL")
    assert price is None

    await adapter.close()
