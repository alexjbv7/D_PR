"""
Tests — AlpacaEquityUniverse & EQUITY_UNIVERSE_BASE.
=====================================================

All tests run without network access.  Live-price filtering is tested by
monkeypatching alpaca-py SDK calls.

Covers
------
* `EQUITY_UNIVERSE_BASE` integrity (≥ 100 symbols, unique, required keys).
* `CRYPTO_UNIVERSE` / `ETF_UNIVERSE` structure.
* `AlpacaEquityUniverse.get_base` respects include_etfs / include_crypto flags.
* `AlpacaEquityUniverse.symbols` deduplicates.
* `AlpacaEquityUniverse.by_sector` groups correctly.
* `AlpacaEquityUniverse.filter_by_tier` — mega + large.
* `AlpacaEquityUniverse.filter_by_price` — removes below threshold.
* `AlpacaEquityUniverse.build` — max_symbols cap, sector stats.
* `build_top_n` convenience function.
* `UniverseManifest` counts.
"""
from __future__ import annotations

from datetime import timezone
from unittest.mock import MagicMock, patch

import pytest

from research.universe.alpaca_equity_universe import (
    CRYPTO_UNIVERSE,
    EQUITY_UNIVERSE_BASE,
    ETF_UNIVERSE,
    AlpacaEquityUniverse,
    UniverseManifest,
    build_top_n,
)


# ---------------------------------------------------------------------------
# EQUITY_UNIVERSE_BASE integrity
# ---------------------------------------------------------------------------

def test_base_universe_min_size():
    """Must have at least 100 equity symbols — MVP universe."""
    assert len(EQUITY_UNIVERSE_BASE) >= 100


def test_base_universe_no_duplicates():
    symbols = [r["symbol"] for r in EQUITY_UNIVERSE_BASE]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols in EQUITY_UNIVERSE_BASE"


def test_base_universe_required_keys():
    required = {"symbol", "sector", "market_cap_tier", "added"}
    for row in EQUITY_UNIVERSE_BASE:
        missing = required - row.keys()
        assert not missing, f"{row['symbol']} missing keys: {missing}"


def test_base_universe_market_cap_tiers():
    valid_tiers = {"mega", "large", "mid"}
    for row in EQUITY_UNIVERSE_BASE:
        assert row["market_cap_tier"] in valid_tiers, (
            f"{row['symbol']} has invalid tier: {row['market_cap_tier']}"
        )


def test_base_universe_added_date_format():
    """All 'added' dates must be YYYY-MM-DD format."""
    import re
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for row in EQUITY_UNIVERSE_BASE:
        assert pattern.match(row["added"]), (
            f"{row['symbol']}: bad 'added' date: {row['added']}"
        )


def test_base_universe_sector_coverage():
    """At least 5 distinct sectors must be represented."""
    sectors = {r["sector"] for r in EQUITY_UNIVERSE_BASE}
    assert len(sectors) >= 5, f"Only {len(sectors)} sectors: {sectors}"


def test_base_universe_has_mega_caps():
    mega = [r for r in EQUITY_UNIVERSE_BASE if r["market_cap_tier"] == "mega"]
    assert len(mega) >= 5, "Expected at least 5 mega-cap symbols"


def test_known_symbols_present():
    syms = {r["symbol"] for r in EQUITY_UNIVERSE_BASE}
    for expected in ("AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "XOM"):
        assert expected in syms, f"{expected} missing from base universe"


# ---------------------------------------------------------------------------
# CRYPTO_UNIVERSE / ETF_UNIVERSE
# ---------------------------------------------------------------------------

def test_crypto_universe_slash_format():
    for row in CRYPTO_UNIVERSE:
        assert "/" in row["symbol"], (
            f"Crypto symbol should be in Alpaca format (BTC/USD): {row['symbol']}"
        )


def test_crypto_universe_has_btc_eth():
    syms = {r["symbol"] for r in CRYPTO_UNIVERSE}
    assert "BTC/USD" in syms
    assert "ETH/USD" in syms


def test_etf_universe_no_slash():
    for row in ETF_UNIVERSE:
        assert "/" not in row["symbol"]


def test_etf_universe_has_spy_qqq():
    syms = {r["symbol"] for r in ETF_UNIVERSE}
    assert "SPY" in syms
    assert "QQQ" in syms


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse.get_base
# ---------------------------------------------------------------------------

def test_get_base_default_no_etf_no_crypto():
    u = AlpacaEquityUniverse()
    rows = u.get_base()
    symbols = {r["symbol"] for r in rows}
    assert "SPY"     not in symbols, "ETF should not be in default base"
    assert "BTC/USD" not in symbols, "Crypto should not be in default base"


def test_get_base_with_etfs():
    u = AlpacaEquityUniverse(include_etfs=True)
    rows = u.get_base()
    symbols = {r["symbol"] for r in rows}
    assert "SPY" in symbols


def test_get_base_with_crypto():
    u = AlpacaEquityUniverse(include_crypto=True)
    rows = u.get_base()
    symbols = {r["symbol"] for r in rows}
    assert "BTC/USD" in symbols


def test_get_base_with_both():
    u = AlpacaEquityUniverse(include_etfs=True, include_crypto=True)
    rows = u.get_base()
    symbols = {r["symbol"] for r in rows}
    assert "SPY"     in symbols
    assert "BTC/USD" in symbols


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse.symbols
# ---------------------------------------------------------------------------

def test_symbols_deduplicates():
    u = AlpacaEquityUniverse(include_etfs=True, include_crypto=True)
    syms = u.symbols(include_etfs=True, include_crypto=True)
    assert len(syms) == len(set(syms)), "symbols() must deduplicate"


def test_symbols_returns_strings():
    u = AlpacaEquityUniverse()
    for sym in u.symbols():
        assert isinstance(sym, str)


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse.by_sector
# ---------------------------------------------------------------------------

def test_by_sector_returns_dict():
    u = AlpacaEquityUniverse()
    groups = u.by_sector()
    assert isinstance(groups, dict)
    assert len(groups) >= 5


def test_by_sector_aapl_in_tech():
    u = AlpacaEquityUniverse()
    groups = u.by_sector()
    assert "AAPL" in groups.get("Technology", [])


def test_by_sector_all_symbols_covered():
    u = AlpacaEquityUniverse()
    all_syms = {r["symbol"] for r in u.get_base()}
    in_groups: set[str] = set()
    for group_syms in u.by_sector().values():
        in_groups.update(group_syms)
    assert all_syms == in_groups


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse.filter_by_tier
# ---------------------------------------------------------------------------

def test_filter_by_tier_default_mega_large():
    u = AlpacaEquityUniverse()
    rows = u.get_base()
    filtered = u.filter_by_tier(rows)
    tiers = {r["market_cap_tier"] for r in filtered}
    assert "mid" not in tiers
    assert len(filtered) >= 80, "Expected many mega+large symbols"


def test_filter_by_tier_mega_only():
    u = AlpacaEquityUniverse()
    rows = u.get_base()
    filtered = u.filter_by_tier(rows, tiers=["mega"])
    assert all(r["market_cap_tier"] == "mega" for r in filtered)
    assert len(filtered) >= 5


def test_filter_by_tier_unknown_tier_returns_empty():
    u = AlpacaEquityUniverse()
    rows = u.get_base()
    filtered = u.filter_by_tier(rows, tiers=["nano"])
    assert filtered == []


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse.filter_by_price (mocked)
# ---------------------------------------------------------------------------

def test_filter_by_price_no_api_key_returns_all():
    u = AlpacaEquityUniverse(api_key=None)
    symbols = ["AAPL", "MSFT", "BTC/USD"]
    result  = u.filter_by_price(symbols, min_price=5.0)
    assert result == symbols, "Without api_key all symbols should pass through"


def test_filter_by_price_removes_below_threshold(monkeypatch):
    import research.universe.alpaca_equity_universe as uni_mod

    # Stub the StockLatestQuoteRequest import inside filter_by_price
    class _FakeQuote:
        def __init__(self, bid, ask):
            self.bid_price = bid
            self.ask_price = ask

    class _FakeClient:
        def get_stock_latest_quote(self, req):
            prices = {"CHEAP": _FakeQuote(0.5, 0.9), "PRICEY": _FakeQuote(100, 101)}
            return {req.symbol_or_symbols: prices.get(req.symbol_or_symbols)}

    def _fake_time():
        class _T:
            def sleep(self, _): pass
        return _T()

    monkeypatch.setattr(uni_mod, "time_module", _fake_time)

    with patch("alpaca.data.historical.StockHistoricalDataClient", return_value=_FakeClient()), \
         patch("alpaca.data.requests.StockLatestQuoteRequest",
               type("SLQ", (), {"__init__": lambda s, **kw: s.__dict__.update(kw)})):
        u = AlpacaEquityUniverse(api_key="test-key", min_price=5.0)
        result = u.filter_by_price(["CHEAP", "PRICEY"], min_price=5.0)

    # CHEAP mid = 0.7 < 5.0 should be removed
    assert "CHEAP"  not in result
    assert "PRICEY" in result


def test_filter_by_price_crypto_passes_through(monkeypatch):
    """Crypto symbols with '/' bypass the price filter."""
    import research.universe.alpaca_equity_universe as uni_mod

    def _fake_time():
        class _T:
            def sleep(self, _): pass
        return _T()

    monkeypatch.setattr(uni_mod, "time_module", _fake_time)

    with patch("alpaca.data.historical.StockHistoricalDataClient",
               return_value=MagicMock()):
        u = AlpacaEquityUniverse(api_key="test-key")
        result = u.filter_by_price(["BTC/USD", "ETH/USD"], min_price=100_000)

    assert "BTC/USD" in result
    assert "ETH/USD" in result


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse.build
# ---------------------------------------------------------------------------

def test_build_default():
    u = AlpacaEquityUniverse()
    manifest = u.build(max_symbols=200)
    assert isinstance(manifest, UniverseManifest)
    assert len(manifest.symbols) <= 200
    assert len(manifest.symbols) >= 50


def test_build_max_symbols_cap():
    u = AlpacaEquityUniverse()
    manifest = u.build(max_symbols=30)
    assert len(manifest.symbols) <= 30


def test_build_no_duplicates():
    u = AlpacaEquityUniverse(include_etfs=True, include_crypto=True)
    manifest = u.build(max_symbols=300)
    assert len(manifest.symbols) == len(set(manifest.symbols))


def test_build_sectors_populated():
    u = AlpacaEquityUniverse()
    manifest = u.build()
    assert len(manifest.sectors) >= 5


def test_build_n_equity_n_crypto_counts():
    u = AlpacaEquityUniverse(include_crypto=True, include_etfs=True)
    manifest = u.build(max_symbols=500)
    assert manifest.n_equity  >= 100
    assert manifest.n_crypto  >= 1
    assert manifest.n_etf     >= 1
    assert manifest.n_equity + manifest.n_etf + manifest.n_crypto == len(manifest.symbols)


def test_build_metadata_matches_symbols():
    u = AlpacaEquityUniverse()
    manifest = u.build(max_symbols=50)
    meta_syms = [r["symbol"] for r in manifest.metadata]
    assert meta_syms == manifest.symbols


def test_build_ts_is_utc_iso():
    u = AlpacaEquityUniverse()
    manifest = u.build()
    # Should parse without error
    from datetime import datetime
    dt = datetime.fromisoformat(manifest.built_at)
    assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# build_top_n
# ---------------------------------------------------------------------------

def test_build_top_n_default():
    syms = build_top_n(200)
    assert isinstance(syms, list)
    assert len(syms) <= 200
    assert len(syms) >= 50


def test_build_top_n_with_crypto():
    # Use n > len(EQUITY_UNIVERSE_BASE) so crypto symbols (appended after
    # equities) are not cut off by the max_symbols cap.
    syms = build_top_n(300, include_crypto=True)
    has_crypto = any("/" in s for s in syms)
    assert has_crypto, "Expected at least one crypto symbol"


def test_build_top_n_small():
    syms = build_top_n(10)
    assert len(syms) == 10


def test_build_top_n_no_duplicates():
    syms = build_top_n(200)
    assert len(syms) == len(set(syms))
