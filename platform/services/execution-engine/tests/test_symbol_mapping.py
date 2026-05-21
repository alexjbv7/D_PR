"""Tests for canonical ↔ Alpaca symbol translation."""
from __future__ import annotations

import pytest

from app.brokers._symbol_mapping import from_alpaca, is_crypto, to_alpaca


# ---------------------------------------------------------------------------
# is_crypto
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDC", "SOLUSD", "BTC/USD", "ETH/USDT"])
def test_is_crypto_true(sym):
    assert is_crypto(sym) is True


@pytest.mark.parametrize("sym", ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"])
def test_is_crypto_false_for_equities(sym):
    assert is_crypto(sym) is False


# ---------------------------------------------------------------------------
# to_alpaca
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical, alpaca", [
    ("BTCUSDT", "BTC/USD"),
    ("ETHUSDT", "ETH/USD"),
    ("ETHUSDC", "ETH/USD"),
    ("SOLUSDT", "SOL/USD"),
    ("DOGEUSDT", "DOGE/USD"),
])
def test_to_alpaca_crypto_pairs(canonical, alpaca):
    assert to_alpaca(canonical) == alpaca


@pytest.mark.parametrize("equity", ["AAPL", "MSFT", "SPY", "QQQ"])
def test_to_alpaca_equities_unchanged(equity):
    assert to_alpaca(equity) == equity


def test_to_alpaca_passthrough_when_already_slashed():
    assert to_alpaca("BTC/USD") == "BTC/USD"


# ---------------------------------------------------------------------------
# from_alpaca
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpaca, canonical", [
    ("BTC/USD",  "BTCUSDT"),
    ("ETH/USD",  "ETHUSDT"),
    ("SOL/USD",  "SOLUSDT"),
])
def test_from_alpaca_crypto_pairs(alpaca, canonical):
    assert from_alpaca(alpaca) == canonical


@pytest.mark.parametrize("equity", ["AAPL", "MSFT", "SPY"])
def test_from_alpaca_equities_unchanged(equity):
    assert from_alpaca(equity) == equity


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", ["BTCUSDT", "ETHUSDT", "AAPL", "SPY"])
def test_round_trip(canonical):
    assert from_alpaca(to_alpaca(canonical)) == canonical
