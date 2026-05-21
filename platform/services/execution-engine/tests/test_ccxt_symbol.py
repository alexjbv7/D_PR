"""Canonical ↔ CCXT symbol translation tests."""
from __future__ import annotations

import pytest

from app.brokers._ccxt_symbol import from_ccxt, to_ccxt


# ---------------------------------------------------------------------------
# to_ccxt — spot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical, ccxt_sym", [
    ("BTCUSDT", "BTC/USDT"),
    ("ETHUSDT", "ETH/USDT"),
    ("ETHUSDC", "ETH/USDC"),
    ("SOLUSDT", "SOL/USDT"),
    ("DOGEUSDT", "DOGE/USDT"),
])
def test_to_ccxt_spot(canonical, ccxt_sym):
    assert to_ccxt(canonical) == ccxt_sym


# ---------------------------------------------------------------------------
# to_ccxt — perp (.P suffix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical, ccxt_sym", [
    ("BTCUSDT.P", "BTC/USDT:USDT"),
    ("ETHUSDT.P", "ETH/USDT:USDT"),
])
def test_to_ccxt_perp(canonical, ccxt_sym):
    assert to_ccxt(canonical) == ccxt_sym


# ---------------------------------------------------------------------------
# to_ccxt — passthrough
# ---------------------------------------------------------------------------

def test_to_ccxt_passthrough_when_already_slashed():
    assert to_ccxt("BTC/USDT") == "BTC/USDT"


def test_to_ccxt_passthrough_when_already_perp():
    assert to_ccxt("BTC/USDT:USDT") == "BTC/USDT:USDT"


# ---------------------------------------------------------------------------
# from_ccxt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ccxt_sym, canonical", [
    ("BTC/USDT",       "BTCUSDT"),
    ("ETH/USDT",       "ETHUSDT"),
    ("ETH/USDC",       "ETHUSDC"),
    ("BTC/USDT:USDT",  "BTCUSDT.P"),
    ("ETH/USDT:USDT",  "ETHUSDT.P"),
])
def test_from_ccxt(ccxt_sym, canonical):
    assert from_ccxt(ccxt_sym) == canonical


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCUSDT.P", "ETHUSDT.P",
])
def test_round_trip(canonical):
    assert from_ccxt(to_ccxt(canonical)) == canonical


# ---------------------------------------------------------------------------
# Crucially: CCXT keeps USDT (unlike Alpaca which collapses to USD)
# ---------------------------------------------------------------------------

def test_ccxt_preserves_usdt_unlike_alpaca():
    """Regression guard: do NOT collapse USDT → USD for ccxt."""
    assert to_ccxt("BTCUSDT") == "BTC/USDT"
    assert to_ccxt("BTCUSDT") != "BTC/USD"
