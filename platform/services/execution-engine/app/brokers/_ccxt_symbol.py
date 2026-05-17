"""
Canonical ↔ CCXT symbol translation.

CCXT uses a unified ``BASE/QUOTE`` format (with optional ``:SETTLE`` for
perpetual swaps), unlike Alpaca which collapses USDT/USDC into USD.

Examples
--------
* canonical  ``BTCUSDT``   ↔ ccxt spot  ``BTC/USDT``
* canonical  ``ETHUSDC``   ↔ ccxt spot  ``ETH/USDC``
* canonical  ``BTCUSDT.P`` ↔ ccxt perp  ``BTC/USDT:USDT``  (``.P`` is our suffix)

Equities are not supported via CCXT and pass through unchanged; the routing
layer is expected to never dispatch equities to a CCXT adapter.
"""
from __future__ import annotations

# Common quote currencies, ordered longest-first so we never match "USD"
# inside "USDT" / "USDC".
_QUOTES: tuple[str, ...] = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP")

_PERP_SUFFIX = ".P"


def to_ccxt(symbol: str) -> str:
    """
    Translate canonical symbol → CCXT format.

    Examples
    --------
    >>> to_ccxt("BTCUSDT")
    'BTC/USDT'
    >>> to_ccxt("ETHUSDC")
    'ETH/USDC'
    >>> to_ccxt("BTCUSDT.P")
    'BTC/USDT:USDT'
    >>> to_ccxt("BTC/USDT")
    'BTC/USDT'
    """
    if "/" in symbol or ":" in symbol:
        return symbol.upper()

    sym = symbol.upper()
    is_perp = sym.endswith(_PERP_SUFFIX)
    if is_perp:
        sym = sym[: -len(_PERP_SUFFIX)]

    for q in _QUOTES:
        if sym.endswith(q) and len(sym) > len(q):
            base = sym[: -len(q)]
            spot = f"{base}/{q}"
            return f"{spot}:{q}" if is_perp else spot

    return sym                              # no recognised quote — pass through


def from_ccxt(symbol: str) -> str:
    """
    Translate CCXT symbol → canonical form.

    Examples
    --------
    >>> from_ccxt("BTC/USDT")
    'BTCUSDT'
    >>> from_ccxt("BTC/USDT:USDT")
    'BTCUSDT.P'
    >>> from_ccxt("ETH/USDC")
    'ETHUSDC'
    """
    sym = symbol.upper()
    is_perp = ":" in sym
    if is_perp:
        sym = sym.split(":", 1)[0]          # drop settle suffix

    if "/" not in sym:
        return sym

    base, quote = sym.split("/", 1)
    canonical = f"{base}{quote}"
    return f"{canonical}{_PERP_SUFFIX}" if is_perp else canonical
