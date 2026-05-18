"""Symbol classification helpers — shared across services (no broker imports)."""
from __future__ import annotations

import re

_EQUITY_RE = re.compile(r"^[A-Z]{1,5}$")
_CRYPTO_QUOTES = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP")


def is_equity(symbol: str) -> bool:
    """
    True if ``symbol`` looks like a US equity / ETF ticker (1–5 uppercase letters).

    Mirrors ``platform/services/execution-engine/app/routing.py:is_equity``.
    """
    sym = symbol.upper()
    if "/" in sym or ":" in sym:
        return False
    for quote in _CRYPTO_QUOTES:
        if sym.endswith(quote) and len(sym) > len(quote):
            return False
    return bool(_EQUITY_RE.match(sym))
