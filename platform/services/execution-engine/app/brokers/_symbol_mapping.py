"""
Symbol translation between internal canonical form and broker-native form.

Internal canonical form (matches Binance / strategy-orchestrator):
  - Crypto pairs: "BTCUSDT", "ETHUSDT", "SOLUSDC"
  - Equities:     "AAPL", "MSFT"
  - Index ETFs:   "SPY", "QQQ"

Alpaca native form:
  - Crypto pairs: "BTC/USD", "ETH/USD"  (uses USD, not USDT)
  - Equities:     "AAPL", "MSFT"        (unchanged)

CCXT uses yet another form ("BTC/USDT") — handled in the CCXT adapter.
"""
from __future__ import annotations

# Crypto base assets supported by Alpaca (as of 2025).
# Source: https://docs.alpaca.markets/docs/crypto-trading
_ALPACA_CRYPTO_BASES: set[str] = {
    "AAVE", "AVAX", "BAT", "BCH", "BTC", "CRV", "DOGE", "DOT", "ETH", "GRT",
    "LINK", "LTC", "MKR", "PEPE", "SHIB", "SOL", "SUSHI", "UNI", "USDC",
    "USDT", "XRP", "XTZ", "YFI",
}

# Common quote-asset suffixes ordered by length (longest first to avoid
# matching "USD" inside "USDC" / "USDT").
_QUOTES: tuple[str, ...] = ("USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP")


def is_crypto(symbol: str) -> bool:
    """Heuristic: True if ``symbol`` looks like a crypto pair."""
    if "/" in symbol:
        return True
    for q in _QUOTES:
        if symbol.endswith(q) and len(symbol) > len(q):
            base = symbol[: -len(q)]
            if base in _ALPACA_CRYPTO_BASES or base.isalpha() and len(base) >= 3:
                return True
    return False


def to_alpaca(symbol: str) -> str:
    """
    Translate canonical symbol → Alpaca format.

    Examples
    --------
    >>> to_alpaca("BTCUSDT")
    'BTC/USD'
    >>> to_alpaca("AAPL")
    'AAPL'
    >>> to_alpaca("ETH/USD")
    'ETH/USD'
    """
    if "/" in symbol:                       # already split
        return symbol.upper()

    if not is_crypto(symbol):               # equity / ETF
        return symbol.upper()

    sym = symbol.upper()
    for q in _QUOTES:
        if sym.endswith(q):
            base = sym[: -len(q)]
            # Alpaca normalises USDT/USDC pairs to USD for spot-like trading
            quote = "USD" if q in ("USDT", "USDC") else q
            return f"{base}/{quote}"
    return sym                              # fallback (unlikely)


def from_alpaca(symbol: str) -> str:
    """
    Translate Alpaca symbol → canonical form.

    Examples
    --------
    >>> from_alpaca("BTC/USD")
    'BTCUSDT'
    >>> from_alpaca("AAPL")
    'AAPL'
    """
    if "/" not in symbol:                   # equity / ETF
        return symbol.upper()

    base, quote = symbol.upper().split("/", 1)
    # Canonical = USDT for USD-quoted crypto (matches Binance perp/spot)
    canonical_quote = "USDT" if quote == "USD" else quote
    return f"{base}{canonical_quote}"
