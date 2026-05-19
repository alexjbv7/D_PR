"""
AlpacaEquityUniverse — top-200 US equity universe for Alpaca paper trading.
=============================================================================

Provides a curated, date-stamped list of liquid US equities tradable via
Alpaca Markets.  The base universe is hardcoded from well-known S&P 500
constituents to avoid survivorship bias in research (symbols are documented
with the date they were added).

Design decisions
----------------
* **Hardcoded base list** avoids a live API call at import time.
  The list is intentionally stable (major S&P 500 / NDX constituents) to
  minimise survivorship-bias concerns within a single walk-forward run.
* **Optional liquidity filter** — :meth:`~AlpacaEquityUniverse.filter_by_price`
  uses Alpaca's Data API to remove illiquid / low-price symbols.  Requires
  network access; skip in unit tests via monkeypatching.
* **Crypto is separate** — crypto symbols (``BTC/USD``, ``ETH/USD``, …) come
  from :data:`CRYPTO_UNIVERSE` and are treated as a distinct asset class.

Survivorship-bias note (§1.1 Capa A, architecture doc)
-------------------------------------------------------
The hardcoded list reflects *live* S&P 500 constituents as of 2026-05-18.
For strict survivorship-bias-free research, pair this list with a dated
constituent file (e.g. Compustat or S&P directly).  This module is an MVP
approximation; correct usage is documented in the walk-forward notebooks.

References
----------
* Architecture doc §1.1 (Capa A — Datos), §5 (Semana 1 roadmap)
* CLAUDE.md §4.2 (Swing strategy, equities + crypto universe)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timezone, datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector constants
# ---------------------------------------------------------------------------

TECH    = "Technology"
FINANCE = "Financials"
HEALTH  = "Healthcare"
ENERGY  = "Energy"
INDUST  = "Industrials"
CONS_D  = "Consumer Discretionary"
CONS_S  = "Consumer Staples"
COMM    = "Communication Services"
MATERI  = "Materials"
UTILS   = "Utilities"
REALES  = "Real Estate"


# ---------------------------------------------------------------------------
# Base universe
# ---------------------------------------------------------------------------

# Each entry: {"symbol": str, "sector": str, "market_cap_tier": str,
#              "added": YYYY-MM-DD, "notes": str}
# market_cap_tier: "mega" (> $200B), "large" (> $10B), "mid" ($2–10B)

EQUITY_UNIVERSE_BASE: list[dict] = [
    # ── Technology (mega + large cap) ──────────────────────────────────────
    {"symbol": "AAPL",  "sector": TECH,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "MSFT",  "sector": TECH,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "NVDA",  "sector": TECH,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "AVGO",  "sector": TECH,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "ORCL",  "sector": TECH,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "AMD",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "QCOM",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "TXN",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CSCO",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "IBM",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "AMAT",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "LRCX",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "KLAC",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "INTU",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ADBE",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CRM",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "NOW",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "PANW",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CRWD",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ANET",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MRVL",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MU",    "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "INTC",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "HPE",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "STX",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "WDC",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "PLTR",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "APP",   "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SNOW",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DDOG",  "sector": TECH,    "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Communication Services ─────────────────────────────────────────────
    {"symbol": "GOOGL", "sector": COMM,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "GOOG",  "sector": COMM,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "META",  "sector": COMM,    "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "NFLX",  "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DIS",   "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CMCSA", "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "TMUS",  "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "VZ",    "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "T",     "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "EA",    "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "TTD",   "sector": COMM,    "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Consumer Discretionary ─────────────────────────────────────────────
    {"symbol": "AMZN",  "sector": CONS_D,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "TSLA",  "sector": CONS_D,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "HD",    "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MCD",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "NKE",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SBUX",  "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "TJX",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "LOW",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "BKNG",  "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ABNB",  "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CMG",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "LULU",  "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ORLY",  "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "AZO",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ROST",  "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "HLT",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MAR",   "sector": CONS_D,  "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Financials ─────────────────────────────────────────────────────────
    {"symbol": "BRK.B", "sector": FINANCE, "market_cap_tier": "mega",  "added": "2026-05-18", "notes": "Berkshire Hathaway B"},
    {"symbol": "JPM",   "sector": FINANCE, "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "V",     "sector": FINANCE, "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "MA",    "sector": FINANCE, "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "BAC",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "WFC",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "GS",    "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MS",    "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "BLK",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SCHW",  "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SPGI",  "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MCO",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "AXP",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "COF",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "USB",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "PNC",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ICE",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CME",   "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CB",    "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "BK",    "sector": FINANCE, "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Healthcare ─────────────────────────────────────────────────────────
    {"symbol": "UNH",   "sector": HEALTH,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "LLY",   "sector": HEALTH,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "JNJ",   "sector": HEALTH,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "ABBV",  "sector": HEALTH,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "MRK",   "sector": HEALTH,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "TMO",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ABT",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DHR",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "PFE",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "AMGN",  "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "GILD",  "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ISRG",  "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SYK",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MDT",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "EW",    "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "BSX",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CI",    "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CVS",   "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "VRTX",  "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "REGN",  "sector": HEALTH,  "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Industrials ────────────────────────────────────────────────────────
    {"symbol": "CAT",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "GE",    "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "HON",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "UPS",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "LMT",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "RTX",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DE",    "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "BA",    "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "FDX",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "NSC",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CSX",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "UNP",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "WM",    "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "GD",    "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "NOC",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "ETN",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "EMR",   "sector": INDUST,  "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Energy ─────────────────────────────────────────────────────────────
    {"symbol": "XOM",   "sector": ENERGY,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "CVX",   "sector": ENERGY,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "COP",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "EOG",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SLB",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "PSX",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "VLO",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MPC",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "OXY",   "sector": ENERGY,  "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Consumer Staples ───────────────────────────────────────────────────
    {"symbol": "WMT",   "sector": CONS_S,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "COST",  "sector": CONS_S,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "PG",    "sector": CONS_S,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "KO",    "sector": CONS_S,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "PEP",   "sector": CONS_S,  "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "PM",    "sector": CONS_S,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MO",    "sector": CONS_S,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CL",    "sector": CONS_S,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "MDLZ",  "sector": CONS_S,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "KHC",   "sector": CONS_S,  "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Materials ──────────────────────────────────────────────────────────
    {"symbol": "LIN",   "sector": MATERI,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "APD",   "sector": MATERI,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "FCX",   "sector": MATERI,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "NEM",   "sector": MATERI,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DOW",   "sector": MATERI,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "NUE",   "sector": MATERI,  "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Utilities ──────────────────────────────────────────────────────────
    {"symbol": "NEE",   "sector": UTILS,   "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DUK",   "sector": UTILS,   "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "SO",    "sector": UTILS,   "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "D",     "sector": UTILS,   "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "AEP",   "sector": UTILS,   "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "EXC",   "sector": UTILS,   "market_cap_tier": "large", "added": "2026-05-18"},

    # ── Real Estate ────────────────────────────────────────────────────────
    {"symbol": "AMT",   "sector": REALES,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "PLD",   "sector": REALES,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "EQIX",  "sector": REALES,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "CCI",   "sector": REALES,  "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DLR",   "sector": REALES,  "market_cap_tier": "large", "added": "2026-05-18"},
]

# Liquid ETFs commonly traded via Alpaca (not equities, tracked separately)
ETF_UNIVERSE: list[dict] = [
    {"symbol": "SPY",   "sector": "ETF",   "market_cap_tier": "mega",  "added": "2026-05-18", "notes": "S&P 500"},
    {"symbol": "QQQ",   "sector": "ETF",   "market_cap_tier": "mega",  "added": "2026-05-18", "notes": "Nasdaq 100"},
    {"symbol": "IWM",   "sector": "ETF",   "market_cap_tier": "large", "added": "2026-05-18", "notes": "Russell 2000"},
    {"symbol": "GLD",   "sector": "ETF",   "market_cap_tier": "large", "added": "2026-05-18", "notes": "Gold"},
    {"symbol": "TLT",   "sector": "ETF",   "market_cap_tier": "large", "added": "2026-05-18", "notes": "20Y Treasury"},
    {"symbol": "XLF",   "sector": "ETF",   "market_cap_tier": "large", "added": "2026-05-18", "notes": "Financials"},
    {"symbol": "XLK",   "sector": "ETF",   "market_cap_tier": "large", "added": "2026-05-18", "notes": "Tech"},
    {"symbol": "XLE",   "sector": "ETF",   "market_cap_tier": "large", "added": "2026-05-18", "notes": "Energy"},
]

# Crypto symbols via Alpaca (24/7, Alpaca format)
CRYPTO_UNIVERSE: list[dict] = [
    {"symbol": "BTC/USD",  "sector": "Crypto", "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "ETH/USD",  "sector": "Crypto", "market_cap_tier": "mega",  "added": "2026-05-18"},
    {"symbol": "SOL/USD",  "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "BNB/USD",  "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "XRP/USD",  "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "AVAX/USD", "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "LINK/USD", "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DOT/USD",  "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "UNI/USD",  "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
    {"symbol": "DOGE/USD", "sector": "Crypto", "market_cap_tier": "large", "added": "2026-05-18"},
]


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class UniverseManifest:
    """
    Result of :meth:`AlpacaEquityUniverse.build`.

    Attributes
    ----------
    built_at : str
        UTC ISO-8601 timestamp of when the universe was built.
    symbols : list[str]
        Final symbol list (tickers only).
    metadata : list[dict]
        Full metadata rows (symbol, sector, market_cap_tier, added, …).
    n_equity : int
        Number of equity symbols.
    n_etf : int
        Number of ETF symbols.
    n_crypto : int
        Number of crypto symbols.
    sectors : dict[str, int]
        Count per sector.
    """
    built_at:  str
    symbols:   list[str]
    metadata:  list[dict]
    n_equity:  int
    n_etf:     int
    n_crypto:  int
    sectors:   dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AlpacaEquityUniverse
# ---------------------------------------------------------------------------

class AlpacaEquityUniverse:
    """
    Curated Alpaca tradable universe with optional liquidity filters.

    Parameters
    ----------
    include_etfs : bool
        Add ETFs from :data:`ETF_UNIVERSE` to the base list (default False).
    include_crypto : bool
        Add crypto symbols from :data:`CRYPTO_UNIVERSE` (default False).
    min_price : float
        Minimum price filter (applied when *api_key* is provided and
        :meth:`filter_by_price` is called).
    api_key : str, optional
        Alpaca API key for live price filtering.
    api_secret : str, optional
        Alpaca API secret.
    feed : str
        Alpaca data feed: ``"iex"`` (free) or ``"sip"`` (paid).

    Examples
    --------
    Offline (no API call)::

        universe = AlpacaEquityUniverse(include_crypto=True)
        manifest = universe.build(max_symbols=50)
        print(manifest.symbols[:5])

    With live price filter::

        universe = AlpacaEquityUniverse(
            api_key="MY_KEY", api_secret="MY_SECRET", min_price=5.0
        )
        manifest = universe.build(max_symbols=200)
    """

    def __init__(
        self,
        include_etfs:    bool  = False,
        include_crypto:  bool  = False,
        min_price:       float = 1.0,
        api_key:         Optional[str] = None,
        api_secret:      Optional[str] = None,
        feed:            str   = "iex",
    ):
        self._include_etfs   = include_etfs
        self._include_crypto = include_crypto
        self._min_price      = min_price
        self._api_key        = api_key
        self._api_secret     = api_secret
        self._feed           = feed

    # -----------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------

    def get_base(self) -> list[dict]:
        """
        Return the raw (unfiltered) universe rows.

        Returns
        -------
        list[dict]
            One dict per symbol with keys: symbol, sector, market_cap_tier,
            added, and optionally notes.
        """
        rows: list[dict] = list(EQUITY_UNIVERSE_BASE)
        if self._include_etfs:
            rows.extend(ETF_UNIVERSE)
        if self._include_crypto:
            rows.extend(CRYPTO_UNIVERSE)
        return rows

    def symbols(self, include_etfs: bool = False, include_crypto: bool = False) -> list[str]:
        """Return a flat list of symbol strings."""
        rows = list(EQUITY_UNIVERSE_BASE)
        if include_etfs or self._include_etfs:
            rows.extend(ETF_UNIVERSE)
        if include_crypto or self._include_crypto:
            rows.extend(CRYPTO_UNIVERSE)
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for r in rows:
            s = r["symbol"]
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def by_sector(self) -> dict[str, list[str]]:
        """Return symbols grouped by sector (includes selected asset classes)."""
        rows = self.get_base()
        groups: dict[str, list[str]] = {}
        for r in rows:
            sector = r["sector"]
            groups.setdefault(sector, []).append(r["symbol"])
        return groups

    # -----------------------------------------------------------------------
    # Filters
    # -----------------------------------------------------------------------

    def filter_by_tier(
        self,
        rows: list[dict],
        tiers: list[str] | None = None,
    ) -> list[dict]:
        """
        Filter rows by ``market_cap_tier``.

        Parameters
        ----------
        rows : list[dict]
            Universe rows (from :meth:`get_base`).
        tiers : list[str], optional
            Allowed tiers.  Defaults to ``["mega", "large"]``.

        Returns
        -------
        list[dict]
        """
        allowed = set(tiers or ["mega", "large"])
        return [r for r in rows if r.get("market_cap_tier", "") in allowed]

    def filter_by_price(
        self,
        symbols: list[str],
        min_price: Optional[float] = None,
    ) -> list[str]:
        """
        Remove symbols whose current price is below *min_price*.

        Requires ``api_key`` to be set.  Silently skips symbols where the
        price cannot be fetched (network errors, symbol not found).

        Parameters
        ----------
        symbols : list[str]
            List of tickers to filter.
        min_price : float, optional
            Minimum price threshold.  Falls back to ``self._min_price``.

        Returns
        -------
        list[str]
            Symbols whose price >= *min_price*.
        """
        threshold = min_price if min_price is not None else self._min_price

        if not self._api_key:
            logger.warning(
                "filter_by_price: no api_key — returning all %d symbols unfiltered",
                len(symbols),
            )
            return symbols

        try:
            from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
            from alpaca.data.requests import StockLatestQuoteRequest       # type: ignore
        except ImportError:  # pragma: no cover
            logger.warning("filter_by_price: alpaca-py not installed — skipping filter")
            return symbols

        client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._api_secret or None,
        )

        passed: list[str] = []
        for sym in symbols:
            if "/" in sym:            # crypto — skip price filter
                passed.append(sym)
                continue
            try:
                req = StockLatestQuoteRequest(
                    symbol_or_symbols=sym, feed=self._feed
                )
                res = client.get_stock_latest_quote(req)
                quote = res.get(sym) if hasattr(res, "get") else res
                if quote is None:
                    logger.debug("filter_by_price: no quote for %s — keeping", sym)
                    passed.append(sym)
                    continue
                mid = ((quote.ask_price or 0) + (quote.bid_price or 0)) / 2
                if mid >= threshold:
                    passed.append(sym)
                else:
                    logger.info(
                        "filter_by_price: removing %s price=%.2f < %.2f",
                        sym, mid, threshold,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("filter_by_price: error for %s: %s — keeping", sym, exc)
                passed.append(sym)

            time_module().sleep(0.35)   # rate-limit guard

        return passed

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def build(
        self,
        max_symbols: int = 200,
        filter_price: bool = False,
        tiers: list[str] | None = None,
    ) -> UniverseManifest:
        """
        Build the final universe up to *max_symbols* entries.

        Parameters
        ----------
        max_symbols : int
            Hard cap on the returned universe size (default 200).
        filter_price : bool
            When ``True``, call :meth:`filter_by_price` (requires
            ``api_key``).  Adds ~0.35 s × n network calls.
        tiers : list[str], optional
            Passed to :meth:`filter_by_tier`.  Defaults to mega + large.

        Returns
        -------
        UniverseManifest
        """
        rows = self.get_base()
        rows = self.filter_by_tier(rows, tiers)

        # Deduplicate
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in rows:
            if r["symbol"] not in seen:
                seen.add(r["symbol"])
                deduped.append(r)

        if filter_price and self._api_key:
            syms = [r["symbol"] for r in deduped]
            passed = self.filter_by_price(syms)
            passed_set = set(passed)
            deduped = [r for r in deduped if r["symbol"] in passed_set]

        final = deduped[:max_symbols]

        # Statistics
        sectors: dict[str, int] = {}
        n_equity = n_etf = n_crypto = 0
        for r in final:
            sector = r["sector"]
            sectors[sector] = sectors.get(sector, 0) + 1
            if sector == "ETF":
                n_etf += 1
            elif sector == "Crypto":
                n_crypto += 1
            else:
                n_equity += 1

        return UniverseManifest(
            built_at  = datetime.now(tz=timezone.utc).isoformat(),
            symbols   = [r["symbol"] for r in final],
            metadata  = final,
            n_equity  = n_equity,
            n_etf     = n_etf,
            n_crypto  = n_crypto,
            sectors   = sectors,
        )

    # -----------------------------------------------------------------------
    # Class methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs: object) -> "AlpacaEquityUniverse":
        """Build with credentials from ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET``."""
        import os
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except ImportError:
            pass
        return cls(
            api_key    = os.environ.get("ALPACA_API_KEY"),
            api_secret = os.environ.get("ALPACA_API_SECRET"),
            **kwargs,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_top_n(
    n: int = 200,
    include_etfs: bool = False,
    include_crypto: bool = False,
) -> list[str]:
    """
    Return the top-N symbols from the base universe (no API call required).

    Parameters
    ----------
    n : int
        Maximum number of symbols to return.
    include_etfs, include_crypto : bool
        Extend with ETFs / crypto symbols.

    Returns
    -------
    list[str]
    """
    universe = AlpacaEquityUniverse(
        include_etfs=include_etfs,
        include_crypto=include_crypto,
    )
    manifest = universe.build(max_symbols=n)
    return manifest.symbols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_module():
    """Late import of ``time`` to allow mocking in tests."""
    import time
    return time
