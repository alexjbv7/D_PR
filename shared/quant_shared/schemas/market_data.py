"""
Tipos canónicos de market data — tick, OHLCV, orderbook.

Usados tanto en research/ (backtesting con datos históricos)
como en platform/ (streaming en tiempo real).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class NormalizedTick:
    """Tick normalizado desde cualquier exchange."""
    symbol:         str
    venue:          str
    price:          float
    volume:         float
    ts:             datetime

    # Microestructura (opcional)
    bid:            Optional[float] = None
    ask:            Optional[float] = None
    spread_bps:     float = 0.0
    ob_imbalance:   float = 0.0
    funding_rate:   float = 0.0
    open_interest:  float = 0.0
    vwap:           float = 0.0

    @property
    def mid_price(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.price


@dataclass
class OHLCVBar:
    """Bar OHLCV completo."""
    symbol:    str
    timeframe: str              # "1m" | "5m" | "1h" | "4h" | "1d"
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    ts:        datetime         # timestamp de apertura del bar
    venue:     str = ""
    vwap:      float = 0.0
    trades:    int = 0


@dataclass
class OrderBook:
    """Snapshot de orderbook."""
    symbol:  str
    venue:   str
    ts:      datetime
    bids:    list[tuple[float, float]] = field(default_factory=list)  # [(price, qty), ...]
    asks:    list[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread_bps(self) -> float:
        if self.best_bid and self.best_ask and self.mid_price:
            return (self.best_ask - self.best_bid) / self.mid_price * 10_000
        return 0.0

    @property
    def imbalance(self) -> float:
        """(bid_vol_top5 - ask_vol_top5) / (bid_vol_top5 + ask_vol_top5)."""
        bid_vol = sum(q for _, q in self.bids[:5])
        ask_vol = sum(q for _, q in self.asks[:5])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
