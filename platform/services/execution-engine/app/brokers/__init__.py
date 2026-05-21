"""
Broker adapters — concrete implementations of BrokerAdapter ABC.

Available:
  AlpacaAdapter     — equities + crypto via Alpaca Markets (PASO C)
  AlpacaMarketData  — OHLCV bars, quotes, snapshots via Alpaca Data API
  CCXTAdapter       — Binance / Bybit / Kraken via ccxt (PASO D)
"""
from .base import BrokerAdapter, AccountInfo, BrokerError, BrokerTimeoutError
from .alpaca import AlpacaAdapter, AlpacaConfig
from ._alpaca.market_data import AlpacaMarketData
from .ccxt_adapter import CCXTAdapter, CCXTConfig

__all__ = [
    "BrokerAdapter", "AccountInfo", "BrokerError", "BrokerTimeoutError",
    "AlpacaAdapter", "AlpacaConfig", "AlpacaMarketData",
    "CCXTAdapter",   "CCXTConfig",
]
