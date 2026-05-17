"""
Broker adapters — concrete implementations of BrokerAdapter ABC.

Available:
  AlpacaAdapter  — equities + crypto via Alpaca Markets (PASO C)
  CCXTAdapter    — Binance / Bybit / Kraken via ccxt (PASO D)
"""
from .base import BrokerAdapter, AccountInfo, BrokerError, BrokerTimeoutError

__all__ = ["BrokerAdapter", "AccountInfo", "BrokerError", "BrokerTimeoutError"]
