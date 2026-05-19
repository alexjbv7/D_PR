"""
research.data
=============

Ingestion utilities for historical market data.

Modules
-------
alpaca_bars
    Download OHLCV bars from Alpaca Data API and persist to partitioned Parquet.
"""
from .alpaca_bars import AlpacaBarsIngestor, IngestReport, SymbolReport

__all__ = ["AlpacaBarsIngestor", "IngestReport", "SymbolReport"]
