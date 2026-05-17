"""
OpenBB Async Client — Núcleo de datos financieros multi-activo.
===============================================================
Wrapper asíncrono sobre OpenBB SDK que:
- Ejecuta llamadas bloqueantes en threadpool (OpenBB es sync)
- Cachea resultados en Redis con TTL por tipo de dato
- Normaliza a schemas estándar
- Expone métodos para acciones, crypto, FX, macro, opciones, sentimiento

Datos disponibles:
  - OHLCV histórico y en tiempo real
  - Fundamentals (earnings, balances, ratios)
  - Opciones (IV, cadena, skew)
  - Noticias con sentimiento
  - Macro indicators
  - ETFs
  - Order flow proxy
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Optional

import pandas as pd
from openbb import obb

from libs.shared.redis_client import RedisCache, TTL

logger = logging.getLogger(__name__)

OPENBB_TOKEN = os.getenv("OPENBB_PAT", "")


class OpenBBClient:
    """
    Cliente OpenBB asíncrono con cache Redis.

    client = OpenBBClient(cache)
    df = await client.get_ohlcv("BTCUSDT", "crypto", "1d", limit=365)
    news = await client.get_news(["BTC", "ETH"])
    """

    def __init__(self, cache: RedisCache):
        self._cache = cache
        self._loop = asyncio.get_event_loop()
        if OPENBB_TOKEN:
            obb.account.login(pat=OPENBB_TOKEN)

    async def _run_sync(self, func, *args, **kwargs) -> Any:
        """Ejecuta llamada síncrona de OpenBB en threadpool."""
        fn = partial(func, *args, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, fn)

    # ── OHLCV ──────────────────────────────────────────────────────────

    # yfinance valid intervals — map unsupported ones to the closest valid
    _YFINANCE_INTERVAL_MAP: dict[str, str] = {
        "4h": "1h",
        "2h": "1h",
        "3h": "1h",
        "6h": "1d",
        "8h": "1d",
        "12h": "1d",
        "3d": "1d",
        "2d": "1d",
    }

    @staticmethod
    def _normalize_crypto_symbol(symbol: str) -> str:
        """Convert BTCUSDT → BTC-USD for yfinance."""
        symbol = symbol.upper()
        for quote in ("USDT", "BUSD", "USD", "BTC", "ETH", "BNB"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[: -len(quote)]
                return f"{base}-USD"
        return symbol

    def _normalize_interval(self, interval: str, venue: str) -> str:
        """Map intervals unsupported by a provider to the closest valid one."""
        if venue == "yfinance":
            return self._YFINANCE_INTERVAL_MAP.get(interval, interval)
        return interval

    async def get_ohlcv(
        self,
        symbol: str,
        asset_type: str = "crypto",
        interval: str = "1d",
        limit: int = 365,
        venue: str = "yfinance",
    ) -> pd.DataFrame:
        """
        Obtiene OHLCV normalizado para cualquier activo.

        asset_type: "crypto" | "equity" | "forex" | "futures"
        venue: OpenBB provider — "yfinance" | "fmp" | "tiingo"
        """
        cache_key = f"ohlcv:{asset_type}:{symbol}:{interval}:{limit}"
        cached = await self._cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            mapped_interval = self._normalize_interval(interval, venue)
            if asset_type == "crypto":
                yf_symbol = self._normalize_crypto_symbol(symbol) if venue == "yfinance" else symbol
                result = await self._run_sync(
                    obb.crypto.price.historical,
                    symbol=yf_symbol,
                    interval=mapped_interval,
                    limit=limit,
                    provider=venue,
                )
            elif asset_type == "equity":
                result = await self._run_sync(
                    obb.equity.price.historical,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                )
            elif asset_type == "forex":
                result = await self._run_sync(
                    obb.forex.price.historical,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                )
            else:
                raise ValueError(f"Unknown asset_type: {asset_type}")

            df = result.to_df()
            df.columns = [c.lower() for c in df.columns]
            await self._cache.set(cache_key, df.to_dict("records"), ttl=TTL["ohlcv"])
            return df

        except Exception as exc:
            logger.error("OpenBB OHLCV error %s %s: %s", asset_type, symbol, exc)
            return pd.DataFrame()

    # ── NOTICIAS ───────────────────────────────────────────────────────

    async def get_news(
        self,
        symbols: list[str],
        limit: int = 50,
    ) -> list[dict]:
        """
        Obtiene noticias recientes con metadata básico.
        Retorna lista de dicts: {headline, url, date, source, symbols}.
        """
        cache_key = f"news:{','.join(sorted(symbols))}:{limit}"
        cached = await self._cache.get(cache_key)
        if cached:
            return cached

        try:
            result = await self._run_sync(
                obb.news.world, symbols=",".join(symbols), limit=limit
            )
            articles = []
            for item in result.results:
                articles.append({
                    "headline":  getattr(item, "title", ""),
                    "url":       getattr(item, "url", ""),
                    "date":      str(getattr(item, "date", "")),
                    "source":    getattr(item, "source", ""),
                    "symbols":   getattr(item, "symbols", []) or symbols,
                    "sentiment": getattr(item, "sentiment", None),
                })
            await self._cache.set(cache_key, articles, ttl=TTL["macro"])
            return articles
        except Exception as exc:
            logger.error("OpenBB news error %s: %s", symbols, exc)
            return []

    # ── OPCIONES ───────────────────────────────────────────────────────

    async def get_options_chain(
        self, symbol: str, expiry: Optional[str] = None
    ) -> pd.DataFrame:
        """Cadena de opciones completa para equity."""
        cache_key = f"options:{symbol}:{expiry}"
        cached = await self._cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            kwargs = {"symbol": symbol}
            if expiry:
                kwargs["expiry"] = expiry
            result = await self._run_sync(obb.derivatives.options.chains, **kwargs)
            df = result.to_df()
            await self._cache.set(cache_key, df.to_dict("records"), ttl=300)
            return df
        except Exception as exc:
            logger.error("OpenBB options error %s: %s", symbol, exc)
            return pd.DataFrame()

    async def get_implied_volatility_surface(self, symbol: str) -> dict:
        """Superficie de volatilidad implícita por strike y expiry."""
        chain = await self.get_options_chain(symbol)
        if chain.empty:
            return {}
        try:
            surface = (
                chain.groupby(["expiry", "strike"])["implied_volatility"]
                .mean()
                .unstack(level=0)
                .to_dict()
            )
            return surface
        except Exception:
            return {}

    # ── MACRO ──────────────────────────────────────────────────────────

    async def get_macro_indicator(
        self, indicator: str, start_date: str = "2000-01-01"
    ) -> pd.DataFrame:
        """
        Obtiene serie macro de OpenBB (FRED proxy u otras fuentes).
        indicator: "cpi", "gdp", "unemployment", "fed_funds", etc.
        """
        cache_key = f"macro:openbb:{indicator}"
        cached = await self._cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            result = await self._run_sync(
                obb.economy.indicators,
                indicators=indicator,
                start_date=start_date,
            )
            df = result.to_df()
            await self._cache.set(cache_key, df.to_dict("records"), ttl=TTL["macro"])
            return df
        except Exception as exc:
            logger.error("OpenBB macro error %s: %s", indicator, exc)
            return pd.DataFrame()

    # ── ETFs ───────────────────────────────────────────────────────────

    async def get_etf_holdings(self, symbol: str) -> pd.DataFrame:
        """Holdings de un ETF — útil para exposición sectorial."""
        cache_key = f"etf:holdings:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            result = await self._run_sync(obb.etf.holdings, symbol=symbol)
            df = result.to_df()
            await self._cache.set(cache_key, df.to_dict("records"), ttl=TTL["macro"])
            return df
        except Exception as exc:
            logger.error("OpenBB ETF holdings error %s: %s", symbol, exc)
            return pd.DataFrame()

    # ── FUNDAMENTALS ───────────────────────────────────────────────────

    async def get_earnings(self, symbol: str, limit: int = 8) -> pd.DataFrame:
        """Earnings historicos y estimados."""
        cache_key = f"earnings:{symbol}:{limit}"
        cached = await self._cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            result = await self._run_sync(
                obb.equity.estimates.historical, symbol=symbol, limit=limit
            )
            df = result.to_df()
            await self._cache.set(cache_key, df.to_dict("records"), ttl=TTL["macro"])
            return df
        except Exception as exc:
            logger.error("OpenBB earnings error %s: %s", symbol, exc)
            return pd.DataFrame()

    async def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        """Ratios fundamentales: P/E, P/B, ROE, etc."""
        cache_key = f"ratios:{symbol}"
        try:
            result = await self._run_sync(
                obb.equity.fundamental.ratios, symbol=symbol
            )
            return result.to_df()
        except Exception as exc:
            logger.error("OpenBB ratios error %s: %s", symbol, exc)
            return pd.DataFrame()

    # ── VOLATILITY ─────────────────────────────────────────────────────

    async def get_vix(self) -> float:
        """VIX spot level."""
        cache_key = "macro:vix"
        cached = await self._cache.get(cache_key)
        if cached:
            return float(cached)
        try:
            df = await self.get_ohlcv("^VIX", "equity", "1d", limit=2)
            val = float(df["close"].iloc[-1]) if not df.empty else 20.0
            await self._cache.set(cache_key, val, ttl=300)
            return val
        except Exception:
            return 20.0

    async def get_realized_vol(
        self, symbol: str, window: int = 20, asset_type: str = "crypto"
    ) -> float:
        """Volatilidad realizada en ventana de `window` días."""
        df = await self.get_ohlcv(symbol, asset_type, "1d", limit=window + 5)
        if df.empty or "close" not in df.columns:
            return 0.0
        returns = df["close"].pct_change().dropna()
        return float(returns.std() * (252 ** 0.5))

    # ── ORDER FLOW PROXY ───────────────────────────────────────────────

    async def get_dark_pool_data(self, symbol: str) -> pd.DataFrame:
        """Dark pool y actividad institucional (si disponible)."""
        try:
            result = await self._run_sync(
                obb.equity.darkpool.otc, symbol=symbol
            )
            return result.to_df()
        except Exception:
            return pd.DataFrame()
