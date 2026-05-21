"""
OpenBBClient — Wrapper asíncrono sobre el SDK de OpenBB Platform 4.7.1.

Responsabilidades:
  - Ejecutar llamadas síncronas de OpenBB en threadpool (run_in_executor)
  - Cache Redis por endpoint + parámetros con TTL por tipo de dato
  - Fallback automático entre providers si el primario falla
  - Rate limiting por semáforo concurrente por provider
  - Logging estructurado de errores y métricas

No exponer el objeto `obb` fuera de este módulo.
Todo acceso al SDK pasa por métodos de esta clase.

Providers disponibles (por categoría):
  crypto:     coingecko, yfinance, coinmarketcap
  macro:      fred, oecd, imf
  equity:     fmp, yfinance, polygon
  options:    deribit, intrinio
  regulators: sec, fmp
  news:       fmp, benzinga
"""
from __future__ import annotations

import asyncio
import logging
from asyncio import Semaphore
from functools import partial
from typing import Any, Optional

import pandas as pd

from .cache import ResponseCache
from .config import Settings

logger = logging.getLogger(__name__)

# Límites de concurrencia por provider (requests simultáneos)
_PROVIDER_SEMAPHORES: dict[str, Semaphore] = {
    "fred":        Semaphore(8),
    "coingecko":   Semaphore(4),
    "fmp":         Semaphore(8),
    "polygon":     Semaphore(10),
    "deribit":     Semaphore(6),
    "sec":         Semaphore(4),
    "yfinance":    Semaphore(3),
    "oecd":        Semaphore(4),
    "intrinio":    Semaphore(4),
}

# Fallback de providers por categoría
_PROVIDER_PRIORITY: dict[str, list[str]] = {
    "crypto_ohlcv":  ["coingecko", "yfinance"],
    "macro_series":  ["fred", "oecd"],
    "equity_price":  ["fmp", "yfinance", "polygon"],
    "options":       ["deribit", "intrinio"],
    "sec_filings":   ["sec", "fmp"],
    "news":          ["fmp", "benzinga"],
    "yield_curve":   ["fred"],
    "cot":           ["cftc"],
}


class OpenBBClient:
    """
    Cliente OpenBB asíncrono con cache Redis y fallback entre providers.

    Usage
    -----
    client = OpenBBClient(settings, cache)
    await client.configure()
    ohlcv = await client.get_crypto_ohlcv("BTC", interval="1d")
    """

    def __init__(self, settings: Settings, cache: ResponseCache):
        self._settings = settings
        self._cache    = cache
        self._obb: Any = None  # lazy import — evitar costo en import time

    async def configure(self) -> None:
        """Inyectar API keys en OpenBB Platform. Llamar al startup del servicio."""
        try:
            from openbb import obb as _obb
            self._obb = _obb

            if self._settings.fred_api_key:
                self._obb.user.credentials.fred_api_key = self._settings.fred_api_key
            if self._settings.fmp_api_key:
                self._obb.user.credentials.fmp_api_key = self._settings.fmp_api_key
            if self._settings.polygon_api_key:
                self._obb.user.credentials.polygon_api_key = self._settings.polygon_api_key
            if self._settings.openbb_pat:
                self._obb.account.login(pat=self._settings.openbb_pat)

            logger.info("openbb_client.configured providers=fred,fmp,polygon,coingecko")
        except ImportError:
            logger.warning("openbb_client.import_failed — SDK not installed; stubs active")

    # ── Util ──────────────────────────────────────────────────────────

    async def _run(self, func, *args, **kwargs) -> Any:
        """Ejecuta función síncrona de OpenBB en threadpool."""
        fn = partial(func, *args, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, fn)

    async def _run_with_limit(
        self, provider: str, func, *args, **kwargs
    ) -> Any:
        """Ejecuta con rate limit por provider."""
        sem = _PROVIDER_SEMAPHORES.get(provider, Semaphore(5))
        async with sem:
            return await self._run(func, *args, **kwargs)

    async def _with_fallback(
        self, category: str, func_map: dict[str, Any], *args, **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Intenta llamar a cada provider en orden de prioridad.
        Retorna el primer resultado exitoso, o DataFrame vacío.
        """
        providers = _PROVIDER_PRIORITY.get(category, ["yfinance"])
        for prov in providers:
            fn = func_map.get(prov)
            if fn is None:
                continue
            try:
                result = await self._run_with_limit(prov, fn, provider=prov)
                return result.to_df() if hasattr(result, "to_df") else result
            except Exception as exc:
                logger.warning(
                    "openbb_client.provider_failed category=%s provider=%s: %s",
                    category, prov, exc,
                )
        logger.error("openbb_client.all_providers_failed category=%s", category)
        return pd.DataFrame()

    # ── Crypto ────────────────────────────────────────────────────────

    async def get_crypto_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: str = "2020-01-01",
        provider: Optional[str] = None,
    ) -> list[dict]:
        """
        OHLCV histórico de crypto.

        Parameters
        ----------
        symbol    : ticker sin /USD, ej "BTC", "ETH"
        interval  : "1m" | "5m" | "1h" | "1d"
        start_date: ISO 8601
        provider  : fuerza un provider específico; si None usa fallback

        Returns
        -------
        list[dict] con keys: date, open, high, low, close, volume
        """
        cache_key = f"crypto:{interval}:{symbol}:{start_date}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        providers = [provider] if provider else _PROVIDER_PRIORITY["crypto_ohlcv"]
        symbol_pair = f"{symbol}USD"

        for prov in providers:
            try:
                result = await self._run_with_limit(
                    prov,
                    self._obb.crypto.price.historical,
                    symbol=symbol_pair,
                    start_date=start_date,
                    interval=interval,
                    provider=prov,
                )
                df = result.to_df()
                df.columns = [c.lower() for c in df.columns]
                data = df.reset_index().to_dict(orient="records")

                ttl = 60 if interval in ("1m", "5m") else 3600
                await self._cache.set(cache_key, data, ttl=ttl)
                return data
            except Exception as exc:
                logger.warning(
                    "openbb_client.crypto_ohlcv_failed symbol=%s provider=%s: %s",
                    symbol, prov, exc,
                )

        return []

    async def get_crypto_funding_rate(
        self,
        symbol: str,
        provider: str = "deribit",
    ) -> list[dict]:
        """Funding rate histórico de perpetuales."""
        cache_key = f"futures:funding:{symbol}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                provider,
                self._obb.derivatives.futures.historical,
                symbol=symbol,
                provider=provider,
            )
            data = result.to_df().reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=28800)
            return data
        except Exception as exc:
            logger.warning("openbb_client.funding_rate_failed symbol=%s: %s", symbol, exc)
            return []

    # ── Macro / FRED ──────────────────────────────────────────────────

    async def get_fred_series(
        self,
        series_id: str,
        start_date: str = "2015-01-01",
        provider: str = "fred",
    ) -> list[dict]:
        """
        Serie macroeconómica desde FRED.

        Returns
        -------
        list[dict] con keys: date, value
        Ordenado cronológicamente (más antiguo primero).
        """
        cache_key = f"macro:daily:{series_id}:{start_date}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                provider,
                self._obb.economy.fred_series,
                symbol=series_id,
                start_date=start_date,
                provider=provider,
            )
            df = result.to_df().reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Normalizar columna de fecha
            date_col = next(
                (c for c in df.columns if "date" in c or "time" in c), df.columns[0]
            )
            value_col = next(
                (c for c in df.columns if c not in (date_col,)), "value"
            )
            data = [
                {"date": str(row[date_col]), "value": float(row[value_col])}
                for _, row in df.iterrows()
                if row[value_col] is not None and str(row[value_col]) != "nan"
            ]
            data.sort(key=lambda x: x["date"])

            # TTL según frecuencia (daily → 1d, mensual → 1 semana)
            ttl = 86_400 if len(data) > 100 else 604_800
            await self._cache.set(cache_key, data, ttl=ttl)
            return data

        except Exception as exc:
            logger.warning(
                "openbb_client.fred_series_failed series=%s: %s", series_id, exc
            )
            return []

    async def get_yield_curve(
        self,
        date: Optional[str] = None,
        country: str = "us",
        provider: str = "fred",
    ) -> list[dict]:
        """Curva de rendimientos del gobierno (US por defecto)."""
        cache_key = f"yield_curve:{country}:{date or 'latest'}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            kwargs: dict = {"country": country, "provider": provider}
            if date:
                kwargs["date"] = date

            result = await self._run_with_limit(
                provider,
                self._obb.fixedincome.government.yield_curve,
                **kwargs,
            )
            data = result.to_df().reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=14_400)
            return data

        except Exception as exc:
            logger.warning("openbb_client.yield_curve_failed: %s", exc)
            return []

    # ── Derivados ─────────────────────────────────────────────────────

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        provider: str = "deribit",
    ) -> list[dict]:
        """Cadena de opciones (BTC, ETH) desde Deribit."""
        cache_key = f"options:{symbol}:{expiration or 'all'}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            kwargs: dict = {"symbol": symbol, "provider": provider}
            if expiration:
                kwargs["expiration"] = expiration

            result = await self._run_with_limit(
                provider,
                self._obb.derivatives.options.chains,
                **kwargs,
            )
            df = result.to_df()
            data = df.reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=300)
            return data

        except Exception as exc:
            logger.warning("openbb_client.options_chain_failed symbol=%s: %s", symbol, exc)
            return []

    async def get_put_call_ratio(self, symbol: str) -> Optional[float]:
        """Put/call ratio sintético calculado desde la cadena de opciones."""
        chain = await self.get_options_chain(symbol)
        if not chain:
            return None
        try:
            puts  = sum(1 for c in chain if str(c.get("option_type", "")).lower() == "put")
            calls = sum(1 for c in chain if str(c.get("option_type", "")).lower() == "call")
            if calls == 0:
                return None
            return round(puts / calls, 4)
        except Exception:
            return None

    # ── Regulators ────────────────────────────────────────────────────

    async def get_sec_filings(
        self,
        symbol: str,
        form_type: str = "10-K",
        limit: int = 10,
        provider: str = "sec",
    ) -> list[dict]:
        """Filings de SEC EDGAR para un ticker."""
        cache_key = f"sec:{symbol}:{form_type}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                provider,
                self._obb.equity.fundamental.filings,
                symbol=symbol,
                form_type=form_type,
                limit=limit,
                provider=provider,
            )
            data = result.to_df().reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=3600)
            return data

        except Exception as exc:
            logger.warning("openbb_client.sec_filings_failed symbol=%s: %s", symbol, exc)
            return []

    async def get_sec_rss_litigation(self) -> list[dict]:
        """Alertas de litigios SEC en tiempo real."""
        cache_key = "sec:rss:litigation"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                "sec",
                self._obb.regulators.sec.rss_litigation,
                provider="sec",
            )
            data = result.to_df().reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=3600)
            return data

        except Exception as exc:
            logger.warning("openbb_client.sec_rss_failed: %s", exc)
            return []

    async def get_cftc_cot(
        self,
        report_type: str = "legacy_fut",
        provider: str = "cftc",
    ) -> list[dict]:
        """CFTC Commitments of Traders — posiciones institucionales."""
        cache_key = f"cot:{report_type}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                provider,
                self._obb.regulators.cftc.cot,
                report_type=report_type,
                provider=provider,
            )
            data = result.to_df().reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=604_800)  # 1 semana
            return data

        except Exception as exc:
            logger.warning("openbb_client.cftc_cot_failed: %s", exc)
            return []

    # ── News ──────────────────────────────────────────────────────────

    async def get_news(
        self,
        symbols: list[str],
        limit: int = 20,
        provider: str = "fmp",
    ) -> list[dict]:
        """Noticias financieras recientes para una lista de símbolos."""
        cache_key = f"news:{','.join(sorted(symbols))}:{limit}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                provider,
                self._obb.news.world.news,
                symbols=",".join(symbols),
                limit=limit,
                provider=provider,
            )
            articles = []
            for item in result.results:
                articles.append({
                    "headline": getattr(item, "title",     ""),
                    "url":      getattr(item, "url",       ""),
                    "date":     str(getattr(item, "date",  "")),
                    "source":   getattr(item, "source",    ""),
                    "symbols":  getattr(item, "symbols",   []) or symbols,
                    "sentiment": getattr(item, "sentiment", None),
                })
            await self._cache.set(cache_key, articles, ttl=300)
            return articles

        except Exception as exc:
            logger.warning("openbb_client.news_failed: %s", exc)
            return []

    # ── Equity / ETF ──────────────────────────────────────────────────

    async def get_institutional_positions(
        self,
        symbol: str,
        provider: str = "fmp",
    ) -> list[dict]:
        """Posiciones 13F — holdings institucionales en ETFs de crypto."""
        cache_key = f"sec:13f:{symbol}"
        if cached := await self._cache.get(cache_key):
            return cached

        if self._obb is None:
            return []

        try:
            result = await self._run_with_limit(
                provider,
                self._obb.equity.ownership.institutional,
                symbol=symbol,
                provider=provider,
            )
            data = result.to_df().reset_index().to_dict(orient="records")
            await self._cache.set(cache_key, data, ttl=86_400)
            return data

        except Exception as exc:
            logger.warning(
                "openbb_client.institutional_positions_failed symbol=%s: %s", symbol, exc
            )
            return []
