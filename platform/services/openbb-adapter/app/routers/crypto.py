"""
Router /crypto — OHLCV histórico y funding rates de crypto.

Endpoints:
  GET /crypto/ohlcv/{symbol}     — OHLCV histórico
  GET /crypto/funding/{symbol}   — Funding rate histórico de perpetuales
  GET /crypto/news               — Noticias de crypto
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..client import OpenBBClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crypto", tags=["crypto"])


def _get_client() -> OpenBBClient:
    from ..main import get_obb_client
    return get_obb_client()


@router.get("/ohlcv/{symbol}", summary="OHLCV histórico de crypto")
async def get_crypto_ohlcv(
    symbol: str,
    interval: str = Query("1d", description="Intervalo: 1m | 5m | 15m | 1h | 4h | 1d"),
    start_date: str = Query("2020-01-01", description="Fecha inicio ISO 8601"),
    provider: Optional[str] = Query(None, description="Forzar provider: coingecko | yfinance"),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    OHLCV histórico de crypto.

    Cache TTL: 60s para 1m/5m, 3600s para 1h/1d.
    Fallback automático: coingecko → yfinance.
    """
    symbol = symbol.upper()
    data = await client.get_crypto_ohlcv(symbol, interval, start_date, provider)
    if not data:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo obtener OHLCV para {symbol}/{interval}",
        )
    return data


@router.get("/funding/{symbol}", summary="Funding rate histórico de perpetuales")
async def get_funding_rate(
    symbol: str,
    provider: str = Query("deribit", description="Provider: deribit | bybit"),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Funding rate histórico de futuros perpetuales.
    Actualización: cada 8 horas (alineado con periodos de funding).
    """
    symbol = symbol.upper()
    data = await client.get_crypto_funding_rate(symbol, provider)
    if not data:
        raise HTTPException(
            status_code=503,
            detail=f"Funding rate no disponible para {symbol}",
        )
    return data


@router.get("/news", summary="Noticias de crypto")
async def get_crypto_news(
    symbols: str = Query("BTC,ETH", description="Tickers separados por coma"),
    limit: int = Query(20, ge=1, le=100),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """Noticias financieras recientes para crypto. Cache TTL: 5 min."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return await client.get_news(symbol_list, limit)
