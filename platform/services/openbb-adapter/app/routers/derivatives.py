"""
Router /derivatives — Opciones y futuros crypto via OpenBB Deribit.

Endpoints:
  GET /derivatives/options/{symbol}        — cadena de opciones
  GET /derivatives/options/{symbol}/pcr    — put/call ratio
  GET /derivatives/futures/{symbol}/curve  — curva de futuros (basis)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..client import OpenBBClient
from ..models import PutCallRatioResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/derivatives", tags=["derivatives"])


def _get_client() -> OpenBBClient:
    from ..main import get_obb_client
    return get_obb_client()


@router.get("/options/{symbol}", summary="Cadena de opciones")
async def get_options_chain(
    symbol: str,
    expiration: Optional[str] = Query(None, description="Fecha de vencimiento YYYY-MM-DD"),
    provider: str = Query("deribit", description="Provider: deribit | intrinio"),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Cadena de opciones completa para BTC, ETH, SOL.

    Incluye: IV implícita, delta, gamma, theta, OI, volumen.
    Cache TTL: 5 minutos.
    """
    symbol = symbol.upper()
    data = await client.get_options_chain(symbol, expiration, provider)
    if not data:
        raise HTTPException(
            status_code=503,
            detail=f"Cadena de opciones no disponible para {symbol}",
        )
    return data


@router.get("/options/{symbol}/pcr", summary="Put/Call Ratio")
async def get_put_call_ratio(
    symbol: str,
    client: OpenBBClient = Depends(_get_client),
) -> PutCallRatioResponse:
    """
    Put/Call ratio sintético calculado desde la cadena de opciones.

    PCR > 1.2 → sentimiento bajista.
    PCR < 0.7 → sentimiento alcista.
    """
    symbol = symbol.upper()
    chain = await client.get_options_chain(symbol)
    if not chain:
        raise HTTPException(503, f"No se pudo calcular PCR para {symbol}")

    puts  = sum(1 for c in chain if str(c.get("option_type", "")).lower() == "put")
    calls = sum(1 for c in chain if str(c.get("option_type", "")).lower() == "call")
    pcr   = round(puts / calls, 4) if calls > 0 else None

    return PutCallRatioResponse(
        symbol=symbol,
        put_call_ratio=pcr,
        puts=puts,
        calls=calls,
        ts=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/futures/{symbol}/curve", summary="Curva de futuros (basis)")
async def get_futures_curve(
    symbol: str,
    provider: str = Query("deribit", description="Provider: deribit"),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Curva de futuros de crypto — spread spot vs futuros (contango/backwardation).

    Basis positivo (contango) → expectativa alcista.
    Basis negativo (backwardation) → demanda spot > futuros.
    """
    symbol = symbol.upper()
    data = await client.get_crypto_funding_rate(symbol, provider)
    if not data:
        raise HTTPException(503, f"Curva de futuros no disponible para {symbol}")
    return data
