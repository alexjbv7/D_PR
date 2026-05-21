"""
Router /regulators — SEC EDGAR y CFTC COT reports via OpenBB.

Endpoints:
  GET /regulators/sec/filings/{ticker}      — filings SEC por empresa
  GET /regulators/sec/rss_litigation        — alertas RSS de litigios
  GET /regulators/cftc/cot                  — Commitments of Traders
  GET /regulators/institutional/{ticker}    — posiciones 13F
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..client import OpenBBClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regulators", tags=["regulators"])


def _get_client() -> OpenBBClient:
    from ..main import get_obb_client
    return get_obb_client()


@router.get("/sec/filings/{ticker}", summary="SEC filings por ticker")
async def get_sec_filings(
    ticker: str,
    form_type: str = Query("10-K", description="Form type: 10-K | 10-Q | 8-K | S-1"),
    limit: int = Query(10, ge=1, le=50),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Filings de SEC EDGAR para un ticker.

    Fuente: SEC EDGAR (gratuito, no requiere API key).
    Cache TTL: 1 hora.
    """
    ticker = ticker.upper()
    data = await client.get_sec_filings(ticker, form_type, limit)
    if not data:
        raise HTTPException(404, f"No se encontraron filings {form_type} para {ticker}")
    return data


@router.get("/sec/rss_litigation", summary="Alertas RSS de litigios SEC")
async def get_rss_litigation(
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Alertas en tiempo real del RSS de litigios de la SEC.
    Señal de riesgo regulatorio para el sistema de trading.
    Cache TTL: 1 hora.
    """
    data = await client.get_sec_rss_litigation()
    return data or []


@router.get("/cftc/cot", summary="CFTC Commitments of Traders")
async def get_cot_report(
    report_type: str = Query(
        "legacy_fut",
        description="Tipo: legacy_fut | disaggregated_fut | tff_fut",
    ),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    CFTC COT Report — posiciones institucionales en futuros regulados.

    Net speculative positivo → institucionales largos → señal alcista.
    Cache TTL: 1 semana (datos semanales del CFTC).
    """
    data = await client.get_cftc_cot(report_type)
    if not data:
        raise HTTPException(503, "CFTC COT no disponible en este momento")
    return data


@router.get("/institutional/{ticker}", summary="Posiciones 13F institucionales")
async def get_institutional_positions(
    ticker: str,
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Holdings institucionales 13F para ETFs de crypto (IBIT, GBTC, MSTR).

    Indica adopción institucional de BTC.
    Cache TTL: 24 horas.
    """
    ticker = ticker.upper()
    data = await client.get_institutional_positions(ticker)
    if not data:
        raise HTTPException(404, f"No se encontraron posiciones 13F para {ticker}")
    return data
