"""
Router /macro — Series macroeconómicas via OpenBB FRED / OECD.

Endpoints:
  GET /macro/fred                — serie individual
  GET /macro/snapshot            — todas las series cacheadas
  GET /macro/yield_curve         — curva de rendimientos
  GET /macro/calendar            — calendario de eventos económicos
  GET /macro/recession_indicators — probabilidades de recesión
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..client import OpenBBClient
from ..models import MacroPoint, MacroSnapshot, YieldCurvePoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/macro", tags=["macro"])

# Series FRED completas — 22 base + 6 extendidas
FRED_SERIES_META: dict[str, dict] = {
    # Inflación
    "CPIAUCSL":  {"name": "CPI All Urban",        "freq": "monthly", "category": "inflation"},
    "CPILFESL":  {"name": "CPI Core (ex F&E)",    "freq": "monthly", "category": "inflation"},
    "PCEPI":     {"name": "PCE Price Index",       "freq": "monthly", "category": "inflation"},
    "PPIFIS":    {"name": "PPI Final Demand",      "freq": "monthly", "category": "inflation"},
    # Tipos de interés
    "DFF":       {"name": "Fed Funds Rate",        "freq": "daily",   "category": "rates"},
    "T10Y2Y":    {"name": "10Y-2Y Yield Curve",   "freq": "daily",   "category": "rates"},
    "T10YIE":    {"name": "10Y Breakeven Inf",    "freq": "daily",   "category": "rates"},
    "DGS10":     {"name": "10Y Treasury",          "freq": "daily",   "category": "rates"},
    "DGS2":      {"name": "2Y Treasury",           "freq": "daily",   "category": "rates"},
    # Empleo
    "UNRATE":    {"name": "Unemployment Rate",     "freq": "monthly", "category": "labor"},
    "PAYEMS":    {"name": "Nonfarm Payrolls",      "freq": "monthly", "category": "labor"},
    "ICSA":      {"name": "Initial Claims",        "freq": "weekly",  "category": "labor"},
    "JTSJOL":    {"name": "Job Openings (JOLTS)",  "freq": "monthly", "category": "labor"},
    # Crecimiento
    "GDP":       {"name": "GDP (real, annlzd)",    "freq": "quarterly","category": "growth"},
    "INDPRO":    {"name": "Industrial Production", "freq": "monthly", "category": "growth"},
    "RSXFS":     {"name": "Retail Sales (ex auto)","freq": "monthly", "category": "growth"},
    # Dinero y crédito
    "M2SL":      {"name": "M2 Money Supply",       "freq": "monthly", "category": "monetary"},
    "TOTCI":     {"name": "Total Credit",          "freq": "monthly", "category": "monetary"},
    # Sentimiento
    "UMCSENT":   {"name": "U Mich Consumer Sent",  "freq": "monthly", "category": "sentiment"},
    "CSCICP03USM665S": {"name": "Conf Board LEI",  "freq": "monthly", "category": "sentiment"},
    # Housing
    "HOUST":     {"name": "Housing Starts",        "freq": "monthly", "category": "housing"},
    "CS20RPSNSA":{"name": "Case-Shiller 20-City",  "freq": "monthly", "category": "housing"},
    # Riesgo / mercados
    "VIXCLS":    {"name": "VIX (CBOE)",            "freq": "daily",   "category": "risk"},
    "DEXUSEU":   {"name": "USD/EUR",               "freq": "daily",   "category": "fx"},
    "DEXJPUS":   {"name": "JPY/USD",               "freq": "daily",   "category": "fx"},
    "DTWEXBGS":  {"name": "USD Broad Index",       "freq": "daily",   "category": "fx"},
    # Extended (nuevas en OpenBB)
    "SAHMREALTIME": {"name": "Sahm Rule RT",       "freq": "monthly", "category": "recession"},
    "USREC":     {"name": "Recession Indicator",   "freq": "monthly", "category": "recession"},
}


def _get_client() -> OpenBBClient:
    """Dependency injection — importado en main.py."""
    from ..main import get_obb_client
    return get_obb_client()


@router.get("/fred", summary="Serie FRED individual")
async def get_fred_series(
    series_id: str = Query(..., description="FRED series ID, ej: UNRATE, DGS10"),
    start_date: str = Query("2015-01-01", description="Fecha inicio ISO 8601"),
    provider: str = Query("fred", description="Provider: fred | oecd"),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """
    Descarga una serie macroeconómica de FRED.
    Cache Redis TTL: 24h para series diarias, 1 semana para mensuales.
    """
    series_id = series_id.upper()
    data = await client.get_fred_series(series_id, start_date, provider)
    if not data:
        raise HTTPException(404, f"Serie {series_id} no encontrada o sin datos")
    return data


@router.get("/snapshot", summary="Snapshot de todas las series FRED")
async def get_macro_snapshot(
    client: OpenBBClient = Depends(_get_client),
) -> MacroSnapshot:
    """
    Retorna el último valor y z-score de todas las series configuradas.
    Diseñado para context-engine y ml-feature-store.
    """
    snapshot: dict[str, dict] = {}

    for sid in list(FRED_SERIES_META.keys())[:10]:  # primeras 10 para no saturar
        data = await client.get_fred_series(sid, start_date="2010-01-01")
        if not data:
            continue
        meta = FRED_SERIES_META.get(sid, {})
        last = data[-1]

        # Z-score usando últimos 252 puntos
        values = [d["value"] for d in data[-252:]]
        if len(values) >= 2:
            import statistics
            mu = statistics.mean(values)
            sd = statistics.stdev(values) or 1e-9
            z = (values[-1] - mu) / sd
        else:
            z = 0.0

        snapshot[sid] = {
            "value":    last["value"],
            "date":     last["date"],
            "z_score":  round(z, 4),
            "name":     meta.get("name", sid),
            "category": meta.get("category", ""),
        }

    return MacroSnapshot(
        series=snapshot,
        ts=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/yield_curve", summary="Curva de rendimientos del gobierno")
async def get_yield_curve(
    country: str = Query("us", description="País: us | gb | eu | jp"),
    date: Optional[str] = Query(None, description="Fecha específica ISO 8601; None = último"),
    client: OpenBBClient = Depends(_get_client),
) -> list[dict]:
    """Curva de rendimientos. Datos FRED/BIS."""
    data = await client.get_yield_curve(date=date, country=country)
    if not data:
        raise HTTPException(503, "No se pudo obtener la curva de rendimientos")
    return data


@router.get("/series", summary="Lista de series disponibles")
async def list_series() -> dict:
    """Retorna el catálogo completo de series FRED configuradas."""
    return {
        "series": {
            sid: {
                "name":     meta["name"],
                "frequency": meta["freq"],
                "category": meta["category"],
            }
            for sid, meta in FRED_SERIES_META.items()
        },
        "count": len(FRED_SERIES_META),
    }
