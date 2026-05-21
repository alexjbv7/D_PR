"""
Modelos Pydantic de request/response para el openbb-adapter.

Todos los modelos de respuesta son schemas aplanados — no re-exponen
la estructura interna de OBBject para mantener la interfaz estable
ante cambios de versión de OpenBB.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Crypto ────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    date:   str
    open:   Optional[float] = None
    high:   Optional[float] = None
    low:    Optional[float] = None
    close:  Optional[float] = None
    volume: Optional[float] = None


class FundingRate(BaseModel):
    date:         str
    symbol:       str
    funding_rate: Optional[float] = None


# ── Macro ─────────────────────────────────────────────────────────────

class MacroPoint(BaseModel):
    date:  str
    value: float


class YieldCurvePoint(BaseModel):
    maturity: str
    rate:     float


class MacroSnapshot(BaseModel):
    """Snapshot de todas las series FRED cacheadas."""
    series: dict[str, dict]
    ts:     str


# ── Derivados ────────────────────────────────────────────────────────

class OptionsContract(BaseModel):
    symbol:           str
    expiration:       Optional[str]   = None
    strike:           Optional[float] = None
    option_type:      Optional[str]   = None  # "call" | "put"
    implied_volatility: Optional[float] = None
    delta:            Optional[float] = None
    gamma:            Optional[float] = None
    theta:            Optional[float] = None
    open_interest:    Optional[int]   = None
    volume:           Optional[int]   = None


class PutCallRatioResponse(BaseModel):
    symbol:          str
    put_call_ratio:  Optional[float]
    puts:            int
    calls:           int
    ts:              str


class FuturesCurvePoint(BaseModel):
    expiration: str
    price:      float
    basis_bps:  Optional[float] = None


# ── Regulators ───────────────────────────────────────────────────────

class SECFiling(BaseModel):
    filed_at:    Optional[str] = None
    symbol:      Optional[str] = None
    form_type:   Optional[str] = None
    filing_url:  Optional[str] = None
    description: Optional[str] = None


class COTReport(BaseModel):
    date:              str
    symbol:            Optional[str] = None
    net_speculative:   Optional[float] = None
    net_commercial:    Optional[float] = None
    long_speculative:  Optional[float] = None
    short_speculative: Optional[float] = None


# ── News ──────────────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    headline:  str
    url:       str
    date:      str
    source:    str
    symbols:   list[str] = Field(default_factory=list)
    sentiment: Optional[float] = None


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    service:       str = "openbb-adapter"
    openbb_ready:  bool
    cache_ready:   bool
    ts:            str
