"""Client for Alpaca GET /v2/assets to build the equity universe."""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY_ID   = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET   = os.getenv("APCA_API_SECRET_KEY", "")

_HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


async def fetch_assets(status: str = "active") -> list[dict[str, Any]]:
    """
    Fetch all US equity assets with the given ``status`` from Alpaca /v2/assets.

    Parameters
    ----------
    status : str
        ``"active"`` or ``"inactive"``.

    Returns
    -------
    list[dict]
        Raw Alpaca asset dicts.
    """
    url = f"{ALPACA_BASE_URL}/v2/assets"
    params: dict[str, str] = {
        "status":      status,
        "asset_class": "us_equity",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params, headers=_HEADERS)
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json()
    logger.info("alpaca.assets.fetched status=%s count=%d", status, len(data))
    return data


async def fetch_all_assets() -> list[dict[str, Any]]:
    """Fetch active + inactive assets and merge into a single list."""
    active   = await fetch_assets("active")
    inactive = await fetch_assets("inactive")
    return active + inactive
