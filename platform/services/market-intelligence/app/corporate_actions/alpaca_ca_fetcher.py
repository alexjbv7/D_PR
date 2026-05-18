"""Client for Alpaca GET /v2/corporate-actions/announcements.

Endpoint reference:
  https://docs.alpaca.markets/reference/get-v2-corporate-actions-announcements
  GET /v2/corporate-actions/announcements
    ?ca_types=forward_split,reverse_split,stock_dividend,cash_dividend,merger,spinoff,name_change
    &since=YYYY-MM-DD
    &until=YYYY-MM-DD   (optional)
    &symbol=AAPL        (optional, omit for all)
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_KEY_ID   = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET   = os.getenv("APCA_API_SECRET_KEY", "")
UTC = timezone.utc

_HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

# All CA types the Alpaca API accepts
_CA_TYPES = (
    "forward_split,reverse_split,stock_dividend,cash_dividend,"
    "merger,spinoff,name_change"
)


async def fetch_announcements(
    since: date | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch corporate action announcements from Alpaca.

    Parameters
    ----------
    since : date, optional
        Look back from this date.  Defaults to 7 days ago (overlap window
        so late-published Alpaca events are captured).
    symbol : str, optional
        Filter by symbol.  ``None`` fetches all symbols.

    Returns
    -------
    list[dict]
        Raw Alpaca announcement dicts.
    """
    if since is None:
        since = (datetime.now(tz=UTC) - timedelta(days=7)).date()

    params: dict[str, str] = {
        "ca_types": _CA_TYPES,
        "since":    since.isoformat(),
    }
    if symbol:
        params["symbol"] = symbol

    url = f"{ALPACA_BASE_URL}/v2/corporate-actions/announcements"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params, headers=_HEADERS)
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json()

    logger.info("alpaca.ca.fetched since=%s count=%d", since, len(data))
    return data
