"""Client for Alpaca GET /v2/corporate-actions/announcements.

Endpoint reference:
  https://docs.alpaca.markets/reference/get-v2-corporate-actions-announcements
  GET /v2/corporate-actions/announcements
    ?ca_types=forward_split,reverse_split,stock_dividend,cash_dividend,merger,spinoff,name_change
    &since=YYYY-MM-DD
    &until=YYYY-MM-DD   (optional)
    &symbol=AAPL        (optional, omit for all)
    &page_token=...      (pagination)
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

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

_CA_TYPE_ALIASES: dict[str, str] = {
    "forward_split":  "forward_split",
    "reverse_split":  "reverse_split",
    "stock_dividend": "stock_dividend",
    "cash_dividend":  "cash_dividend",
    "dividend":       "cash_dividend",
    "cash":           "cash_dividend",
    "merger":         "merger",
    "spinoff":        "spinoff",
    "name_change":    "name_change",
    "split":          "forward_split",  # refined below when rates present
    "stock_split":    "forward_split",
}


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _parse_ex_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    try:
        d = date.fromisoformat(str(value))
        return datetime(d.year, d.month, d.day, tzinfo=UTC)
    except ValueError:
        return None


def _classify_split(
    raw_type: str,
    old_rate: Optional[Decimal],
    new_rate: Optional[Decimal],
) -> str:
    """Return ``forward_split`` or ``reverse_split`` from split rates."""
    if old_rate is not None and new_rate is not None and new_rate < old_rate:
        return "reverse_split"
    return "forward_split"


def _normalize_ca_type(
    raw_type: str,
    old_rate: Optional[Decimal],
    new_rate: Optional[Decimal],
) -> str:
    key = raw_type.lower().replace(" ", "_")
    if key in ("split", "stock_split") or "split" in key:
        return _classify_split(raw_type, old_rate, new_rate)
    return _CA_TYPE_ALIASES.get(key, key)


def parse_announcement(ann: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Parse a raw Alpaca announcement dict into a normalized structure.

    Returns ``None`` and logs a WARNING when required fields are missing
    (malformed payload) — callers should skip and continue the batch.

    Normalized keys
    ---------------
    symbol, ca_type, ex_ts (UTC), split_ratio, split_from, split_to,
    cash_amount, stock_amount, alpaca_id, raw.
    """
    symbol = ann.get("symbol")
    if not symbol:
        logger.warning(
            "alpaca.ca.parse_skip reason=missing_symbol ann_id=%s",
            ann.get("id"),
        )
        return None

    ex_ts = _parse_ex_ts(ann.get("ex_date"))
    if ex_ts is None:
        logger.warning(
            "alpaca.ca.parse_skip reason=missing_ex_date symbol=%s ann_id=%s",
            symbol,
            ann.get("id"),
        )
        return None

    raw_ca_type = str(ann.get("ca_type", ""))
    old_rate = _to_decimal(ann.get("old_rate"))
    new_rate = _to_decimal(ann.get("new_rate"))
    cash_amount = _to_decimal(ann.get("cash"))

    ca_type = _normalize_ca_type(raw_ca_type, old_rate, new_rate)

    split_ratio: Optional[Decimal] = None
    split_from: Optional[Decimal] = None
    split_to: Optional[Decimal] = None
    stock_amount: Optional[Decimal] = None

    if ca_type in ("forward_split", "reverse_split"):
        if old_rate is None or new_rate is None or old_rate == 0:
            logger.warning(
                "alpaca.ca.parse_skip reason=invalid_split_rates symbol=%s",
                symbol,
            )
            return None
        split_from = old_rate
        split_to = new_rate
        split_ratio = new_rate / old_rate
    elif ca_type == "stock_dividend":
        stock_amount = _to_decimal(ann.get("stock_amount")) or new_rate
        if stock_amount is None:
            logger.warning(
                "alpaca.ca.parse_skip reason=missing_stock_amount symbol=%s",
                symbol,
            )
            return None
    elif ca_type == "cash_dividend":
        if cash_amount is None:
            logger.warning(
                "alpaca.ca.parse_skip reason=missing_cash_amount symbol=%s",
                symbol,
            )
            return None

    return {
        "symbol":       str(symbol),
        "ca_type":      ca_type,
        "ex_ts":        ex_ts,
        "split_ratio":  split_ratio,
        "split_from":   split_from,
        "split_to":     split_to,
        "cash_amount":  cash_amount if ca_type == "cash_dividend" else None,
        "stock_amount": stock_amount,
        "alpaca_id":    str(ann.get("id", "")) or None,
        "raw":          ann,
    }


def _extract_page(body: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Return (items, next_page_token) from Alpaca JSON body."""
    if isinstance(body, list):
        return body, None
    if isinstance(body, dict):
        items = body.get("corporate_actions") or body.get("announcements") or []
        token = body.get("next_page_token") or None
        return list(items), token
    return [], None


async def fetch_announcements(
    since: date | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch corporate action announcements from Alpaca (raw payloads).

    Follows ``next_page_token`` until exhausted.

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

    url = f"{ALPACA_BASE_URL}/v2/corporate-actions/announcements"
    all_rows: list[dict[str, Any]] = []
    page_token: Optional[str] = None

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params: dict[str, str] = {
                "ca_types": _CA_TYPES,
                "since":    since.isoformat(),
            }
            if symbol:
                params["symbol"] = symbol
            if page_token:
                params["page_token"] = page_token

            resp = await client.get(url, params=params, headers=_HEADERS)
            resp.raise_for_status()
            page, page_token = _extract_page(resp.json())
            all_rows.extend(page)

            if not page_token:
                break

    logger.info("alpaca.ca.fetched since=%s count=%d", since, len(all_rows))
    return all_rows


async def fetch_parsed_announcements(
    since: date | None = None,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch announcements and parse each; skip malformed rows with WARN.

    Returns only successfully parsed dicts (see :func:`parse_announcement`).
    """
    raw_list = await fetch_announcements(since=since, symbol=symbol)
    parsed: list[dict[str, Any]] = []
    for ann in raw_list:
        item = parse_announcement(ann)
        if item is not None:
            parsed.append(item)
    return parsed
