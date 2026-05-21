"""Unit tests for Alpaca corporate-actions fetcher — parsing and HTTP pagination.

Uses ``respx`` to mock HTTP; no live Alpaca calls.
"""
from __future__ import annotations

import logging
import sys
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
import pytest
import respx

UTC = timezone.utc

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.corporate_actions.alpaca_ca_fetcher import (  # noqa: E402
    ALPACA_BASE_URL,
    fetch_announcements,
    fetch_parsed_announcements,
    parse_announcement,
)

_CA_URL = f"{ALPACA_BASE_URL}/v2/corporate-actions/announcements"


# ---------------------------------------------------------------------------
# Helpers — mock Alpaca announcement payloads
# ---------------------------------------------------------------------------

def _split_ann(
    symbol: str = "AAPL",
    old_rate: str = "1",
    new_rate: str = "4",
    ca_type: str = "split",
    ex_date: str = "2020-08-31",
    ann_id: str = "ann-001",
) -> dict[str, Any]:
    return {
        "id": ann_id,
        "ca_type": ca_type,
        "symbol": symbol,
        "ex_date": ex_date,
        "old_rate": old_rate,
        "new_rate": new_rate,
    }


def _cash_div_ann(
    symbol: str = "AAPL",
    cash: str = "0.24",
    ex_date: str = "2024-03-15",
) -> dict[str, Any]:
    return {
        "id": "ann-cash-001",
        "ca_type": "cash_dividend",
        "symbol": symbol,
        "ex_date": ex_date,
        "cash": cash,
    }


def _stock_div_ann(
    symbol: str = "MSFT",
    stock_amount: str = "0.10",
    ex_date: str = "2024-06-01",
) -> dict[str, Any]:
    return {
        "id": "ann-stock-001",
        "ca_type": "stock_dividend",
        "symbol": symbol,
        "ex_date": ex_date,
        "stock_amount": stock_amount,
        "new_rate": stock_amount,
    }


# ---------------------------------------------------------------------------
# 1. Forward split 4:1
# ---------------------------------------------------------------------------

def test_parse_forward_split_4_to_1() -> None:
    raw = _split_ann(old_rate="1", new_rate="4", ca_type="split")
    parsed = parse_announcement(raw)

    assert parsed is not None
    assert parsed["symbol"] == "AAPL"
    assert parsed["ca_type"] == "forward_split"
    assert parsed["split_ratio"] == Decimal("4")
    assert parsed["split_from"] == Decimal("1")
    assert parsed["split_to"] == Decimal("4")
    assert parsed["ex_ts"] == datetime(2020, 8, 31, tzinfo=UTC)
    assert parsed["cash_amount"] is None


# ---------------------------------------------------------------------------
# 2. Reverse split 1:2
# ---------------------------------------------------------------------------

def test_parse_reverse_split_1_to_2() -> None:
    raw = _split_ann(old_rate="2", new_rate="1", ca_type="split")
    parsed = parse_announcement(raw)

    assert parsed is not None
    assert parsed["ca_type"] == "reverse_split"
    assert parsed["split_ratio"] == Decimal("0.5")
    assert parsed["split_to"] < parsed["split_from"]
    assert parsed["split_from"] == Decimal("2")
    assert parsed["split_to"] == Decimal("1")


# ---------------------------------------------------------------------------
# 3. Cash dividend
# ---------------------------------------------------------------------------

def test_parse_cash_dividend() -> None:
    raw = _cash_div_ann(cash="0.24")
    parsed = parse_announcement(raw)

    assert parsed is not None
    assert parsed["ca_type"] == "cash_dividend"
    assert parsed["cash_amount"] == Decimal("0.24")
    assert parsed["split_ratio"] is None
    assert parsed["ex_ts"] == datetime(2024, 3, 15, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 4. Stock dividend
# ---------------------------------------------------------------------------

def test_parse_stock_dividend() -> None:
    raw = _stock_div_ann(stock_amount="0.10")
    parsed = parse_announcement(raw)

    assert parsed is not None
    assert parsed["ca_type"] == "stock_dividend"
    assert parsed["stock_amount"] == Decimal("0.10")
    assert parsed["split_ratio"] is None


# ---------------------------------------------------------------------------
# 5. Pagination — next_page_token
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_pagination_follows_next_token() -> None:
    page1 = {
        "corporate_actions": [_split_ann(ann_id="p1", symbol="AAPL")],
        "next_page_token": "token-page-2",
    }
    page2 = {
        "corporate_actions": [_split_ann(ann_id="p2", symbol="MSFT", old_rate="1", new_rate="2")],
    }

    route = respx.get(_CA_URL)
    route.side_effect = [
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ]

    since = date(2024, 1, 1)
    rows = await fetch_announcements(since=since)

    assert len(rows) == 2
    assert route.call_count == 2
    # Second request must include page_token
    second_params = dict(route.calls[1].request.url.params)
    assert second_params.get("page_token") == "token-page-2"


# ---------------------------------------------------------------------------
# 6. Invalid payload — skip with WARN, batch continues
# ---------------------------------------------------------------------------

def test_invalid_payload_raises_or_skips(caplog: pytest.LogCaptureFixture) -> None:
    """Missing ex_date → parse returns None; valid row still parsed."""
    bad = {
        "id": "bad-001",
        "ca_type": "split",
        "symbol": "BADTICKER",
        "old_rate": "1",
        "new_rate": "4",
        # ex_date intentionally missing
    }
    good = _split_ann(symbol="GOOD", ann_id="good-001")

    with caplog.at_level(logging.WARNING):
        assert parse_announcement(bad) is None
        good_parsed = parse_announcement(good)

    assert good_parsed is not None
    assert good_parsed["symbol"] == "GOOD"

    assert any(
        "parse_skip" in r.message and "ex_date" in r.message
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


@pytest.mark.asyncio
async def test_fetch_parsed_skips_invalid_in_batch(caplog: pytest.LogCaptureFixture) -> None:
    """fetch_parsed_announcements processes full batch; one bad row skipped."""
    bad = {"id": "x", "ca_type": "cash_dividend", "symbol": "X", "cash": "1.0"}
    good = _cash_div_ann(symbol="VALID", cash="0.50")

    with respx.mock:
        respx.get(_CA_URL).mock(
            return_value=httpx.Response(200, json=[bad, good])
        )
        with caplog.at_level(logging.WARNING):
            parsed = await fetch_parsed_announcements(since=date(2024, 1, 1))

    assert len(parsed) == 1
    assert parsed[0]["symbol"] == "VALID"
    assert parsed[0]["cash_amount"] == Decimal("0.50")
