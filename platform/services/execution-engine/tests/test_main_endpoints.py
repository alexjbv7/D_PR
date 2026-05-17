"""
FastAPI endpoint tests.

The lifespan is bypassed: we manually inject an AppState into ``app.state``
so we can test endpoints without spinning up asyncpg / Kafka / brokers.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from quant_shared.schemas.orders import (
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
)

from app.brokers.base import AccountInfo, BrokerError
from app.main import app, AppState
from app.repository import MemoryRepository
from app.routing import Router
from app.service import ExecutionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_state() -> AppState:
    state = AppState()

    alpaca = MagicMock()
    alpaca.venue       = "alpaca"
    alpaca.get_account = AsyncMock(return_value=AccountInfo(
        account_id="acc-1",
        venue="alpaca",
        equity=Decimal("100000"),
        cash=Decimal("80000"),
        is_paper=True,
        pnl_day=Decimal("250"),
        currency="USD",
    ))

    router = Router(default_equity="alpaca", default_crypto="binance")
    router.register(alpaca)

    repo = MemoryRepository()
    risk_gate = MagicMock()
    reconciler = MagicMock()
    reconciler._kill_switch_tripped     = False
    reconciler._consecutive_failures    = 0

    state.router     = router
    state.repository = repo
    state.risk_gate  = risk_gate
    state.reconciler = reconciler
    state.service    = ExecutionService(
        router=router, risk_gate=risk_gate, repository=repo,
    )
    return state


@pytest.fixture
def client():
    """TestClient with the lifespan bypassed."""
    state = _build_state()
    app.state.app_state = state
    with TestClient(app) as c:
        # TestClient.__enter__ runs the lifespan, which would also overwrite
        # app.state.app_state.  Restore our manual state after entering.
        app.state.app_state = state
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"]      == "ok"
    assert body["service"]     == "execution-engine"
    assert body["kill_switch"] is False
    assert "alpaca" in body["venues"]
    assert "counters" in body


# ---------------------------------------------------------------------------
# /api/positions
# ---------------------------------------------------------------------------

def test_positions_empty_initially(client):
    r = client.get("/api/positions")
    assert r.status_code == 200
    assert r.json() == {"count": 0, "positions": []}


def test_positions_returns_persisted(client):
    state: AppState = app.state.app_state
    pos = Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"),
        venue="binance",
    )
    import asyncio
    asyncio.run(state.repository.upsert_position(pos))

    r = client.get("/api/positions")
    body = r.json()
    assert body["count"] == 1
    assert body["positions"][0]["symbol"] == "BTCUSDT"


def test_positions_filters_by_venue(client):
    state: AppState = app.state.app_state
    import asyncio
    asyncio.run(state.repository.upsert_position(Position(
        symbol="AAPL", side=OrderSide.BUY,
        qty=Decimal("10"), avg_entry=Decimal("170"), venue="alpaca",
    )))
    asyncio.run(state.repository.upsert_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), avg_entry=Decimal("65000"), venue="binance",
    )))
    r = client.get("/api/positions?venue=alpaca")
    body = r.json()
    assert body["count"] == 1
    assert body["positions"][0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# /api/orders/recent
# ---------------------------------------------------------------------------

def test_orders_recent_empty(client):
    r = client.get("/api/orders/recent")
    assert r.status_code == 200
    assert r.json() == {"count": 0, "orders": []}


def test_orders_recent_limit_param(client):
    state: AppState = app.state.app_state
    import asyncio
    for i in range(5):
        r = OrderResult(
            symbol="BTCUSDT", side=OrderSide.BUY,
            status=OrderStatus.SUBMITTED, qty=Decimal("0.01"),
            venue="binance",
        )
        asyncio.run(state.repository.save_result(r))
    body = client.get("/api/orders/recent?limit=3").json()
    assert body["count"] == 3


# ---------------------------------------------------------------------------
# /api/account/{venue}
# ---------------------------------------------------------------------------

def test_account_known_venue(client):
    r = client.get("/api/account/alpaca")
    assert r.status_code == 200
    body = r.json()
    assert body["venue"]     == "alpaca"
    assert body["equity"]    == "100000"
    assert body["is_paper"]  is True


def test_account_unknown_venue_404(client):
    r = client.get("/api/account/kraken")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/kill_switch
# ---------------------------------------------------------------------------

def test_kill_switch_trip(client):
    r = client.post("/api/kill_switch/trip")
    assert r.status_code == 200
    assert r.json() == {"kill_switch": True}
    # state should be persistent across requests
    assert client.get("/health").json()["kill_switch"] is True


def test_kill_switch_reset(client):
    client.post("/api/kill_switch/trip")
    r = client.post("/api/kill_switch/reset")
    assert r.status_code == 200
    assert r.json() == {"kill_switch": False}


def test_kill_switch_bad_action(client):
    r = client.post("/api/kill_switch/explode")
    assert r.status_code == 400
