"""RiskGate integration — market calendar (Semana 3)."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from quant_shared.schemas.orders import OrderIntent, OrderSide, OrderType

from app.brokers.base import AccountInfo
from app.repository import MemoryRepository
from app.risk_gate import RiskConfig, RiskGate

UTC = timezone.utc


@pytest.fixture
def repo():
    return MemoryRepository()


@pytest.fixture
def paper_account():
    return AccountInfo(
        account_id="acc",
        venue="alpaca",
        equity=Decimal("100000"),
        cash=Decimal("5000"),
        pnl_day=Decimal("0"),
        is_paper=True,
    )


def _cfg() -> RiskConfig:
    return RiskConfig(
        require_paper=False,
        per_symbol_cap_pct=1.0,
        per_venue_cap_pct=1.0,
        min_cash_buffer_pct=0.01,
    )


def _intent(
    symbol: str,
    ts: datetime,
    qty: str = "10",
    price: str = "100",
    venue: str = "alpaca",
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=Decimal(qty),
        order_type=OrderType.LIMIT,
        limit_price=Decimal(price),
        venue=venue,
        ts=ts,
    )


async def test_equity_approved_during_rth(repo, paper_account):
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 19, 14, 0, tzinfo=UTC)  # Tuesday RTH EDT
    decision = await gate.evaluate(
        _intent("AAPL", ts, qty="1", price="100"),
        paper_account,
    )
    assert decision.approved is True


async def test_equity_rejected_post_rth(repo, paper_account):
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 19, 21, 0, tzinfo=UTC)
    decision = await gate.evaluate(_intent("AAPL", ts), paper_account)
    assert decision.approved is False
    assert decision.breach == "market_closed"


async def test_equity_rejected_weekend(repo, paper_account):
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 16, 15, 0, tzinfo=UTC)  # Saturday
    decision = await gate.evaluate(_intent("AAPL", ts), paper_account)
    assert decision.approved is False
    assert decision.breach == "market_closed"


async def test_crypto_approved_weekend(repo, paper_account):
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 16, 3, 0, tzinfo=UTC)
    decision = await gate.evaluate(
        _intent("BTCUSDT", ts, qty="0.01", price="65000", venue="binance"),
        paper_account,
    )
    assert decision.approved is True


async def test_market_closed_before_cash_buffer(repo, paper_account):
    """Closed market must short-circuit before cash_buffer breach."""
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 16, 15, 0, tzinfo=UTC)  # Saturday — market closed
    intent = _intent("AAPL", ts, qty="100", price="1000")  # would breach 50% buffer
    decision = await gate.evaluate(intent, paper_account)
    assert decision.approved is False
    assert decision.breach == "market_closed"
