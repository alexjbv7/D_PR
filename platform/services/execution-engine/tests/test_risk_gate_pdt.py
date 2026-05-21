"""RiskGate integration — PDT + extended hours (Semana 6)."""
from __future__ import annotations

from datetime import datetime, time, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_shared.calendar import market_calendar
from quant_shared.schemas.orders import OrderIntent, OrderSide, OrderType

from app.brokers.base import AccountInfo
from app.repository import MemoryRepository
from app.risk_gate import RiskConfig, RiskGate

UTC = timezone.utc
_ET = ZoneInfo("America/New_York")


def now_utc() -> datetime:
    return datetime(2026, 5, 19, 14, 0, tzinfo=UTC)


@pytest.fixture
def repo() -> MemoryRepository:
    return MemoryRepository()


@pytest.fixture
def paper_account() -> AccountInfo:
    return AccountInfo(
        account_id="acc-pdt",
        venue="alpaca",
        equity=Decimal("25000"),
        cash=Decimal("20000"),
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
    symbol: str = "AAPL",
    *,
    ts: datetime | None = None,
    order_type: OrderType = OrderType.LIMIT,
    extended_hours: bool = False,
    qty: str = "1",
    price: str = "100",
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=Decimal(qty),
        order_type=order_type,
        limit_price=Decimal(price) if order_type != OrderType.MARKET else None,
        extended_hours=extended_hours,
        venue="alpaca",
        ts=ts or now_utc(),
    )


async def _seed_day_trades(repo: MemoryRepository, account_id: str, n: int) -> None:
    for i in range(n):
        dates = market_calendar.last_n_trading_dates("AAPL", now_utc(), n=i + 2)
        d = dates[-(i + 1)]
        ts = datetime.combine(d, time(12, 0), tzinfo=_ET).astimezone(UTC)
        await repo.record_position_action(
            account_id, f"SYM{i}", OrderSide.BUY, Decimal("1"), None, f"b{i}", ts,
        )
        await repo.record_position_action(
            account_id, f"SYM{i}", OrderSide.SELL, Decimal("1"), None, f"s{i}", ts,
        )


async def test_pdt_blocks_after_3_in_5d(repo, paper_account) -> None:
    await _seed_day_trades(repo, paper_account.account_id, 3)
    gate = RiskGate(_cfg(), repo)
    decision = await gate.evaluate(_intent("TSLA"), paper_account)
    assert decision.approved is False
    assert decision.breach == "pdt_rule"


async def test_pdt_skips_crypto(repo, paper_account) -> None:
    await _seed_day_trades(repo, paper_account.account_id, 5)
    gate = RiskGate(_cfg(), repo)
    decision = await gate.evaluate(
        _intent("BTCUSDT", ts=now_utc(), qty="0.01", price="65000"),
        paper_account,
    )
    assert decision.approved is True


async def test_extended_hours_blocks_market(repo, paper_account) -> None:
    gate = RiskGate(_cfg(), repo)
    decision = await gate.evaluate(
        _intent(order_type=OrderType.MARKET, extended_hours=True, qty="1"),
        paper_account,
    )
    assert decision.approved is False
    assert decision.breach == "extended_hours_requires_limit"


async def test_extended_hours_limit_passes(repo, paper_account) -> None:
    gate = RiskGate(_cfg(), repo)
    decision = await gate.evaluate(
        _intent(order_type=OrderType.LIMIT, extended_hours=True),
        paper_account,
    )
    assert decision.approved is True


async def test_extended_hours_limit_passes_pre_market(repo, paper_account) -> None:
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)  # 08:00 ET
    decision = await gate.evaluate(
        _intent(order_type=OrderType.LIMIT, extended_hours=True, ts=ts),
        paper_account,
    )
    assert decision.approved is True


async def test_extended_hours_limit_blocks_overnight(repo, paper_account) -> None:
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 20, 2, 30, tzinfo=UTC)  # 22:30 ET previous day
    decision = await gate.evaluate(
        _intent(order_type=OrderType.LIMIT, extended_hours=True, ts=ts),
        paper_account,
    )
    assert decision.approved is False
    assert decision.breach == "market_closed"


async def test_market_closed_before_pdt(repo, paper_account) -> None:
    """Weekend equity intent → market_closed, not pdt_rule."""
    await _seed_day_trades(repo, paper_account.account_id, 3)
    gate = RiskGate(_cfg(), repo)
    ts = datetime(2026, 5, 16, 15, 0, tzinfo=UTC)
    decision = await gate.evaluate(_intent(ts=ts), paper_account)
    assert decision.approved is False
    assert decision.breach == "market_closed"
