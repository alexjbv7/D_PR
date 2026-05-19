"""Unit tests for PDTTracker (Semana 6)."""
from __future__ import annotations

from datetime import datetime, time, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from quant_shared.calendar import market_calendar
from quant_shared.schemas.orders import OrderSide

from app._pdt.pdt_tracker import PDTTracker
from app.repository import MemoryRepository

UTC = timezone.utc
_ET = ZoneInfo("America/New_York")


def now_utc() -> datetime:
    return datetime(2026, 5, 19, 14, 0, tzinfo=UTC)


@pytest.fixture
def repo() -> MemoryRepository:
    return MemoryRepository()


@pytest.fixture
def tracker(repo: MemoryRepository) -> PDTTracker:
    return PDTTracker(repo)


async def _insert_action(
    repo: MemoryRepository,
    account_id: str,
    symbol: str,
    side: OrderSide,
    trade_date_et,
) -> None:
    ts = datetime.combine(trade_date_et, time(12, 0), tzinfo=_ET).astimezone(UTC)
    await repo.record_position_action(
        account_id=account_id,
        symbol=symbol,
        side=side,
        qty=Decimal("1"),
        notional=None,
        fill_id=f"fill-{symbol}-{side.value}-{trade_date_et}",
        ts_utc=ts,
    )


async def _insert_day_trade(
    repo: MemoryRepository,
    account_id: str,
    symbol: str,
    trading_days_ago: int,
) -> None:
    dates = market_calendar.last_n_trading_dates("AAPL", now_utc(), n=trading_days_ago + 2)
    if len(dates) < trading_days_ago:
        pytest.skip("not enough trading history in calendar cache")
    trade_date = dates[-trading_days_ago]
    await _insert_action(repo, account_id, symbol, OrderSide.BUY, trade_date)
    await _insert_action(repo, account_id, symbol, OrderSide.SELL, trade_date)


async def test_pdt_no_history_returns_under_limit(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    decision = await tracker.check("acc1", "AAPL", Decimal("25000"), now_utc())
    assert decision.blocked is False
    assert decision.count_last_5d == 0


async def test_pdt_3_day_trades_in_5d_blocks_next(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    await _insert_day_trade(repo, "acc1", "AAPL", trading_days_ago=4)
    await _insert_day_trade(repo, "acc1", "MSFT", trading_days_ago=3)
    await _insert_day_trade(repo, "acc1", "GOOG", trading_days_ago=1)
    decision = await tracker.check("acc1", "TSLA", Decimal("25000"), now_utc())
    assert decision.blocked is True
    assert decision.count_last_5d == 3


async def test_pdt_2_day_trades_does_not_block(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    await _insert_day_trade(repo, "acc1", "AAPL", trading_days_ago=4)
    await _insert_day_trade(repo, "acc1", "MSFT", trading_days_ago=2)
    decision = await tracker.check("acc1", "TSLA", Decimal("25000"), now_utc())
    assert decision.blocked is False
    assert decision.count_last_5d == 2


async def test_pdt_old_day_trade_falls_out_of_window(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    await _insert_day_trade(repo, "acc1", "AAPL", trading_days_ago=7)
    await _insert_day_trade(repo, "acc1", "MSFT", trading_days_ago=2)
    decision = await tracker.check("acc1", "TSLA", Decimal("25000"), now_utc())
    assert decision.count_last_5d == 1


async def test_pdt_skips_when_equity_above_threshold(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    for i in range(5):
        await _insert_day_trade(repo, "acc1", f"SYM{i}", trading_days_ago=i + 1)
    decision = await tracker.check("acc1", "TSLA", Decimal("50000"), now_utc())
    assert decision.blocked is False
    assert decision.reason == "above_threshold"


async def test_pdt_skips_crypto(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    for i in range(5):
        await _insert_day_trade(repo, "acc1", "AAPL", trading_days_ago=i + 1)
    decision = await tracker.check("acc1", "BTCUSDT", Decimal("25000"), now_utc())
    assert decision.blocked is False
    assert decision.reason == "not_equity"


async def test_pdt_same_symbol_multiple_fills_same_day_counts_once(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    dates = market_calendar.last_n_trading_dates("AAPL", now_utc(), n=1)
    assert dates
    today = dates[-1]
    for i in range(5):
        await _insert_action(repo, "acc1", "AAPL", OrderSide.BUY, today)
    for i in range(5):
        await _insert_action(repo, "acc1", "AAPL", OrderSide.SELL, today)
    await _insert_day_trade(repo, "acc1", "MSFT", trading_days_ago=2)
    await _insert_day_trade(repo, "acc1", "GOOG", trading_days_ago=3)
    decision = await tracker.check("acc1", "TSLA", Decimal("25000"), now_utc())
    assert decision.count_last_5d == 3


async def test_pdt_equity_exactly_at_buffer_threshold_blocks(
    tracker: PDTTracker, repo: MemoryRepository,
) -> None:
    for i in range(3):
        await _insert_day_trade(repo, "acc1", f"SYM{i}", trading_days_ago=i + 1)
    decision = await tracker.check("acc1", "TSLA", Decimal("25999"), now_utc())
    assert decision.blocked is True
