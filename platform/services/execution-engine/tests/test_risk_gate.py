"""
Tests for RiskGate.

Each test populates a MemoryRepository with the relevant positions, builds
a fresh OrderIntent + AccountInfo, and asserts on the RiskDecision.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
)

from app.brokers.base import AccountInfo
from app.repository import MemoryRepository
from app.risk_gate import RiskConfig, RiskGate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo():
    return MemoryRepository()


@pytest.fixture
def paper_account():
    return AccountInfo(
        account_id="acc",
        venue="binance",
        equity=Decimal("100000"),
        cash=Decimal("80000"),
        pnl_day=Decimal("0"),
        is_paper=True,
    )


@pytest.fixture
def live_account():
    return AccountInfo(
        account_id="acc",
        venue="binance",
        equity=Decimal("100000"),
        cash=Decimal("80000"),
        is_paper=False,
    )


def _intent(qty="0.05", price="65000", venue="binance", symbol="BTCUSDT") -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=Decimal(qty),
        order_type=OrderType.LIMIT,
        limit_price=Decimal(price),
        venue=venue,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_out_of_range():
    with pytest.raises(ValueError, match="per_symbol_cap_pct"):
        RiskConfig(per_symbol_cap_pct=1.5)


def test_config_defaults_sensible():
    cfg = RiskConfig()
    assert 0 < cfg.per_symbol_cap_pct < cfg.per_venue_cap_pct
    assert cfg.require_paper is True


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_approves_small_first_trade(repo, paper_account):
    gate   = RiskGate(RiskConfig(), repo)
    intent = _intent(qty="0.01")          # notional 650 << 5 % of 100k = 5000

    decision = await gate.evaluate(intent, paper_account)
    assert decision.approved is True


async def test_decision_persisted_on_approval(repo, paper_account):
    gate   = RiskGate(RiskConfig(), repo)
    intent = _intent(qty="0.01")

    await gate.evaluate(intent, paper_account)

    stored = await repo.get_intent(intent.intent_id)
    assert stored is not None


# ---------------------------------------------------------------------------
# Live blocker
# ---------------------------------------------------------------------------

async def test_rejects_live_account_when_require_paper(repo, live_account):
    gate     = RiskGate(RiskConfig(require_paper=True), repo)
    intent   = _intent(qty="0.001")
    decision = await gate.evaluate(intent, live_account)

    assert decision.approved is False
    assert decision.breach   == "require_paper"


async def test_allows_live_when_require_paper_false(repo, live_account):
    gate     = RiskGate(RiskConfig(require_paper=False), repo)
    intent   = _intent(qty="0.001")
    decision = await gate.evaluate(intent, live_account)
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Daily DD
# ---------------------------------------------------------------------------

async def test_rejects_when_daily_dd_breached(repo, paper_account):
    paper_account.pnl_day = Decimal("-4000")          # -4 % of 100k
    gate     = RiskGate(RiskConfig(daily_dd_kill_pct=0.03), repo)
    intent   = _intent(qty="0.001")
    decision = await gate.evaluate(intent, paper_account)

    assert decision.approved is False
    assert decision.breach   == "daily_dd"


async def test_allows_when_dd_within_threshold(repo, paper_account):
    paper_account.pnl_day = Decimal("-2000")          # -2 %, threshold -3 %
    gate     = RiskGate(RiskConfig(daily_dd_kill_pct=0.03), repo)
    decision = await gate.evaluate(_intent(qty="0.001"), paper_account)
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Per-symbol cap
# ---------------------------------------------------------------------------

async def test_rejects_when_symbol_cap_breached(repo, paper_account):
    # 5 % of 100k = 5000 cap.  Existing 4500 + new 1000 = 5500 → breach
    await repo.upsert_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.075"), avg_entry=Decimal("60000"),     # notional 4500
        venue="binance",
    ))
    gate     = RiskGate(RiskConfig(per_symbol_cap_pct=0.05), repo)
    intent   = _intent(qty="0.02", price="50000")             # notional 1000
    decision = await gate.evaluate(intent, paper_account)

    assert decision.approved is False
    assert decision.breach   == "per_symbol_cap"


async def test_other_symbol_not_counted_against_cap(repo, paper_account):
    await repo.upsert_position(Position(
        symbol="ETHUSDT", side=OrderSide.BUY,
        qty=Decimal("1"), avg_entry=Decimal("5000"),          # notional 5000
        venue="binance",
    ))
    gate     = RiskGate(RiskConfig(per_symbol_cap_pct=0.05), repo)
    intent   = _intent(symbol="BTCUSDT", qty="0.01")          # notional 650
    decision = await gate.evaluate(intent, paper_account)
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Per-venue cap
# ---------------------------------------------------------------------------

async def test_rejects_when_venue_cap_breached(repo, paper_account):
    # 50 % cap = 50_000.  Stack two positions to 49500 on binance, then add 1000.
    await repo.upsert_position(Position(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.5"), avg_entry=Decimal("60000"),       # 30000
        venue="binance",
    ))
    await repo.upsert_position(Position(
        symbol="ETHUSDT", side=OrderSide.BUY,
        qty=Decimal("4"), avg_entry=Decimal("4875"),           # 19500
        venue="binance",
    ))
    gate   = RiskGate(RiskConfig(per_venue_cap_pct=0.50,
                                  per_symbol_cap_pct=1.0), repo)
    intent = _intent(symbol="SOLUSDT", qty="6", price="200")  # notional 1200
    decision = await gate.evaluate(intent, paper_account)

    assert decision.approved is False
    assert decision.breach   == "per_venue_cap"


# ---------------------------------------------------------------------------
# Cash buffer
# ---------------------------------------------------------------------------

async def test_rejects_when_cash_buffer_breached(repo, paper_account):
    # buffer = 10 % of 100k = 10000.  Cash = 80000.  Allowed spend = 70000.
    gate     = RiskGate(RiskConfig(min_cash_buffer_pct=0.10,
                                    per_symbol_cap_pct=1.0,
                                    per_venue_cap_pct=1.0), repo)
    intent   = _intent(qty="2", price="40000")                 # notional 80000
    decision = await gate.evaluate(intent, paper_account)

    assert decision.approved is False
    assert decision.breach   == "cash_buffer"


async def test_sell_does_not_consume_cash_buffer(repo, paper_account):
    gate = RiskGate(RiskConfig(min_cash_buffer_pct=0.10,
                                per_symbol_cap_pct=1.0,
                                per_venue_cap_pct=1.0), repo)
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.SELL,
        qty=Decimal("2"), order_type=OrderType.LIMIT,
        limit_price=Decimal("40000"), venue="binance",
    )
    decision = await gate.evaluate(intent, paper_account)
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Market order (no limit_price) — notional cannot be pre-validated → passes
# ---------------------------------------------------------------------------

async def test_market_order_skips_notional_checks(repo, paper_account):
    gate = RiskGate(RiskConfig(), repo)
    intent = OrderIntent(
        symbol="BTCUSDT", side=OrderSide.BUY,
        qty=Decimal("0.01"), order_type=OrderType.MARKET,
        venue="binance",       # no limit_price
    )
    decision = await gate.evaluate(intent, paper_account)
    assert decision.approved is True


# ---------------------------------------------------------------------------
# Kill switch (P1-002)
# ---------------------------------------------------------------------------

async def test_kill_switch_blocks_all_intents(repo, paper_account):
    """trip_kill_switch() must be checked before every other rule."""
    gate   = RiskGate(RiskConfig(), repo)
    gate.trip_kill_switch()
    intent = _intent(qty="0.001")          # would otherwise pass all checks

    decision = await gate.evaluate(intent, paper_account)

    assert decision.approved is False
    assert decision.breach   == "kill_switch"


async def test_kill_switch_idempotent_trip(repo, paper_account):
    gate = RiskGate(RiskConfig(), repo)
    gate.trip_kill_switch()
    gate.trip_kill_switch()                # second trip must not raise
    assert gate._kill_switch_active is True


async def test_kill_switch_reset_re_enables_gate(repo, paper_account):
    gate   = RiskGate(RiskConfig(), repo)
    gate.trip_kill_switch()
    gate.reset_kill_switch()
    intent = _intent(qty="0.001")

    decision = await gate.evaluate(intent, paper_account)

    assert decision.approved is True


async def test_kill_switch_not_active_by_default(repo):
    gate = RiskGate(RiskConfig(), repo)
    assert gate._kill_switch_active is False
