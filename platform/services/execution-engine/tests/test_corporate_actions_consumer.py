"""E2E tests for the corporate actions Kafka consumer.

All tests use MemoryRepository — no database required.

Scenarios
---------
1. Forward split 4:1 on AAPL:  qty=10  → qty=40,  avg_entry=100 → 25
2. Forward split 2:1 on AAPL:  qty=10  → qty=20,  avg_entry=100 → 50
3. Reverse split 1:2 on AAPL:  qty=10  → qty=5,   avg_entry=100 → 200
4. Reverse split 1:2 odd qty:  qty=11  → qty=5.5  WARN + residual preserved
5. Stock dividend 10%:          qty=10  → qty=11,  avg_entry=100 → ~90.90
6. Cash dividend:               no change to position
7. Merger rename:               symbol updated
8. Idempotence:                 second application of same ca_id is no-op
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pytest

from quant_shared.schemas.events import CorporateActionEvent
from quant_shared.schemas.orders import OrderSide, Position

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Import consumer and repo from execution-engine
# ---------------------------------------------------------------------------
import sys, os
_EE = os.path.join(os.path.dirname(__file__), "..")
if _EE not in sys.path:
    sys.path.insert(0, _EE)

from app.corporate_actions_consumer import apply_to_positions, _adjust_position  # noqa: E402
from app.repository import MemoryRepository  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo() -> MemoryRepository:
    return MemoryRepository()


def _pos(
    symbol: str = "AAPL",
    qty: str = "10",
    avg_entry: str = "100",
    venue: str = "alpaca",
) -> Position:
    return Position(
        symbol=symbol,
        side=OrderSide.BUY,
        qty=Decimal(qty),
        avg_entry=Decimal(avg_entry),
        venue=venue,
    )


def _split_event(
    symbol: str = "AAPL",
    ca_type: str = "forward_split",
    split_from: str = "1",
    split_to: str = "4",
    ca_id: str = "ca-test-001",
) -> CorporateActionEvent:
    ratio = Decimal(split_to) / Decimal(split_from)
    return CorporateActionEvent(
        ca_id=ca_id,
        symbol=symbol,
        ca_type=ca_type,  # type: ignore[arg-type]
        ex_ts=datetime(2020, 8, 31, tzinfo=UTC),
        split_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Forward split tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forward_split_4_1(repo: MemoryRepository):
    """AAPL qty=10 avg_entry=100 + 4:1 split → qty=40 avg_entry=25."""
    pos = _pos(qty="10", avg_entry="100")
    await repo.upsert_position(pos)

    event = _split_event(split_from="1", split_to="4", ca_id="ca-fwd-4-1")
    await apply_to_positions(event, repo)

    positions = await repo.get_open_positions_for_symbol("AAPL")
    assert len(positions) == 1
    updated = positions[0]
    assert updated.qty == Decimal("40")
    assert updated.avg_entry == Decimal("25")


@pytest.mark.asyncio
async def test_forward_split_2_1(repo: MemoryRepository):
    """AAPL qty=10 avg_entry=100 + 2:1 split → qty=20 avg_entry=50."""
    pos = _pos(qty="10", avg_entry="100")
    await repo.upsert_position(pos)

    event = _split_event(split_from="1", split_to="2", ca_id="ca-fwd-2-1")
    await apply_to_positions(event, repo)

    positions = await repo.get_open_positions_for_symbol("AAPL")
    updated = positions[0]
    assert updated.qty       == Decimal("20")
    assert updated.avg_entry == Decimal("50")


# ---------------------------------------------------------------------------
# Reverse split tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reverse_split_1_2(repo: MemoryRepository):
    """AAPL qty=10 avg_entry=100 + 1:2 reverse → qty=5 avg_entry=200."""
    pos = _pos(qty="10", avg_entry="100")
    await repo.upsert_position(pos)

    event = _split_event(ca_type="reverse_split", split_from="2", split_to="1", ca_id="ca-rev-1-2")
    await apply_to_positions(event, repo)

    positions = await repo.get_open_positions_for_symbol("AAPL")
    updated = positions[0]
    assert updated.qty       == Decimal("5")
    assert updated.avg_entry == Decimal("200")


@pytest.mark.asyncio
async def test_reverse_split_fractional_residual_warns(
    repo: MemoryRepository,
    capsys: pytest.CaptureFixture,
):
    """Reverse split 1:2 with odd qty=11 → qty=5.5, WARN logged."""
    pos = _pos(qty="11", avg_entry="100")
    await repo.upsert_position(pos)

    event = _split_event(ca_type="reverse_split", split_from="2", split_to="1", ca_id="ca-rev-frac")
    await apply_to_positions(event, repo)

    positions = await repo.get_open_positions_for_symbol("AAPL")
    updated = positions[0]
    # qty = 11 * 0.5 = 5.5
    assert updated.qty == Decimal("5.5")
    # avg_entry = 100 / 0.5 = 200
    assert updated.avg_entry == Decimal("200")

    # structlog writes to stderr; check it contains the warning keyword
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "fractional" in output.lower(), (
        f"Expected a warning about fractional residual in structlog output. Got:\n{output}"
    )


# ---------------------------------------------------------------------------
# Stock dividend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stock_dividend_10pct(repo: MemoryRepository):
    """Stock dividend 10%: qty*1.10, avg_entry/1.10."""
    pos = _pos(qty="10", avg_entry="100")
    await repo.upsert_position(pos)

    event = CorporateActionEvent(
        ca_id="ca-stock-div-001",
        symbol="AAPL",
        ca_type="stock_dividend",
        ex_ts=datetime(2024, 3, 15, tzinfo=UTC),
        stock_amount=Decimal("0.10"),
    )
    await apply_to_positions(event, repo)

    positions = await repo.get_open_positions_for_symbol("AAPL")
    updated = positions[0]
    assert updated.qty == Decimal("10") * Decimal("1.10")
    expected_avg = Decimal("100") / Decimal("1.10")
    # Allow small floating-point tolerance via Decimal rounding
    assert abs(updated.avg_entry - expected_avg) < Decimal("1e-8")


# ---------------------------------------------------------------------------
# Cash dividend — no change
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cash_dividend_no_position_change(repo: MemoryRepository):
    """Cash dividend must not alter qty or avg_entry."""
    pos = _pos(qty="10", avg_entry="100")
    await repo.upsert_position(pos)

    event = CorporateActionEvent(
        ca_id="ca-cash-div-001",
        symbol="AAPL",
        ca_type="cash_dividend",
        ex_ts=datetime(2024, 3, 1, tzinfo=UTC),
        cash_amount=Decimal("0.82"),
    )
    await apply_to_positions(event, repo)

    positions = await repo.get_open_positions_for_symbol("AAPL")
    unchanged = positions[0]
    assert unchanged.qty       == Decimal("10")
    assert unchanged.avg_entry == Decimal("100")


# ---------------------------------------------------------------------------
# Merger rename
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_merger_renames_position(repo: MemoryRepository):
    """Merger AAPL → APPLE_NEW renames position symbol."""
    pos = _pos(symbol="AAPL", qty="5", avg_entry="150")
    await repo.upsert_position(pos)

    event = CorporateActionEvent(
        ca_id="ca-merger-001",
        symbol="AAPL",
        ca_type="merger",
        ex_ts=datetime(2025, 1, 10, tzinfo=UTC),
        new_symbol="APPLE_NEW",
    )
    await apply_to_positions(event, repo)

    # Old symbol gone
    old = await repo.get_open_positions_for_symbol("AAPL")
    assert len(old) == 0

    # New symbol present
    new = await repo.get_open_positions_for_symbol("APPLE_NEW")
    assert len(new) == 1
    assert new[0].qty       == Decimal("5")
    assert new[0].avg_entry == Decimal("150")


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotence_same_ca_id(repo: MemoryRepository):
    """Emitting the same ca_id twice does not double-adjust the position."""
    pos = _pos(qty="10", avg_entry="100")
    await repo.upsert_position(pos)

    event = _split_event(split_from="1", split_to="2", ca_id="ca-idem-001")

    await apply_to_positions(event, repo)  # first: qty=20
    await apply_to_positions(event, repo)  # second: no-op

    positions = await repo.get_open_positions_for_symbol("AAPL")
    assert positions[0].qty == Decimal("20"), "Second application must be a no-op"


# ---------------------------------------------------------------------------
# _adjust_position pure function tests
# ---------------------------------------------------------------------------

def test_adjust_position_forward_split_pure():
    """Pure _adjust_position correctly calculates for forward split."""
    pos = _pos(qty="10", avg_entry="100")
    event = _split_event(split_from="1", split_to="4", ca_id="x")
    result = _adjust_position(pos, event)
    assert result.qty       == Decimal("40")
    assert result.avg_entry == Decimal("25")


def test_adjust_position_cash_dividend_returns_same():
    """Cash dividend returns original position (identity)."""
    pos = _pos()
    event = CorporateActionEvent(
        ca_id="x",
        symbol="AAPL",
        ca_type="cash_dividend",
        ex_ts=datetime(2024, 1, 1, tzinfo=UTC),
    )
    result = _adjust_position(pos, event)
    assert result.qty       == pos.qty
    assert result.avg_entry == pos.avg_entry


def test_adjust_position_spinoff_warns_noop(capsys: pytest.CaptureFixture):
    """Spinoff returns original position with WARNING logged to structlog."""
    pos = _pos()
    event = CorporateActionEvent(
        ca_id="x",
        symbol="AAPL",
        ca_type="spinoff",
        ex_ts=datetime(2025, 6, 1, tzinfo=UTC),
    )
    result = _adjust_position(pos, event)

    assert result.qty == pos.qty

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "spinoff" in output.lower(), f"Expected spinoff warning in structlog output. Got:\n{output}"
