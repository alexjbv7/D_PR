"""
Tests for ExecutionService — the end-to-end signal-to-result pipeline.

Uses MemoryRepository and a Router with mocked adapters so we never touch
real brokers or Postgres.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_shared.schemas.orders import (
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)

from app.brokers.base import AccountInfo, BrokerError
from app.repository import MemoryRepository
from app.risk_gate import RiskConfig, RiskGate
from app.routing import Router
from app.service import ExecutionService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter(venue: str, *, equity="100000", price="1000",
                  is_paper=True) -> MagicMock:
    a = MagicMock()
    a.venue          = venue
    a.connect        = AsyncMock()
    a.close          = AsyncMock()
    a.get_account    = AsyncMock(return_value=AccountInfo(
        account_id=f"{venue}-acc",
        venue=venue,
        equity=Decimal(equity),
        cash=Decimal(equity),
        is_paper=is_paper,
    ))
    a.get_last_price = AsyncMock(return_value=Decimal(price))
    a.get_positions  = AsyncMock(return_value=[])
    def _submit(intent: OrderIntent) -> OrderResult:
        qty = intent.qty if intent.qty is not None else Decimal("1")
        return OrderResult(
            intent_id=intent.intent_id,
            broker_id=f"{venue}-001",
            symbol=intent.symbol,
            side=intent.side,
            status=OrderStatus.SUBMITTED,
            qty=qty,
            venue=venue,
        )

    a.submit = AsyncMock(side_effect=_submit)
    return a


@pytest.fixture
def binance():
    return _make_adapter("binance")


@pytest.fixture
def router(binance):
    r = Router(default_equity="alpaca", default_crypto="binance")
    r.register(binance)
    return r


@pytest.fixture
def repo():
    return MemoryRepository()


@pytest.fixture
def service(router, repo):
    gate = RiskGate(RiskConfig(require_paper=True), repo)
    return ExecutionService(
        router=router,
        risk_gate=gate,
        repository=repo,
        account_refresh_sec=60,
    )


def _signal(**overrides):
    base = {
        "event_id":      "sig-1",
        "strategy":      "regime_adaptive",
        "symbol":        "BTCUSDT",
        "direction":     1,
        "p_win":         0.6,
        "position_size": 0.02,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_handle_signal_full_flow(service, binance):
    result = await service.handle_signal(_signal())

    assert result is not None
    assert result.symbol  == "BTCUSDT"
    assert result.side    == OrderSide.BUY
    assert result.venue   == "binance"
    binance.submit.assert_awaited_once()


async def test_counters_incremented(service):
    await service.handle_signal(_signal())
    stats = service.stats()
    assert stats["signals_seen"]  == 1
    assert stats["intents_built"] == 1
    assert stats["approved"]      == 1
    assert stats["submitted"]     == 1
    assert stats["rejected"]      == 0


async def test_result_persisted_to_repository(service, repo):
    await service.handle_signal(_signal())
    recent = await repo.list_recent_results()
    assert len(recent) == 1


async def test_result_emitter_called(router, repo):
    gate = RiskGate(RiskConfig(), repo)
    emitter = AsyncMock()
    service = ExecutionService(
        router=router, risk_gate=gate, repository=repo,
        result_emitter=emitter,
    )
    await service.handle_signal(_signal())
    emitter.assert_awaited_once()


# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------

async def test_flat_signal_no_submit(service, binance):
    result = await service.handle_signal(_signal(direction=0))
    assert result is None
    binance.submit.assert_not_called()


async def test_missing_price_no_submit(service, binance):
    binance.get_last_price.return_value = None
    result = await service.handle_signal(_signal())
    assert result is None
    binance.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Risk rejections
# ---------------------------------------------------------------------------

async def test_rejected_by_live_blocker(router, repo, binance):
    binance.get_account.return_value = AccountInfo(
        account_id="x", venue="binance",
        equity=Decimal("100000"), cash=Decimal("100000"),
        is_paper=False,            # live!
    )
    gate = RiskGate(RiskConfig(require_paper=True), repo)
    svc  = ExecutionService(router=router, risk_gate=gate, repository=repo)

    result = await svc.handle_signal(_signal())
    assert result is None
    binance.submit.assert_not_called()
    assert svc.stats()["rejected"] == 1


# ---------------------------------------------------------------------------
# Broker errors
# ---------------------------------------------------------------------------

async def test_submit_error_counted(service, binance):
    binance.submit.side_effect = BrokerError("rate limit")
    result = await service.handle_signal(_signal())
    assert result is None
    assert service.stats()["submit_errors"] == 1


async def test_account_error_skips_signal(service, binance):
    binance.get_account.side_effect = BrokerError("auth")
    result = await service.handle_signal(_signal())
    assert result is None
    binance.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Account cache
# ---------------------------------------------------------------------------

async def test_account_cached_within_window(service, binance):
    await service.handle_signal(_signal())
    await service.handle_signal(_signal(event_id="sig-2"))
    # cache window is 60s ; should only fetch once
    assert binance.get_account.await_count == 1


async def test_default_venue_resolution_for_equity(repo):
    alpaca = _make_adapter("alpaca", price="170")
    router = Router(default_equity="alpaca", default_crypto="binance")
    router.register(alpaca)

    gate = RiskGate(RiskConfig(), repo)
    svc  = ExecutionService(router=router, risk_gate=gate, repository=repo)

    sig = _signal(symbol="AAPL")
    sig.pop("venue", None)
    sig["ts"] = "2026-05-19T14:00:00+00:00"
    result = await svc.handle_signal(sig)
    assert result is not None
    assert result.venue == "alpaca"
