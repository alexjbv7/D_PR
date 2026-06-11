"""
Property tests for FirmRiskGate (ADR-009 / ADR-042) — CPU.

El test del kill switch step-0 es el cierre del gap P1-002: una señal con
p_win=0.99 se deniega igual; el riesgo de firma SIEMPRE gana sobre el alfa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quant_shared.contracts import RiskAction, RiskGate, SizeDecision
from quant_shared.schemas.signals import SignalDirection, TradeSignal
from risk.gate import AccountSnapshot, FirmRiskGate, RiskLimits


def _signal(
    direction: SignalDirection = SignalDirection.LONG,
    p_win: float = 0.60,
    symbol: str = "SPY",
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol, direction=direction, p_win=p_win, p_win_raw=p_win
    )


def _size(weight: float) -> SizeDecision:
    return SizeDecision(target_weight=weight, method="test", reason="test")


@pytest.fixture
def gate() -> FirmRiskGate:
    g = FirmRiskGate(RiskLimits())
    g.update(AccountSnapshot(equity=100_000.0))
    return g


# ---------------------------------------------------------------------------
# Contrato + ALLOW en condiciones normales
# ---------------------------------------------------------------------------


def test_implements_contract(gate: FirmRiskGate) -> None:
    assert isinstance(gate, RiskGate)


def test_allow_within_limits(gate: FirmRiskGate) -> None:
    decision = gate.check(_signal(), _size(0.03))
    assert decision.action == RiskAction.ALLOW
    assert decision.max_weight == pytest.approx(0.03)
    assert "ok" in decision.reason


# ---------------------------------------------------------------------------
# STEP-0 — kill switch (gap P1-002)
# ---------------------------------------------------------------------------


def test_kill_switch_denies_everything_step0(gate: FirmRiskGate) -> None:
    """Con el switch tripped, CUALQUIER señal (p_win=0.99) → DENY."""
    gate.update(AccountSnapshot(kill_switch_tripped=True))

    decision = gate.check(_signal(p_win=0.99), _size(0.01))
    assert decision.action == RiskAction.DENY
    assert decision.max_weight == 0.0
    assert "kill_switch" in decision.reason

    # También las señales de cierre: step-0 no evalúa nada más
    flat = gate.check(_signal(direction=SignalDirection.FLAT), _size(0.0))
    assert flat.action == RiskAction.DENY


def test_kill_switch_is_sticky_until_explicit_reset(gate: FirmRiskGate) -> None:
    gate.update(AccountSnapshot(kill_switch_tripped=True))
    # Un snapshot posterior "limpio" NO rearma el switch
    gate.update(AccountSnapshot(kill_switch_tripped=False))
    assert gate.check(_signal(), _size(0.01)).action == RiskAction.DENY

    gate.reset_kill_switch()
    assert gate.check(_signal(), _size(0.01)).action == RiskAction.ALLOW


# ---------------------------------------------------------------------------
# Drawdown (§12.2)
# ---------------------------------------------------------------------------


def test_daily_dd_auto_trips_kill_switch(gate: FirmRiskGate) -> None:
    gate.update(AccountSnapshot(drawdown_daily=0.04))  # > 3%

    first = gate.check(_signal(), _size(0.01))
    assert first.action == RiskAction.DENY
    assert "drawdown_daily" in first.reason
    assert gate.kill_switch_tripped

    # El trip es pegajoso: con snapshot ya limpio, sigue denegando (step-0)
    gate.update(AccountSnapshot(drawdown_daily=0.0))
    second = gate.check(_signal(p_win=0.99), _size(0.01))
    assert second.action == RiskAction.DENY
    assert "kill_switch" in second.reason


def test_weekly_dd_close_only(gate: FirmRiskGate) -> None:
    gate.update(AccountSnapshot(
        drawdown_weekly=0.08,                      # > 7%
        exposure_by_symbol={"SPY": 0.04},          # long existente
    ))

    # Apertura/incremento (mismo signo) → DENY
    add = gate.check(_signal(direction=SignalDirection.LONG), _size(0.05))
    assert add.action == RiskAction.DENY
    assert "drawdown_weekly" in add.reason

    # Apertura en símbolo sin exposición → DENY
    new = gate.check(_signal(symbol="QQQ"), _size(0.03))
    assert new.action == RiskAction.DENY

    # Cierre (FLAT) → permitido
    close = gate.check(_signal(direction=SignalDirection.FLAT), _size(0.0))
    assert close.action == RiskAction.ALLOW

    # Reducción (dirección opuesta) → permitida, capada a la exposición actual
    reduce = gate.check(_signal(direction=SignalDirection.SHORT), _size(0.10))
    assert reduce.action == RiskAction.REDUCE
    assert reduce.max_weight == pytest.approx(0.04)


def test_monthly_dd_freezes_everything(gate: FirmRiskGate) -> None:
    gate.update(AccountSnapshot(
        drawdown_monthly=0.13,                     # > 12%
        drawdown_weekly=0.08,                      # semanal también roto
        exposure_by_symbol={"SPY": 0.04},
    ))
    # Freeze domina sobre el solo-cierre semanal: incluso el cierre se deniega
    decision = gate.check(_signal(direction=SignalDirection.FLAT), _size(0.0))
    assert decision.action == RiskAction.DENY
    assert "drawdown_monthly" in decision.reason


# ---------------------------------------------------------------------------
# Cap por símbolo (§12.1)
# ---------------------------------------------------------------------------


def test_reduce_per_symbol_cap(gate: FirmRiskGate) -> None:
    decision = gate.check(_signal(), _size(0.20))
    assert decision.action == RiskAction.REDUCE
    assert decision.max_weight == pytest.approx(0.05)
    assert "per_symbol_cap" in decision.reason


# ---------------------------------------------------------------------------
# Leverage bruto (§12.6)
# ---------------------------------------------------------------------------


def test_leverage_reduce_to_headroom(gate: FirmRiskGate) -> None:
    gate.update(AccountSnapshot(
        exposure_by_symbol={"AAPL": 0.50, "MSFT": 0.47},  # gross resto = 0.97
    ))
    decision = gate.check(_signal(symbol="SPY"), _size(0.05))
    assert decision.action == RiskAction.REDUCE
    assert decision.max_weight == pytest.approx(0.03)     # headroom 1.0 - 0.97
    assert "leverage_cap" in decision.reason


def test_leverage_deny_without_headroom(gate: FirmRiskGate) -> None:
    gate.update(AccountSnapshot(
        exposure_by_symbol={"AAPL": 0.60, "MSFT": 0.40},  # gross resto = 1.0
    ))
    decision = gate.check(_signal(symbol="SPY"), _size(0.02))
    assert decision.action == RiskAction.DENY
    assert "leverage_cap" in decision.reason


def test_leverage_cap_configurable_per_asset_class() -> None:
    """3.0 crypto (§12.6): la misma exposición pasa con el cap alto."""
    gate = FirmRiskGate(RiskLimits(leverage_cap=3.0, per_symbol_cap=0.50))
    gate.update(AccountSnapshot(exposure_by_symbol={"BTC/USD": 1.5, "ETH/USD": 1.0}))
    decision = gate.check(_signal(symbol="SOL/USD"), _size(0.30))
    assert decision.action == RiskAction.ALLOW


# ---------------------------------------------------------------------------
# Config sin magic numbers
# ---------------------------------------------------------------------------


def test_limits_validation() -> None:
    with pytest.raises(ValueError, match="per_symbol_cap"):
        RiskLimits(per_symbol_cap=1.5)
    with pytest.raises(ValueError, match="leverage_cap"):
        RiskLimits(leverage_cap=0.0)
