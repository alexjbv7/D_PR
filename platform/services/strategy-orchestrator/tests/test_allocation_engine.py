"""Tests for AllocationEngine — fractional Kelly + drawdown brakes."""
import pytest
from app.allocation_engine import AllocationEngine, AllocationConfig


@pytest.fixture
def engine():
    cfg = AllocationConfig(
        total_capital=100_000,
        kelly_fraction=0.25,
        max_leverage=2.0,
        dd_soft_brake=0.05,
        dd_hard_brake=0.10,
    )
    return AllocationEngine(config=cfg)


def _vols(*strategies):
    return {s: 0.20 for s in strategies}   # 20% annualized vol


class TestPositionSizeFromSignal:
    def test_high_confidence_signal(self, engine):
        size = engine.position_size_from_signal(
            p_win=0.65, stop_loss_pct=0.02, capital=100_000
        )
        assert size > 0
        assert size <= 100_000 * 0.25

    def test_very_low_pwin_zero_position(self, engine):
        # Kelly break-even at stop=0.02 is ~p_win≈0.019; below that → 0
        size = engine.position_size_from_signal(
            p_win=0.01, stop_loss_pct=0.02, capital=100_000
        )
        assert size == 0.0

    def test_size_increases_with_pwin(self, engine):
        # Higher win probability → larger position
        size_lo = engine.position_size_from_signal(0.55, 0.02, 100_000)
        size_hi = engine.position_size_from_signal(0.75, 0.02, 100_000)
        assert size_hi > size_lo

    def test_respects_capital_proportionality(self, engine):
        size_small = engine.position_size_from_signal(0.7, 0.02, 10_000)
        size_large = engine.position_size_from_signal(0.7, 0.02, 100_000)
        assert size_large == pytest.approx(size_small * 10, rel=1e-2)


class TestComputeAllocations:
    def test_hard_brake_at_max_drawdown(self, engine):
        allocs = {"strat_a": 0.6, "strat_b": 0.4}
        result = engine.compute(
            allocations=allocs,
            realized_vol=_vols("strat_a", "strat_b"),
            correlations={},
            current_drawdown=0.10,
            portfolio_value=100_000,
        )
        total = sum(result.values())
        assert total == 0.0

    def test_soft_brake_reduces_size(self, engine):
        # Soft brake ramp is between dd_soft_brake (0.05) and dd_hard_brake (0.10).
        # At dd=0.05 multiplier is still 1.0; at dd=0.07 it starts reducing.
        allocs = {"strat_a": 0.6, "strat_b": 0.4}
        vols = _vols("strat_a", "strat_b")
        result_nodraw = engine.compute(allocs, vols, {}, 0.0,  100_000)
        result_draw   = engine.compute(allocs, vols, {}, 0.07, 100_000)  # mid-ramp
        assert sum(result_draw.values()) < sum(result_nodraw.values())

    def test_correlation_penalty_reduces_notional(self, engine):
        allocs = {"strat_a": 0.5, "strat_b": 0.5}
        vols = _vols("strat_a", "strat_b")
        no_corr  = engine.compute(allocs, vols, {}, 0.0, 100_000)
        high_corr = engine.compute(allocs, vols, {"strat_a:strat_b": 0.85}, 0.0, 100_000)
        assert sum(high_corr.values()) <= sum(no_corr.values())

    def test_leverage_cap(self, engine):
        allocs = {"strat_a": 1.0}
        vols = {"strat_a": 0.001}  # very low vol → large scaling
        result = engine.compute(
            allocations=allocs,
            realized_vol=vols,
            correlations={},
            current_drawdown=0.0,
            portfolio_value=100_000,
        )
        total = sum(result.values())
        assert total <= 100_000 * 2.0

    def test_all_allocations_non_negative(self, engine):
        allocs = {"a": 0.4, "b": 0.3, "c": 0.3}
        vols = _vols("a", "b", "c")
        result = engine.compute(allocs, vols, {}, 0.02, 100_000)
        for v in result.values():
            assert v >= 0


class TestDrawdownBrake:
    @pytest.mark.parametrize("dd,expected_range", [
        (0.00, (0.9, 1.1)),   # no brake: multiplier = 1
        (0.05, (0.9, 1.1)),   # at soft brake start: still 1
        (0.07, (0.3, 0.7)),   # mid-ramp: ~50%
        (0.10, (0.0, 0.01)),  # hard brake: 0
    ])
    def test_brake_multiplier(self, engine, dd, expected_range):
        brake = engine._drawdown_multiplier(dd)
        lo, hi = expected_range
        assert lo <= brake <= hi, f"dd={dd} → brake={brake} not in [{lo},{hi}]"

    def test_brake_monotonically_decreasing(self, engine):
        dds = [0.0, 0.02, 0.05, 0.07, 0.09, 0.10]
        brakes = [engine._drawdown_multiplier(d) for d in dds]
        for i in range(len(brakes) - 1):
            assert brakes[i] >= brakes[i + 1]
