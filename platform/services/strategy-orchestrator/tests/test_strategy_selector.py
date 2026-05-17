"""Tests for StrategySelector — Thompson Sampling bandit."""
import pytest
from app.strategy_selector import StrategySelector, StrategyPerformance


ALL_STRATEGIES = [
    "momentum_ml",
    "mean_reversion_funding",
    "regime_adaptive",
    "whale_follow",
]


@pytest.fixture
def selector():
    return StrategySelector(strategies=ALL_STRATEGIES, seed=42)


class TestSelect:
    def test_returns_allocations_summing_to_one(self, selector):
        allocs = selector.select(
            available=ALL_STRATEGIES,
            recession_prob=0.2,
            market_regime="bull_trend",
            vol_percentile=0.5,
            max_active=3,
        )
        total = sum(allocs.values())
        assert total == pytest.approx(1.0, abs=1e-3)  # softmax may have minor float drift

    def test_max_active_respected(self, selector):
        allocs = selector.select(ALL_STRATEGIES, 0.2, "bull_trend", 0.5, max_active=2)
        assert len(allocs) <= 2

    def test_recession_blocks_momentum_ml(self, selector):
        """High recession prob → momentum_ml should not be selected."""
        allocs = selector.select(ALL_STRATEGIES, recession_prob=0.80,
                                  market_regime="recession", vol_percentile=0.9)
        assert "momentum_ml" not in allocs

    def test_recession_blocks_regime_adaptive(self, selector):
        allocs = selector.select(ALL_STRATEGIES, recession_prob=0.80,
                                  market_regime="recession", vol_percentile=0.9)
        assert "regime_adaptive" not in allocs

    def test_available_subset(self, selector):
        subset = ["momentum_ml", "whale_follow"]
        allocs = selector.select(subset, 0.2, "bull_trend", 0.5, max_active=3)
        # Only strategies from subset can appear
        for s in allocs:
            assert s in subset

    def test_empty_available_returns_empty(self, selector):
        allocs = selector.select([], 0.2, "bull_trend", 0.5)
        assert allocs == {}

    def test_allocations_between_0_and_1(self, selector):
        allocs = selector.select(ALL_STRATEGIES, 0.2, "bull_trend", 0.5, max_active=4)
        for v in allocs.values():
            assert 0.0 < v <= 1.0

    def test_only_initialized_strategies_selected(self, selector):
        # Selector is initialized with ALL_STRATEGIES only
        allocs = selector.select(ALL_STRATEGIES, 0.2, "bull_trend", 0.5, max_active=4)
        for s in allocs:
            assert s in ALL_STRATEGIES


class TestRecordOutcome:
    def test_win_increments_wins(self, selector):
        s = "momentum_ml"
        perf_before = selector._perf[s].wins
        selector.record_outcome(s, won=True, pnl=100.0)
        assert selector._perf[s].wins > perf_before

    def test_loss_increments_losses(self, selector):
        s = "momentum_ml"
        perf_before = selector._perf[s].losses
        selector.record_outcome(s, won=False, pnl=-50.0)
        assert selector._perf[s].losses > perf_before

    def test_repeated_wins_raise_win_rate(self, selector):
        s = "regime_adaptive"
        wr_before = selector._perf[s].win_rate
        for _ in range(10):
            selector.record_outcome(s, won=True, pnl=50.0)
        assert selector._perf[s].win_rate > wr_before

    def test_decay_applied(self, selector):
        s = "whale_follow"
        selector._perf[s].wins   = 100.0
        selector._perf[s].losses = 100.0
        selector.record_outcome(s, won=True, pnl=10.0)
        # Decay ×0.99 means both should be < 100 + 1 = 101 but wins ≈ 99+1=100
        assert selector._perf[s].losses < 100.0   # decayed

    def test_pnl_accumulated(self, selector):
        s = "mean_reversion_funding"
        selector.record_outcome(s, won=True, pnl=200.0)
        assert selector._perf[s].total_pnl == pytest.approx(200.0)

    def test_unknown_strategy_no_crash(self, selector):
        # Should not raise, just warn
        selector.record_outcome("nonexistent", won=True, pnl=10.0)


class TestStrategyPerformance:
    def test_sample_in_0_1(self):
        perf = StrategyPerformance(name="test", wins=5.0, losses=3.0)
        for _ in range(50):
            s = perf.sample
            assert 0.0 <= s <= 1.0

    def test_expected_value(self):
        perf = StrategyPerformance(name="test", wins=8.0, losses=2.0)
        assert perf.expected_value == pytest.approx(0.8)

    def test_win_rate(self):
        perf = StrategyPerformance(name="test", wins=3.0, losses=1.0)
        assert perf.win_rate == pytest.approx(0.75)

    def test_update_win(self):
        perf = StrategyPerformance(name="x", wins=1.0, losses=1.0)
        perf.update(won=True, pnl=100.0, decay=0.99)
        assert perf.wins  == pytest.approx(0.99 + 1.0)
        assert perf.losses == pytest.approx(0.99)
