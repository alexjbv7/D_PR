"""Tests for LiquidityScanner — squeeze risk classification."""
import pytest
from app.liquidity_scanner import LiquidityScanner, LiquiditySnapshot


@pytest.fixture
def scanner():
    return LiquidityScanner(window=100)


def _make_data(**kwargs):
    base = {
        "symbol":            "BTCUSDT",
        "open_interest_usd": 10_000_000,
        "long_liq_1h":       0,
        "short_liq_1h":      0,
        "exchange_reserve":  100_000,
        "funding_rate":      0.0001,
    }
    base.update(kwargs)
    return base


class TestProcess:
    def test_returns_snapshot(self, scanner):
        snap = scanner.process(_make_data())
        assert isinstance(snap, LiquiditySnapshot)
        assert snap.symbol == "BTCUSDT"

    def test_liq_imbalance_all_longs(self, scanner):
        snap = scanner.process(_make_data(long_liq_1h=1_000_000, short_liq_1h=0))
        assert snap.liq_imbalance == pytest.approx(1.0, abs=1e-3)

    def test_liq_imbalance_all_shorts(self, scanner):
        snap = scanner.process(_make_data(long_liq_1h=0, short_liq_1h=1_000_000))
        assert snap.liq_imbalance == pytest.approx(-1.0, abs=1e-3)

    def test_liq_imbalance_balanced(self, scanner):
        snap = scanner.process(_make_data(long_liq_1h=500_000, short_liq_1h=500_000))
        assert snap.liq_imbalance == pytest.approx(0.0, abs=1e-3)

    def test_z_score_zero_on_first_tick(self, scanner):
        snap = scanner.process(_make_data())
        assert snap.oi_z_score == 0.0
        assert snap.reserve_z_score == 0.0

    def test_z_score_nonzero_after_history(self, scanner):
        # Feed 30 identical ticks then one very different
        for _ in range(30):
            scanner.process(_make_data(open_interest_usd=10_000_000))
        snap = scanner.process(_make_data(open_interest_usd=50_000_000))
        assert snap.oi_z_score > 1.0  # clearly above mean

    def test_squeeze_none_baseline(self, scanner):
        snap = scanner.process(_make_data())
        assert snap.squeeze_risk == "none"

    def test_to_dict(self, scanner):
        snap = scanner.process(_make_data())
        d = snap.to_dict()
        assert "squeeze_risk" in d
        assert "oi_z_score" in d


class TestClassifySqueeze:
    @pytest.mark.parametrize("oi_z,fund_z,liq_imbal,reserve_z,expected", [
        # None: no signals
        (0.0,  0.0,  0.0,  0.0,   "none"),
        # Low: single mild signal
        (2.1,  0.0,  0.0,  0.0,   "low"),
        # Medium: two signals
        (2.1,  2.6,  0.0,  0.0,   "medium"),
        # Medium: oi_z+fund_z+liq_imbal = 1+1+0.5 = 2.5 → "medium"
        (2.1,  2.6,  0.8,  0.0,   "medium"),
        # Critical: all signals
        (3.1,  4.1,  0.8, -2.5,   "critical"),
        # High funding but no OI
        (0.0,  4.1,  0.0,  0.0,   "medium"),
    ])
    def test_classification(self, oi_z, fund_z, liq_imbal, reserve_z, expected):
        result = LiquidityScanner._classify_squeeze(oi_z, fund_z, liq_imbal, reserve_z)
        assert result == expected

    def test_symmetric_funding(self):
        """Negative funding (short squeeze) should count same as positive."""
        pos = LiquidityScanner._classify_squeeze(0, 3.0, 0, 0)
        neg = LiquidityScanner._classify_squeeze(0, -3.0, 0, 0)
        assert pos == neg
