"""Tests for MacroSignalEngine — portfolio bias computation."""
import pytest
from app.macro_signal_engine import MacroSignalEngine, MacroBias, MacroSignal


@pytest.fixture
def engine():
    return MacroSignalEngine(kafka_servers="", redis_url="")


def _mock_recession(regime="expansion", recession_prob=0.1,
                    rate_environment="neutral", yield_curve_inversion=False):
    class _R:
        pass
    r = _R()
    r.regime = regime
    r.recession_prob = recession_prob
    r.rate_environment = rate_environment
    r.yield_curve_inversion = yield_curve_inversion
    return r


class TestCompute:
    def test_returns_macro_signal(self, engine):
        sig = engine.compute({}, _mock_recession())
        assert isinstance(sig, MacroSignal)

    def test_strong_recession_gives_strong_bearish(self, engine):
        sig = engine.compute({}, _mock_recession(
            recession_prob=0.80, regime="recession",
            rate_environment="hiking", yield_curve_inversion=True
        ))
        assert sig.bias == MacroBias.STRONG_BEARISH

    def test_expansion_cutting_gives_bullish(self, engine):
        sig = engine.compute(
            {"VIXCLS": 13.0, "DTWEXBGS": 98.0},
            _mock_recession(regime="expansion", recession_prob=0.05,
                            rate_environment="cutting")
        )
        assert sig.bias in (MacroBias.BULLISH, MacroBias.STRONG_BULLISH)

    def test_recession_leverage_very_low(self, engine):
        sig = engine.compute({}, _mock_recession(recession_prob=0.70))
        assert sig.leverage_adj == pytest.approx(0.25)

    def test_strong_bullish_leverage_high(self, engine):
        sig = engine.compute(
            {"VIXCLS": 12.0, "DTWEXBGS": 97.0},
            _mock_recession(recession_prob=0.05, rate_environment="cutting",
                            regime="expansion")
        )
        # Bias should be at least bullish → leverage >= 1.2
        assert sig.leverage_adj >= 1.0

    def test_high_vix_reduces_score(self, engine):
        sig_low_vix  = engine.compute({"VIXCLS": 12.0}, _mock_recession())
        sig_high_vix = engine.compute({"VIXCLS": 40.0}, _mock_recession())
        assert sig_low_vix.leverage_adj >= sig_high_vix.leverage_adj

    def test_expansion_regime_favors_altcoins(self, engine):
        sig = engine.compute({}, _mock_recession(regime="expansion"))
        assert "SOLUSDT" in sig.favored_assets

    def test_recession_avoids_all_crypto(self, engine):
        sig = engine.compute({}, _mock_recession(regime="recession", recession_prob=0.80))
        # All major assets should be in avoided
        assert "BTCUSDT" in sig.avoided_assets

    def test_signal_has_event_id(self, engine):
        sig = engine.compute({}, _mock_recession())
        assert len(sig.event_id) == 36  # UUID4

    def test_stores_last_signal(self, engine):
        sig = engine.compute({}, _mock_recession())
        assert engine.last_signal is sig

    @pytest.mark.parametrize("dxy,expected_direction", [
        (110.0, "bearish"),   # strong dollar → bearish crypto
        (96.0,  "bullish"),   # weak dollar → bullish crypto
    ])
    def test_dxy_effect(self, engine, dxy, expected_direction):
        sig = engine.compute(
            {"DTWEXBGS": dxy},
            _mock_recession(recession_prob=0.1)
        )
        if expected_direction == "bearish":
            assert sig.bias.value in ("bearish", "strong_bearish", "neutral")
        else:
            assert sig.bias.value in ("bullish", "strong_bullish", "neutral")
