"""Tests for MarketStateEngine — composite risk score."""
import pytest
from app.market_state_engine import MarketStateEngine, MarketState


@pytest.fixture
def engine():
    return MarketStateEngine(kafka_servers="", redis_url="")


def _mock_regime(regime_id=0, label="bull_trend", confidence=0.9):
    class _R:
        pass
    r = _R()
    r.regime_id = regime_id
    r.label = label
    r.confidence = confidence
    return r


def _mock_liquidity(squeeze="none", oi_z=0.0, funding_z=0.0):
    class _L:
        pass
    l = _L()
    l.squeeze_risk = squeeze
    l.oi_z_score = oi_z
    l.funding_z = funding_z
    return l


def _mock_macro(bias="neutral", leverage=1.0, rec_prob=0.1):
    class _M:
        pass
    m = _M()
    m.bias = bias
    m.leverage_adj = leverage
    m.recession_prob = rec_prob
    m.rate_environment = "neutral"
    m.yield_curve_inversion = False
    return m


def _mock_anomaly(severity, atype="price"):
    class _A:
        pass
    a = _A()
    a.severity = severity
    a.anomaly_type = atype
    return a


class TestBuild:
    def test_returns_market_state(self, engine):
        s = engine.build()
        assert isinstance(s, MarketState)

    def test_baseline_low_risk(self, engine):
        s = engine.build(
            regime=_mock_regime(0, "bull_trend"),
            liquidity=_mock_liquidity("none"),
            macro=_mock_macro("bullish", 1.2, 0.05),
        )
        assert s.composite_risk_score < 30
        assert s.allow_new_longs is True
        assert s.defensive_mode is False

    def test_critical_squeeze_raises_score(self, engine):
        s = engine.build(liquidity=_mock_liquidity("critical"))
        assert s.composite_risk_score >= 30

    def test_critical_anomaly_raises_score(self, engine):
        anomalies = [_mock_anomaly("critical"), _mock_anomaly("critical")]
        s = engine.build(anomalies=anomalies)
        assert s.anomalies_critical == 2
        assert s.composite_risk_score >= 30

    def test_full_crisis_triggers_defensive(self, engine):
        s = engine.build(
            regime=_mock_regime(4, "crisis"),
            liquidity=_mock_liquidity("critical"),
            anomalies=[_mock_anomaly("critical")] * 2,
            macro=_mock_macro("strong_bearish", 0.25, 0.75),
        )
        assert s.composite_risk_score > 85
        assert s.defensive_mode is True
        assert s.allow_new_longs is False

    def test_score_clamped_0_100(self, engine):
        s = engine.build(
            regime=_mock_regime(4, "crisis"),
            liquidity=_mock_liquidity("critical"),
            anomalies=[_mock_anomaly("critical")] * 5,
            macro=_mock_macro("strong_bearish", 0.25, 0.80),
            recession_market_prob=0.80,
        )
        assert 0.0 <= s.composite_risk_score <= 100.0

    def test_whale_sentiment_stored(self, engine):
        s = engine.build(whale_sentiment=0.7, whale_confidence=0.9)
        assert s.whale_sentiment == pytest.approx(0.7)
        assert s.whale_confidence == pytest.approx(0.9)

    def test_polymarket_probs_stored(self, engine):
        s = engine.build(btc_up_prob=0.65, recession_market_prob=0.25)
        assert s.btc_up_prob == pytest.approx(0.65)
        assert s.recession_market_prob == pytest.approx(0.25)

    def test_summary_nonempty(self, engine):
        s = engine.build()
        assert len(s.summary) > 0

    def test_to_dict_serializable(self, engine):
        import json
        s = engine.build()
        d = s.to_dict()
        json.dumps(d)   # must not raise

    @pytest.mark.parametrize("squeeze,expected_min_score", [
        ("none",     0),
        ("low",      5),
        ("medium",  10),
        ("high",    20),
        ("critical", 30),
    ])
    def test_squeeze_score_contribution(self, engine, squeeze, expected_min_score):
        s = engine.build(liquidity=_mock_liquidity(squeeze))
        assert s.composite_risk_score >= expected_min_score
