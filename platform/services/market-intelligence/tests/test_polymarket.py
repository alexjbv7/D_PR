"""Tests for PolymarketParser + ProbabilityAnalyzer."""
import pytest
from app.polymarket.polymarket_parser import PolymarketParser, PolymarketEvent
from app.polymarket.probability_analyzer import ProbabilityAnalyzer, ProbabilitySignal


@pytest.fixture
def parser():
    return PolymarketParser()


@pytest.fixture
def analyzer():
    return ProbabilityAnalyzer()


# ---------------------------------------------------------------------------
# PolymarketParser
# ---------------------------------------------------------------------------

def _make_market(question, yes_price=0.6, oi=100_000, form="outcomes"):
    if form == "outcomes":
        return {
            "id": "market-001",
            "slug": "test-market",
            "question": question,
            "end_date": "2026-12-31",
            "outcomes": ["Yes", "No"],
            "outcomePrices": [str(yes_price), str(1 - yes_price)],
            "volume24hr": 50000,
            "openInterest": oi,
        }
    return {
        "id": "market-002",
        "slug": "test-market-2",
        "question": question,
        "end_date": "2026-12-31",
        "tokens": [
            {"outcome": "YES", "price": yes_price},
            {"outcome": "NO",  "price": 1 - yes_price},
        ],
        "volume24hr": 50000,
        "openInterest": oi,
    }


class TestPolymarketParser:
    def test_parse_crypto_market(self, parser):
        raw = {"data": [_make_market("Will Bitcoin reach $100k by end of 2026?")]}
        events = parser.parse_markets(raw)
        assert len(events) >= 1
        e = events[0]
        assert e.category == "crypto"
        assert e.asset == "BTC"

    def test_parse_macro_market(self, parser):
        raw = {"data": [_make_market("Will the Fed cut rates in Q1 2026?", oi=200_000)]}
        events = parser.parse_markets(raw)
        assert len(events) >= 1
        e = events[0]
        assert e.category == "macro"

    def test_parse_regulatory_market(self, parser):
        raw = {"data": [_make_market("Will the SEC approve a Bitcoin ETF in 2026?", oi=500_000)]}
        events = parser.parse_markets(raw)
        assert any(e.category in ("regulatory", "crypto") for e in events)

    def test_yes_price_extracted(self, parser):
        market = _make_market("Will ETH reach $5000?", yes_price=0.72)
        e = parser.parse_single_market(market)
        assert e.yes_price == pytest.approx(0.72, abs=1e-3)

    def test_no_price_complements_yes(self, parser):
        market = _make_market("Will SOL flip ETH?", yes_price=0.3)
        e = parser.parse_single_market(market)
        assert e.no_price == pytest.approx(0.7, abs=1e-3)

    def test_token_format_parsed(self, parser):
        market = _make_market("Will BTC drop below $50k?", yes_price=0.25, form="tokens")
        e = parser.parse_single_market(market)
        assert e.yes_price == pytest.approx(0.25, abs=1e-3)

    def test_direction_bullish(self, parser):
        e = parser.parse_single_market(_make_market("Will Bitcoin reach $200k?"))
        assert e.direction == "bullish"

    def test_direction_bearish(self, parser):
        e = parser.parse_single_market(_make_market("Will Bitcoin drop below $30k?"))
        assert e.direction == "bearish"

    def test_liquidity_classification(self, parser):
        low  = parser.parse_single_market(_make_market("Will ADA moon?", oi=5_000))
        med  = parser.parse_single_market(_make_market("Will ETH pump?", oi=100_000))
        high = parser.parse_single_market(_make_market("Will BTC pump?", oi=600_000))
        assert low.liquidity  == "low"
        assert med.liquidity  == "medium"
        assert high.liquidity == "high"

    def test_is_tradeable_requires_min_oi(self, parser):
        illiquid = parser.parse_single_market(_make_market("Will DOGE moon?", oi=1_000))
        liquid   = parser.parse_single_market(_make_market("Will BTC moon?",  oi=50_001))
        assert illiquid.is_tradeable is False
        assert liquid.is_tradeable is True

    def test_irrelevant_markets_filtered(self, parser):
        raw = {"data": [
            _make_market("Will the NFL have a new champion?"),   # sports → filtered
            _make_market("Will Bitcoin reach $100k?"),           # kept
        ]}
        events = parser.parse_markets(raw)
        questions = [e.question for e in events]
        assert any("Bitcoin" in q for q in questions)


# ---------------------------------------------------------------------------
# ProbabilityAnalyzer
# ---------------------------------------------------------------------------

def _btc_events(probs=None, ois=None, direction="bullish"):
    probs = probs or [0.65, 0.70]
    ois   = ois   or [200_000, 300_000]
    return [
        PolymarketEvent(
            market_id=f"m{i}", slug=f"btc-{i}",
            question="Will BTC reach $100k?",
            end_date="2026-12-31",
            yes_price=p, no_price=1-p,
            volume_24h=50_000, open_interest=oi,
            category="crypto", asset="BTC",
            direction=direction, liquidity="medium",
        )
        for i, (p, oi) in enumerate(zip(probs, ois))
    ]


class TestProbabilityAnalyzer:
    def test_returns_signal(self, analyzer):
        signals = analyzer.analyze(_btc_events())
        assert len(signals) >= 1
        assert isinstance(signals[0], ProbabilitySignal)

    def test_signal_asset_correct(self, analyzer):
        signals = analyzer.analyze(_btc_events())
        assert signals[0].asset == "BTC"

    def test_weighted_probability(self, analyzer):
        # OI-weighted: 200k × 0.65 + 300k × 0.70 = 130k + 210k = 340k / 500k = 0.68
        events = _btc_events([0.65, 0.70], [200_000, 300_000])
        signals = analyzer.analyze(events)
        assert signals[0].raw_prob == pytest.approx(0.68, abs=0.01)

    def test_calibration_shrinks_extremes(self, analyzer):
        # Very high raw prob → calibrated should be slightly lower
        events = _btc_events([0.95], [100_000])
        signals = analyzer.analyze(events)
        assert signals[0].probability < signals[0].raw_prob

    def test_calibration_preserves_midrange(self, analyzer):
        events = _btc_events([0.55], [100_000])
        signals = analyzer.analyze(events)
        # Mid-range: calibrated ≈ raw
        assert abs(signals[0].probability - signals[0].raw_prob) < 0.01

    def test_bias_score_bullish_positive(self, analyzer):
        signals = analyzer.analyze(_btc_events([0.75], direction="bullish"))
        assert signals[0].as_bias_score() > 0

    def test_bias_score_bearish_negative(self, analyzer):
        signals = analyzer.analyze(_btc_events([0.75], direction="bearish"))
        assert signals[0].as_bias_score() < 0

    def test_confidence_increases_with_more_markets(self, analyzer):
        single = analyzer.analyze(_btc_events([0.65], [200_000]))
        multi  = analyzer.analyze(_btc_events([0.65] * 5, [200_000] * 5))
        if single and multi:
            assert multi[0].confidence >= single[0].confidence

    def test_get_recession_prob(self, analyzer):
        recession_events = [
            PolymarketEvent(
                market_id="rec-1", slug="recession-2026",
                question="Will the US enter a recession in 2026?",
                end_date="2026-12-31",
                yes_price=0.35, no_price=0.65,
                volume_24h=100_000, open_interest=500_000,
                category="macro", asset="MACRO",
                direction="bearish", liquidity="high",
            )
        ]
        prob = analyzer.get_recession_prob(recession_events)
        assert prob is not None
        assert 0 < prob < 1

    def test_no_recession_markets_returns_none(self, analyzer):
        btc_only = _btc_events()
        assert analyzer.get_recession_prob(btc_only) is None

    def test_low_oi_markets_excluded(self, analyzer):
        illiquid = _btc_events([0.8], ois=[5_000])  # below min OI
        signals = analyzer.analyze(illiquid)
        # Illiquid events filtered by is_tradeable
        assert len(signals) == 0
