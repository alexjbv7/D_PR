"""Tests for SentimentClassifier — financial text sentiment."""
import pytest
from app.sentiment_classifier import SentimentClassifier, SentimentResult


@pytest.fixture
def clf():
    c = SentimentClassifier(use_finbert=False)
    return c


class TestClassify:
    def test_returns_sentiment_result(self, clf):
        r = clf.classify("Bitcoin price rises to new all-time high.")
        assert isinstance(r, SentimentResult)

    def test_bullish_text_positive(self, clf):
        r = clf.classify(
            "Bitcoin surpassed all-time high as institutional adoption grows. "
            "Coinbase reported record revenue and earnings beat expectations."
        )
        assert r.score > 0
        assert r.label in ("positive", "mixed")

    def test_bearish_text_negative(self, clf):
        r = clf.classify(
            "SEC charges Coinbase with fraud and violation. Bankruptcy risk increases. "
            "Exchange hacked, $100M stolen."
        )
        assert r.score < 0
        assert r.label in ("negative", "mixed")

    def test_neutral_text(self, clf):
        r = clf.classify("The price of Bitcoin is currently $65,000.")
        assert r.label in ("neutral", "mixed", "positive", "negative")

    def test_score_between_minus1_and_1(self, clf):
        texts = [
            "Bitcoin surges to all-time high",
            "Market crash wipes out gains",
            "Trading volume remains steady",
            "SEC investigation launched into exchange",
        ]
        for text in texts:
            r = clf.classify(text)
            assert -1.0 <= r.score <= 1.0

    def test_confidence_between_0_and_1(self, clf):
        r = clf.classify("Bitcoin price up 10%")
        assert 0.0 <= r.confidence <= 1.0

    def test_method_is_string(self, clf):
        r = clf.classify("Test")
        assert isinstance(r.method, str)
        assert len(r.method) > 0


class TestSignalDetection:
    def test_regulatory_action_detected(self, clf):
        r = clf.classify(
            "The SEC filed charges and investigation against the exchange for fraud."
        )
        assert "regulatory_action" in r.signals
        assert r.is_market_moving is True

    def test_etf_signal_detected(self, clf):
        r = clf.classify(
            "BlackRock files for Bitcoin spot ETF approval with the SEC."
        )
        assert "etf_signal" in r.signals

    def test_ma_event_detected(self, clf):
        r = clf.classify(
            "Coinbase announced the acquisition of a blockchain analytics firm."
        )
        assert "ma_event" in r.signals

    def test_insolvency_detected(self, clf):
        r = clf.classify(
            "Exchange files for Chapter 11 bankruptcy protection amid insolvency concerns."
        )
        assert "insolvency" in r.signals
        assert r.is_market_moving is True

    def test_security_incident_detected(self, clf):
        r = clf.classify(
            "Exchange suffered a hack; $50M in cryptocurrency was stolen."
        )
        assert "security_incident" in r.signals

    def test_institutional_crypto_detected(self, clf):
        r = clf.classify(
            "BlackRock increases Bitcoin holdings by 15% in Q1 2026."
        )
        assert "institutional_crypto" in r.signals

    def test_no_signal_plain_text(self, clf):
        r = clf.classify("The weather is sunny today in New York.")
        assert len(r.signals) == 0
        assert r.is_market_moving is False


class TestClassifyBatch:
    def test_batch_returns_same_length(self, clf):
        texts = [
            "Bitcoin price soars.",
            "Market crashes hard.",
            "Volume is average.",
        ]
        results = clf.classify_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_to_dict_serializable(self, clf):
        import json
        r = clf.classify("Test text for serialization.")
        d = r.to_dict()
        json.dumps(d)  # must not raise
