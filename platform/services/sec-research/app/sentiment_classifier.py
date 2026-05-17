"""
SentimentClassifier — Clasificación de sentimiento de texto financiero.

Combina tres enfoques complementarios:
  1. FinBERT-style word lists (rápido, sin GPU)
  2. VADER con calibración financiera (built-in)
  3. Regex patterns para señales específicas (M&A, regulatorio, earnings)

El output es un SentimentResult con score normalizado -1 a +1 y
señales específicas relevantes para crypto (regulatory, institutional,
market-moving events).

Decisión de diseño:
  - No dependency en modelos HuggingFace por default (alto costo en RAM).
  - Si HuggingFace disponible → usa FinBERT para textos > 100 palabras.
  - Ensemble: FinBERT×0.6 + VADER×0.2 + lexicon×0.2
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Financial sentiment lexicons (curated for crypto/macro context)
# ---------------------------------------------------------------------------

_BULLISH_TERMS = {
    # Strong positive
    "record", "all-time high", "ath", "surpass", "exceed", "beat",
    "approval", "approved", "launch", "partnership", "integration",
    "adoption", "institutional", "etf", "accumulate", "buy",
    "upgrade", "outperform", "overweight", "bullish", "growth",
    "profit", "revenue", "earnings beat", "guidance raised",
    # Regulatory positive
    "regulatory clarity", "legal", "compliant", "licensed", "approved",
    # Macro positive
    "rate cut", "dovish", "stimulus", "qe", "liquidity",
}

_BEARISH_TERMS = {
    # Strong negative
    "ban", "banned", "prohibition", "crackdown", "lawsuit", "sec charges",
    "fraud", "hacked", "hack", "exploit", "vulnerability", "breach",
    "bankruptcy", "insolvent", "collapse", "crash", "miss", "shortfall",
    "guidance cut", "downgrade", "underperform", "underweight", "bearish",
    "loss", "deficit", "writedown", "impairment",
    # Regulatory negative
    "investigation", "subpoena", "enforcement", "fine", "penalty",
    "violation", "probe", "scrutiny", "sanction",
    # Macro negative
    "rate hike", "hawkish", "tightening", "inflation", "recession",
    "stagflation", "default", "crisis",
}

# Patterns that indicate market-moving events (regardless of sentiment)
_MARKET_MOVING_PATTERNS = [
    (r"\b(sec|cftc|doj|ofac)\b.*\b(charges?|lawsuit|investigation)\b", "regulatory_action"),
    (r"\b(etf|exchange.traded fund)\b.*\b(approv|appli|fil)", "etf_signal"),
    (r"\b(merger|acquisition|acqui[rs]|takeover|buyout)\b", "ma_event"),
    (r"\b(earnings?|eps|revenue)\b.*\b(beat|miss|exceed|below)\b", "earnings_signal"),
    (r"\b(bankruptcy|chapter 11|insolvency|liquidat)\b", "insolvency"),
    (r"\b(hack|exploit|breach|stolen|theft)\b", "security_incident"),
    (r"\b(partnership|integrat|collaborat)\b.*\b(bitcoin|ethereum|crypto|blockchain)\b", "crypto_adoption"),
    (r"\b(institutional|blackrock|fidelity|vanguard|jpmorgan|goldman)\b.*\b(bitcoin|crypto|btc)\b", "institutional_crypto"),
]


@dataclass
class SentimentResult:
    text_snippet:  str            # first 100 chars
    score:         float          # -1 to +1
    label:         str            # positive | negative | neutral | mixed
    confidence:    float          # 0-1
    method:        str            # lexicon | vader | finbert | ensemble
    signals:       list[str] = field(default_factory=list)  # market-moving signal types
    is_market_moving: bool = False

    def to_dict(self) -> dict:
        return {
            "text_snippet":    self.text_snippet[:100],
            "score":           round(self.score, 4),
            "label":           self.label,
            "confidence":      round(self.confidence, 3),
            "method":          self.method,
            "signals":         self.signals,
            "is_market_moving": self.is_market_moving,
        }


class SentimentClassifier:
    """
    Multi-method financial sentiment classifier.

    Usage
    -----
    clf = SentimentClassifier()
    await clf.load()
    result = clf.classify("Coinbase receives SEC Wells Notice...")
    """

    def __init__(self, use_finbert: bool = False):
        self._use_finbert = use_finbert
        self._finbert     = None
        self._vader       = None

    async def load(self) -> None:
        """Load optional heavy models (called once at startup)."""
        # Try VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("sentiment_classifier.vader_loaded")
        except ImportError:
            logger.info("sentiment_classifier.vader_unavailable")

        # Try FinBERT (optional, GPU-heavy)
        if self._use_finbert:
            try:
                from transformers import pipeline
                self._finbert = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    top_k=None,
                )
                logger.info("sentiment_classifier.finbert_loaded")
            except Exception as e:
                logger.warning("sentiment_classifier.finbert_unavailable", error=str(e))

    def classify(self, text: str) -> SentimentResult:
        """Classify sentiment of financial text."""
        text_lower = text.lower()

        # 1. Lexicon score
        lex_score = self._lexicon_score(text_lower)

        # 2. VADER score
        vader_score = self._vader_score(text) if self._vader else None

        # 3. FinBERT score (sync wrapper)
        finbert_score = None
        if self._finbert and len(text.split()) >= 10:
            finbert_score = self._finbert_score(text)

        # Ensemble
        score, method = self._ensemble(lex_score, vader_score, finbert_score)

        # Label
        label = self._score_to_label(score)

        # Confidence: based on how many methods agree
        confidence = self._compute_confidence(lex_score, vader_score, finbert_score, score)

        # Market-moving signals
        signals = self._detect_signals(text_lower)
        is_mm   = len(signals) > 0

        return SentimentResult(
            text_snippet=text[:100],
            score=round(score, 4),
            label=label,
            confidence=round(confidence, 3),
            method=method,
            signals=signals,
            is_market_moving=is_mm,
        )

    def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.classify(t) for t in texts]

    # ------------------------------------------------------------------

    @staticmethod
    def _lexicon_score(text_lower: str) -> float:
        bull_hits = sum(1 for t in _BULLISH_TERMS if t in text_lower)
        bear_hits = sum(1 for t in _BEARISH_TERMS if t in text_lower)
        total = bull_hits + bear_hits
        if total == 0:
            return 0.0
        return (bull_hits - bear_hits) / total

    def _vader_score(self, text: str) -> Optional[float]:
        if not self._vader:
            return None
        scores = self._vader.polarity_scores(text)
        return scores["compound"]

    def _finbert_score(self, text: str) -> Optional[float]:
        if not self._finbert:
            return None
        try:
            # Truncate to 512 tokens (FinBERT limit)
            short = " ".join(text.split()[:400])
            results = self._finbert(short)[0]
            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            score = sum(r["score"] * label_map.get(r["label"], 0) for r in results)
            return score
        except Exception as e:
            logger.debug("finbert.classify_error", error=str(e))
            return None

    @staticmethod
    def _ensemble(
        lex: float,
        vader: Optional[float],
        finbert: Optional[float],
    ) -> tuple[float, str]:
        if finbert is not None and vader is not None:
            score  = lex * 0.2 + vader * 0.2 + finbert * 0.6
            method = "ensemble"
        elif finbert is not None:
            score  = lex * 0.3 + finbert * 0.7
            method = "finbert+lexicon"
        elif vader is not None:
            score  = lex * 0.4 + vader * 0.6
            method = "vader+lexicon"
        else:
            score  = lex
            method = "lexicon"
        return score, method

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.25:
            return "positive"
        if score <= -0.25:
            return "negative"
        if abs(score) < 0.05:
            return "neutral"
        return "mixed"

    @staticmethod
    def _compute_confidence(
        lex: float,
        vader: Optional[float],
        finbert: Optional[float],
        final: float,
    ) -> float:
        scores = [s for s in [lex, vader, finbert] if s is not None]
        if len(scores) <= 1:
            return 0.6
        # Confidence = 1 - std of component scores (normalized)
        import statistics
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        return max(0.3, min(1.0, 1.0 - std))

    @staticmethod
    def _detect_signals(text_lower: str) -> list[str]:
        detected = []
        for pattern, signal_type in _MARKET_MOVING_PATTERNS:
            if re.search(pattern, text_lower):
                detected.append(signal_type)
        return detected
