"""
ProbabilityAnalyzer — Convierte precios de Polymarket en señales de trading.

Lógica:
  1. Filtra mercados por asset, calidad y proximidad temporal
  2. Combina múltiples mercados del mismo asset (Bayesian update)
  3. Genera ProbabilitySignal con prior actualizado y confianza
  4. Calibra la probabilidad contra historical accuracy de Polymarket

Calibración empírica (Polymarket 2023-2024):
  - Bien calibrado para p ∈ [0.20, 0.80]
  - Sobreestima tail events (p < 0.10 o p > 0.90)
  - Aplica shrinkage toward 0.5 para extremos

Integración con MacroSignalEngine:
  - P(recesión) de Polymarket ajusta recession_prob
  - P(BTC > X) ajusta favored_assets score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from .polymarket_parser import PolymarketEvent

logger = structlog.get_logger(__name__)


@dataclass
class ProbabilitySignal:
    asset:         str              # BTC | ETH | SOL | MACRO | etc.
    probability:   float           # 0-1, calibrated
    raw_prob:      float           # pre-calibration
    direction:     Optional[str]   # bullish | bearish | neutral
    confidence:    float           # 0-1 based on OI and market count
    horizon_days:  Optional[float] # days to resolution
    num_markets:   int             # markets aggregated
    description:   str

    def as_bias_score(self) -> float:
        """
        Convert to a directional score for use in signal engines.
        Returns -1 to +1: +1 = strongly bullish, -1 = strongly bearish.
        """
        if self.direction == "bullish":
            return (self.probability - 0.5) * 2 * self.confidence
        elif self.direction == "bearish":
            # High P(bearish event) → negative score
            return -(self.probability - 0.5) * 2 * self.confidence
        return 0.0

    def to_dict(self) -> dict:
        return {
            "asset":        self.asset,
            "probability":  round(self.probability, 4),
            "raw_prob":     round(self.raw_prob, 4),
            "direction":    self.direction,
            "confidence":   round(self.confidence, 3),
            "horizon_days": round(self.horizon_days, 1) if self.horizon_days else None,
            "num_markets":  self.num_markets,
            "bias_score":   round(self.as_bias_score(), 4),
            "description":  self.description,
        }


class ProbabilityAnalyzer:
    """
    Aggregates Polymarket events into asset-level probability signals.

    Usage
    -----
    analyzer = ProbabilityAnalyzer()
    signals = analyzer.analyze(events)
    macro_recession_prob = analyzer.get_recession_prob(events)
    """

    # Polymarket calibration: shrink extreme probs toward 0.5
    # Based on empirical analysis of resolved markets
    _SHRINKAGE = {
        "extreme_lo": 0.08,   # p < this → shrink toward 0.15
        "extreme_hi": 0.92,   # p > this → shrink toward 0.85
        "shrink_lo":  0.15,
        "shrink_hi":  0.85,
        "shrink_alpha": 0.4,  # blend factor: 0=raw, 1=full shrink
    }

    # Max days to resolution to include in signals (too far → less signal)
    _MAX_HORIZON_DAYS = 90

    def analyze(self, events: list[PolymarketEvent]) -> list[ProbabilitySignal]:
        """
        Analyze all events and return per-asset probability signals.
        """
        # Group by asset
        by_asset: dict[str, list[PolymarketEvent]] = {}
        for evt in events:
            if not evt.is_tradeable:
                continue
            key = evt.asset or "OTHER"
            by_asset.setdefault(key, []).append(evt)

        signals = []
        for asset, asset_events in by_asset.items():
            sig = self._aggregate_asset_signal(asset, asset_events)
            if sig:
                signals.append(sig)

        logger.info("probability_analyzer.signals",
                    count=len(signals),
                    assets=[s.asset for s in signals])
        return signals

    def get_recession_prob(self, events: list[PolymarketEvent]) -> Optional[float]:
        """
        Extract market-implied recession probability from MACRO markets.
        Returns None if no relevant markets found.
        """
        recession_keywords = ["recession", "gdp contraction", "negative gdp",
                               "economic downturn"]
        macro_events = [
            e for e in events
            if e.category == "macro"
            and e.is_tradeable
            and any(kw in e.question.lower() for kw in recession_keywords)
        ]
        if not macro_events:
            return None

        # Weight by OI
        total_oi = sum(e.open_interest for e in macro_events)
        if total_oi == 0:
            return None

        weighted_prob = sum(
            e.probability * e.open_interest / total_oi
            for e in macro_events
        )
        calibrated = self._calibrate(weighted_prob)
        logger.info("recession_prob.polymarket",
                    raw=round(weighted_prob, 4),
                    calibrated=round(calibrated, 4),
                    markets=len(macro_events))
        return calibrated

    def get_etf_approval_prob(self, events: list[PolymarketEvent]) -> dict[str, float]:
        """
        Extract ETF approval probabilities by asset.
        Returns {asset: prob} dict.
        """
        etf_events = [
            e for e in events
            if "etf" in e.question.lower() and e.direction == "bullish"
        ]
        result: dict[str, float] = {}
        by_asset: dict[str, list[PolymarketEvent]] = {}
        for e in etf_events:
            key = e.asset or "UNKNOWN"
            by_asset.setdefault(key, []).append(e)

        for asset, evts in by_asset.items():
            total_oi = sum(e.open_interest for e in evts)
            if total_oi > 0:
                w_prob = sum(e.probability * e.open_interest / total_oi for e in evts)
                result[asset] = round(self._calibrate(w_prob), 4)
        return result

    # ------------------------------------------------------------------

    def _aggregate_asset_signal(
        self, asset: str, events: list[PolymarketEvent]
    ) -> Optional[ProbabilitySignal]:
        if not events:
            return None

        # Filter by horizon
        near_term = [e for e in events if self._days_to_resolution(e) <= self._MAX_HORIZON_DAYS]
        if not near_term:
            near_term = events  # fallback: use all

        # Separate bullish / bearish markets
        bull_events = [e for e in near_term if e.direction == "bullish"]
        bear_events = [e for e in near_term if e.direction == "bearish"]

        total_oi = sum(e.open_interest for e in near_term) + 1e-9

        # Weighted probability
        if bull_events:
            # Use bullish events (P(price > X))
            dominant_events = bull_events
            direction = "bullish"
        elif bear_events:
            dominant_events = bear_events
            direction = "bearish"
        else:
            dominant_events = near_term
            direction = "neutral"

        dom_oi = sum(e.open_interest for e in dominant_events) + 1e-9
        raw_prob = sum(
            e.probability * e.open_interest / dom_oi
            for e in dominant_events
        )
        calibrated_prob = self._calibrate(raw_prob)

        # Confidence: based on total OI and number of markets
        confidence = min(1.0, (len(near_term) / 5) * 0.5 + (min(total_oi, 1_000_000) / 1_000_000) * 0.5)

        # Horizon
        horizons = [self._days_to_resolution(e) for e in near_term if self._days_to_resolution(e) > 0]
        avg_horizon = sum(horizons) / len(horizons) if horizons else None

        # Description: use highest-OI market's question
        top = max(near_term, key=lambda e: e.open_interest)

        return ProbabilitySignal(
            asset=asset,
            probability=round(calibrated_prob, 4),
            raw_prob=round(raw_prob, 4),
            direction=direction,
            confidence=round(confidence, 3),
            horizon_days=round(avg_horizon, 1) if avg_horizon else None,
            num_markets=len(near_term),
            description=f"[{top.slug}] {top.question[:80]}",
        )

    def _calibrate(self, p: float) -> float:
        """Apply shrinkage calibration for extreme probabilities."""
        s = self._SHRINKAGE
        alpha = s["shrink_alpha"]
        if p < s["extreme_lo"]:
            return p * (1 - alpha) + s["shrink_lo"] * alpha
        if p > s["extreme_hi"]:
            return p * (1 - alpha) + s["shrink_hi"] * alpha
        return p

    @staticmethod
    def _days_to_resolution(evt: PolymarketEvent) -> float:
        """Days from now to event resolution. Returns 999 if unknown."""
        if not evt.end_date:
            return 999.0
        try:
            # Handle ISO format and simple date strings
            end_str = evt.end_date.replace("Z", "+00:00")
            end_dt = datetime.fromisoformat(end_str)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (end_dt - now).total_seconds() / 86400
            return max(0.0, delta)
        except (ValueError, AttributeError):
            return 999.0
