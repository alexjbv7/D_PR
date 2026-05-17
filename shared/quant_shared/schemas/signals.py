"""
TradeSignal — schema canónico de señal de trading.

Compartido entre research/ (backtesting genera señales) y
platform/ (strategy-orchestrator emite señales).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SignalDirection(str, Enum):
    LONG  = "long"
    SHORT = "short"
    FLAT  = "flat"


class PositionSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


@dataclass
class TradeSignal:
    """Señal de trading generada por el sistema."""

    symbol:        str
    direction:     SignalDirection
    p_win:         float           # probabilidad calibrada [0, 1]
    p_win_raw:     float           # probabilidad antes de meta-labeler

    # Sizing
    kelly_fraction: float = 0.0   # fracción Kelly recomendada
    size_usd:       float = 0.0   # tamaño en USD (si se calcula)

    # Contexto
    strategy:       str   = ""
    regime:         str   = ""
    model_version:  str   = ""
    feature_set_hash: str = ""

    # Metadata
    signal_id:      str   = ""
    ts:             datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    notes:          str   = ""

    # Risk
    stop_loss_pct:  float = 0.02   # 2% default
    take_profit_pct: float = 0.04  # 2:1 R:R default
    confidence:     float = 0.0    # 0-1, confianza adicional del meta-labeler

    def is_actionable(self, min_p_win: float = 0.52, min_confidence: float = 0.5) -> bool:
        """True si la señal supera umbrales mínimos para ejecutar."""
        return (
            self.direction != SignalDirection.FLAT
            and self.p_win >= min_p_win
            and self.confidence >= min_confidence
        )

    def to_dict(self) -> dict:
        return {
            "signal_id":   self.signal_id,
            "symbol":      self.symbol,
            "direction":   self.direction.value,
            "p_win":       round(self.p_win, 4),
            "p_win_raw":   round(self.p_win_raw, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "size_usd":    round(self.size_usd, 2),
            "strategy":    self.strategy,
            "regime":      self.regime,
            "model_version": self.model_version,
            "confidence":  round(self.confidence, 4),
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "ts":          self.ts.isoformat(),
            "notes":       self.notes,
        }
