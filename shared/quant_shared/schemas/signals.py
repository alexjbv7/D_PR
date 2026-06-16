"""
TradeSignal — schema canónico de señal de trading.

Compartido entre research/ (backtesting genera señales) y
platform/ (strategy-orchestrator emite señales).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class UncalibratedSignalError(ValueError):
    """
    Se intentó dimensionar capital (Kelly) a partir de un ``p_win`` NO calibrado.

    Blocker duro del arbitraje (Documento D, R-02): "Calibrar p_win antes de
    exponerlo al PositionSizer = blocker duro". El softmax de Q-values del DQN es
    un proxy ordinal; alimentarlo a Kelly puede producir tamaños de posición sin
    acotar (riesgo de ruina). Calibra la señal (IsotonicCalibrator OOS por fold) y
    marca ``p_win_calibrated=True`` antes de sizing.
    """


def require_calibrated_signal(signal: "TradeSignal") -> None:
    """
    Lanza ``UncalibratedSignalError`` si ``signal.p_win`` no está calibrado.

    Punto único de enforcement del guard de calibración. Cualquier capa que
    convierta una ``TradeSignal`` en input de Kelly/sizing DEBE llamar esto primero
    (ver ``portfolio.sizing.edge_posterior_from_signal``).
    """
    if not getattr(signal, "p_win_calibrated", False):
        raise UncalibratedSignalError(
            f"Señal '{getattr(signal, 'strategy', '?')}' sobre "
            f"'{getattr(signal, 'symbol', '?')}' tiene p_win sin calibrar "
            f"(p_win_calibrated=False). No se permite Kelly/sizing sobre p_win crudo "
            f"(arbitraje D / R-02). Usa vol-target puro o calibra primero."
        )


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
    p_win:         float           # P(win) en [0,1]. Calibrada SOLO si p_win_calibrated=True.
                                   # Si es False es un proxy ordinal (p.ej. softmax de Q-values):
                                   # NO tiene sentido frecuentista y NO debe alimentar Kelly (arbitraje D, R-02).
    p_win_raw:     float           # probabilidad antes de meta-labeler / calibración
    # Honestidad de calibración (guard anti-ruina, arbitraje D / R-02.a):
    # True SOLO cuando un calibrador OOS (IsotonicCalibrator) fijó p_win. El productor
    # de la señal es responsable de ponerlo en True; por defecto se asume NO calibrado.
    p_win_calibrated: bool = False

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

    def require_calibrated(self) -> None:
        """
        Guard anti-ruina (arbitraje D / R-02.b): lanza ``UncalibratedSignalError``
        si esta señal no tiene ``p_win`` calibrado. Llamar SIEMPRE antes de exponer
        ``p_win`` a cualquier cálculo de Kelly / position sizing.
        """
        require_calibrated_signal(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id":   self.signal_id,
            "symbol":      self.symbol,
            "direction":   self.direction.value,
            "p_win":       round(self.p_win, 4),
            "p_win_raw":   round(self.p_win_raw, 4),
            "p_win_calibrated": self.p_win_calibrated,
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
