"""
Risk Management Module
======================
Implementaciones profesionales de gestión de riesgo:
- Position sizing (fixed, Kelly fraccional, vol-targeting)
- Stop loss dinámico (ATR-based, vol-based, trailing)
- Control de drawdown a nivel portfolio
- Risk parity para múltiples assets

PRINCIPIO: Tu primer trabajo no es ganar, es no quebrar. Sin gestión de riesgo,
ni el mejor modelo del mundo te salva.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# ============================================================================
# POSITION SIZING
# ============================================================================

def fixed_fraction_size(equity: float, fraction: float = 0.95) -> float:
    """Position size fija como fracción del equity."""
    return equity * fraction


def kelly_fractional_size(
    win_prob: float,
    win_loss_ratio: float,
    kelly_fraction: float = 0.25,
    cap: float = 1.0,
) -> float:
    """
    Kelly fraccional. NUNCA uses kelly_fraction=1.0 (Kelly completo) en producción.

    f* = (p*b - q) / b
    donde p = prob ganar, q = 1-p, b = win/loss ratio
    """
    if win_prob <= 0 or win_loss_ratio <= 0:
        return 0.0
    q = 1 - win_prob
    full_kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * kelly_fraction, cap)


def vol_target_size(
    equity: float,
    asset_volatility: float,
    target_volatility: float = 0.15,
    leverage_cap: float = 1.0,
) -> float:
    """
    Position sizing por volatility targeting.

    Mantiene la volatilidad anualizada del portfolio en `target_volatility`.
    Si la vol del activo es 0.30 (30% anual) y queremos 0.15 → posición 50% del equity.
    """
    if asset_volatility <= 0:
        return 0.0
    weight = target_volatility / asset_volatility
    return equity * min(weight, leverage_cap)


# ============================================================================
# STOP LOSS
# ============================================================================

@dataclass
class StopLoss:
    """Configuración de stop loss."""
    initial_atr_mult: float = 2.0    # Stop a 2 ATR del entry
    trailing: bool = True             # Mover stop con el precio (solo en favor)
    take_profit_atr_mult: float = 4.0 # Take profit a 4 ATR (R:R = 2:1)


class StopManager:
    """
    Gestor de stops dinámicos por trade.

    Uso:
        sm = StopManager(StopLoss(initial_atr_mult=2.0, trailing=True))
        sm.open_position(entry_price=100, atr=2, direction=1)
        # En cada nueva barra:
        sm.update(current_price=102, current_atr=2.1)
        if sm.is_stopped(current_low=99):
            ...
    """
    def __init__(self, config: StopLoss):
        self.config = config
        self.entry_price: Optional[float] = None
        self.direction: int = 0  # 1 long, -1 short
        self.stop_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.entry_atr: float = 0.0

    def open_position(self, entry_price: float, atr: float, direction: int):
        self.entry_price = entry_price
        self.direction = direction
        self.entry_atr = atr
        self.stop_price = entry_price - direction * self.config.initial_atr_mult * atr
        self.tp_price = entry_price + direction * self.config.take_profit_atr_mult * atr

    def update(self, current_price: float, current_atr: float):
        """Actualiza stop trailing si aplica."""
        if not self.config.trailing or self.entry_price is None:
            return

        if self.direction == 1:
            # Long: stop sube si el precio sube
            new_stop = current_price - self.config.initial_atr_mult * current_atr
            if new_stop > self.stop_price:
                self.stop_price = new_stop
        elif self.direction == -1:
            new_stop = current_price + self.config.initial_atr_mult * current_atr
            if new_stop < self.stop_price:
                self.stop_price = new_stop

    def check_exit(self, high: float, low: float) -> Optional[str]:
        """
        Comprueba si stop o take-profit fueron tocados durante la barra.
        Returns: 'stop', 'tp', o None.
        """
        if self.entry_price is None:
            return None
        if self.direction == 1:
            if low <= self.stop_price:
                return 'stop'
            if high >= self.tp_price:
                return 'tp'
        elif self.direction == -1:
            if high >= self.stop_price:
                return 'stop'
            if low <= self.tp_price:
                return 'tp'
        return None

    def close(self):
        self.entry_price = None
        self.direction = 0
        self.stop_price = None
        self.tp_price = None


# ============================================================================
# CONTROL DE DRAWDOWN A NIVEL PORTFOLIO
# ============================================================================

@dataclass
class DrawdownGuard:
    """
    Reduce o pausa el trading cuando el drawdown supera umbrales.

    Profesional: hedge funds tienen mandatos de ~10-15% MDD máximo. Si llegas a 10%,
    reduces tamaño 50%. Si llegas a 15%, pausas hasta nuevo high.
    """
    soft_dd_threshold: float = 0.10   # 10% DD: reduce sizing
    soft_dd_multiplier: float = 0.5   # a la mitad
    hard_dd_threshold: float = 0.20   # 20% DD: pausa total

    def __post_init__(self):
        self._equity_peak = 0.0
        self._paused = False

    def update(self, current_equity: float) -> float:
        """
        Returns: multiplicador a aplicar al position sizing (0.0 = pausa).
        """
        self._equity_peak = max(self._equity_peak, current_equity)
        if self._equity_peak == 0:
            return 1.0

        drawdown = (current_equity - self._equity_peak) / self._equity_peak

        if drawdown <= -self.hard_dd_threshold:
            self._paused = True
            return 0.0

        # Si estaba pausado, requerir recuperación a nuevo high
        if self._paused:
            if current_equity >= self._equity_peak * 0.99:
                self._paused = False
            else:
                return 0.0

        if drawdown <= -self.soft_dd_threshold:
            return self.soft_dd_multiplier

        return 1.0


# ============================================================================
# COMBINADOR: POSITION SIZER COMPLETO
# ============================================================================

class IntegratedRiskManager:
    """
    Sizer integrado que combina:
    - Volatility targeting
    - Drawdown guard
    - Cap por trade
    - Cap por exposición total

    Esta es la función que pasas como `position_sizer` al Backtester.
    """
    def __init__(
        self,
        target_vol: float = 0.15,
        max_position_pct: float = 0.95,
        soft_dd: float = 0.10,
        hard_dd: float = 0.20,
    ):
        self.target_vol = target_vol
        self.max_position_pct = max_position_pct
        self.dd_guard = DrawdownGuard(
            soft_dd_threshold=soft_dd,
            hard_dd_threshold=hard_dd,
        )

    def __call__(
        self,
        signal: float,
        equity_pct: float,  # 1.0 al inicio, equity normalizado
        price: float,
        recent_vol: float,
    ) -> float:
        if signal == 0 or pd.isna(recent_vol):
            return 0.0

        # Convertir vol por barra a anualizada (asumiendo barras horarias)
        # Para 1h: ~24*365 = 8760 barras/año
        # Adapta este factor a tu timeframe
        annualized_vol = recent_vol * np.sqrt(8760)

        if annualized_vol <= 0:
            return 0.0

        # Vol-targeted weight
        weight = self.target_vol / annualized_vol
        weight = min(weight, self.max_position_pct)

        # Aplicar drawdown guard (multiplicador 0-1)
        # equity_pct aquí es proxy; en producción pasas equity real
        dd_mult = self.dd_guard.update(equity_pct)

        return weight * dd_mult * abs(signal) * np.sign(signal)
