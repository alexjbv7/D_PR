"""
AllocationEngine — Kelly-adjusted capital allocation entre estrategias activas.

Recibe las fracciones de Thompson Sampling del StrategySelector y las ajusta
por volatilidad de portfolio, drawdown actual y correlación inter-estrategia.

Decisiones de diseño:
  - Fractional Kelly (0.25) como base: nunca full Kelly en producción.
  - Vol scaling: target_vol / realized_vol ajusta el leverage efectivo.
  - Drawdown brake: si DD > umbral, reduce sizing exponencialmente.
  - Correlation penalty: pares con corr > 0.7 se tratan como 1 posición.
  - Output: dict {strategy → qty_in_base_currency}, listo para executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AllocationConfig:
    total_capital:     float = 10_000.0    # USD
    kelly_fraction:    float = 0.25        # fractional Kelly
    target_vol:        float = 0.15        # annualized vol target
    max_leverage:      float = 2.0
    max_single_alloc:  float = 0.40        # max fraction to any one strategy
    min_single_alloc:  float = 0.05        # min fraction (below = skip)
    dd_soft_brake:     float = 0.05        # 5% DD → start reducing
    dd_hard_brake:     float = 0.10        # 10% DD → 0 new positions
    corr_threshold:    float = 0.70        # treat as same position above
    risk_free_rate:    float = 0.05        # annualized


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AllocationEngine:
    """
    Converts Thompson-Sampling allocations (fractions) into position sizes
    after adjusting for volatility, drawdown, and correlation.

    Usage
    -----
    engine = AllocationEngine(config)
    sizes = engine.compute(
        allocations={"momentum_ml": 0.6, "whale_follow": 0.4},
        realized_vol={"momentum_ml": 0.18, "whale_follow": 0.22},
        correlations={"momentum_ml:whale_follow": 0.45},
        current_drawdown=0.03,
        portfolio_value=12_000,
    )
    # → {"momentum_ml": 2880.0, "whale_follow": 1440.0}  (USD notional)
    """

    def __init__(self, config: Optional[AllocationConfig] = None):
        self.cfg = config or AllocationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        allocations:       dict[str, float],    # {strategy: fraction 0-1}
        realized_vol:      dict[str, float],    # {strategy: annualized vol}
        correlations:      dict[str, float],    # {"s1:s2": corr}
        current_drawdown:  float = 0.0,         # current DD fraction (positive)
        portfolio_value:   float | None = None,
    ) -> dict[str, float]:
        """
        Returns {strategy: USD notional} sized for current conditions.
        """
        capital = portfolio_value or self.cfg.total_capital

        # 1. Hard brake — no new positions
        if current_drawdown >= self.cfg.dd_hard_brake:
            logger.warning("allocation.hard_brake", dd=current_drawdown)
            return {s: 0.0 for s in allocations}

        # 2. Normalize allocations
        total = sum(allocations.values())
        if total <= 0:
            return {}
        norm = {s: v / total for s, v in allocations.items()}

        # 3. Cap at max single allocation
        norm = self._cap_allocations(norm)

        # 4. Correlation penalty — reduce correlated pairs
        norm = self._apply_correlation_penalty(norm, correlations)

        # 5. Kelly fraction
        norm = {s: v * self.cfg.kelly_fraction for s, v in norm.items()}

        # 6. Vol scaling per strategy
        norm = self._apply_vol_scaling(norm, realized_vol)

        # 7. Drawdown soft brake (smooth reduction)
        dd_mult = self._drawdown_multiplier(current_drawdown)
        norm = {s: v * dd_mult for s, v in norm.items()}

        # 8. Leverage cap
        total_alloc = sum(norm.values())
        if total_alloc > self.cfg.max_leverage:
            scale = self.cfg.max_leverage / total_alloc
            norm = {s: v * scale for s, v in norm.items()}

        # 9. Drop below min
        norm = {s: v for s, v in norm.items() if v >= self.cfg.min_single_alloc}

        # 10. Convert to USD notional
        sizes = {s: round(v * capital, 2) for s, v in norm.items()}

        logger.info("allocation.computed",
                    sizes=sizes, dd=current_drawdown,
                    dd_mult=round(dd_mult, 3), total_notional=sum(sizes.values()))
        return sizes

    def position_size_from_signal(
        self,
        p_win:         float,
        stop_loss_pct: float,
        capital:       float | None = None,
        *,
        p_win_calibrated: bool = False,
    ) -> float:
        """
        Single-trade Kelly sizing from p_win and stop loss.

        f* = (p_win * (1/stop) - (1-p_win)) / (1/stop)
        Applied at kelly_fraction of full Kelly.

        Parameters
        ----------
        p_win_calibrated : bool
            Must be True. Kelly on uncalibrated p_win (e.g. softmax of Q-values)
            is forbidden (R-02 / triaje Y-003). Callers that size from a
            TradeSignal should pass ``signal.p_win_calibrated``.
        """
        if not p_win_calibrated:
            # Y-003: never feed raw ordinal confidence into Kelly.
            from quant_shared.schemas.signals import UncalibratedSignalError
            raise UncalibratedSignalError(
                "position_size_from_signal: p_win_calibrated=False — "
                "Kelly over uncalibrated p_win is blocked (Y-003 / R-02). "
                "Calibrate OOS and pass p_win_calibrated=True."
            )
        cap = capital or self.cfg.total_capital
        if stop_loss_pct <= 0 or p_win <= 0:
            return 0.0
        b = 1.0 / stop_loss_pct   # odds ratio
        q = 1.0 - p_win
        kelly_full = (p_win * b - q) / b
        kelly_full = max(0.0, kelly_full)
        fractional = kelly_full * self.cfg.kelly_fraction
        notional = min(fractional * cap,
                       self.cfg.max_single_alloc * cap)
        return round(notional, 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cap_allocations(self, norm: dict[str, float]) -> dict[str, float]:
        """Cap any strategy at max_single_alloc and renormalize."""
        capped = {s: min(v, self.cfg.max_single_alloc) for s, v in norm.items()}
        total = sum(capped.values())
        if total <= 0:
            return capped
        return {s: v / total for s, v in capped.items()}

    def _apply_correlation_penalty(
        self,
        norm:         dict[str, float],
        correlations: dict[str, float],
    ) -> dict[str, float]:
        """
        For each pair with corr > threshold, reduce the smaller allocation
        by (corr - threshold) / (1 - threshold).
        """
        strategies = list(norm.keys())
        result = dict(norm)
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1:]:
                key = f"{s1}:{s2}"
                alt = f"{s2}:{s1}"
                corr = correlations.get(key, correlations.get(alt, 0.0))
                if corr > self.cfg.corr_threshold:
                    penalty = (corr - self.cfg.corr_threshold) / (1.0 - self.cfg.corr_threshold)
                    # penalize the smaller allocation
                    if result[s1] < result[s2]:
                        result[s1] *= (1.0 - penalty)
                    else:
                        result[s2] *= (1.0 - penalty)
        return result

    def _apply_vol_scaling(
        self,
        norm:         dict[str, float],
        realized_vol: dict[str, float],
    ) -> dict[str, float]:
        """Scale each allocation by target_vol / realized_vol."""
        result = {}
        for s, alloc in norm.items():
            rv = realized_vol.get(s, self.cfg.target_vol)
            rv = max(rv, 0.01)   # floor
            scale = min(self.cfg.target_vol / rv, 2.0)  # cap upward scale
            result[s] = alloc * scale
        return result

    def _drawdown_multiplier(self, dd: float) -> float:
        """
        Linear ramp from 1.0 (no reduction) at dd_soft_brake
        to 0.0 at dd_hard_brake.
        """
        if dd <= self.cfg.dd_soft_brake:
            return 1.0
        if dd >= self.cfg.dd_hard_brake:
            return 0.0
        span = self.cfg.dd_hard_brake - self.cfg.dd_soft_brake
        return 1.0 - (dd - self.cfg.dd_soft_brake) / span
