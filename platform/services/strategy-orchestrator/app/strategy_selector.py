"""
StrategySelector — Selección dinámica de estrategias via Thompson Sampling.

Mantiene un modelo Beta(α, β) por estrategia donde:
  α = wins en ventana 90d (decayed)
  β = losses en ventana 90d (decayed)

Incorpora filtros de régimen macro y market regime para desactivar
estrategias que históricamente underperforman en el contexto actual.

Decisiones de diseño:
  - Thompson Sampling sobre Bayesian bandits: exploración/explotación natural.
  - Decay exponencial (λ=0.99 diario) para ponderar recencia.
  - Filtros deterministas primero, bandit segundo: si el macro dice recesión,
    no se explora en estrategias agresivas sin importar el prior.
  - RL no decide los filtros — reglas explícitas de riesgo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class StrategyPerformance:
    """Estado Beta(α, β) de una estrategia."""
    name:         str
    wins:         float = 1.0   # pseudo-count (prior α)
    losses:       float = 1.0   # pseudo-count (prior β)
    total_pnl:    float = 0.0
    last_updated: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        return self.wins / (self.wins + self.losses)

    @property
    def sample(self) -> float:
        """Thompson sample from Beta(α, β)."""
        return float(np.random.beta(self.wins, self.losses))

    @property
    def expected_value(self) -> float:
        return self.wins / (self.wins + self.losses)

    def update(self, won: bool, pnl: float, decay: float = 0.99) -> None:
        """Apply decay then update with new observation."""
        self.wins   *= decay
        self.losses *= decay
        if won:
            self.wins   += 1.0
        else:
            self.losses += 1.0
        self.total_pnl += pnl
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class RegimeFilter:
    """Restricciones de qué estrategias corren en cada régimen."""
    # Estrategias bloqueadas por régimen macro
    recession_blocked:  list[str] = field(default_factory=lambda: [
        "momentum_ml", "regime_adaptive"
    ])
    high_vol_blocked:   list[str] = field(default_factory=lambda: [
        "mean_reversion_funding"
    ])
    # Estrategias favorecidas por régimen de mercado
    bull_trend_boost:   list[str] = field(default_factory=lambda: [
        "momentum_ml", "whale_follow"
    ])
    bear_trend_boost:   list[str] = field(default_factory=lambda: [
        "mean_reversion_funding"
    ])
    high_vol_boost:     list[str] = field(default_factory=lambda: [
        "regime_adaptive"
    ])


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class StrategySelector:
    """
    Selecciona estrategias activas y sus allocations via Thompson Sampling
    con filtros de régimen deterministas.

    Usage
    -----
    selector = StrategySelector(strategies=["momentum_ml", ...])
    allocations = selector.select(
        available=["momentum_ml", "mean_reversion_funding", ...],
        recession_prob=0.2,
        market_regime="bull_trend",
        max_active=3,
    )
    selector.record_outcome("momentum_ml", won=True, pnl=0.034)
    """

    def __init__(
        self,
        strategies: list[str],
        decay_rate:       float = 0.99,
        regime_filter:    Optional[RegimeFilter] = None,
        recession_thresh: float = 0.60,
        high_vol_thresh:  float = 0.75,
        boost_mult:       float = 1.5,
        seed:             Optional[int] = None,
    ):
        self._decay         = decay_rate
        self._rf            = regime_filter or RegimeFilter()
        self._rec_thresh    = recession_thresh
        self._hv_thresh     = high_vol_thresh
        self._boost_mult    = boost_mult
        self._rng           = np.random.default_rng(seed)

        # Beta bandit state per strategy
        self._perf: dict[str, StrategyPerformance] = {
            s: StrategyPerformance(name=s) for s in strategies
        }

        logger.info("strategy_selector.init", strategies=strategies)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        available:      list[str],
        recession_prob: float = 0.0,
        market_regime:  str   = "unknown",
        vol_percentile: float = 0.5,
        max_active:     int   = 3,
    ) -> dict[str, float]:
        """
        Returns dict {strategy_name: allocation_fraction} that sums to 1.0.
        Only strategies in `available` are considered.
        """
        candidates = self._apply_filters(
            available, recession_prob, market_regime, vol_percentile
        )

        if not candidates:
            logger.warning("strategy_selector.no_candidates",
                           recession_prob=recession_prob,
                           regime=market_regime)
            return {}

        # Thompson samples with regime boost
        scores = {}
        for name in candidates:
            perf = self._perf.get(name)
            if perf is None:
                scores[name] = 0.5
                continue
            sample = perf.sample
            # Apply boost
            if market_regime == "bull_trend" and name in self._rf.bull_trend_boost:
                sample = min(1.0, sample * self._boost_mult)
            elif market_regime == "bear_trend" and name in self._rf.bear_trend_boost:
                sample = min(1.0, sample * self._boost_mult)
            elif vol_percentile > self._hv_thresh and name in self._rf.high_vol_boost:
                sample = min(1.0, sample * self._boost_mult)
            scores[name] = sample

        # Top-K by sample score
        sorted_strats = sorted(scores, key=lambda x: scores[x], reverse=True)
        selected = sorted_strats[:max_active]

        # Softmax normalization of raw scores for allocations
        raw = np.array([scores[s] for s in selected])
        exp_raw = np.exp(raw - raw.max())  # numerically stable
        allocs = exp_raw / exp_raw.sum()

        allocations = {s: float(round(a, 4)) for s, a in zip(selected, allocs)}

        logger.info("strategy_selector.selected",
                    allocations=allocations,
                    recession_prob=recession_prob,
                    regime=market_regime)
        return allocations

    def record_outcome(self, strategy: str, won: bool, pnl: float = 0.0) -> None:
        """Update Beta posterior after observing a trade outcome."""
        if strategy not in self._perf:
            self._perf[strategy] = StrategyPerformance(name=strategy)
        self._perf[strategy].update(won=won, pnl=pnl, decay=self._decay)
        logger.debug("strategy_selector.outcome",
                     strategy=strategy, won=won, pnl=pnl,
                     new_wr=self._perf[strategy].win_rate)

    def get_stats(self) -> dict[str, dict]:
        return {
            name: {
                "win_rate":    round(p.win_rate, 4),
                "wins":        round(p.wins, 2),
                "losses":      round(p.losses, 2),
                "total_pnl":   round(p.total_pnl, 4),
                "last_updated": p.last_updated.isoformat() if p.last_updated else None,
            }
            for name, p in self._perf.items()
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        available:      list[str],
        recession_prob: float,
        market_regime:  str,
        vol_percentile: float,
    ) -> list[str]:
        """Remove strategies blocked by current regime context."""
        blocked: set[str] = set()

        if recession_prob >= self._rec_thresh:
            blocked.update(self._rf.recession_blocked)
            logger.info("strategy_selector.filter.recession",
                        prob=recession_prob, blocked=list(blocked))

        if vol_percentile >= self._hv_thresh:
            blocked.update(self._rf.high_vol_blocked)

        return [s for s in available if s not in blocked]
