"""
BetaPosterior — immutable Beta(α, β) with exponential decay.

Mathematical contract
---------------------
* α, β  : Decimal, both ≥ 1.0 (floor enforced after decay).
* mean  : α / (α + β).
* var   : αβ / ((α + β)² · (α + β + 1)).
* decay : (α, β) → (α · 0.99^Δd, β · 0.99^Δd) where Δd = days since last_update_ts.
* sample: Thompson draw via numpy.Generator.beta() with decay applied at read time.

Immutability
------------
The class is a frozen dataclass.  All "mutations" return a new instance.
This makes the object trivially thread-/coroutine-safe and avoids the
"posterior aged between writes" bug where decay applied at save time
but not at load time produces inconsistent posteriors.

Decay choice (ADR-032)
----------------------
DECAY_PER_DAY = 0.99 → half-life ≈ 69 days, weight at 90 days ≈ 0.404.
Matches CLAUDE.md §4.10 multi-agent "ventana 90d decayed".

Floor at Beta(1, 1) — uniform distribution.  Without a floor, the posterior
can collapse to (0, 0) under aggressive decay + low evidence, producing
degenerate samples / NaN variance.

Concurrency
-----------
Persistence (load/save) is delegated to AllocatorRepository.  This class
is pure logic; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Literal

import numpy as np

__all__ = ["BetaPosterior", "DECAY_PER_DAY", "MIN_ALPHA", "MIN_BETA"]

# Decimal precision: 28 is the default for Python's Decimal; explicit for clarity.
getcontext().prec = 28

DECAY_PER_DAY: Decimal = Decimal("0.99")
MIN_ALPHA:     Decimal = Decimal("1.0")
MIN_BETA:      Decimal = Decimal("1.0")

_OUTCOME = Literal["win", "loss"]


@dataclass(frozen=True)
class BetaPosterior:
    """Immutable Beta(α, β) posterior with lazy exponential decay.

    Parameters
    ----------
    alpha : Decimal
        Pseudo-wins (incl. prior).  Must be ≥ MIN_ALPHA.
    beta : Decimal
        Pseudo-losses (incl. prior).  Must be ≥ MIN_BETA.
    last_update_ts : datetime
        Timezone-aware UTC timestamp of the last (decayed-then-)update.
    """

    alpha:          Decimal
    beta:           Decimal
    last_update_ts: datetime

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:  # noqa: D401
        if self.last_update_ts.tzinfo is None:
            raise ValueError("BetaPosterior.last_update_ts must be tz-aware UTC")
        if self.alpha < MIN_ALPHA:
            raise ValueError(f"alpha={self.alpha} < MIN_ALPHA={MIN_ALPHA}")
        if self.beta < MIN_BETA:
            raise ValueError(f"beta={self.beta} < MIN_BETA={MIN_BETA}")

    # ------------------------------------------------------------------
    # Decay & update (return new instance)
    # ------------------------------------------------------------------

    def decayed_to(self, ts: datetime) -> "BetaPosterior":
        """Return the posterior with decay applied from last_update_ts → ts.

        If ts ≤ last_update_ts the original is returned unchanged
        (no decay backwards in time).
        """
        if ts < self.last_update_ts:
            return self
        delta_seconds = (ts - self.last_update_ts).total_seconds()
        if delta_seconds == 0:
            return self
        days   = Decimal(str(delta_seconds / 86400.0))
        factor = DECAY_PER_DAY ** days
        new_alpha = max(MIN_ALPHA, self.alpha * factor)
        new_beta  = max(MIN_BETA,  self.beta  * factor)
        return replace(
            self,
            alpha=new_alpha,
            beta=new_beta,
            last_update_ts=ts,
        )

    def update(self, outcome: _OUTCOME, ts: datetime) -> "BetaPosterior":
        """Apply decay up to ts, then increment α or β by 1 based on outcome.

        Returns a new instance — never mutates self.
        """
        d = self.decayed_to(ts)
        if outcome == "win":
            return replace(d, alpha=d.alpha + Decimal("1"))
        if outcome == "loss":
            return replace(d, beta=d.beta + Decimal("1"))
        raise ValueError(f"outcome must be 'win' or 'loss', got {outcome!r}")

    # ------------------------------------------------------------------
    # Sampling & summary stats (apply decay first, then compute)
    # ------------------------------------------------------------------

    def sample(self, ts: datetime, rng: np.random.Generator) -> float:
        """Thompson sample from the decayed posterior at ts.

        Parameters
        ----------
        ts : datetime
            Wall-clock used to apply decay.  Use the same ts for all
            horizons in one decision to make samples comparable.
        rng : numpy.random.Generator
            Seedable RNG; pass np.random.default_rng(seed) for reproducibility.
        """
        d = self.decayed_to(ts)
        return float(rng.beta(float(d.alpha), float(d.beta)))

    @property
    def mean(self) -> Decimal:
        total = self.alpha + self.beta
        if total <= 0:
            return Decimal("0.5")
        return self.alpha / total

    @property
    def variance(self) -> Decimal:
        a, b  = self.alpha, self.beta
        total = a + b
        if total <= 0:
            return Decimal("0")
        return (a * b) / (total * total * (total + Decimal("1")))
