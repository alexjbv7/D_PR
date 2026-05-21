"""
MultiFactorConfirmation — 4-layer gate (AND aggregation).

Order (ADR-033 — cheapest first)
--------------------------------
1. has_primary_direction  : in-process O(1) check on the signal payload.
2. regime_stable          : O(1) Redis GET (context-engine cache).
3. macro_coherent         : O(1) Redis GET (macroeconomic cache).
4. meta_label_confident   : ~10 ms inference call.

If any factor fails, the remaining (more expensive) factors are NOT evaluated.
This is observable in tests (test_early_reject_*) via the spy on
meta_labeler.predict — it must NOT be called when an earlier factor fails.

Why AND, not weighted vote
--------------------------
A weighted vote with a near-pass elsewhere can compensate for a hard
risk-control failure (e.g. regime breakdown).  We refuse that trade-off.
If a regime is unstable, no amount of model confidence justifies executing.

This is also consistent with CLAUDE.md §12.7 "kill switch" semantics:
any single critical condition stops execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Protocol, runtime_checkable

import structlog

from .factors import (
    MacroSnapshot,
    RegimeSnapshot,
    has_primary_direction,
    macro_coherent,
    meta_label_confident,
    regime_stable,
)

logger = structlog.get_logger(__name__)


# ----------------------------------------------------------------------
# Dependencies (Protocols)
# ----------------------------------------------------------------------

@runtime_checkable
class RegimeRepository(Protocol):
    """Minimal interface required from the regime data source."""
    async def get_current(self, symbol: str, ts: datetime) -> Optional[RegimeSnapshot]: ...


@runtime_checkable
class MacroRepository(Protocol):
    """Minimal interface required from the macro data source."""
    async def get_current(self, ts: datetime) -> Optional[MacroSnapshot]: ...


@runtime_checkable
class MetaLabeler(Protocol):
    """Minimal interface for the secondary classifier."""
    async def predict(
        self,
        signal: Any,
        symbol: str,
        ts:     datetime,
    ) -> Decimal: ...


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ConfirmationResult:
    """Aggregated outcome of one confirmation pass.

    passed : bool
        True iff every factor passed.
    rejected_by : str or None
        Name of the first failing factor; None if passed.
    p_correct : Decimal or None
        Meta-labeler output — populated only if all earlier factors passed.
    """
    passed:      bool
    rejected_by: Optional[str]
    p_correct:   Optional[Decimal] = None


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

class MultiFactorConfirmation:
    """4-factor AND gate with early rejection.

    Parameters
    ----------
    regime_repo : RegimeRepository
    macro_repo  : MacroRepository
    meta_labeler : MetaLabeler
    regime_threshold : Decimal
        Default 0.6 — see CLAUDE.md §13.4 stability_60bar.
    meta_threshold : Decimal
        Default 0.55 — minimum p_correct from the secondary classifier.
    """

    def __init__(
        self,
        regime_repo:      RegimeRepository,
        macro_repo:       MacroRepository,
        meta_labeler:     MetaLabeler,
        regime_threshold: Decimal = Decimal("0.6"),
        meta_threshold:   Decimal = Decimal("0.55"),
    ) -> None:
        self._regime_repo  = regime_repo
        self._macro_repo   = macro_repo
        self._meta_labeler = meta_labeler
        self._regime_thr   = regime_threshold
        self._meta_thr     = meta_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def confirm(
        self,
        signal:           Any,
        symbol:           str,
        direction:        int,
        ts:               datetime,
        symbol_is_crypto: bool = False,
    ) -> ConfirmationResult:
        """Run all factors in canonical order; stop at first failure.

        Parameters
        ----------
        signal : opaque
            Signal payload passed through to the meta-labeler.
        symbol : str
        direction : int  (-1 / 0 / +1)
        ts : datetime
        symbol_is_crypto : bool
            Used by the macro-coherence factor.
        """
        # Factor 1 — primary direction.
        if not has_primary_direction(direction):
            return ConfirmationResult(False, "primary_flat")

        # Factor 2 — regime stability.
        regime = await self._regime_repo.get_current(symbol, ts)
        if not regime_stable(regime, self._regime_thr):
            return ConfirmationResult(False, "regime_unstable")

        # Factor 3 — macro coherence.
        macro = await self._macro_repo.get_current(ts)
        if not macro_coherent(direction, symbol_is_crypto, macro):
            return ConfirmationResult(False, "macro_incoherent")

        # Factor 4 — meta-labeler (expensive; only reached when others pass).
        p_correct = await self._meta_labeler.predict(signal, symbol, ts)
        if not meta_label_confident(p_correct, self._meta_thr):
            return ConfirmationResult(False, "meta_low_confidence", p_correct)

        return ConfirmationResult(True, None, p_correct)


__all__ = ["MultiFactorConfirmation", "ConfirmationResult"]
