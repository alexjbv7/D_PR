"""
Pure factor functions used by MultiFactorConfirmation.

Each factor returns a Boolean (pass / fail).  They are kept as free functions
(not methods) so they can be unit-tested in isolation and composed in
different orders by experimental confirmation policies.

Factors (in canonical early-reject order)
-----------------------------------------
1. ``has_primary_direction(signal)``     — direction ∈ {-1, +1}
2. ``regime_stable(regime, threshold)``  — stability_60bar ≥ threshold
3. ``macro_coherent(direction, symbol_is_crypto, macro)`` — direction agrees with macro
4. ``meta_label_confident(p_correct, threshold)`` — meta-labeler p_correct ≥ threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

__all__ = [
    "RegimeSnapshot",
    "MacroSnapshot",
    "has_primary_direction",
    "regime_stable",
    "macro_coherent",
    "meta_label_confident",
]


@dataclass(frozen=True)
class RegimeSnapshot:
    """Minimal regime view for confirmation.

    stability_60bar : Decimal in [0, 1]
        Stability of the regime probability vector over the last 60 bars.
        Computed by context-engine (GMM) as the inverse mean entropy of the
        rolling probability vector.
    label : str
        Regime label (informational; not used in the gate but logged).
    """
    stability_60bar: Decimal
    label:           str = "unknown"


@dataclass(frozen=True)
class MacroSnapshot:
    """Macro context used for coherence checks.

    regime : str
        One of {"risk_on", "risk_off", "neutral"}.  Source: macroeconomic
        service.
    """
    regime: str = "neutral"


# ----------------------------------------------------------------------
# Factors
# ----------------------------------------------------------------------

def has_primary_direction(direction: int) -> bool:
    """True if the primary signal indicates long (+1) or short (-1)."""
    return direction in (-1, 1)


def regime_stable(
    regime:   Optional[RegimeSnapshot],
    threshold: Decimal = Decimal("0.6"),
) -> bool:
    """True if regime is sufficiently stable to trust the primary signal."""
    if regime is None:
        return False
    return regime.stability_60bar >= threshold


def macro_coherent(
    direction:        int,
    symbol_is_crypto: bool,
    macro:            Optional[MacroSnapshot],
) -> bool:
    """Coherence rules:

    * Crypto symbols → always coherent (MVP: no macro coupling assumed).
    * Long equity + macro=risk_off → incoherent.
    * Short equity + macro=risk_on → incoherent (informational; conservative).
    * Otherwise → coherent.
    """
    if symbol_is_crypto:
        return True
    if macro is None:
        return True  # benefit of the doubt when macro is unavailable
    if direction > 0 and macro.regime == "risk_off":
        return False
    if direction < 0 and macro.regime == "risk_on":
        return False
    return True


def meta_label_confident(
    p_correct: Any,
    threshold: Decimal = Decimal("0.55"),
) -> bool:
    """True if meta-labeler probability ≥ threshold."""
    if p_correct is None:
        return False
    try:
        return Decimal(str(p_correct)) >= threshold
    except Exception:  # noqa: BLE001
        return False
