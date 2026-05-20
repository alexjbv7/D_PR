"""
Multi-factor confirmation — 4-layer gate before the allocator is consulted.

See ADR-033 for the ordering rationale (cheapest first → early reject).

Public API
----------
MultiFactorConfirmation   — orchestrator
ConfirmationResult        — outcome dataclass
factors                   — pure functions, one per factor
"""
from .multi_factor import MultiFactorConfirmation, ConfirmationResult
from . import factors

__all__ = [
    "MultiFactorConfirmation",
    "ConfirmationResult",
    "factors",
]
