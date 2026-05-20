"""
Allocator sub-package — Thompson sampling over horizons (intraday/swing/daily).

Modules
-------
posterior        — BetaPosterior (immutable, lazy decay 0.99/day, Decimal precision)
thompson         — ThompsonAllocator.choose() decides which horizon executes
repository       — AllocatorRepository: state load/save + idempotent audit log
update_consumer  — Async consumer of execution.result → posterior updates

Design (ADR-032)
----------------
* Warm-start priors Beta(20, 20) → mean=0.5, var≈6.1e-3 (informative but not rigid).
* Decay 0.99/day → half-life ≈ 69 days, weight at 90d ≈ 0.40.
* Floor at Beta(1, 1) to prevent the posterior from collapsing to a point mass.
* Lazy decay: applied on every consult (sample/mean/variance), not at write.
  This keeps writes O(1) and reads correct for variable wall-clock gaps.

The allocator's choose() is on the hot path (latency budget < 5 ms p99).
All DB I/O is restricted to the update_consumer (off the hot path); choose()
hits only a per-process LRU cache backed by AllocatorRepository.
"""
from .posterior import BetaPosterior, DECAY_PER_DAY, MIN_ALPHA, MIN_BETA
from .thompson  import ThompsonAllocator, AllocatorDecision

__all__ = [
    "BetaPosterior",
    "DECAY_PER_DAY",
    "MIN_ALPHA",
    "MIN_BETA",
    "ThompsonAllocator",
    "AllocatorDecision",
]
