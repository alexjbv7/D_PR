"""
ThompsonAllocator — picks the horizon that executes when ≥1 signal is confirmed.

Decision flow
-------------
1. For each confirmed horizon, sample p ~ Beta(α, β) decayed to ts.
2. Pick the horizon with the highest sample.
3. Emit an AllocatorDecisionEvent (every call, for audit).
4. Return the decision; the caller acquires the lock and emits the final signal.

Hot-path constraints
--------------------
* No DB I/O — repository.load() reads the per-process cache.
* No await chains beyond the cache + Kafka producer (fire-and-forget).
* Sample cost dominates: numpy.random.Generator.beta() ≈ 1 µs per call.
* p99 budget: 5 ms (see test_allocator_latency).

Coordinator role
----------------
This class is the *only* component allowed to set the "chosen horizon".
The same-symbol/direction lock (locks.horizon_lock) prevents double opens,
but cross-horizon arbitration is purely the allocator's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Optional

import numpy as np
import structlog

from .posterior  import BetaPosterior
from .repository import AllocatorRepository

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AllocatorDecision:
    """Outcome of a single ThompsonAllocator.choose() call.

    chosen_horizon : str or None
        Horizon picked, or None if nothing was confirmed / all rejected.
    samples : mapping
        Per-horizon Thompson sample (for audit & dashboard).
    rejected_by : str or None
        Reason set when chosen_horizon is None ("no_confirmed", etc.).
    """
    chosen_horizon: Optional[str]
    samples:        Mapping[str, float]
    rejected_by:    Optional[str]


class ThompsonAllocator:
    """Bandit-style horizon selector.

    Parameters
    ----------
    repo : AllocatorRepository
        Persistent state store; load() is cached, so the hot-path read is O(1).
    rng : numpy.random.Generator
        Seedable RNG.  Use np.random.default_rng(seed) for tests.
    kafka_producer : object, optional
        Async producer with ``send_and_wait(topic, key, value)``.  Optional
        because tests can omit it.
    decision_topic : str
        Kafka topic for AllocatorDecisionEvent.
    """

    def __init__(
        self,
        repo:           AllocatorRepository,
        rng:            np.random.Generator,
        kafka_producer: Any = None,
        decision_topic: str = "los_ojos.allocator.decisions",
    ) -> None:
        self._repo     = repo
        self._rng      = rng
        self._producer = kafka_producer
        self._topic    = decision_topic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def choose(
        self,
        symbol:             str,
        direction:          int,
        confirmed_horizons: Mapping[str, Any],
        ts:                 datetime,
    ) -> AllocatorDecision:
        """Pick the horizon that executes.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. "AAPL", "BTCUSDT").
        direction : int
            -1 = short, 0 = flat (should not happen here), +1 = long.
        confirmed_horizons : mapping
            {horizon_name: signal_payload} — signals that already passed
            MultiFactorConfirmation.  Values are opaque to this class.
        ts : datetime
            Decision wall-clock (UTC).  Used uniformly for decay across
            horizons so samples are comparable.

        Returns
        -------
        AllocatorDecision
        """
        if not confirmed_horizons:
            decision = AllocatorDecision(
                chosen_horizon=None, samples={}, rejected_by="no_confirmed",
            )
            await self._publish_decision(symbol, direction, decision)
            return decision

        samples: dict[str, float] = {}
        for horizon in confirmed_horizons:
            posterior: BetaPosterior = await self._repo.load(horizon)
            samples[horizon] = posterior.sample(ts, self._rng)

        chosen = max(samples, key=lambda h: samples[h])
        decision = AllocatorDecision(
            chosen_horizon=chosen, samples=samples, rejected_by=None,
        )
        await self._publish_decision(symbol, direction, decision)
        return decision

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _publish_decision(
        self,
        symbol:    str,
        direction: int,
        decision:  AllocatorDecision,
    ) -> None:
        """Fire-and-forget Kafka emit; never raises."""
        if self._producer is None:
            return
        import json
        from datetime import timezone
        payload = {
            "event_type":     "AllocatorDecisionEvent",
            "symbol":         symbol,
            "direction":      direction,
            "chosen_horizon": decision.chosen_horizon,
            "samples": {
                h: str(Decimal(str(v))) for h, v in decision.samples.items()
            },
            "rejected_by":    decision.rejected_by,
            "ts":             datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            await self._producer.send_and_wait(
                self._topic,
                value=json.dumps(payload).encode(),
                key=symbol.encode(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "thompson.publish_decision.error",
                symbol=symbol, error=str(exc),
            )
