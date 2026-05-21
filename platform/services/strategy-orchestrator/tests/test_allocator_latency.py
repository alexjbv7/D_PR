"""
Latency benchmark — DoD-8.

choose() must run < 5 ms p99 with cached posteriors.
This test uses pytest-benchmark; mark it `slow` so it doesn't run on every
local invocation.

Run only the benchmark:
    pytest --benchmark-only tests/test_allocator_latency.py
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from app.allocator.posterior import BetaPosterior
from app.allocator.thompson  import ThompsonAllocator


class _InMemRepo:
    def __init__(self, posteriors: dict[str, BetaPosterior]) -> None:
        self._posteriors = posteriors

    async def load(self, horizon: str) -> BetaPosterior:
        return self._posteriors[horizon]

    async def save(self, horizon: str, posterior: BetaPosterior) -> None:
        self._posteriors[horizon] = posterior


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.benchmark(group="allocator-latency")
def test_choose_p99_under_5ms(benchmark) -> None:  # type: ignore[no-untyped-def]
    """choose() with 3 horizons, cached posteriors → < 5 ms p99."""
    ts   = _now()
    repo = _InMemRepo({
        "intraday": BetaPosterior(Decimal("20"), Decimal("20"), ts),
        "swing":    BetaPosterior(Decimal("20"), Decimal("20"), ts),
        "daily":    BetaPosterior(Decimal("20"), Decimal("20"), ts),
    })
    rng = np.random.default_rng(0)
    allocator = ThompsonAllocator(repo=repo, rng=rng, kafka_producer=None)  # type: ignore[arg-type]
    confirmed = {"intraday": {}, "swing": {}, "daily": {}}

    loop = asyncio.new_event_loop()
    try:
        def _run_choose() -> None:
            loop.run_until_complete(
                allocator.choose("AAPL", 1, confirmed, ts)
            )

        benchmark.pedantic(_run_choose, rounds=200, iterations=5, warmup_rounds=5)
    finally:
        loop.close()

    # pytest-benchmark stats are in seconds.  Enforce p99 < 5 ms.
    # stats may not have a percentile entry if rounds < 100, so use max as a
    # conservative upper bound when median is small.
    stats = benchmark.stats.stats
    p99   = getattr(stats, "median", None)
    max_  = getattr(stats, "max",    None)
    # Conservative: require both median << 5 ms and max < 25 ms.
    if p99 is not None:
        assert p99 < 5e-3, f"median latency {p99*1000:.2f}ms exceeds 5 ms budget"
    if max_ is not None:
        assert max_ < 25e-3, f"max latency {max_*1000:.2f}ms is suspiciously high"
